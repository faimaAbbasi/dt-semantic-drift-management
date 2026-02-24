const fs = require("fs");
const csv = require("csv-parser");
const stringSimilarity = require("string-similarity");
const natural = require("natural");
const path = require("path");
const axios = require("axios");
const moment = require("moment");
const unzipper = require("unzipper");

const TfIdf = natural.TfIdf;

// ************************ General Entities ************************

const generalEntities = {
  "Building": ["building", "site", "location"],
  "Room": ["room", "area", "zone"],
  "Controller": ["controller", "device", "system", "level"],
  "TemperatureSensor": ["temperature", "unit"],
  "HumiditySensor": ["humidity", "humidity_unit"],
  "PressureSensor": ["pressure", "pressure_unit"],
  "Alarm": ["alarm", "status", "alert", "active"]
};

// ************************ Inference ************************

const SIMILARITY_THRESHOLD = 0.4;

const predictEntity = (attributeName) => {
  attributeName = attributeName.toLowerCase();
  for (const [entity, keywords] of Object.entries(generalEntities)) {
    const matches = stringSimilarity.findBestMatch(attributeName, keywords);
    if (matches.bestMatch.rating > SIMILARITY_THRESHOLD) {
      return entity;
    }
  }
  return "Unknown";
};

const processCSV = (filePath, callback) => {
  const attributeToEntityMap = {};
  let columnsExtracted = false;

  fs.createReadStream(filePath)
    .pipe(csv())
    .on("data", (row) => {
      if (!columnsExtracted) {
        const headers = Object.keys(row).filter(h => h !== "id" && h !== "date_time");
        headers.forEach(col => {
          attributeToEntityMap[col] = predictEntity(col);
        });
        columnsExtracted = true;
      }
    })
    .on("end", () => callback(null, attributeToEntityMap))
    .on("error", (error) => callback(error, null));
};

const recommendEntitiesForUnknowns = async (unknownAttributes) => {
  if (unknownAttributes.length === 0) return [];
  const { kmeans } = await import("ml-kmeans");
  const tfidf = new TfIdf();
  unknownAttributes.forEach(attr => tfidf.addDocument(attr));

  const vectors = unknownAttributes.map((_, i) => {
    const vec = [];
    tfidf.listTerms(i).forEach(term => vec.push(term.tfidf));
    return vec.length > 0 ? vec : [0];
  });

  const k = Math.min(Object.keys(generalEntities).length, unknownAttributes.length);
  const kmResult = kmeans(vectors, k);

  const clusters = {};
  kmResult.clusters.forEach((clusterIndex, i) => {
    if (!clusters[clusterIndex]) clusters[clusterIndex] = [];
    clusters[clusterIndex].push(unknownAttributes[i]);
  });

  const clusterEntityLabels = {};
  for (const [clusterIndex, attrs] of Object.entries(clusters)) {
    const combinedText = attrs.join(" ").toLowerCase();

    let bestEntity = "Unknown";
    let bestScore = 0;

    for (const [entity, keywords] of Object.entries(generalEntities)) {
      const matches = stringSimilarity.findBestMatch(combinedText, keywords);
      if (matches.bestMatch.rating > bestScore) {
        bestScore = matches.bestMatch.rating;
        bestEntity = entity;
      }
    }
    clusterEntityLabels[clusterIndex] = bestEntity;
  }

  return unknownAttributes.map((attr, i) => {
    const clusterIndex = kmResult.clusters[i];
    return { attribute: attr, recommendedEntity: clusterEntityLabels[clusterIndex] };
  });
};

// ************************ Type Inference ************************

const inferDataType = (value) => {
  if (!value || value === "") return "isString";

  const val = value.toString().toLowerCase().trim();

  if (["true", "false", "yes", "no", "on", "off"].includes(val)) return "isBoolean";
  if (!isNaN(Number(val)) && isFinite(val)) return "isNumber";

  const dateFormats = ["YYYY-MM-DD", "MM/DD/YYYY", "DD-MM-YYYY", moment.ISO_8601];
  if (moment(val, dateFormats, true).isValid()) return "isDate";

  return "isString";
};

const inferAttributeTypes = (filePath, attributes) => {
  return new Promise((resolve, reject) => {
    const sampleSize = 5;
    const samples = {};
    attributes.forEach(attr => samples[attr] = []);

    fs.createReadStream(filePath)
      .pipe(csv())
      .on("data", (row) => {
        attributes.forEach(attr => {
          if (samples[attr].length < sampleSize && row[attr]) {
            samples[attr].push(row[attr]);
          }
        });
      })
      .on("end", () => {
        const inferredTypes = {};
        for (const [attr, values] of Object.entries(samples)) {
          const typeCounts = { isString: 0, isNumber: 0, isBoolean: 0, isDate: 0 };
          values.forEach(val => {
            const inferred = inferDataType(val);
            typeCounts[inferred]++;
          });
          const bestType = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])[0][0];
          inferredTypes[attr] = bestType;
        }
        resolve(inferredTypes);
      })
      .on("error", (err) => reject(err));
  });
};

// ************************ Ollama Description ************************

async function generateDescription(promptText) {
  try {
    const response = await axios.post("http://localhost:11434/api/generate", {
      model: "llama3",
      prompt: `Provide a concise, very brief one-sentence description for: ${promptText}. Please donot add any extra text in response`,
      stream: false
    });

    return response.data.response.trim().replace(/\n/g, " ");
  } catch (err) {
    console.error("Ollama description generation error:", err.message);
    return "No description available.";
  }
}

// ************************ Save New Entities ************************

const saveNewEntitiesWithAttributes = async (newAttributesWithNewEntities, newDataFile, outputFile) => {
  const attributes = newAttributesWithNewEntities.map(obj => obj.attribute);
  const inferredTypes = await inferAttributeTypes(newDataFile, attributes);

  const entityMap = {};
  for (const { attribute, entity } of newAttributesWithNewEntities) {
    if (!entityMap[entity]) {
      entityMap[entity] = { attributes: [], attrNames: [] };
    }
    const type = inferredTypes[attribute] || "isString";
    entityMap[entity].attributes.push(`${attribute}: ${type}`);
    entityMap[entity].attrNames.push(attribute);
  }

  const formattedOutput = [];

  for (const [entityName, { attributes }] of Object.entries(entityMap)) {
    const description = await generateDescription(`Entity "${entityName}" in a building and air quality sensor automation system`);
    formattedOutput.push({
      name: entityName,
      primitive_attributes: attributes,
      description
    });
  }

  const outputPath = path.join(__dirname, '..', 'output', outputFile);
  fs.writeFileSync(outputPath, JSON.stringify(formattedOutput, null, 2), "utf-8");
  console.log(`\n Saved new entities with inferred types & descriptions to: ${outputPath}\n`);
};

// ************************ Structural Drift ************************

const identify_structural_drift = (historicalFile, newFile, callback) => {
  processCSV(historicalFile, (err, historicalMap) => {
    if (err) return callback(err, null);

    processCSV(newFile, async (err, newMap) => {
      if (err) return callback(err, null);

      console.log(`\n Step # 01 --> Identifying Data Drift...\n`);

      const newAttributesWithExistingEntities = [];
      const newAttributesWithNewEntities = [];
      const unknownAttributes = [];

      Object.entries(newMap).forEach(([attribute, entity]) => {
        if (!(attribute in historicalMap)) {
          if (entity === "Unknown") {
            unknownAttributes.push(attribute);
          } else if (Object.values(historicalMap).includes(entity)) {
            newAttributesWithExistingEntities.push({ attribute, entity });
          } else {
            newAttributesWithNewEntities.push({ attribute, entity });
          }
        }
      });

      const totalNew =
        newAttributesWithExistingEntities.length +
        newAttributesWithNewEntities.length +
        unknownAttributes.length;

      const driftScore = totalNew === 0 ? 0 :
        (newAttributesWithNewEntities.length +
          unknownAttributes.length +
          0.5 * newAttributesWithExistingEntities.length) / totalNew;

      const recommendations = await recommendEntitiesForUnknowns(unknownAttributes);

      console.log("  - New Attributes with Existing Entities: \n", newAttributesWithExistingEntities);
      console.log("\n  - New Attributes with New Entities: \n", newAttributesWithNewEntities);
      console.log("\n  - Drift Score:", driftScore.toFixed(2));

      //await saveNewEntitiesWithAttributes(newAttributesWithNewEntities, newFile, "new_entities.json");

      callback(null, {
        newAttributesWithExistingEntities,
        newAttributesWithNewEntities,
        unknownAttributes,
        recommendations,
        driftScore: Number(driftScore.toFixed(2))
      });
    });
  });
};

module.exports = { identify_structural_drift, generateDescription };
