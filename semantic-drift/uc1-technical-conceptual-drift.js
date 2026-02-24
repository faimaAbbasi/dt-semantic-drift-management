const { describe, it } = require('mocha');
const fs = require("fs");
const path = require("path");
const unzipper = require("unzipper");
const csv = require("csv-parser");
const JSMF = require('jsmf-core'), 
      nav = require('jsmf-magellan'),
      check = require('jsmf-check/src/index.js'),
      core = require('jsmf-core');

const should = require('should');  
const discoverer = require('jsmf-core/src/indexdiscovery');
const { Cardinality } = require('jsmf-core/src/Cardinality')
const Model = JSMF.Model;
const Class = JSMF.Class;
const instances = [];


// ******************* Importing Metamodel, Utilities & Drift Metrics ***************************
const { MM, M, getInstances, getElement, addElement, doesEntityExist, save_metamodel} = require('../model-metamodel-layer/model-metamodel.js');
const { identify_structural_drift} = require('./structural-drift.js');
const historicalDataZip = "../data-layer/historicaldata.zip";
const newDataZip = "../data-layer/newdata.zip";


async function extractZip(zipPath) {
  const folder = path.dirname(zipPath);
  const zipName = path.basename(zipPath, ".zip");
  const outputCSV = path.join(folder, zipName + ".csv");

  // Delete existing output csv to avoid stale files
  if (fs.existsSync(outputCSV)) fs.unlinkSync(outputCSV);

  return new Promise((resolve, reject) => {
    fs.createReadStream(zipPath)
      .pipe(unzipper.Parse())
      .on("entry", entry => {
        const fileName = entry.path;

        if (fileName.toLowerCase().endsWith(".csv")) {
          // Extract this CSV file into outputCSV
          entry.pipe(fs.createWriteStream(outputCSV))
            .on("finish", () => resolve(outputCSV));
        } else {
          entry.autodrain();
        }
      })
      .on("error", reject)
      .on("close", () => resolve(outputCSV));
  });
}


// ******************* Identifying Technical Conceptual Drift ***************************
function identify_technical_drift(instance, classType){
    const definedAttributes = Object.keys(classType.attributes);
    console.log("\n Defined Metamodel Attributes:", definedAttributes);
    const instanceAttributes = Object.keys(instance);
    console.log("\n Defined Instance Attributes:", instanceAttributes);
    const instancePrimitiveAttrs = [];
    const instanceReferences = [];

    instanceAttributes.forEach(attr => {
        let value = instance[attr];  
        let type = typeof value;  
        if (type === "string" || type === "number" || type === "boolean") {
            instancePrimitiveAttrs.push(attr);
        } else if (type === "object" && value !== null) {
            instanceReferences.push(attr);
        }
    });

    console.log("\n Primitive Instance Attributes:", instancePrimitiveAttrs);
    console.log("\n Reference Instance Attributes:", instanceReferences);
    const missingAttributes = definedAttributes.filter(attr => !instancePrimitiveAttrs.includes(attr));
    const extraAttributes = instancePrimitiveAttrs.filter(attr => !definedAttributes.includes(attr));

    const definedSet  = new Set( definedAttributes );
    const instanceSet = new Set( instancePrimitiveAttrs );

    let intersectionSize = 0;
    for (const attr of definedSet) {
        if (instanceSet.has(attr)) intersectionSize++;
    }

    const unionSize = new Set([ ...definedSet, ...instanceSet ]).size;

    const jaccardSimilarity = unionSize > 0
        ? intersectionSize / unionSize
        : 1; 
    const jaccardDrift = 1 - jaccardSimilarity;

    console.log(`\n Missing Attributes:`, missingAttributes);
    console.log(` Extra Attributes:`, extraAttributes);

    console.log(`\n Jaccard Similarity: ${jaccardSimilarity.toFixed(3)}`);
    console.log(` Jaccard‐based Drift Score (1 – similarity): ${jaccardDrift.toFixed(3)}`);


    if (missingAttributes.length === 0 && extraAttributes.length === 0) {
        console.log(`\n Class instance validation successful.`);
        isValid=true;
    } else {
        console.log(`\n Class instance validation failed!`);
        if (missingAttributes.length > 0) console.log("\n Technical Drift Detected (Missing Attributes Identified):", missingAttributes);
        if (extraAttributes.length > 0) console.log("\n Technical Drift Detected (Extra Attributes Identified):", extraAttributes);
        isValid=false;
    }
    return isValid

}

// ******************* Updating Metamodel Layer using ArchetypalDiscovery ***************************
function updateMetamodel(isValid, instance, classType, entity){
    if (!isValid) {
        const res = discoverer.archetypalDiscovery(entity,instance,classType)
        discoverer.updateClass(classType,res)
        console.log("\n", entity,"Class After Updation:",classType)
        save_metamodel('metamodel-v2.json')

    } else {
        console.log("No update required.");
    }
}


// ******************* Updating Model Layer using Flexible Metamodel Approach ***************************
function updateModel(row, attribute, entity) {
    let flexibleInstance = null;
    let flexibleClass = null;
    const classMap = {
        Building: MM.Building,
        Room: MM.Room,
        Controller: MM.Controller,
        TempSensor: MM.TempSensor,
        HumSensor: MM.HumSensor,
        Alarm: MM.Alarm,
    };
    
    if (classMap[entity]) {
        classMap[entity].setFlexible(true);
        flexibleClass = classMap[entity];
    }

    const building = MM.Building.newInstance();
    building.id = row.Building_id;

    const room = MM.Room.newInstance();
    room.id = row.Room_id;

    const tempSensor = MM.TempSensor.newInstance();
    tempSensor.value = parseFloat(row.temperature);
    tempSensor.unit = row.temperature_unit;

    const humSensor = MM.HumSensor.newInstance();
    humSensor.value = parseFloat(row.humidity);
    humSensor.unit = row.humidity_unit;

    const proxSensor = MM.ProxSensor.newInstance();
    proxSensor.value = parseFloat(row.proximity);
    proxSensor.unit = row.proximity_unit;

    const controller = MM.Controller.newInstance();
    controller.id = row.Controller_id;
    controller[attribute] = row.load_level;


    const alarm = MM.Alarm.newInstance();
    alarm.isActive = row.IsActive.toLowerCase() === "true";

    building.rooms=room
    room.controllers=controller
    controller.alarms=alarm
    controller.tempsensors=tempSensor
    controller.humsensors=humSensor
    controller.proximitySensor=proxSensor

    const instances = [building, room, controller, tempSensor, humSensor, proxSensor, alarm];
    M.setModellingElements(instances);
    flexibleInstance = instances.find(inst => flexibleClass && inst instanceof flexibleClass);
    if (flexibleInstance && flexibleClass) {
        // Step # 04 --> Identify Technical Drift (Bottom Up Approach)
        console.log(`\n Step # 04 --> Identifying Technical Conceptual Drift !!!\n`);
        const valid = identify_technical_drift(flexibleInstance, flexibleClass);

        // Step # 05 --> Update Metamodel
        console.log(`\n Step # 05 --> Updating Metamodel Layer !!!\n`);
        updateMetamodel(valid, flexibleInstance, flexibleClass, entity);
    } 

}

// ******************* Use-Case # 01 ***************************

// Load Level Attribute Starts to appear in new data for Controller Entity

// Step # 02 --> Identify Structural Drift
(async function main() {
  try {
    const historicalCSV = await extractZip(historicalDataZip);
    const newCSV = await extractZip(newDataZip);

    console.log("Historical CSV:", historicalCSV);
    console.log("New CSV:", newCSV);
    identify_structural_drift(historicalCSV, newCSV, (err, result) => {
      if (err) {
        console.error("Error processing files:", err);
        return;
      }

      if (result.newAttributesWithExistingEntities.length > 0) {
        console.log("\n Step # 02 --> Structural Drift Detected ");

        result.newAttributesWithExistingEntities.forEach(({ attribute, entity }) => {
          console.log(`\n Attribute: ${attribute} -> Entity: ${entity}`);

          let rowCount = 0;
          fs.createReadStream(newCSV)
            .pipe(csv())
            .on("data", (row) => {
              if (rowCount < 1) {
                console.log(`\n Step # 03 --> Updating Model Layer by Re-instantiating the Entities!!`);
                updateModel(row, attribute, entity);
                rowCount++;
              }
            })
            .on("error", err => console.error("CSV read error:", err));
        });

      } else {
        console.log("No Drift Detected!");
      }
    });

  } catch (err) {
    console.error("Fatal Error:", err);
  }
})();