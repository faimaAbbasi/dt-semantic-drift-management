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
const { MM, M, getInstances, getElement, addElement, doesEntityExist, save_metamodel, add_element} = require('../model-metamodel-layer/model-metamodel.js');
const { identify_structural_drift, generateDescription} = require('./structural-drift.js');
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

async function update_metamodel(instance, classType, entity){
  const cls = getElement(entity)
  if (cls == null) {
    //console.log('cls is null or undefined')
    const res = discoverer.archetypalDiscovery(entity,instance,classType)
    discoverer.updateClass(classType,res)
    const desc = await generateDescription(`Entity "${entity}" in a building and air quality sensor automation system`)
    add_element(classType, entity, desc)
  } else {
    console.log('cls exists', cls)
  }
}

function update_model(row, attributes) {
    console.log(attributes)
    const uniqueEntities = [...new Set(attributes.map(item => item.entity))][0];
    //console.log(uniqueEntities);
    const newInstance = new JSMF.Thing();
    newInstance.value=parseFloat(row.pressure);
    newInstance.unit=row.pressure_unit;
    console.log("\n Newly Created Instance: \n")
    console.log(newInstance); 
    classType=newInstance.conformsTo();
    //console.log(classType)

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

    const alarm = MM.Alarm.newInstance();
    alarm.isActive = row.IsActive.toLowerCase() === "true";

    building.rooms = room;
    room.controllers = controller;
    controller.alarms = alarm;
    controller.tempsensors = tempSensor;
    controller.humsensors = humSensor;
    controller.proximitySensor = proxSensor;

    const instances = [building, room, controller, tempSensor, humSensor, proxSensor, alarm, newInstance];
    M.setModellingElements(instances); 
    console.log(`\n Step # 04 --> Updating Metamodel Layer by adding new Entities!!`);
    update_metamodel(newInstance, classType, uniqueEntities)
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
      if (result.newAttributesWithNewEntities.length > 0) {
        console.log("\n Step # 02 --> Structural Drift Detected ");

        result.newAttributesWithNewEntities.forEach(({ attribute, entity }) => {
          console.log(`\n Attribute: ${attribute} -> Entity: ${entity}`);
        });
        let rowCount = 0;
        fs.createReadStream(newCSV)
          .pipe(csv())
          .on("data", (row) => {
            if (rowCount < 1) {
              console.log(`\n Step # 03 --> Updating Model Layer by Re-instantiating the Entities!!`);
              update_model(row, result.newAttributesWithNewEntities);
              rowCount++;
              }
            })
        .on("error", err => console.error("CSV read error:", err));
      } else {
        console.log("No Drift Detected!");
      }
    });
  } catch (err) {
    console.error("Fatal Error:", err);
  }
})();