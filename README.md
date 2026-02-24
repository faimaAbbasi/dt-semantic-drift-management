## Semantic Drift Management in Digital Twin 

This repository demonstrates systematic approach for semantic dift management in **a multi-layered, model-driven digital twin (DT)**, adopting a three-step approach: (i) identification, (ii) evaluation and (iii) propagation. Submitted to a conference, it provides source code, datasets, and documentation for the air quality use case set up.

It also includes reproducible experiments for:

- We employ a bottom-up approach with a case-based generalization strategy to systematically analyze and manage different variants of semantic drift in model-driven, multi-layered DTs, emerging from data. 
- We provide methods to identify different variants of semantic drift and propagate changes to correct semantic drift across abstraction layers, exploiting features and constructs of heterogeneous model alignment approach (see **[details](https://arxiv.org/abs/2512.15281)**). 
- We provide different methods to quantify or evaluate variants of semantic drift across each abstraction layer.

### Air Quality Usecase 

Air quality in a room is closely related to the concentration of occupants and the potential spread of viruses, making it important for organizations to monitor and manage it to ensure safety and productivity. Key metrics include CO₂ levels, humidity, and temperature, and exceeding defined thresholds can negatively impact people in the space.

In the baseline system, a **Building** consists of multiple **Rooms**, each equipped with **Controllers** connected to **Sensors** (to monitor air quality) and **Alarms** (to notify occupants). Detecting threshold violations allows the system to take appropriate measures to maintain healthy indoor air quality. Airquality use case is adapted from **[here](https://github.com/derlehner/IndoorAirQuality_DigitalTwin_Exemplar)**. We integrated and mapped an **[air quality dataset](https://doi.org/10.17605/OSF.IO/BAEW7)** of 2.5 million one-minute samples from 10 sensors to the DT structure for studying heterogeneous model alignment and semantic consistency across all layers. We consider two semantic drift scenarios from airquality usecase stated as follows:

- **Drift Case 1 - Load Level:** A new functionality is added in a controller to measure how intensely a controller is operating based on environmental and occupancy conditions. It helps in performance tracking, balancing, and predictive maintenance. It is calculated using insights from the other sensors, i.e., proximity, humidity and temperature, linked with controller. A sample load-level attribute calculation is shown below, where weights like 1.5 and 0.5 represent the relative impact of environmental factors on the controller's workload. 

```bash
score = proximity value + (1.5 * (current temperature - optimal setpoint)) + (0.5 * (current humidity - optimal setpoint))
load-level = "Low" if score < 10 else "Moderate" if score <= 20 else "High"
```

- **Drift Case 2 - Pressure Sensor:** A new sensor type is added under the \emph{Controller} through \emph{pressensors} association. It holds value and unit properties, enabling the system to monitor air pressure per room. This supports enhanced air quality management by improving sensor calibration, detecting blockages, or managing airflow.


### Project Hierarchy
Below is folder structure for project:
```
dt-model-alignment/
│
├── data-layer/                          # DT Data Layer for Air Quality Usecase Setup (Historical and New Data)
├── evaluation/                          # Experiment related to Evaluation of SSM-OM on OAEI Testcases
├── model-metamodel-layer/               # DT Model and Metamodel Layer for Air Quality Usecase Setup
│   ├── metamodel-flexibility.js              # Change Adaptation
│   └── model-metamodel.js                    # DT Model-Metamodel layer 
├── node modules                           
├── ontology-layer/                      # DT ontology Layer for Air Quality Usecase Setup                             
│   └── metamodel-ontology-matching.py        # Semantic and Structure-aware Metamodel Ontology Matching (SSM-OM)
├── output/                              # Results in .csv files for all experiments for matching tasks
├── semantic-drift/                      # Semantic Drift Management in multi-layered, model-driven DT
│   └── structural-drift.js                   # Structural drift identification and evaluation
│   └── uc1-knowledge-conceptual-drift.py     # Knowledge Conceptual drift evaluation and propagation for Drift Scenario 1 
│   └── uc1-technical-conceptual-drift.py     # Technical Conceptual drift evaluation and propagation for Drift Scenario 1 
│   └── uc2-technical-conceptual-drift.py     # Technical Conceptual drift evaluation and propagation for Drift Scenario 2 
├── testcases/                           # OAEI Testcases Used for Evaluation
├── .gitattributes 
└── README.md
└── package.json
└── package-lock.json
└── requirement.txt
```
### Installation
Follow are steps to set up and run this project locally.

#### 1. Clone Repository
```bash
git clone https://github.com/faimaAbbasi/dt-model-alignment.git
cd dt-model-alignment
```

#### 2. Virtual Environment
In order not to conflict with already installed packages on your machine, it is recommended to use a virtual environment (e.g. **[venv](https://docs.python.org/3/library/venv.html)**) for reproducibility. 
```bash
python -m venv venv
```
Activate virtual environment on Windows:
```bash
venv\Scripts\activate
```
On Linux:
```bash
source venv/bin/activate
```

#### 3. Install Required Packages
We recommend using ```Python 3.12.2```. Make sure your ```requirements.txt``` file is in the project’s root folder, then run:
```bash
pip install -r requirement.txt
```
For npm
```npm
npm install
```
### Usage and Examples
Follow these steps to set-up digital twin and manage semantic drift systematically.

#### 1. Digital Twin Set-up
To set up the DT model-metamodel layer for the air quality use case with data, run the following file:
```bash
cd model-metamodel-layer
node model-metamodel.js      
```
Below is a simple example illustrating the ```Controller``` concept in the air quality metamodel using JSMF constructs. See <a href='model-metamodel-layer/model-metamodel.js'> model-metamodel.js </a> for details:
```python
const Model = JSMF.Model;
const Class = JSMF.Class;
const airquality = new Model("Airquality-MM");

const Controller = Class.newInstance("Controller");
Controller.addAttribute("id", String);
Controller.setDescription("A device managing sensors and alarms.")

Controller.setReference("alarms", Alarm, JSMF.Cardinality.any);
Controller.setReference("tempsensors", TempSensor, JSMF.Cardinality.any);
airquality.setModellingElements([Building, Room, TempSensor, Proximity, HumiditySensor, PressureSensor, Controller, Alarm]);     
```
The example below illustrates bottom-up change management in the metamodel. Note that changes first occur in the data, which then trigger updates in the model and metamodel layers in DT. Details are in **<a href='model-metamodel-layer/metamodel-flexibility.js'> metamodel-flexibility.js </a>**
```python
airquality.Controller.setFlexible(true)

const controller1=airquality.Controller.newInstance()
controller1.id=row.Controller_id
controller1.load_level=row.load_level # new attribute appeared in data
airquality_model.setModellingElements([controller1]);

const res = discoverer.archetypalDiscovery("Controller", controller1, airquality.Controller)
discoverer.updateClass(airquality.Controller,res)    
```
To set up DT ontology layer and map metamodel to traget **[Brick](https://brickschema.org/)** ontology, for air quality usecase using **SSM-OM**, execute the following file:
```bash
cd ontology-layer
python metamodel-ontology-matching.py     
```
To evaluate the performance of **SSM-OM** on different test cases from **[OAEI tracks](https://oaei.ontologymatching.org/)** tracks, run the scripts in the **[evaluation](evaluation)** folder for each test case. The test cases themselves are available in **[testcases](testcases)** folder. A simple execution example is shown below:
```bash
cd evaluation
python anatomy-track-ontology-mapping.py     
```

#### 2. Semantic Drift Management
To identify and evaluate structural drift from data, execute the following script.
```bash
cd semantic-drift
node structural-drift.js     
```
To evaluate and propagte technical conceptual drift for Drift Case 1, execute the following script.
```bash
cd semantic-drift
node uc1-technical-conceptual-drift.js     
```
To evaluate and propagte knowledge conceptual drift for Drift Case 1, execute the following script.
```bash
cd semantic-drift
node uc1-knowledge-conceptual-drift.py     
```
To evaluate and propagte technical conceptual drift for Drift Case 2, execute the following script.
```bash
cd semantic-drift
node uc2-technical-conceptual-drift.js     
```








