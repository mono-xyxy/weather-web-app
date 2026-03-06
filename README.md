# 🌦️ AI Predictive Weather Forecaster & Synthetic Data Engine

An end-to-end Big Data & Machine Learning pipeline that programmatically generates synthetic climate datasets at scale, trains a suite of AI models entirely on that data, and validates the model's accuracy against the **Real World** through a highly dynamic UI inspired by modern Dribbble shots.

---

## 🚀 Features

* **Distributed Synthetic Generation**: Utilizes PySpark to algorithmically generate millions of continuous, logically sound weather records.
* **Ensemble Machine Learning**: Implements `scikit-learn`'s Random Forest algorithms (Classification & Regression) built on top of the generated dataset locally to perform weather forecasting.
* **Live Ground Truth Verification**: Integrates directly with the `Open-Meteo REST API` to cross-verify the synthetic-trained AI against accurate real-world data across the globe.
* **Geo-Temporal Dynamic UI**: Features a bespoke Vanilla CSS/HTML Glassmorphism dashboard that intelligently changes its entire core color palette based on the target city's latitude and the calculated season of the requested prediction!

---

## 🛠️ Architecture

1. **`synthetic_weather_pipeline.py` (Big Data Generation)**
   * Built for Hadoop/Spark Ecosystems.
   * Leverages `spark.range()` to spawn partitions logically formatted with weather bounds mathematically using normal distributions (`randn()`).
   * Scalable from 500,000 local test rows directly up to Billions of rows on HDFS via YARN orchestration.

2. **`ai_weather_app.py` (AI Neural Interface)**
   * A Python `Flask` application serving as the UI and orchestration backend.
   * Bootstraps 5 Random Forest AI models straight into memory on startup.
   * Exposes a fast interface that computes deviation matrices (`Delta Difference Analyzer`) to measure how well the AI guesses reality.

---

## 💻 Tech Stack

* **Core Pipeline**: Apache Spark, PySpark, Pandas
* **AI Algorithms**: Scikit-Learn (*Random Forest Regressor/Classifier*)
* **Web Serving Backend**: Flask
* **Frontend Design**: HTML5, Vanilla CSS3, JavaScript
* **Live Data Engine**: Open-Meteo API

---

## ⚙️ Installation & Setup

1. **Install Requirements**
   Ensure Python 3.8+ is installed. Then run:
   ```bash
   pip install pyspark pandas pyarrow scikit-learn flask requests
   ```

2. **Step 1: Generate the Dataset**
   First, synthesize the weather dataset by running the PySpark pipeline. 
   *(Note: This creates a 500k row `.csv` locally. For Billions of rows natively, you must execute within a YARN cluster mapping to HDFS directories.)*
   ```bash
   python synthetic_weather_pipeline.py
   ```
   **Output**: `synthetic_weather_output_million.csv` will be generated in your directory.

3. **Step 2: Start the AI Dashboard**
   Launch the web server. This will import the generated data, train the Models, and expose the UI.
   ```bash
   python ai_weather_app.py
   ```
   
4. **Step 3: Access the Interface**
   Navigate to your local browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🎨 Theme Mechanics

The User Interface isn't static! It includes an **Autochromatic Engine** that interprets what season it currently is at the target location:

* **Location Checking**: Maps negative latitudes (Southern Hemisphere like Sydney) vs northern coordinates (Northern Hemisphere like London).
* **Seasonal Palettes**:
  * 🧊 **Winter**: Transitions to icy blues and frosted accents.
  * 🌸 **Spring**: Shifts toward bright pastel greens.
  * ☀️ **Summer**: Activates a warm peach-sunset scheme.
  * 🍂 **Autumn**: Adopts a sleek gray-beige minimalist look.

---

*Architected by Advanced Agentic UI Synthesis.*
