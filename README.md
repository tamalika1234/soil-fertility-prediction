# Soil Fertility Predictor
Predict soil fertility class based on soil nutrient values using machine learning.

## Project Overview
This project predicts the **fertility class of soil** (Low, Medium, High) based on nutrient values such as Nitrogen (N), Phosphorus (P), Potassium (K), pH, EC, and other trace elements.  
The goal is to provide a simple interface for farmers, researchers, and agronomists to quickly assess soil fertility and make informed crop management decisions.

The project uses a **Random Forest Classifier** trained on a soil nutrient dataset for accurate predictions.

## Features Used
- **N** – Nitrogen content
- **P** – Phosphorus content
- **K** – Potassium content
- **pH** – Soil pH level
- **EC** – Electrical Conductivity
- **OC** – Organic Carbon
- **S** – Sulfur content
- **Zn** – Zinc content
- **Fe** – Iron content
- **Cu** – Copper content
- **Mn** – Manganese content
- **B** – Boron content

## Folder Structure

soil-quality-project/
│
├── dataset1.csv # Dataset with soil nutrient values and fertility labels
├── soil_fertility_model.py # Script to train the Random Forest model
├── app.py # Streamlit deployment script
├── soil_fertility_model.pkl # Saved trained model
├── scaler.pkl # Saved StandardScaler
├── label_encoder.pkl # Saved LabelEncoder for fertility classes
├── fertility_mapping.pkl # Numeric -> class name mapping
├── README.md # Project documentation

##How to Run

1. **Clone the repository:**

git clone https://github.com/YourUsername/soil-quality-project.git
cd soil-quality-project

2. ** Create a virtual environment and install dependencies:**

python -m venv venv
venv\Scripts\activate       # Windows
# OR
source venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
3. **Run the Streamlit app:**

streamlit run app.py

4. **Use the App:**

Enter soil nutrient values in the input fields.

Click  Predict Fertility.

Get predictions like:

vbnet
Copy code
Predicted Soil Fertility Class: Medium Fertility 
Notes
All .pkl model files are included and fully usable, even if GitHub cannot preview them.

The app handles missing values and automatically scales numeric features.

Emoji icons indicate fertility levels:

Low 

Medium 

High 

Author
Tamalika Ghosh – B.Tech ECE | Aspiring Full-Stack & Software Developer

