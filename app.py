import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------------
# Load model, scaler, and mapping
# ------------------------------
try:
    model = joblib.load("soil_fertility_model.pkl")
    scaler = joblib.load("scaler.pkl")
    fertility_mapping = joblib.load("fertility_mapping.pkl")
    le = joblib.load("label_encoder.pkl")
except FileNotFoundError as e:
    st.error(f"Model files not found. Please run soil_fertility_model.py first: {e}")
    st.stop()

# ------------------------------
# Load dataset to get min/max for input clipping
# ------------------------------
try:
    data = pd.read_csv("dataset1.csv")
    numeric_cols = ['N','P','K','pH','EC','OC','S','Zn','Fe','Cu','Mn','B']
    data_min = data[numeric_cols].min().values
    data_max = data[numeric_cols].max().values
    
    # Override EC range based on domain knowledge
    ec_index = numeric_cols.index('EC')
    data_min[ec_index] = 0.10
    data_max[ec_index] = 0.95
    
except FileNotFoundError:
    st.error("Dataset file (dataset1.csv) not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Define optimal ranges for each nutrient (example values - adjust based on your domain knowledge)
optimal_ranges = {
    'N': (50, 100),
    'P': (20, 40),
    'K': (40, 80),
    'pH': (6.0, 7.5),
    'EC': (0.4, 0.8),
    'OC': (0.6, 1.2),
    'S': (15, 30),
    'Zn': (1.0, 2.5),
    'Fe': (2.5, 4.5),
    'Cu': (0.6, 1.2),
    'Mn': (1.5, 2.8),
    'B': (0.7, 1.5)
}

# ------------------------------
# Streamlit app title
# ------------------------------
st.title("🌱 Soil Fertility Prediction")
st.write("Enter soil nutrient values to predict fertility class.")

# ------------------------------
# Input fields
# ------------------------------
st.subheader("Enter Soil Nutrient Values")
inputs = {}
default_values = {
    "N": 75.0, "P": 30.0, "K": 60.0,
    "pH": 6.8, "EC": 0.6, "OC": 0.9,
    "S": 22.0, "Zn": 1.8, "Fe": 3.5,
    "Cu": 0.9, "Mn": 2.0, "B": 1.1
}

col1, col2, col3 = st.columns(3)

for i, col in enumerate(numeric_cols):
    # Organize inputs in columns for better layout
    if i % 3 == 0:
        current_col = col1
    elif i % 3 == 1:
        current_col = col2
    else:
        current_col = col3
        
    with current_col:
        default_val_clipped = np.clip(default_values[col], data_min[i], data_max[i])
        user_val = st.number_input(
            f"{col}",
            min_value=float(data_min[i]),
            max_value=float(data_max[i]),
            value=float(default_val_clipped),
            step=0.1,
            help=f"Optimal range: {optimal_ranges[col][0]} - {optimal_ranges[col][1]}"
        )
        inputs[col] = np.clip(user_val, data_min[i], data_max[i])

# ------------------------------
# Predict button
# ------------------------------
if st.button("🔮 Predict Fertility", type="primary"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([inputs], columns=numeric_cols)

    # Scale features
    features_scaled = scaler.transform(input_df)

    # Predict numeric class
    prediction = model.predict(features_scaled)[0]

    # Map numeric prediction to original class name
    fertility_class_name = fertility_mapping[prediction]

    # Map to user-friendly display (with emoji)
    fertility_display = {
        "Low": "Low Fertility 🌵",
        "Medium": "Medium Fertility 🌱",
        "High": "High Fertility 🌳"
    }
    
    # Handle case where class name is not in fertility_display
    fertility_class = fertility_display.get(fertility_class_name, f"{fertility_class_name} Fertility")
    
    # Get prediction probabilities
    probabilities = model.predict_proba(features_scaled)[0]

    # Display result
    st.success(f"✅ Predicted Soil Fertility Class: **{fertility_class}**")
    
    # Show confidence scores
    st.subheader("Prediction Confidence")
    conf_cols = st.columns(len(le.classes_))
    for i, cls in enumerate(le.classes_):
        display_name = fertility_display.get(cls, f"{cls} Fertility")
        with conf_cols[i]:
            st.metric(display_name, f"{probabilities[i]*100:.1f}%")
    
    # ------------------------------
    # Nutrient Level Chart
    # ------------------------------
    st.subheader("Nutrient Levels vs Optimal Range")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot optimal ranges
    for i, nutrient in enumerate(numeric_cols):
        optimal_min, optimal_max = optimal_ranges[nutrient]
        ax.barh(nutrient, optimal_max - optimal_min, left=optimal_min, 
                color='lightgreen', alpha=0.5, label='Optimal Range' if i == 0 else "")
    
    # Plot actual values
    actual_values = [inputs[nutrient] for nutrient in numeric_cols]
    ax.scatter(actual_values, numeric_cols, color='red', s=100, zorder=5, label='Your Soil Value')
    
    # Customize chart
    ax.set_xlabel('Value')
    ax.set_title('Soil Nutrient Levels Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Add interpretation
    st.info("""
    **Chart Interpretation:**
    - The green bars show the optimal range for each nutrient
    - The red dots show your soil's actual values
    - Values within the green range are optimal for plant growth
    """)
    
    # Optional: show scaled features
    with st.expander("Show Technical Details"):
        st.write("Scaled Features (model input):")
        st.write(pd.DataFrame(features_scaled, columns=numeric_cols))
        st.write("Model used: Random Forest Classifier")