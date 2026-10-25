import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="SuperKart Sales Forecaster",
    page_icon="🛒",
    layout="wide"
)

# Load the serialized model pipeline safely
@st.cache_resource
def load_model():
    model_path = "superkart_model.joblib"
    if not os.path.exists(model_path):
        # Fallback search if directory structure differs inside container
        model_path = os.path.join("backend_files", "superkart_model.joblib")
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("Model pipeline loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model pipeline: {e}")
    st.stop()

# App layout & Title
st.title("🛒 SuperKart Store Sales Forecasting App")
st.write("Enter the product and store attributes below to forecast total sales revenue.")

# Layout inputs using Streamlit columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Product Information")
    product_weight = st.number_input("Product Weight", min_value=0.0, max_value=50.0, value=12.6, step=0.1)
    product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    product_allocated_area = st.number_input("Product Allocated Area Ratio", min_value=0.0, max_value=1.0, value=0.027, step=0.001)
    product_mrp = st.number_input("Product Maximum Retail Price (MRP)", min_value=0.0, max_value=1000.0, value=117.0, step=1.0)
    product_id_char = st.selectbox("Product ID Category Prefix", ["FD", "DR", "NC"])
    product_type_category = st.selectbox("Product Type Class", ["Non Perishables", "Perishables"])

with col2:
    st.subheader("🏪 Store Information")
    store_size = st.selectbox("Store Size", ["Medium", "High", "Low"])
    store_location_city_type = st.selectbox("Store City Location Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
    store_age_years = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=16, step=1)

# Predict button
if st.button("🔮 Forecast Total Revenue"):
    # Map inputs exactly matching training DataFrame schema
    input_data = pd.DataFrame({
        'Product_Weight': [product_weight],
        'Product_Sugar_Content': [product_sugar_content],
        'Product_Allocated_Area': [product_allocated_area],
        'Product_MRP': [product_mrp],
        'Store_Size': [store_size],
        'Store_Location_City_Type': [store_location_city_type],
        'Store_Type': [store_type],
        'Product_Id_char': [product_id_char],
        'Store_Age_Years': [store_age_years],
        'Product_Type_Category': [product_type_category]
    })
    
    try:
        prediction = model.predict(input_data)[0]
        st.metric(label="Predicted Sales Revenue ($)", value=f"${prediction:,.2f}")
    except Exception as e:
        st.error(f"Inference failed: {e}")
