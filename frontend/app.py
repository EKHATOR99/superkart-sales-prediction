
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend (resolved via the shared Docker network)
BACKEND_URL = "http://backend:7860"

st.set_page_config(page_title="SuperKart Sales Prediction", page_icon="🛒", layout="centered")

st.title("🛒 SuperKart Sales Prediction System")
st.write("Enter the product and store details below to predict the total sales for that product at that store.")

st.header("Single (Online) Prediction")

col1, col2 = st.columns(2)
with col1:
    Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
    Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027)
    Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
    Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    Product_Id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"], help="FD = Food, DR = Drinks, NC = Non-Consumable")

with col2:
    Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
    Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
    Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category,
}

if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            predicted_sales = result["Sales"]
            st.success(f"Predicted Product Store Sales Total: ₹{predicted_sales:,.2f}")
            if "Sales_95pct_CI_Lower" in result:
                st.caption(
                    f"Approximate 95% prediction interval: "
                    f"₹{result['Sales_95pct_CI_Lower']:,.2f} – ₹{result['Sales_95pct_CI_Upper']:,.2f}"
                )
        else:
            st.error(f"API error ({response.status_code}): {response.json().get('error', response.text)}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Unable to connect to the prediction API: {exc}")

st.divider()
st.header("Batch Prediction")
st.write("Upload a CSV file containing the same feature columns as above for multiple products.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    st.dataframe(pd.read_csv(uploaded_file).head())
    uploaded_file.seek(0)  # reset the file pointer after previewing

    if st.button("Predict for Batch", type="primary"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": uploaded_file},
                timeout=30,
            )
            if response.status_code == 200:
                predictions = response.json()
                result_df = pd.DataFrame(
                    {"Row": list(predictions.keys()), "Predicted_Sales": list(predictions.values())}
                )
                st.success("Batch prediction complete!")
                st.dataframe(result_df)
                st.download_button(
                    "Download predictions as CSV",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name="superkart_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"API error ({response.status_code}): {response.json().get('error', response.text)}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Unable to connect to the prediction API: {exc}")
