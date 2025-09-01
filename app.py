import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title
st.title("🌞 Solar Power Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:", df.head())

    X = df.drop(columns=["power-generated"])
    y = df["power-generated"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    st.write("### Model Performance")
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R2 Score:", r2_score(y_test, y_pred))

    # Prediction for user input
    st.write("### Make a Prediction")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"⚡ Predicted Power: {prediction:.2f}")
