import os
import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open("xgboost_house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

@app.get("/")  
def read_root():  
    return {"message": "API is running"}

# Define expected feature names (as used during training)
expected_columns = ['Size', 'Bedrooms', 'Bathrooms', 'Condition', 'Location_CityB',
                    'Location_CityC', 'Location_CityD', 'Type_Single Family', 'Type_Townhouse',
                    'House Age', 'Time Since Sale']

def preprocess_input(data):
    """Ensure input data matches training features"""
    input_df = pd.DataFrame([data])

    # Rename columns to match training feature names
    input_df.rename(columns={'House_Age': 'House Age', 'Time_Since_Sale': 'Time Since Sale'}, inplace=True)

    # Ensure all one-hot encoding columns exist
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns as 0

    # Reorder columns to match training order
    input_df = input_df[expected_columns]

    return input_df

# Define API route for prediction
@app.post("/predict")
def predict_price(features: dict):
    try:
        # Preprocess input to match training feature names
        input_data = preprocess_input(features)

        # Predict house price
        prediction = model.predict(input_data)

        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render assigns a dynamic port
    uvicorn.run(app, host="0.0.0.0", port=port)