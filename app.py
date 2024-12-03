import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model
import logging

# Load files and initialize model on function startup
stock_risk_data = pd.read_csv("stock_risk_levels.csv")
model = load_model("risk_prediction_model.h5")

# Predefine constants
investment_goals_list = ["Growth", "Stability", "Short-term", "Long-term"]
spending_habits_list = ["Low", "Moderate", "High"]

# Initialize encoders
label_enc_goal = LabelEncoder().fit(investment_goals_list)
label_enc_habits = LabelEncoder().fit(spending_habits_list)

# Predefined values for other inputs
DEFAULT_AGE = 30
DEFAULT_SAVINGS = 20000
DEFAULT_DEBT = 1000
DEFAULT_INVESTMENT_GOALS = "Growth"
DEFAULT_SPENDING_HABITS = "Moderate"

# Azure Function handler
def main(req):
    logging.info("Received request for stock recommendations.")
    try:
        # Parse income from request
        req_body = req.get_json()
        income = req_body.get("income")
        if income is None:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Income value is required"})
            }
        
        # Create input data
        input_data = pd.DataFrame([{
            "Income": income,
            "Age": DEFAULT_AGE,
            "Savings": DEFAULT_SAVINGS,
            "Debt": DEFAULT_DEBT,
            "Investment Goal": DEFAULT_INVESTMENT_GOALS,
            "Spending Habits": DEFAULT_SPENDING_HABITS
        }])
        
        # Encode and normalize features
        input_data["Investment Goal"] = label_enc_goal.transform(input_data["Investment Goal"])
        input_data["Spending Habits"] = label_enc_habits.transform(input_data["Spending Habits"])
        scaler = MinMaxScaler()
        input_data[["Income", "Age", "Savings", "Debt"]] = scaler.fit_transform(
            input_data[["Income", "Age", "Savings", "Debt"]]
        )
        
        # Predict risk level
        predicted_risk_probs = model.predict(input_data)
        predicted_risk_level = int(np.argmax(predicted_risk_probs, axis=1)[0] + 1)
        
        # Filter stocks by predicted risk level
        eligible_stocks = stock_risk_data[stock_risk_data["Risk Level"] == predicted_risk_level]
        recommendations = eligible_stocks.sample(n=min(5, len(eligible_stocks)))
        
        # Format response
        response = recommendations[["Stock Name", "Risk Level"]].to_dict(orient="records")
        return {
            "statusCode": 200,
            "body": json.dumps({"recommendations": response})
        }
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
