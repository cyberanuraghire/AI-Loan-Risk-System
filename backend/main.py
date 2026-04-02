from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import shap
from groq import Groq
import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==============================
# LOAD MODEL FILES
# ==============================
model = joblib.load('../model/loan_model.pkl')
scaler = joblib.load('../model/scaler.pkl')
features = joblib.load('../model/features.pkl')

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# 
client = Groq(api_key="GROK_API_KEY")

# ==============================
# FASTAPI APP
# ==============================
app = FastAPI(title="AI Loan Risk Analyzer")

# ==============================
# INPUT SCHEMA
# ==============================
class LoanData(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Gender: float
    Married: float
    Education: float
    Self_Employed: float
    Property_Area: float

# ==============================
# HOME ROUTE
# ==============================
@app.get("/")
def home():
    return {"message": "Loan Risk API Running 🚀"}

# ==============================
# LLM FUNCTION
# ==============================
def generate_ai_explanation(top_features, risk_level):
    try:
        prompt = f"""
        You are a financial risk analyst.

        The applicant has the following risk factors:
        {top_features}

        Risk level: {risk_level}

        Explain in simple, human-friendly terms why this applicant is risky or safe.
        Keep it short and clear (2-3 lines).
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception:
        return "AI explanation unavailable"

# ==============================
# PREDICTION ROUTE
# ==============================
@app.post("/predict")
def predict(data: LoanData):
    try:
        input_dict = data.dict()

        # Ensure correct feature order
        input_list = [input_dict[f] for f in features]
        input_array = np.array([input_list])

        # Scale
        input_scaled = scaler.transform(input_array)

        # Prediction
        prob = float(model.predict_proba(input_scaled)[0][1])
        pred = int(prob > 0.5)

        # ==============================
        # SHAP EXPLANATION
        # ==============================
        shap_values = explainer.shap_values(input_scaled)
        feature_impact = np.abs(shap_values[0])

        # Top 3 important features
        top_idx = np.argsort(feature_impact)[-3:]
        top_features = [str(features[int(i)]) for i in top_idx]

        # ==============================
        # BUSINESS DECISION
        # ==============================
        if pred == 1:
            decision = "High Risk - Reject Loan"
        else:
            decision = "Low Risk - Approve Loan"

        # Risk level
        if prob < 0.3:
            risk = "Low"
        elif prob < 0.6:
            risk = "Medium"
        else:
            risk = "High"

        # ==============================
        # LLM EXPLANATION
        # ==============================
        ai_explanation = generate_ai_explanation(top_features, risk)

        # ==============================
        # RESPONSE
        # ==============================
        return {
            "decision": str(decision),
            "default_probability_percent": round(prob * 100, 2),
            "risk_level": str(risk),
            "top_factors": top_features,
            "ai_explanation": str(ai_explanation)
        }

    except Exception as e:
        return {"error": str(e)}