import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from groq import AsyncGroq

# Import the model logic
from model import CreditApprovalModel

# ---------------------------------------------------------
# Configuration for loan_data_tree_ready.csv
# ---------------------------------------------------------
DATASET_FILENAME = "loan_data_tree_ready.csv" 
TARGET_COLUMN = "Loan Status"      
SENSITIVE_COLUMN = "Gender"            

app = FastAPI(title="Credit Approval XGBoost API")
groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY", "dummy_key"))

ml_system = CreditApprovalModel(DATASET_FILENAME, TARGET_COLUMN, SENSITIVE_COLUMN)

@app.on_event("startup")
async def startup_event():
    try:
        print("Training XGBoost model on loan_data_tree_ready.csv...")
        ml_system.train_and_evaluate()
        print("Model trained and metrics generated successfully.")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")

# ---------------------------------------------------------
# Next.js API Endpoints
# ---------------------------------------------------------

@app.get("/model-performance")
async def get_performance_report():
    if not ml_system.model:
        raise HTTPException(status_code=500, detail="Model not initialized.")
    return ml_system.performance_report

@app.get("/fairness-report")
async def get_fairness_report():
    if not ml_system.model:
        raise HTTPException(status_code=500, detail="Model not initialized.")
    return ml_system.fairness_report

@app.get("/shap-details/global")
async def get_global_shap():
    if not ml_system.model:
        raise HTTPException(status_code=500, detail="Model not initialized.")
    return {"feature_importance": ml_system.global_shap}

@app.post("/predict")
async def predict_approval(data: Dict[str, Any]):
    if not ml_system.model:
        raise HTTPException(status_code=500, detail="Model not initialized.")

    # 1. Prediction & SHAP
    approval_chance = ml_system.predict(data)
    feature_contributions = ml_system.get_local_shap(data)

    # 2. Extract Top 3 most impactful features for the prompt
    sorted_contributions = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = dict(sorted_contributions[:3])

    # 3. Construct Groq Prompt
    profile_string = "\n".join([f"- {k}: {v}" for k, v in data.items()])
    prompt = f"""
    You are an AI financial assistant. Explain this loan approval prediction in plain English.
    
    Applicant Data:
    {profile_string}
    
    Model Prediction:
    - Approval Chance: {approval_chance}%
    
    Top Contributing Factors (Positive helped approval, Negative hurt approval):
    {top_features}
    
    Provide:
    1. A brief plain English explanation of why they got this score based on the Top Contributing Factors.
    2. 2-3 specific, actionable suggestions to improve their approval chances next time.
    Keep it concise, professional, and do not use the term "SHAP values".
    """

    # 4. Fetch Explanation
    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful financial explainer."},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-20b", 
            temperature=0.5,
            max_tokens=400,
        )
        ai_explanation = chat_completion.choices[0].message.content
    except Exception as e:
        ai_explanation = f"Groq AI explanation unavailable. Error: {str(e)}"

    return {
        "approval_chances_percentage": approval_chance,
        "prediction_class": "Approved" if approval_chance >= 50 else "Rejected",
        "local_shap_details": feature_contributions,
        "ai_explanation_and_suggestions": ai_explanation
    }