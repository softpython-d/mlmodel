import os
import logging
import json
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from groq import AsyncGroq, BadRequestError

# Import the model logic
from model import CreditApprovalModel

# ---------------------------------------------------------
# Configuration for loan_data_tree_ready.csv
# ---------------------------------------------------------
DATASET_FILENAME = "loan_data_tree_ready.csv" 
TARGET_COLUMN = "Loan Status"      
SENSITIVE_COLUMN = "Gender"            

app = FastAPI(title="Credit Approval XGBoost API")
logger = logging.getLogger(__name__)
groq_api_key = os.environ.get("GROQ_API_KEY")
groq_client = AsyncGroq(api_key=groq_api_key) if groq_api_key else None

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

    try:
        approval_chance = ml_system.predict(data)
    except Exception as e:
        logger.exception("Loan approval prediction failed.")
        raise HTTPException(status_code=500, detail="The model could not evaluate this application.") from e

    try:
        feature_contributions = ml_system.get_local_shap(data)
    except Exception:
        logger.exception("Prediction succeeded, but local feature explanations could not be generated.")
        return {
            "approval_chances_percentage": approval_chance,
            "prediction_class": "Approved" if approval_chance >= 50 else "Rejected",
            "local_shap_details": {},
            "ai_explanation_and_suggestions": (
                "The approval estimate is available, but a feature explanation could not be generated. "
                "Please review the estimate alongside the applicant's full circumstances."
            ),
        }

    # 3. Construct a grounded plain-English explanation prompt.
    feature_labels = {
        "Age": "Age",
        "Gender": "Gender code",
        "Person Income": "Annual income",
        "Employee Experience": "Employment experience",
        "Loan Amount": "Requested loan amount",
        "Loan interest Rate": "Loan interest rate",
        "Loan percentage": "Loan-to-income ratio",
        "Credit History": "Credit history length",
        "Credit Score": "Credit score",
        "Previous Loan": "Previous loan indicator",
    }
    top_features = {
        feature_labels.get(name, name.replace("_", " ")): round(float(value), 4)
        for name, value in sorted(
            feature_contributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:3]
    }
    loan_intent = next(
        (
            name.removeprefix("Loan Intent_").replace("_", " ").lower()
            for name, value in data.items()
            if name.startswith("Loan Intent_") and value
        ),
        "not specified",
    )
    applicant_context = {
        "loan_purpose": loan_intent,
        "requested_amount": data.get("Loan Amount"),
        "annual_income": data.get("Person Income"),
        "loan_to_income_ratio": data.get("Loan percentage"),
    }
    prompt = f"""
    Explain a machine-learning loan estimate in plain, simple English.

    Estimate: {approval_chance}% ({'predicted approved' if approval_chance >= 50 else 'predicted declined'})
    Applicant's stated loan context: {json.dumps(applicant_context)}
    Model's three strongest feature contributions: {json.dumps(top_features)}

    The contribution sign shows direction: positive moved this model's score toward approval;
    negative moved it away. These are associations learned by a model, not proof of cause.
    Do not invent facts, lender policies, thresholds, or financial circumstances. Do not claim
    changing a factor will guarantee approval. Do not suggest changing protected traits such
    as gender. Describe a factor only when its name and direction support the description.
    Keep the tone neutral, supportive, and understandable to a general reader.

    Suggest exactly three general loan types to explore that could fit the stated purpose.
    These are categories only, not named lenders, available products, or verified offers.
    Do not invent lender names, interest rates, fees, eligibility rules, or product availability.
    Explain briefly why each category might be worth comparing and one important term to check.
    If the purpose or information is insufficient, say so plainly in the suitability field.

    Return only a JSON object with exactly these fields:
    {{
      "summary": "One or two short sentences describing the estimate and main reasons.",
      "positive_factors": ["At most two short, plain-English points."],
      "negative_factors": ["At most two short, plain-English points."],
      "suggestions": ["At most three cautious, practical suggestions; do not promise an outcome."],
      "loan_recommendations": [
        {{
          "name": "General loan category",
          "why_it_may_fit": "A brief reason to compare this type for the stated purpose.",
          "what_to_check": "One important term or condition to compare."
        }}
      ]
    }}
    """

    ai_explanation = {
        "summary": "The model estimate is available. An AI-written explanation is temporarily unavailable.",
        "positive_factors": [],
        "negative_factors": [],
        "suggestions": [],
        "loan_recommendations": [],
    }
    if groq_client:
        try:
            request = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You explain model outputs clearly and cautiously. Return only valid JSON matching the requested fields.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "model": "openai/gpt-oss-20b",
                "temperature": 0.3,
                "max_tokens": 500,
                "response_format": {"type": "json_object"},
            }
            try:
                chat_completion = await groq_client.chat.completions.create(**request)
            except BadRequestError as error:
                error_body = error.body if isinstance(error.body, dict) else {}
                error_details = error_body.get("error", {})
                if error_details.get("code") != "json_validate_failed":
                    raise
                logger.warning(
                    "Groq rejected constrained JSON output; retrying without response_format."
                )
                request.pop("response_format")
                chat_completion = await groq_client.chat.completions.create(**request)

            content = chat_completion.choices[0].message.content
            if content:
                try:
                    parsed = json.loads(content)
                    if not isinstance(parsed, dict) or not isinstance(parsed.get("summary"), str):
                        raise ValueError("Groq returned an invalid explanation shape.")

                    def string_items(value: Any) -> list[str]:
                        if not isinstance(value, list):
                            return []
                        return [
                            item.strip()
                            for item in value
                            if isinstance(item, str) and item.strip()
                        ][:3]

                    def loan_recommendations(value: Any) -> list[dict[str, str]]:
                        if not isinstance(value, list):
                            return []
                        recommendations = []
                        for item in value:
                            if not isinstance(item, dict):
                                continue
                            fields = (
                                item.get("name"),
                                item.get("why_it_may_fit"),
                                item.get("what_to_check"),
                            )
                            if not all(isinstance(field, str) and field.strip() for field in fields):
                                continue
                            recommendations.append({
                                "name": fields[0].strip(),
                                "why_it_may_fit": fields[1].strip(),
                                "what_to_check": fields[2].strip(),
                            })
                            if len(recommendations) == 3:
                                break
                        return recommendations

                    ai_explanation = {
                        "summary": parsed["summary"].strip(),
                        "positive_factors": string_items(parsed.get("positive_factors")),
                        "negative_factors": string_items(parsed.get("negative_factors")),
                        "suggestions": string_items(parsed.get("suggestions")),
                        "loan_recommendations": loan_recommendations(parsed.get("loan_recommendations")),
                    }
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Groq returned a malformed explanation; returning it as plain text.")
                    ai_explanation["summary"] = content
        except Exception:
            logger.exception("Groq explanation unavailable; returning the model prediction and feature contributions.")
    else:
        logger.warning("GROQ_API_KEY is not configured; returning the model prediction without an AI explanation.")

    return {
        "approval_chances_percentage": approval_chance,
        "prediction_class": "Approved" if approval_chance >= 50 else "Rejected",
        "local_shap_details": feature_contributions,
        "ai_explanation_and_suggestions": ai_explanation
    }