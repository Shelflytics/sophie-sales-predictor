from fastapi import FastAPI

app = FastAPI()

# Commands to run
# 1. pip install "fastapi[standard]" or conda install "fastapi[standard]"
# 2. fastapi dev main.py
# You will be able to find your docs at https://127.0.0.1/docs/

from fastapi import FastAPI
from pydantic import BaseModel
import lightgbm as lgb
import pandas as pd
import joblib

from recommender import get_enhanced_product_rankings

app = FastAPI()

# Load everything
model = lgb.Booster(model_file="./best_lgbm_model.txt")
encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")
processed_data = pd.read_pickle("processed_data.pkl")

class MeowRequest(BaseModel):
    postal_code: str
    top_k: int = 10

@app.post("/api/meow")
async def meow(request: MeowRequest):
    try:
        rankings = get_enhanced_product_rankings(
            postal_code=request.postal_code,
            top_k=request.top_k,
            model=model,
            encoders=encoders,
            processed_data=processed_data,
            feature_columns=feature_columns
        )
        columns = ["Product_ID", "Product_Name", "Category", "predicted_potential"] if "Product_Name" in rankings.columns else ["Product_ID", "Category", "predicted_potential"]
        result = rankings[columns].to_dict(orient="records")

        return {"recommendations": result}
    except Exception as e:
        return {"error": str(e)}


# @app.get("/api/meow")
# def meow(request):
#     data = request.json()
#     postal_code = data["postal_code"]
#     return {"response": "Hello!"}
