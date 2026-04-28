from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from src.explain import explain_prediction

app = FastAPI(title="Email Spam Detection API")

model = joblib.load("models/spam_model.pkl")

class EmailInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Email Spam Detection API Running"}

@app.post("/predict")
def predict(data: EmailInput):

    pred = model.predict([data.text])[0]
    prob = model.predict_proba([data.text])[0].max()

    result = "spam" if pred == 1 else "ham"

    important_words = explain_prediction(data.text)

    return {
        "prediction": result,
        "confidence": round(float(prob), 3),
        "important_words": important_words
    }