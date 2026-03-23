from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

@app.post("/predict")
def predict(size: float, bedrooms: int):
    prediction = model.predict([[size, bedrooms]])
    return {
        "predicted_price": round(prediction[0], 2)
    }
