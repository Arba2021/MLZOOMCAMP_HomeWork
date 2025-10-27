import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained pipeline
with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

# Define input schema using Pydantic
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Create FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Lead Conversion Prediction API is running!"}

@app.post("/predict")
def predict(lead: Lead):
    data = lead.dict()
    probability = model.predict_proba([data])[0, 1]
    return {"conversion_probability": round(probability, 3)}
