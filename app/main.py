from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.model_utils import predict

app = FastAPI(
    title="Personality Type Classifier",
    description=(
        "Predicts whether a person is an Introvert or Extrovert "
        "based on behavioral data. Built for Decryptogen technical assignment."
    ),
    version="1.0.0",
)



class PersonData(BaseModel):
    Time_spent_Alone: float = Field(..., example=5,
        description="Hours spent alone per day (0-11)")
    Stage_fear: str = Field(..., example="No",
        description="Yes or No")
    Social_event_attendance: float = Field(..., example=3,
        description="Social events attended per month (0-10)")
    Going_outside: float = Field(..., example=4,
        description="Times going outside per week (0-7)")
    Drained_after_socializing: str = Field(..., example="No",
        description="Yes or No")
    Friends_circle_size: float = Field(..., example=6,
        description="Number of close friends (0-15)")
    Post_frequency: float = Field(..., example=4,
        description="Social media posts per week (0-10)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time_spent_Alone": 5,
                "Stage_fear": "No",
                "Social_event_attendance": 3,
                "Going_outside": 4,
                "Drained_after_socializing": "No",
                "Friends_circle_size": 6,
                "Post_frequency": 4,
            }
        }
    }



class PredictionResponse(BaseModel):
    prediction: str = Field(..., example="Extrovert")
    confidence: float = Field(..., example=0.87)
    input_received: dict



@app.get("/", summary="Health check")
def root():
    return {
        "status": "online",
        "model": "Stacking Ensemble (LightGBM + XGBoost + RF + SVM)",
        "accuracy": "91.72%",
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict personality type")
def predict_personality(data: PersonData):
    try:
        raw = data.model_dump()
        result = predict(raw)
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "input_received": raw,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
