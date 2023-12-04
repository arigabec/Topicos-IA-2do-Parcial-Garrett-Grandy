from fastapi import (
    FastAPI,
    status,
    HTTPException,
    Depends
)
import csv
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from src.config import get_settings
from functools import cache
from src.sentiment_analysis_model import SentimentAnalysisModel
from src.nlp_analysis import textAnalysis
import spacy

# Colocamos en una lista los datos de cada request de /sentiment
execution_logs = []

_SETTINGS = get_settings()

app = FastAPI(
    title=_SETTINGS.service_name,
    version=_SETTINGS.k_revision
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia del modelo de análisis de sentimiento
sentiment_model = SentimentAnalysisModel()

@cache
def get_nlp():
    return spacy.load("es_core_news_md")

@app.get("/status")
def root():
    return {
        "status": "OK",
        "message": "API is running",
        "model": "",
        "service": "Análisis del sentimiento API es un servicio que permite analizar el sentimiento que expresa un texto.",
        "version": "1.0.0",
        "author": "Camila Grandy Camacho y Ariane Garrett Becerra",
    }

@app.post("/sentiment")
def detect_sentiment(text: str, range: bool = False):
    try:
        label, score, execution_time = sentiment_model.analyze_sentiment(text)
        response_data = {
            "sentiment": label,
            "confidence": score,
            "execution_time": execution_time
        }

        # Agregar el rango de puntuación si el parámetro "range" es True
        if range:
            response_data["range"] = score

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during sentiment analysis: {str(e)}")

@app.post("/analysis")
def generate_analysis(text: str, nlp=Depends(get_nlp)):
    results = textAnalysis(text, "Análisis del sentimiento", nlp)
    return results

@app.get("/reports")
def generate_report():
    if not execution_logs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Por el momento no existen reportes!"
        )

    csv_file_path = "poses_report.csv"

    with open(csv_file_path, mode="w", newline="") as csv_file:
        fieldnames = execution_logs[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(execution_logs)

    return FileResponse(csv_file_path, filename="poses_report.csv", media_type="text/csv")