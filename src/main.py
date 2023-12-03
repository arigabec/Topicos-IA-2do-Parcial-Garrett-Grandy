from fastapi import (
    FastAPI,
    Depends,
    UploadFile, 
    File,
    status,
    HTTPException, 
)
import io
import cv2
import csv
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from src.config import get_settings
import numpy as np
from functools import cache
from PIL import Image

# Colocamos en una lista los datos de cada request de /sentiment
execution_logs = []

_SETTINGS = get_settings()

app = FastAPI(
    title = _SETTINGS.service_name,
    version = _SETTINGS.k_revision
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
def root():
    return {"status": "OK",
            "message": "API is running",
            "model": "",
            "service": "Análisis del sentimiento API es un servicio que permite analizar el sentimiento que expresa un texto.",
            "version": "1.0.0",
            "author": "Camila Grandy Camacho y Ariane Garrett Becerra",
            }

@app.post("/sentiment")
def detect_sentiment():
    return {"sentiment": "OK"}

@app.post("/analysis")
def generate_analysis():
    return {"sentiment": "OK"}

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