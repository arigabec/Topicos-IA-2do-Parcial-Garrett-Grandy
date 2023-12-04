from transformers import pipeline
from datetime import datetime


class SentimentAnalysisModel:
    def __init__(self):
        self.pipe = pipeline("sentiment-analysis")

    def analyze_sentiment(self, text):
        start_time = datetime.now()
        result = self.pipe(text)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000  # en milisegundos

        # Ajustar la puntuación al rango de -1 a 1
        score_normalized = (result[0]['score'] - 0.5) * 2

        # Definir etiquetas personalizadas basadas en el rango de puntuación
        if score_normalized < -0.5:
            label = "NEGATIVE"
        elif score_normalized > 0.5:
            label = "POSITIVE"
        else:
            label = "NEUTRAL"

        return label, score_normalized, execution_time