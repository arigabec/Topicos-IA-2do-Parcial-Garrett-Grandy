# analysis_model.py
import spacy

class AnalysisModel:
    def __init__(self):
        # Cargar el modelo de spaCy para NLP
        self.nlp = spacy.load("es_core_news_sm")

    def perform_analysis(self, text):
        # Realizar el análisis de sentimiento y NLP utilizando spaCy
        doc = self.nlp(text)

        sentiment_value = 0.5  # Modifica esto con la lógica real de tu modelo
        sentiment_category = 'Neutral'  # Modifica esto con la lógica real de tu modelo
        return sentiment_value, sentiment_category, doc
