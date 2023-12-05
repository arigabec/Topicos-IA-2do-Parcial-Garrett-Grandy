import spacy
from src.sentiment_analysis_model import SentimentAnalysisModel

# Instancia del modelo de análisis de sentimiento
sentiment_model = SentimentAnalysisModel()

class AnalysisModel:
    def __init__(self):
        # Cargar el modelo de spaCy para NLP y agregar la extensión SpacyTextBlob
        self.nlp = spacy.load("es_core_news_md")

    def perform_analysis(self, text):
        # Realizar el análisis de sentimiento y NLP utilizando transformers y spacy
        doc = self.nlp(text)
        label, score, execution_time, transformed_scores = sentiment_model.analyze_sentiment(text)

        return score, label, transformed_scores, doc
