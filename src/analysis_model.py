# analysis_model.py
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

class AnalysisModel:
    def __init__(self):
        # Cargar el modelo de spaCy para NLP y agregar la extensión SpacyTextBlob
        self.nlp = spacy.load("es_core_news_sm")
        self.nlp.add_pipe('spacytextblob')

    def perform_analysis(self, text):
        # Realizar el análisis de sentimiento y NLP utilizando spaCy y spacytextblob
        doc = self.nlp(text)

        # Obtener la polaridad y la subjetividad del texto
        sentiment_score = doc._.blob.polarity
        subjectivity = doc._.blob.subjectivity

        # Determinar la categoría de sentimiento
        sentiment_category = self.get_sentiment_category(sentiment_score)

        return sentiment_score, sentiment_category, subjectivity, doc

    def get_sentiment_category(self, sentiment_score):
        # Determinar la categoría de sentimiento basada en la polaridad
        if sentiment_score > 0:
            return "Positivo"
        elif sentiment_score < 0:
            return "Negativo"
        else:
            return "Neutral"
