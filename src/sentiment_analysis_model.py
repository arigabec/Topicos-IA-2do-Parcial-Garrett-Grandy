from transformers import pipeline
from datetime import datetime

class SentimentAnalysisModel:
    def __init__(self):
        self.pipe = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            return_all_scores=True
        )

    def analyze_sentiment(self, text):
        start_time = datetime.now()
        results = self.pipe(text)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000  # en milisegundos

        # Ajustar la puntuación al rango de -1 a 1
        transformed_scores = []
        for percentage_dict in results[0]:
            transformed_value = (2 * percentage_dict["score"]) - 1
            percentage_dict["score"] = transformed_value
            transformed_scores.append(percentage_dict)
        # print(transformed_scores)

        # Obtenemos el mayor valor de scores, que representa a la mejor prediccion del sentimiento
        best_prediction = {}
        if transformed_scores[0]["score"] > transformed_scores[1]["score"] and transformed_scores[0]["score"] > transformed_scores[2]["score"]:
            best_prediction = transformed_scores[0]
        elif transformed_scores[1]["score"] > transformed_scores[0]["score"] and transformed_scores[1]["score"] > transformed_scores[2]["score"]:
            best_prediction = transformed_scores[1]
        else:
            best_prediction = transformed_scores[2]

        return best_prediction["label"], best_prediction["score"], execution_time, transformed_scores