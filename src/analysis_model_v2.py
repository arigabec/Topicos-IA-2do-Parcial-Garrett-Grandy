import openai
import json
from src.config import get_settings
from src.sentiment_analysis_model import SentimentAnalysisModel

_SETTINGS = get_settings()

# Instancia del modelo de análisis de sentimiento
sentiment_model = SentimentAnalysisModel()

class AnalysisModelV2:
    def perform_analysis(self, text):
        # Realizar el análisis de sentimiento y NLP utilizando transformers y openai
        label, score, execution_time, transformed_scores = sentiment_model.analyze_sentiment(text)

        # Definimos la funcion que le pasaremos al modelo de openai para obtener NER
        ner_gpt_function = [
            {
                "name": "find_ner",
                "description": "Extracts named entities and their categories from the input text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": {"type": "string",
                                               "description": "A Named entity extracted from text."},
                                    "category": {"type": "string",
                                                 "description": "Category of the named entity."}
                                }
                            }
                        }
                    }
                },
                "required": ["entities"]
            }
        ]

        # Encontramos NER usando GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": text}],
            functions=ner_gpt_function,
            function_call={"name": "find_ner"},
        )
        cleaned_string = response['choices'][0]['message']['function_call']['arguments'].replace("\\n", "\n")
        parsed_object = json.loads(cleaned_string)
        # print(parsed_object)

        return score, label, transformed_scores, parsed_object