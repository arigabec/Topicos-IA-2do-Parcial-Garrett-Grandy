import openai
import json
import os
from src.config import get_settings
from src.sentiment_analysis_model import SentimentAnalysisModel
from dotenv import load_dotenv

load_dotenv()  # Cargamos el archivo .env

_SETTINGS = get_settings()

# Instancia del modelo de análisis de sentimiento
sentiment_model = SentimentAnalysisModel()

openai.api_key = os.environ.get("OPENAI_API_KEY")

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

        # Definimos la funcion que le pasaremos al modelo de openai para obtener POS tagging
        pos_gpt_function = [
            {
                "name": "find_pos",
                "description": "Perform Part-of-Speech tagging for the input text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "word": {"type": "string",
                                               "description": "Word extracted from text."},
                                    "category": {"type": "string",
                                                 "description": "Category of the named word."}
                                }
                            }
                        }
                    }
                },
                "required": ["entities"]
            }
        ]

        # Encontramos POS usando GPT-4
        response_pos = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": text}],
            functions=pos_gpt_function,
            function_call={"name": "find_pos"},
        )
        cleaned_string_pos = response_pos['choices'][0]['message']['function_call']['arguments'].replace("\\n", "\n")
        parsed_object_pos = json.loads(cleaned_string_pos)
        # print(parsed_object_pos)

        return score, label, transformed_scores, parsed_object, parsed_object_pos