from pydantic import BaseModel

class TextAnalysis(BaseModel):
    length: int # longitud en caracteres
    n_positive_feelings: int # cantidad de sentimientos positivos mencionados
    top_positive_feelings: str # sentimiento positivo más mencionado
    n_negative_feelings: int # cantidad de sentimientos negativos mencionados
    top_negative_feelings: str # sentimiento negativo más mencionado

def textAnalysis(text, title, nlp):
    doc = nlp(text)

    locations_set = set()
    people_set = set()
    org_set = set()

    for ent in doc.ents:
        if ent.label_ == "LOC":
            locations_set.add(ent.text)
        elif ent.label_ == "PER":
            people_set.add(ent.text)
        elif ent.label_ == "ORG":
            org_set.add(ent.text)

    results = textAnalysis(
        url="url",
        length=len(text),
        title=title,
        n_locations=len(locations_set),
        top_location="" if not locations_set else max(locations_set, key=len),
        n_people=len(people_set),
        top_person="" if not people_set else max(people_set, key=len),
        n_org=len(org_set),
        top_org="" if not org_set else max(org_set, key=len)
    )

    return results