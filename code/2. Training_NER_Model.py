import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import json
import random

def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)

def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=3)

def create_training_data(file, type):
    data = load_data(file)
    patterns = []
    for item in data:
        pattern = {
            "label": type,
            "pattern": item
        }
        patterns.append(pattern)
    return(patterns)

def generate_rules(patterns):
    nlp = English()
    ruler = EntityRuler(nlp)
    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler)
    nlp.to_disk("unit_ner")

def test_model(model, text):
    doc = nlp(text)
    results = []
    entities = []
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))
    if len(entities)>0:
        results = [text, {"entities": entities}]
        print(results)
    return (results)

def train_spacy(TRAIN_DATA, iterations):
    nlp = spacy.blank("en")
    ner = nlp.create_pipe("ner")
    ner.add_label("UNIT")
    nlp.add_pipe(ner, name="unit_ner")

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "unit_ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print(f"Starting iteration {str(itn)}")
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update( [text],
                            [annotations],
                            drop=0.2,
                            sgd=optimizer,
                            losses=losses,
                )
            print(losses)
    return(nlp)

#    if "ner" not in nlp.pipe_names:
#        ner = nlp.create_pipe("ner")
#        nlp.add_pipe(ner, last=True)
#    for _, annotations in TRAIN_DATA:
#        for ent in annotations.get("entities"):
#            ner.add_label(ent[2])
#    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
#    with nlp.disable_pipes(*other_pipes):
#        optimizer = nlp.begin_training()
#        for itn in range(iterations):
#            print("Starting iteration" + str(itn))
#            random.shuffle(TRAIN_DATA)
#            losses = {}
#            for text, annotations in TRAIN_DATA:
#                nlp.update([text],
#                           [annotations],
#                           drop=0.2,
#                           sgd=optimizer,
#                           losses=losses)
#                print(losses)
#    return(nlp) #fully trained model


nlp = spacy.load("unit_ner")

TRAIN_DATA = load_data("../data/units_training_data.json")
random.shuffle(TRAIN_DATA)

nlp = train_spacy(TRAIN_DATA, 1)
nlp.to_disk("ner_model")
print(TRAIN_DATA[0:10])