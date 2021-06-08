#Purpose of this script is to create rules-based traing dataset
#with particular focus to UNDP unit organization

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

#The default format for the Spacy Train Data is the following
#TRAIN_DATA = [(text, {"entities": [(start, end, label)]})]

patterns = create_training_data("../data/UNDP_units_original.json", "UNIT")
generate_rules(patterns)

nlp = spacy.load("unit_ner")
TRAIN_DATA = []
with open("../data/undp_jobs.txt", "r", errors='ignore') as f:
    text = f.read()
    hits = []
    results = test_model(nlp,text)
    if results != None:
        TRAIN_DATA.append(results)

print(len(TRAIN_DATA))
save_data("../data/units_training_data.json", TRAIN_DATA)