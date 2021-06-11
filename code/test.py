import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Text here")