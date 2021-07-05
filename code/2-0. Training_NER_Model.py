import spacy

nlp = spacy.blank("en")

ner = nlp.create_pipe("ner")
ner.add_label("unit")

nlp.add_pipe(ner, name="unit")

nlp.to_disk("unit_ner")

