import spacy

new_vocab = ["UNIT"]
main_nlp = spacy.load("en_core_web_lg")

for item in new_vocab:
    main_nlp.vocab.strings.add(item)

unit_nlp = spacy.load("") #actually location of the model
ner = unit_nlp.get_pipe("ner")

main_nlp.add_pipe(ner, name="unit_ner", before="ner")

main_nlp.to_disk("main_model")