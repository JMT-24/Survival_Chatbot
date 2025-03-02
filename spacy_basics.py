import spacy

nlp = spacy.load("en_core_web_sm")

text = "During a power outage, use candles, flashlights, or a generator."

doc = nlp(text)

print("Tokens: ", [token.text for token in doc])

print("POS Tags:")
for token in doc:
    print(f"{token.text}: {token.pos_}")

print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text}: ({ent.label_})")