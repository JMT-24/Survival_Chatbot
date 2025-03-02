import spacy
import json
from spacy.tokens import DocBin
from spacy.training import Example

# Load English model
nlp = spacy.blank("en")

# Define a text classification pipeline
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat")
    textcat.cfg["exclusive_classes"] = True
else:
    textcat = nlp.get_pipe("textcat")

# Load training data
with open("survival_questions.json", "r") as f:
    training_data = json.load(f)

# Add labels
labels = set([entry["label"] for entry in training_data])
for label in labels:
    textcat.add_label(label)

# Convert data into spaCy's format
doc_bin = DocBin()
for entry in training_data:
    text = entry["text"]
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, {"cats": {label: 1.0 if label == entry["label"] else 0.0 for label in labels}})
    doc_bin.add(example.reference)  # Save reference doc (not the raw one)

# Save training data
doc_bin.to_disk("training_data.spacy")

# Train the model
nlp.begin_training()
for epoch in range(10):  # Train for 10 epochs
    losses = {}
    examples = []  # Collect examples for batch training

    for entry in training_data:
        text = entry["text"]
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, {"cats": {label: 1.0 if label == entry["label"] else 0.0 for label in labels}})
        examples.append(example)

    # Update the model with all examples in a batch
    nlp.update(examples, losses=losses)
    print(f"Epoch {epoch+1}, Loss: {losses}")

# Save the trained model
nlp.to_disk("survival_model")