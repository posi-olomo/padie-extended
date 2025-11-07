from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "posi-olomo/padie-extended",
    load_in_8bit = True,
    device_map = "auto")


model.save_pretrained ("models/full/language_detection")
