from transformers import Trainer, AutoModelForSequenceClassifciation, AutTokenizer 
from datasets import load_dataset
import torch 

# Load trained model and tokenizer
model = AutoModelForSequenceClassifciation.from_pretrained("./models/full/language_detection")
tokenizer = AutTokenizer.from_pretrained("./models/full/language_detection")

# Load your dataset
datasets = load_dataset("json", 
                        data_files = {
                            "eval": "datasets/language_detection/eval_dataset.jsonl"
                        },
    )

eval_dataset = datasets["eval"].shuffle(seed=42).select(range(1000))

# Load trainer 
trainer = Trainer(model = model, tokenizer = tokenizer)

# Evaluate on training data
results = trainer.evaluate(eval_dataset = eval_dataset)

print(results)