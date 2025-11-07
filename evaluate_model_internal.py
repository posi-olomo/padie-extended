"""
evaluate_model.py

Evaluates a trained language detection model on a randomized subset of the evaluation dataset.
"""

import random
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODEL_DIR = "./models/full/language_detection"   # Path to your trained model
EVAL_DATA_PATH = "datasets/language_detection/eval_dataset.jsonl"  # Path to eval dataset
SAMPLE_SIZE = 1000
MAX_LENGTH = 64
SEED = 42

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def compute_metrics(pred):
    """Computes evaluation metrics (accuracy and weighted F1)."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def tokenize_function(example, tokenizer):
    """Tokenizes the text field in the dataset."""
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

# ---------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------
def main():
    print("=== Language Detection Model Evaluation ===")
    torch.manual_seed(SEED)
    random.seed(SEED)

    print(f"Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    print("getting label2id from model config if available")
    label2id = getattr(model.config, "label2id", None)
    id2label = getattr(model.config, "id2label", None)

    print(f"Loading evaluation dataset from: {EVAL_DATA_PATH}")
    datasets = load_dataset("json", data_files={"eval": EVAL_DATA_PATH})
    eval_dataset = datasets["eval"]

    # Randomly sample subset
    total = len(eval_dataset)
    sample_size = min(SAMPLE_SIZE, total)
    sample_indices = random.sample(range(total), sample_size)
    eval_ds = eval_dataset.select(sample_indices)
    print(f"Evaluating on {sample_size}/{total} random samples")

    print("Inspecting the dataset label type")
    sample_label = eval_ds[0]["label"]
    print("sample label example:", sample_label, "type:", type(sample_label))

    # 4) If labels are strings and model has label2id, map strings -> ints
    if isinstance(sample_label, str):
    	if label2id is None:
        	# Try to infer mapping from the dataset's label names (if dataset provides a 'labels' feature)
        	# Prefer explicit mapping saved from training — fallback is risky.
        	unique_labels = sorted(list(set(eval_ds["label"])))
        	print("No label2id in model config. Inferred label names:", unique_labels)
        	label2id = {name: i for i, name in enumerate(unique_labels)}
        	id2label = {i: name for name, i in label2id.items()}
        	print("WARNING: You inferred label2id from eval set. Make sure it matches training mapping!")
    	# Map string labels to ints
    	def map_labels_to_ids(example):
        	example["label"] = label2id[example["label"]]
        	return example

    	eval_ds = eval_ds.map(map_labels_to_ids)

    # 5) If labels are lists, flatten as appropriate (common cause of nesting errors)
    elif isinstance(sample_label, (list, tuple)):
    	# If label is like [2] -> turn into 2
    	def flatten_label(example):
        	lab = example["label"]
        	if isinstance(lab, (list, tuple)) and len(lab) == 1:
            		example["label"] = int(lab[0])
        	else:
            		# If truly multi-label, you'll need a different evaluation setup
            		raise ValueError("Detected multi-label data; adapt evaluation to multi-label metrics.")
        	return example

    	eval_ds = eval_ds.map(flatten_label)

    # Tokenize
    tokenized_eval = eval_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Running evaluation ...")
    results = trainer.evaluate(eval_dataset=tokenized_eval)

    print("\n==== Evaluation Results ====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n✅ Evaluation complete.")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()

