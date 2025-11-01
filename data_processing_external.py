import os
import json

# Directory containing the JSON files
DATASET_DIR = "datasets/language_detection/test_dataset"
COMBINED_FILE = "datasets/language_detection/test_dataset/test_dataset.json"

def clean_text(text: str) -> str:
    # Remove loading/trailing spaces and normalize spaces/newlines
    return " ".join(text.split())

def preprocessing_and_combine_datasets(dataset_dir, output_file):
    combined_data = []

    # Iterate through all JSON files in the directory
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(dataset_dir, filename)
            print(f"Processing file: {file_path}")

            # Load the JSON file
            with open(file_path, "r", encoding = "utf-8") as f:
                data = json.load(f)

            # Clean the data
            for obj in data:
                if "text" in obj:
                    obj["text"] = clean_text(obj["text"])

            combined_data.extend(data)

    # Save the combined dataset to a new JSON file
    with open(output_file, "w", encoding = "utf-8") as f:
        json.dump(combined_data, f, ensure_ascii = False, indent = 4)

    print(f"Combined dataset saved to {output_file}")

# Run the preprocessing function
preprocessing_and_combine_datasets(DATASET_DIR, COMBINED_FILE)