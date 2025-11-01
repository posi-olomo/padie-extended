import os
import json

# Directory containing the JSON files 
DATASET_DIR = "datasets/language_detection/test_dataset"
COMBINED_FILE = "datasets/language_detection/test_dataset.json"

def preprocessing_and_combine_datasets(dataset_Dir, output_file):
    combined_data = []

import os
import json

# Directory containing the JSON files
DATASET_DIR = "datasets/intent_recognition"
COMBINED_FILE = "datasets/intent_recognition/combined.json"

def preprocess_and_combine_datasets(dataset_dir, output_file):
    combined_data = []

    # Iterate through all JSON files in the directory
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(dataset_dir, filename)
            print(f"Processing file: {file_path}")

            # Load the JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Replace "language" key with "label" key
            for entry in data:
                if "language" in entry and "label" in entry:
                    entry["language"] = entry["label"]

            # Add the modified data to the combined dataset
            combined_data.extend(data)

    # Save the combined dataset to a new JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

    print(f"Combined dataset saved to: {output_file}")

# Run the preprocessing
preprocess_and_combine_datasets(DATASET_DIR, COMBINED_FILE)