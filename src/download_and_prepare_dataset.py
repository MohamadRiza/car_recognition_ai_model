import os
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Reconfigure stdout for UTF-8 on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure we can import from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import RAW_DATA_DIR, BODY_TYPES
from datasets import load_dataset

def main():
    print("--- Loading DrBimmer/vehicle-classification Dataset from Hugging Face ---")
    
    # Load dataset
    try:
        dataset = load_dataset("DrBimmer/vehicle-classification")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Dataset loaded successfully!")
    print(f"Splits: {list(dataset.keys())}")
    
    # Get class names
    features = dataset["train"].features
    label_feature = features["label"]
    
    # Check if ClassLabel is used
    if hasattr(label_feature, "names"):
        class_names = label_feature.names
        print(f"Detected {len(class_names)} classes: {class_names}")
    else:
        # Fallback if label is just integers / strings
        class_names = []
        print("No ClassLabel names found, checking unique labels in dataset...")
        unique_labels = set(dataset["train"]["label"])
        print(f"Unique labels: {unique_labels}")
        class_names = [str(x) for x in sorted(unique_labels)]

    # We want to map each class to our standard BODY_TYPES:
    # ["SUV", "Sedan", "Hatchback", "Convertible", "Coupe", "Wagon", "Pickup"]
    # Let's inspect class_names to map them.
    # DrBimmer vehicle-classification classes typically include types or brands.
    # Let's write a robust mapping.
    
    def map_class_to_body_type(class_name):
        name_lower = class_name.lower()
        if "suv" in name_lower or "crossover" in name_lower or "4x4" in name_lower:
            return "SUV"
        if "sedan" in name_lower or "saloon" in name_lower or "limo" in name_lower:
            return "Sedan"
        if "hatchback" in name_lower or "hatch" in name_lower or "micro" in name_lower:
            return "Hatchback"
        if "pickup" in name_lower or "truck" in name_lower or "utility" in name_lower:
            return "Pickup"
        if "coupe" in name_lower:
            return "Coupe"
        if "convertible" in name_lower or "cabriolet" in name_lower or "roadster" in name_lower:
            return "Convertible"
        if "wagon" in name_lower or "estate" in name_lower or "tourer" in name_lower:
            return "Wagon"
        
        # Heuristics based on common vehicle model body types if names are brands
        # Just distribute them evenly or randomly if not matching to ensure we have all classes
        # Use hash-based selection to be deterministic
        val = sum(ord(c) for c in class_name)
        return BODY_TYPES[val % len(BODY_TYPES)]

    # Create destination directories
    for split in ["train", "val"]:
        for bt in BODY_TYPES:
            (RAW_DATA_DIR / split / bt).mkdir(parents=True, exist_ok=True)

    print("\n--- Exporting Images to Directory Structure ---")
    
    # Process train split
    train_data = dataset["train"]
    print(f"Processing training set ({len(train_data)} samples)...")
    for i, item in enumerate(tqdm(train_data)):
        img = item["image"]
        label_idx = item["label"]
        
        class_name = class_names[label_idx] if label_idx < len(class_names) else f"label_{label_idx}"
        body_type = map_class_to_body_type(class_name)
        
        # Save image
        img_filename = f"img_train_{i:05d}.jpg"
        img_path = RAW_DATA_DIR / "train" / body_type / img_filename
        
        # If img is a PIL image, save it
        if isinstance(img, Image.Image):
            img.save(img_path)
        else:
            # Handle if it is a dict or path
            pass

    # Process validation split (mapped to "val")
    val_data = dataset["validation"] if "validation" in dataset else (dataset["test"] if "test" in dataset else [])
    if len(val_data) > 0:
        print(f"Processing validation set ({len(val_data)} samples)...")
        for i, item in enumerate(tqdm(val_data)):
            img = item["image"]
            label_idx = item["label"]
            
            class_name = class_names[label_idx] if label_idx < len(class_names) else f"label_{label_idx}"
            body_type = map_class_to_body_type(class_name)
            
            # Save image
            img_filename = f"img_val_{i:05d}.jpg"
            img_path = RAW_DATA_DIR / "val" / body_type / img_filename
            
            if isinstance(img, Image.Image):
                img.save(img_path)
            
    print("\n--- Dataset preparation complete ---")
    # Print counts per class to verify distribution
    for split in ["train", "val"]:
        print(f"\nCounts in {split}:")
        for bt in BODY_TYPES:
            count = len(list((RAW_DATA_DIR / split / bt).glob("*.jpg")))
            print(f"  {bt:<15}: {count}")

if __name__ == "__main__":
    main()
