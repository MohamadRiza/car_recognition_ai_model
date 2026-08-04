import os
import sys
import random
from pathlib import Path
from PIL import Image, ImageDraw

# Reconfigure stdout for UTF-8 on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure we can import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DATA_DIR, CAR_MAKES, BODY_TYPES, FUEL_TYPES, TRANSMISSION_TYPES, CAR_COLORS

def generate_shape_image(body_type: str, color_name: str) -> Image.Image:
    """
    Generate a simple color shape image representing the car body type
    so the CNN can learn to classify them based on shape and color.
    """
    # Create blank canvas
    img = Image.new("RGB", (224, 224), color="white")
    draw = ImageDraw.Draw(img)
    
    # Map color name to RGB
    color_map = {
        "White": (240, 240, 240),
        "Black": (15, 15, 15),
        "Silver": (192, 192, 192),
        "Gray": (128, 128, 128),
        "Red": (220, 20, 60),
        "Blue": (30, 144, 255),
        "Green": (34, 139, 34),
        "Yellow": (255, 215, 0),
        "Orange": (255, 69, 0),
        "Brown": (139, 69, 19),
        "Gold": (218, 165, 32),
        "Other": (147, 112, 219)
    }
    rgb = color_map.get(color_name, (128, 128, 128))
    
    # Add random background clutter
    for _ in range(5):
        x1 = random.randint(0, 200)
        y1 = random.randint(0, 200)
        x2 = x1 + random.randint(10, 40)
        y2 = y1 + random.randint(10, 40)
        draw.rectangle([x1, y1, x2, y2], fill=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
        
    # Draw body type specific shapes (bounding boxes/silhouettes)
    if body_type == "SUV":
        # Tall box in the center
        draw.rectangle([40, 80, 180, 160], fill=rgb)
        draw.rectangle([60, 50, 160, 80], fill=rgb)
    elif body_type == "Sedan":
        # Low long box with cabin
        draw.rectangle([30, 100, 190, 160], fill=rgb)
        draw.rectangle([70, 70, 150, 100], fill=rgb)
    elif body_type == "Hatchback":
        # Short box with sloped back
        draw.rectangle([40, 100, 180, 160], fill=rgb)
        draw.polygon([(80, 70), (140, 70), (180, 100), (80, 100)], fill=rgb)
    elif body_type == "Pickup":
        # Cabin + flat bed
        draw.rectangle([30, 100, 190, 160], fill=rgb)
        draw.rectangle([50, 60, 110, 100], fill=rgb)
    elif body_type == "Coupe":
        # Sleek low cabin
        draw.rectangle([30, 110, 190, 160], fill=rgb)
        draw.polygon([(70, 85), (130, 85), (170, 110), (50, 110)], fill=rgb)
    elif body_type == "Convertible":
        # Low cabin open top
        draw.rectangle([30, 110, 190, 160], fill=rgb)
        draw.rectangle([60, 100, 140, 110], fill=(200, 200, 200)) # windshield
    elif body_type == "Wagon":
        # Long tall back cabin
        draw.rectangle([30, 90, 190, 160], fill=rgb)
        draw.rectangle([60, 60, 175, 90], fill=rgb)
        
    # Draw wheels
    draw.ellipse([50, 150, 80, 180], fill=(0, 0, 0))
    draw.ellipse([140, 150, 170, 180], fill=(0, 0, 0))
    
    return img

def main():
    print("--- Generating Synthetic Car Shape Dataset for Training Verification ---")
    
    num_train_per_class = 20
    num_val_per_class = 5
    
    total_images = 0
    
    for split, count in [("train", num_train_per_class), ("val", num_val_per_class)]:
        print(f"Generating {split} images...")
        
        # Create output directories
        split_dir = RAW_DATA_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate samples for each class
        for bt in BODY_TYPES:
            class_dir = split_dir / bt
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                # Choose make deterministically based on body type
                make_options = {
                    "SUV": ["Toyota", "Ford", "Volkswagen", "Hyundai", "Kia", "Audi"],
                    "Sedan": ["Toyota", "Honda", "BMW", "Mercedes-Benz", "Tesla", "Hyundai", "Kia", "Audi"],
                    "Hatchback": ["Honda", "Volkswagen", "Hyundai"],
                    "Pickup": ["Ford", "Toyota"],
                    "Coupe": ["BMW", "Ford"],
                    "Convertible": ["BMW", "Mercedes-Benz"],
                    "Wagon": ["Audi", "Volkswagen"]
                }.get(bt, ["Toyota"])
                
                make = random.choice(make_options)
                color = random.choice(CAR_COLORS)
                
                # Determine fuel type based on make
                if make == "Tesla":
                    fuel = "Electric"
                elif make in ["Toyota", "Hyundai"]:
                    fuel = "Hybrid"
                else:
                    fuel = random.choice(["Petrol", "Diesel", "Plug-in Hybrid"])
                    
                # Determine transmission based on body type / make
                if bt in ["Coupe", "Pickup"]:
                    transmission = "Manual"
                else:
                    transmission = random.choice(["Automatic", "Semi-Automatic"])

                img = generate_shape_image(bt, color)
                
                # Clean fuel name for filename (e.g. Plug-in Hybrid -> Plug-in-Hybrid)
                fuel_clean = fuel.replace(" ", "-")
                filename = f"car_{make.lower()}_{color.lower()}_{fuel_clean.lower()}_{transmission.lower()}_{i:03d}.jpg"
                
                img_path = class_dir / filename
                img.save(img_path, "JPEG")
                total_images += 1
                
    print(f"Generated {total_images} total images successfully under {RAW_DATA_DIR}!")

if __name__ == "__main__":
    main()
