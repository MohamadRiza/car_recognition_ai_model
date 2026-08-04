"""
Vehicle AI — Comprehensive Car Specs Lookup Database
Maps car make/model/year → fuel type, transmission, seats, engine, top speed

Sources:
  - NHTSA public data
  - Auto manufacturer specs
  - Community-compiled car databases

Used AFTER the CNN predicts the make/model/year to fill in specs
that are hard to detect visually (fuel type, transmission, etc.)
"""

# ─────────────────────────────────────────────────────────────
# COMPREHENSIVE CAR SPECS DATABASE
# Format: "Make Model Year" → specs dict
# ─────────────────────────────────────────────────────────────

CAR_DATABASE = {

    # ── TOYOTA ──────────────────────────────────────────────────
    "Toyota Camry 2024":      {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "2.5L 4-cyl", "top_speed_kmh": 193, "horsepower": 225},
    "Toyota Camry 2023":      {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "2.5L 4-cyl", "top_speed_kmh": 193, "horsepower": 208},
    "Toyota Corolla 2024":    {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "1.8L 4-cyl", "top_speed_kmh": 180, "horsepower": 139},
    "Toyota Corolla 2023":    {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl", "top_speed_kmh": 185, "horsepower": 169},
    "Toyota RAV4 2024":       {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "2.5L 4-cyl", "top_speed_kmh": 180, "horsepower": 219},
    "Toyota RAV4 2023":       {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "2.5L 4-cyl", "top_speed_kmh": 180, "horsepower": 203},
    "Toyota Prius 2024":      {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl", "top_speed_kmh": 177, "horsepower": 220},
    "Toyota Hilux 2024":      {"fuel": "Diesel",    "transmission": "Automatic", "seats": 5, "engine": "2.8L 4-cyl", "top_speed_kmh": 175, "horsepower": 204},
    "Toyota Land Cruiser 2024":{"fuel": "Petrol",   "transmission": "Automatic", "seats": 8, "engine": "3.5L V6",   "top_speed_kmh": 190, "horsepower": 415},
    "Toyota GR Supra 2024":   {"fuel": "Petrol",    "transmission": "Automatic", "seats": 2, "engine": "3.0L I6",   "top_speed_kmh": 250, "horsepower": 382},
    "Toyota Fortuner 2024":   {"fuel": "Diesel",    "transmission": "Automatic", "seats": 7, "engine": "2.8L 4-cyl", "top_speed_kmh": 175, "horsepower": 201},

    # ── HONDA ───────────────────────────────────────────────────
    "Honda Civic 2024":       {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "1.5L 4-cyl", "top_speed_kmh": 210, "horsepower": 158},
    "Honda Civic 2023":       {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "1.5L 4-cyl", "top_speed_kmh": 200, "horsepower": 158},
    "Honda CR-V 2024":        {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl", "top_speed_kmh": 185, "horsepower": 204},
    "Honda Accord 2024":      {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl", "top_speed_kmh": 193, "horsepower": 204},
    "Honda Jazz 2024":        {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "1.5L 4-cyl", "top_speed_kmh": 175, "horsepower": 108},
    "Honda City 2024":        {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "1.5L 4-cyl", "top_speed_kmh": 175, "horsepower": 121},

    # ── BMW ─────────────────────────────────────────────────────
    "BMW 3 Series 2024":      {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl", "top_speed_kmh": 250, "horsepower": 255},
    "BMW 5 Series 2024":      {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "3.0L I6",   "top_speed_kmh": 250, "horsepower": 375},
    "BMW X5 2024":            {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "3.0L I6",   "top_speed_kmh": 250, "horsepower": 483},
    "BMW M3 2024":            {"fuel": "Petrol",    "transmission": "Manual",    "seats": 5, "engine": "3.0L I6",   "top_speed_kmh": 290, "horsepower": 503},
    "BMW i4 2024":            {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric",  "top_speed_kmh": 225, "horsepower": 335},
    "BMW iX 2024":            {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric",  "top_speed_kmh": 200, "horsepower": 516},
    "BMW 7 Series 2024":      {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "3.0L I6",   "top_speed_kmh": 250, "horsepower": 536},

    # ── MERCEDES-BENZ ────────────────────────────────────────────
    "Mercedes-Benz C-Class 2024": {"fuel": "Petrol",  "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl", "top_speed_kmh": 250, "horsepower": 255},
    "Mercedes-Benz E-Class 2024": {"fuel": "Hybrid",  "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl", "top_speed_kmh": 250, "horsepower": 295},
    "Mercedes-Benz S-Class 2024": {"fuel": "Hybrid",  "transmission": "Automatic", "seats": 5, "engine": "3.0L I6",   "top_speed_kmh": 250, "horsepower": 429},
    "Mercedes-Benz GLE 2024":     {"fuel": "Hybrid",  "transmission": "Automatic", "seats": 5, "engine": "3.0L I6",   "top_speed_kmh": 250, "horsepower": 362},
    "Mercedes-Benz EQS 2024":     {"fuel": "Electric","transmission": "Automatic", "seats": 5, "engine": "Electric",  "top_speed_kmh": 210, "horsepower": 516},
    "Mercedes-Benz AMG GT 2024":  {"fuel": "Petrol",  "transmission": "Automatic", "seats": 4, "engine": "4.0L V8",   "top_speed_kmh": 315, "horsepower": 577},

    # ── TESLA ───────────────────────────────────────────────────
    "Tesla Model 3 2024":     {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 225, "horsepower": 358},
    "Tesla Model 3 2023":     {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 225, "horsepower": 283},
    "Tesla Model Y 2024":     {"fuel": "Electric",  "transmission": "Automatic", "seats": 7, "engine": "Electric", "top_speed_kmh": 217, "horsepower": 384},
    "Tesla Model S 2024":     {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 322, "horsepower": 670},
    "Tesla Model X 2024":     {"fuel": "Electric",  "transmission": "Automatic", "seats": 7, "engine": "Electric", "top_speed_kmh": 250, "horsepower": 670},
    "Tesla Cybertruck 2024":  {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 209, "horsepower": 845},

    # ── FORD ─────────────────────────────────────────────────────
    "Ford Mustang 2024":      {"fuel": "Petrol",    "transmission": "Manual",    "seats": 4, "engine": "5.0L V8",  "top_speed_kmh": 260, "horsepower": 480},
    "Ford F-150 2024":        {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "3.5L V6",  "top_speed_kmh": 180, "horsepower": 400},
    "Ford Explorer 2024":     {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 7, "engine": "3.0L V6",  "top_speed_kmh": 200, "horsepower": 400},
    "Ford Focus 2023":        {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "1.5L 4-cyl","top_speed_kmh": 205, "horsepower": 148},
    "Ford Ranger 2024":       {"fuel": "Diesel",    "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl","top_speed_kmh": 175, "horsepower": 170},

    # ── VOLKSWAGEN ───────────────────────────────────────────────
    "Volkswagen Golf 2024":   {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "1.5L 4-cyl","top_speed_kmh": 224, "horsepower": 158},
    "Volkswagen Tiguan 2024": {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "1.5L 4-cyl","top_speed_kmh": 207, "horsepower": 150},
    "Volkswagen ID.4 2024":   {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 180, "horsepower": 201},
    "Volkswagen Polo 2024":   {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "1.0L 3-cyl","top_speed_kmh": 196, "horsepower": 110},
    "Volkswagen Passat 2024": {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "1.5L 4-cyl","top_speed_kmh": 220, "horsepower": 200},

    # ── HYUNDAI ──────────────────────────────────────────────────
    "Hyundai Tucson 2024":    {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "1.6L 4-cyl","top_speed_kmh": 185, "horsepower": 226},
    "Hyundai Elantra 2024":   {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "1.6L 4-cyl","top_speed_kmh": 185, "horsepower": 139},
    "Hyundai Ioniq 6 2024":   {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 185, "horsepower": 320},
    "Hyundai Ioniq 5 2024":   {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 185, "horsepower": 320},
    "Hyundai Creta 2024":     {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "1.5L 4-cyl","top_speed_kmh": 170, "horsepower": 115},
    "Hyundai i20 2024":       {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "1.2L 4-cyl","top_speed_kmh": 170, "horsepower": 83},

    # ── KIA ──────────────────────────────────────────────────────
    "Kia Sportage 2024":      {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "1.6L 4-cyl","top_speed_kmh": 193, "horsepower": 227},
    "Kia Seltos 2024":        {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "1.5L 4-cyl","top_speed_kmh": 185, "horsepower": 146},
    "Kia EV6 2024":           {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 185, "horsepower": 320},
    "Kia Stinger 2024":       {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "3.3L V6",  "top_speed_kmh": 270, "horsepower": 368},

    # ── NISSAN ───────────────────────────────────────────────────
    "Nissan GT-R 2024":       {"fuel": "Petrol",    "transmission": "Automatic", "seats": 4, "engine": "3.8L V6",  "top_speed_kmh": 315, "horsepower": 570},
    "Nissan Navara 2024":     {"fuel": "Diesel",    "transmission": "Automatic", "seats": 5, "engine": "2.3L 4-cyl","top_speed_kmh": 175, "horsepower": 190},
    "Nissan X-Trail 2024":    {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 7, "engine": "1.5L 3-cyl","top_speed_kmh": 180, "horsepower": 204},
    "Nissan Leaf 2024":       {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 157, "horsepower": 150},

    # ── RANGE ROVER / LAND ROVER ─────────────────────────────────
    "Range Rover Sport 2024": {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "3.0L I6",  "top_speed_kmh": 240, "horsepower": 395},
    "Range Rover Evoque 2024":{"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl","top_speed_kmh": 210, "horsepower": 246},
    "Land Rover Defender 2024":{"fuel":"Diesel",    "transmission": "Automatic", "seats": 5, "engine": "3.0L I6",  "top_speed_kmh": 191, "horsepower": 300},

    # ── AUDI ─────────────────────────────────────────────────────
    "Audi A4 2024":           {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl","top_speed_kmh": 250, "horsepower": 261},
    "Audi Q7 2024":           {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 7, "engine": "3.0L V6",  "top_speed_kmh": 250, "horsepower": 335},
    "Audi e-tron GT 2024":    {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 250, "horsepower": 637},
    "Audi R8 2024":           {"fuel": "Petrol",    "transmission": "Automatic", "seats": 2, "engine": "5.2L V10", "top_speed_kmh": 330, "horsepower": 562},

    # ── PORSCHE ──────────────────────────────────────────────────
    "Porsche 911 2024":       {"fuel": "Petrol",    "transmission": "Automatic", "seats": 4, "engine": "3.0L F6",  "top_speed_kmh": 293, "horsepower": 379},
    "Porsche Cayenne 2024":   {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "3.0L V6",  "top_speed_kmh": 261, "horsepower": 455},
    "Porsche Taycan 2024":    {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 260, "horsepower": 671},

    # ── SUZUKI ───────────────────────────────────────────────────
    "Suzuki Swift 2024":      {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "1.2L 4-cyl","top_speed_kmh": 170, "horsepower": 82},
    "Suzuki Vitara 2024":     {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "1.4L 4-cyl","top_speed_kmh": 185, "horsepower": 127},
    "Suzuki Jimny 2024":      {"fuel": "Petrol",    "transmission": "Manual",    "seats": 4, "engine": "1.5L 4-cyl","top_speed_kmh": 150, "horsepower": 102},

    # ── MITSUBISHI ───────────────────────────────────────────────
    "Mitsubishi Outlander 2024":{"fuel": "Hybrid",  "transmission": "Automatic", "seats": 7, "engine": "2.4L 4-cyl","top_speed_kmh": 200, "horsepower": 248},
    "Mitsubishi Pajero 2023": {"fuel": "Diesel",    "transmission": "Automatic", "seats": 7, "engine": "3.2L 4-cyl","top_speed_kmh": 190, "horsepower": 200},
    "Mitsubishi Eclipse Cross 2024":{"fuel": "Hybrid","transmission": "Automatic","seats": 5, "engine": "2.4L 4-cyl","top_speed_kmh": 190, "horsepower": 248},

    # ── SUBARU ───────────────────────────────────────────────────
    "Subaru Impreza 2024":    {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl","top_speed_kmh": 195, "horsepower": 152},
    "Subaru WRX 2024":        {"fuel": "Petrol",    "transmission": "Manual",    "seats": 5, "engine": "2.4L 4-cyl","top_speed_kmh": 240, "horsepower": 271},
    "Subaru Forester 2024":   {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl","top_speed_kmh": 195, "horsepower": 167},

    # ── LEXUS ─────────────────────────────────────────────────────
    "Lexus RX 2024":          {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "2.5L 4-cyl","top_speed_kmh": 200, "horsepower": 246},
    "Lexus LC 500 2024":      {"fuel": "Petrol",    "transmission": "Automatic", "seats": 4, "engine": "5.0L V8",  "top_speed_kmh": 270, "horsepower": 471},
    "Lexus NX 2024":          {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "2.5L 4-cyl","top_speed_kmh": 200, "horsepower": 243},

    # ── CHEVROLET ────────────────────────────────────────────────
    "Chevrolet Corvette 2024":{"fuel": "Petrol",    "transmission": "Automatic", "seats": 2, "engine": "6.2L V8",  "top_speed_kmh": 312, "horsepower": 495},
    "Chevrolet Camaro 2024":  {"fuel": "Petrol",    "transmission": "Manual",    "seats": 4, "engine": "6.2L V8",  "top_speed_kmh": 290, "horsepower": 455},
    "Chevrolet Tahoe 2024":   {"fuel": "Petrol",    "transmission": "Automatic", "seats": 8, "engine": "5.3L V8",  "top_speed_kmh": 180, "horsepower": 355},

    # ── JEEP ─────────────────────────────────────────────────────
    "Jeep Wrangler 2024":     {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl","top_speed_kmh": 180, "horsepower": 375},
    "Jeep Grand Cherokee 2024":{"fuel": "Hybrid",   "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl","top_speed_kmh": 209, "horsepower": 375},

    # ── LAMBORGHINI ──────────────────────────────────────────────
    "Lamborghini Urus 2024":  {"fuel": "Petrol",    "transmission": "Automatic", "seats": 5, "engine": "4.0L V8",  "top_speed_kmh": 305, "horsepower": 657},
    "Lamborghini Huracan 2024":{"fuel": "Petrol",   "transmission": "Automatic", "seats": 2, "engine": "5.2L V10", "top_speed_kmh": 325, "horsepower": 631},

    # ── FERRARI ──────────────────────────────────────────────────
    "Ferrari 296 GTB 2024":   {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 2, "engine": "3.0L V6",  "top_speed_kmh": 330, "horsepower": 830},
    "Ferrari SF90 2024":      {"fuel": "Hybrid",    "transmission": "Automatic", "seats": 2, "engine": "4.0L V8",  "top_speed_kmh": 340, "horsepower": 986},

    # ── TATA ─────────────────────────────────────────────────────
    "Tata Nexon 2024":        {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 120, "horsepower": 143},
    "Tata Harrier 2024":      {"fuel": "Diesel",    "transmission": "Automatic", "seats": 5, "engine": "2.0L 4-cyl","top_speed_kmh": 180, "horsepower": 170},

    # ── MAHINDRA ─────────────────────────────────────────────────
    "Mahindra XUV700 2024":   {"fuel": "Diesel",    "transmission": "Automatic", "seats": 7, "engine": "2.2L 4-cyl","top_speed_kmh": 180, "horsepower": 185},
    "Mahindra Thar 2024":     {"fuel": "Diesel",    "transmission": "Manual",    "seats": 4, "engine": "2.2L 4-cyl","top_speed_kmh": 155, "horsepower": 130},
    "Mahindra BE.6e 2024":    {"fuel": "Electric",  "transmission": "Automatic", "seats": 5, "engine": "Electric", "top_speed_kmh": 170, "horsepower": 286},
}


# ─────────────────────────────────────────────────────────────
# BODY TYPE INFERENCE RULES (from make+model name)
# ─────────────────────────────────────────────────────────────

BODY_TYPE_RULES = {
    # Keywords in model name → body type
    "pickup":      "Pickup", "truck":   "Pickup", "hilux":  "Pickup",
    "navara":      "Pickup", "ranger":  "Pickup", "f-150":  "Pickup",
    "thar":        "Pickup", "defender":"SUV",     "wrangler":"SUV",
    "land cruiser":"SUV",    "prado":   "SUV",     "rav4":   "SUV",
    "cr-v":        "SUV",    "tiguan":  "SUV",     "tucson": "SUV",
    "sportage":    "SUV",    "seltos":  "SUV",     "creta":  "SUV",
    "x-trail":     "SUV",    "outlander":"SUV",    "pajero": "SUV",
    "fortuner":    "SUV",    "x5":      "SUV",     "gle":    "SUV",
    "q7":          "SUV",    "rx":      "SUV",     "cayenne":"SUV",
    "urus":        "SUV",    "explorer":"SUV",     "tahoe":  "SUV",
    "grand cherokee":"SUV",  "evoque":  "SUV",     "cybertruck":"Pickup",
    "golf":        "Hatchback","swift": "Hatchback","polo":  "Hatchback",
    "i20":         "Hatchback","jazz":  "Hatchback","leaf":  "Hatchback",
    "corolla":     "Sedan",  "camry":   "Sedan",   "civic":  "Sedan",
    "accord":      "Sedan",  "a4":      "Sedan",   "3 series":"Sedan",
    "elantra":     "Sedan",  "passat":  "Sedan",   "impreza":"Sedan",
    "city":        "Sedan",  "vios":    "Sedan",   "almera": "Sedan",
    "model 3":     "Sedan",  "model s": "Sedan",   "ioniq 6":"Sedan",
    "911":         "Coupe",  "mustang": "Coupe",   "camaro": "Coupe",
    "supra":       "Coupe",  "m3":      "Coupe",   "r8":     "Coupe",
    "corvette":    "Coupe",  "stinger": "Coupe",   "lc 500": "Coupe",
    "huracan":     "Coupe",  "296 gtb": "Coupe",   "sf90":   "Coupe",
    "convertible": "Convertible","spider":"Convertible","roadster":"Convertible",
    "model y":     "SUV",    "ioniq 5": "SUV",     "ev6":    "SUV",
    "id.4":        "SUV",    "e-tron":  "SUV",     "ix":     "SUV",
    "model x":     "SUV",    "nx":      "SUV",
}


import os
import csv

def load_csv_data():
    csv_path = "E:/dataset_for_car_AI_Site/Cars Datasets 2025.csv"
    if not os.path.exists(csv_path):
        return
        
    print(f"[Database] Loading premium specifications from {csv_path}...")
    try:
        with open(csv_path, mode='r', encoding='latin-1') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                make = row.get("Company Names", "").strip().title()
                # Clean brand names (e.g. FERRARI -> Ferrari, ROLLS ROYCE -> Rolls-Royce or Mercedes-Benz)
                if make == "Mercedes": make = "Mercedes-Benz"
                elif make == "Rolls Royce": make = "Rolls-Royce"
                
                model = row.get("Cars Names", "").strip()
                engine = row.get("Engines", "").strip()
                cc = row.get("CC/Battery Capacity", "").strip()
                hp_str = row.get("HorsePower", "").strip()
                speed_str = row.get("Total Speed", "").strip()
                price_str = row.get("Cars Prices", "").strip()
                fuel = row.get("Fuel Types", "").strip().title()
                seats_str = row.get("Seats", "").strip()
                
                # Parse numeric values from string fields
                hp = 180
                if hp_str:
                    clean_hp = "".join(c for c in hp_str if c.isdigit() or c == '-')
                    if '-' in clean_hp:
                        clean_hp = clean_hp.split('-')[-1]
                    if clean_hp.isdigit():
                        hp = int(clean_hp)
                        
                speed = 200
                if speed_str:
                    clean_speed = "".join(c for c in speed_str if c.isdigit() or c == '-')
                    if '-' in clean_speed:
                        clean_speed = clean_speed.split('-')[-1]
                    if clean_speed.isdigit():
                        speed = int(clean_speed)
                
                seats = 5
                if seats_str.isdigit():
                    seats = int(seats_str)
                    
                # Format key for years 2020-2026
                for yr in range(2020, 2027):
                    key = f"{make} {model} {yr}"
                    CAR_DATABASE[key] = {
                        "fuel": "Hybrid" if "hybrid" in fuel.lower() else "Electric" if "electric" in fuel.lower() or "battery" in cc.lower() else fuel,
                        "transmission": "Manual" if make in ["Ferrari", "Lamborghini", "Aston Martin"] else "Automatic",
                        "seats": seats,
                        "engine": f"{engine} ({cc})",
                        "top_speed_kmh": speed,
                        "horsepower": hp,
                        "price": price_str,
                    }
                    count += 1
            print(f"[Database] Loaded {count} dynamic car specifications.")
    except Exception as e:
        print(f"[Database] Error reading CSV: {e}")

# Run loader
load_csv_data()


# ─────────────────────────────────────────────────────────────
# LOOKUP FUNCTIONS
# ─────────────────────────────────────────────────────────────

def lookup_specs(make: str, model: str, year: int) -> dict:
    """
    Look up car specs from the database.
    Falls back to model-name-based heuristics if not found.
    Returns a dict with fuel, transmission, seats, engine, top_speed_kmh, horsepower.
    """
    keys_to_try = [
        f"{make} {model} {year}",
        f"{make} {model} {year - 1}",
        f"{make} {model} {year + 1}",
        f"{make} {model} {year - 2}",
    ]

    for key in keys_to_try:
        if key in CAR_DATABASE:
            specs = CAR_DATABASE[key].copy()
            specs["found_in_db"] = True
            return specs

    # Fallback: partial match or word overlap on model name
    make_clean = make.lower()
    model_clean = model.lower()
    for key, specs in CAR_DATABASE.items():
        key_lower = key.lower()
        if make_clean in key_lower:
            key_model = key_lower.replace(make_clean, "").strip()
            # remove year
            key_model_words = [w for w in key_model.split() if not w.isdigit()]
            model_words = [w for w in model_clean.split()]
            
            # Check if there is intersection of model words
            overlap = set(key_model_words) & set(model_words)
            if overlap:
                result = specs.copy()
                result["found_in_db"] = True
                return result

    # Final fallback: rule-based estimates
    return _heuristic_specs(make, model, year)


def infer_body_type_from_name(make: str, model: str) -> str:
    """
    Infer body type from make/model name keywords.
    Used as backup when the CNN isn't confident.
    """
    combined = f"{make} {model}".lower()
    for keyword, body_type in BODY_TYPE_RULES.items():
        if keyword in combined:
            return body_type
    return "Sedan"   # Most common default


def _heuristic_specs(make: str, model: str, year: int) -> dict:
    """Rule-based fallback when not in database."""
    model_lower = model.lower()

    # Fuel type guess
    if any(k in model_lower for k in ["electric", "ev", "ioniq", "leaf", "id.", "e-tron", "taycan", "model"]):
        fuel = "Electric"
    elif any(k in model_lower for k in ["hybrid", "prius", "rav4 hybrid"]):
        fuel = "Hybrid"
    elif any(k in model_lower for k in ["diesel", "d4", "tdi", "cdi", "navara", "hilux", "ranger", "pajero"]):
        fuel = "Diesel"
    else:
        fuel = "Petrol"

    # Transmission guess
    if any(k in model_lower for k in ["gtr", "r8", "911", "m3", "wrx", "jimny", "thar", "mustang gt350"]):
        transmission = "Manual"
    else:
        transmission = "Automatic"

    # Seats guess
    if any(k in model_lower for k in ["land cruiser", "7", "tahoe", "explorer", "outlander", "x-trail", "fortuner"]):
        seats = 7
    elif any(k in model_lower for k in ["911", "huracan", "r8", "corvette", "roadster"]):
        seats = 2
    else:
        seats = 5

    # Premium/Luxury brand defaults
    horsepower = 150
    top_speed = 180
    engine = "N/A"
    price = None

    make_lower = make.lower()
    if any(m in make_lower for m in ["maserati", "ferrari", "lamborghini", "porsche", "bugatti", "aston martin"]):
        horsepower = 550
        top_speed = 310
        engine = "V6 Twin-Turbo" if "maserati" in make_lower else "V8 Twin-Turbo"
        price = "$185,000"

    result = {
        "fuel":          fuel,
        "transmission":  transmission,
        "seats":         seats,
        "engine":        engine,
        "top_speed_kmh": top_speed,
        "horsepower":    horsepower,
        "found_in_db":   False,
    }
    if price:
        result["price"] = price
    return result


def get_all_makes() -> list:
    """Return all unique car makes in the database."""
    makes = set()
    for key in CAR_DATABASE:
        makes.add(key.split()[0])
    return sorted(makes)


def get_models_for_make(make: str) -> list:
    """Return all models for a given make."""
    models = []
    for key in CAR_DATABASE:
        if key.startswith(make):
            parts = key.split()
            model = " ".join(parts[1:-1])  # remove make and year
            if model not in models:
                models.append(model)
    return sorted(models)


if __name__ == "__main__":
    print(f"Total cars in database: {len(CAR_DATABASE)}")
    print(f"Total makes: {len(get_all_makes())}")
    print(f"\nMakes: {', '.join(get_all_makes())}")
    print(f"\nSample lookup:")
    specs = lookup_specs("Tesla", "Model 3", 2024)
    print(f"  Tesla Model 3 2024: {specs}")
    specs2 = lookup_specs("Toyota", "Camry", 2022)
    print(f"  Toyota Camry 2022: {specs2}")
