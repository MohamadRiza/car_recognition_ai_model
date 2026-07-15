# Vehicle AI Model 🚗🤖

Custom AI model that replaces Google Gemini for car image analysis in the Vehicle Next.js app.
Built with **PyTorch** + **EfficientNet-B0** (transfer learning) + **FastAPI**.

---

## 📁 Project Structure

```
ai-model/
├── src/
│   ├── config.py     ← All settings (paths, hyperparameters, classes)
│   ├── dataset.py    ← Data pipeline (transforms, DataLoader, label maps)
│   ├── model.py      ← EfficientNet-B0 multi-head architecture
│   ├── train.py      ← Training script
│   ├── predict.py    ← Inference engine
│   └── api.py        ← FastAPI server
├── tests/
│   └── test_api.py   ← API test suite
├── notebooks/        ← Jupyter notebooks for exploration
├── data/
│   ├── raw/          ← Training images (not committed)
│   └── processed/    ← Preprocessed tensors
├── models/saved/     ← Saved checkpoints (.pth, .onnx)
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚡ Quick Start

### 1. Create Virtual Environment

```powershell
cd "d:\A-Projects\Vehicle app Next JS\ai-model"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Copy Environment File

```powershell
Copy-Item .env.example .env
```

### 4. Start the API Server

```powershell
cd src
python api.py
```

The server starts at **http://localhost:8000**
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### 5. Test the API

```powershell
# In a new terminal (with venv active)
python tests/test_api.py

# With a real car image
python tests/test_api.py path/to/car.jpg
```

---

## 🧠 Training the Model

### Step 1: Get Training Data

Option A — Stanford Cars Dataset (via Kaggle):
```powershell
pip install kaggle
# Place your kaggle.json in C:\Users\<you>\.kaggle\
kaggle datasets download -d rickyyyyyyy/torchvision-stanford-cars
# Unzip to data/raw/
```

Option B — Your own images:
```
data/raw/
  train/
    SUV/       ← drop SUV images here
    Sedan/
    Hatchback/
  val/
    SUV/
    Sedan/
    Hatchback/
```

### Step 2: Train

```powershell
cd src
python train.py
python train.py --epochs 30 --lr 5e-5 --batch 8
```

### Step 3: Model is saved to `models/saved/car_model_best.pth`

---

## 🔌 Connecting to Next.js

Once the API server is running, update `action/cars.js` in the Next.js app:

```js
// Replace this Gemini call:
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// With this fetch to our local API:
const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,  // FormData with the image file
});
const result = await response.json();
```

---

## 🧪 API Reference

### `POST /predict`

**Request:** multipart/form-data with `file` field (JPEG/PNG/WebP, max 10MB)

**Response:**
```json
{
  "success": true,
  "data": {
    "make":         "Unknown",
    "model":        "Unknown",
    "year":         2022,
    "color":        "White",
    "price":        "32000",
    "mileage":      "28000",
    "bodytype":     "SUV",
    "fueltype":     "Petrol",
    "transmission": "Automatic",
    "description":  "A sleek white SUV...",
    "confidence":   0.78
  },
  "took_ms": 245.3
}
```

---

## 📊 Model Details

| Component | Details |
|---|---|
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Heads | Body Type (7), Fuel Type (5), Transmission (3), Color (12) |
| Input | 224×224 RGB image |
| Framework | PyTorch 2.3 |
| Device | CPU (Intel HD 4000) |
| Export | ONNX (for production) |

---

## 🗺️ Roadmap

- [x] Environment setup
- [x] Model architecture (EfficientNet-B0 multi-head)
- [x] Training pipeline
- [x] FastAPI server
- [ ] Dataset download & preprocessing
- [ ] Fine-tune on Stanford Cars Dataset
- [ ] Add make/model classification head
- [ ] Export to ONNX
- [ ] Connect to Next.js app
- [ ] Deploy (e.g., Docker + cloud)
