"""
Vehicle AI — API Test Script
Tests the FastAPI server endpoints before connecting to Next.js.

Usage:
    # With the server running (python src/api.py in another terminal):
    python tests/test_api.py
    python tests/test_api.py path/to/car.jpg
"""

import sys
import json
import time
from pathlib import Path

try:
    import requests
    from PIL import Image
    import io
    import numpy as np
except ImportError:
    print("Run: pip install requests pillow numpy")
    sys.exit(1)

BASE_URL = "http://localhost:8000"


def print_header(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def test_health():
    print_header("Health Check — GET /health")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        data = r.json()
        print(f"  Status     : {r.status_code}")
        print(f"  API status : {data.get('status')}")
        print(f"  Model ready: {data.get('model_loaded')}")
        assert r.status_code == 200, "Health check failed!"
        assert data.get("status") == "ok", "Status not OK!"
        print("  ✅ PASSED")
        return True
    except requests.exceptions.ConnectionError:
        print("  ❌ FAILED — Is the server running?")
        print("     Start it with: python src/api.py")
        return False


def test_predict_dummy():
    print_header("Prediction Test — POST /predict (dummy image)")

    # Create a simple test image
    img = Image.new("RGB", (224, 224), color=(180, 100, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    try:
        start = time.time()
        r = requests.post(
            f"{BASE_URL}/predict",
            files={"file": ("test.jpg", buf, "image/jpeg")},
            timeout=60,
        )
        elapsed = (time.time() - start) * 1000
        data = r.json()

        print(f"  Status    : {r.status_code}")
        print(f"  Took      : {elapsed:.0f}ms")

        if r.status_code == 200 and data.get("success"):
            pred = data["data"]
            print(f"\n  Prediction:")
            for k, v in pred.items():
                print(f"    {k:<15}: {v}")
            print(f"\n  ✅ PASSED — Confidence: {pred['confidence']:.1%}")
            return True
        else:
            print(f"  ❌ FAILED: {data}")
            return False

    except requests.exceptions.ConnectionError:
        print("  ❌ FAILED — Server not running")
        return False


def test_predict_real_image(image_path: str):
    print_header(f"Prediction Test — Real Image: {Path(image_path).name}")

    try:
        with open(image_path, "rb") as f:
            content_type = "image/jpeg"
            if image_path.endswith(".png"):   content_type = "image/png"
            elif image_path.endswith(".webp"): content_type = "image/webp"

            start = time.time()
            r = requests.post(
                f"{BASE_URL}/predict",
                files={"file": (Path(image_path).name, f, content_type)},
                timeout=120,
            )
            elapsed = (time.time() - start) * 1000

        data = r.json()
        if r.status_code == 200 and data.get("success"):
            pred = data["data"]
            print(f"\n  Prediction:")
            for k, v in pred.items():
                print(f"    {k:<15}: {v}")
            print(f"\n  Took      : {elapsed:.0f}ms")
            print(f"  ✅ PASSED — Confidence: {pred['confidence']:.1%}")
        else:
            print(f"  ❌ FAILED: {data}")

    except FileNotFoundError:
        print(f"  ❌ File not found: {image_path}")


def test_invalid_file():
    print_header("Validation Test — Invalid File Type")
    try:
        r = requests.post(
            f"{BASE_URL}/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            timeout=10,
        )
        print(f"  Status: {r.status_code}")
        assert r.status_code == 400, "Expected 400 for invalid file"
        print("  ✅ PASSED — Correctly rejected invalid file")
    except requests.exceptions.ConnectionError:
        print("  ❌ Server not running")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print("\nVehicle AI - API Test Suite")
    print(f"   Target: {BASE_URL}")

    all_passed = True
    all_passed &= test_health()

    if all_passed:
        all_passed &= test_predict_dummy()
        test_invalid_file()

        if len(sys.argv) > 1:
            test_predict_real_image(sys.argv[1])

    print(f"\n{'='*50}")
    print(f"  {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print(f"{'='*50}\n")
