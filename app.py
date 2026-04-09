"""
PhishDetect Backend — Flask REST API
POST /scan-email  →  phishing classification + detailed analysis
"""
from train_model import CombinedFeatures, SecurityFeatureExtractor
import pickle, re, json, os, sys, traceback
from flask import Flask, request, jsonify, send_file

app = Flask(__name__, static_folder='.', static_url_path='')

# ─── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'phish_model.pkl')
model_data = None

def load_model():
    global model_data
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        print(f"[PhishDetect] Model loaded (v{model_data.get('version','?')})")
    except Exception as e:
        print(f"[PhishDetect] ERROR loading model: {e}")
        sys.exit(1)

# ─── Indicator detection ─────────────────────────────────────────────────────
