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
INDICATOR_RULES = [
    {
        'id': 'urgent_language',
        'label': 'Urgent / threatening language',
        'patterns': [
            r'\burgent\b', r'\bimmediately\b', r'\baction required\b',
            r'\bwithin 24 hours?\b', r'\bwithin 48 hours?\b',
            r'\bact now\b', r'\bwarning\b', r'\balert\b',
            r'\bdo not ignore\b', r'\bfinal notice\b', r'\blast chance\b',
        ],
    },
    {
        'id': 'account_threat',
        'label': 'Account suspension / closure threat',
        'patterns': [
            r'\baccount.*suspend', r'\bsuspend.*account',
            r'\baccount.*block', r'\bblock.*account',
            r'\bpermanent.*clos', r'\bterminat.*account',
            r'\baccount.*at risk\b',
        ],
    },
    {
