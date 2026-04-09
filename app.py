"""
PhishDetect Backend — Flask REST API
POST /scan-email  →  phishing classification + detailed analysis
"""
from train_model import CombinedFeatures, SecurityFeatureExtractor
import pickle, re, json, os, sys, traceback
from flask import Flask, request, jsonify, send_file

app = Flask(__name__, static_folder='.', static_url_path='')

# ─── Load model ──────────────────────────────────────────────────────────────
