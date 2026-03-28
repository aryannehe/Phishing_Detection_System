"""
PhishDetect ML Training Script
Uses TF-IDF + engineered features + Voting Ensemble for maximum accuracy.
No external dataset download needed — uses a large synthetic corpus that mirrors
real phishing and legitimate email patterns.
"""

import pickle, re, string, math
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# ─── Phishing indicator patterns ────────────────────────────────────────────
URGENT_PATTERNS = [
    r'\burgent\b', r'\bimmediately\b', r'\baction required\b',
    r'\baccount.*suspend', r'\bverif', r'\bconfirm.*identity',
    r'\bwithin 24 hours?\b', r'\bwithin 48 hours?\b', r'\bexpire[sd]?\b',
    r'\blimited time\b', r'\bact now\b', r'\bwarning\b', r'\balert\b',
    r'\bsuspicious activity\b', r'\bunusual.*activit', r'\bsecurity.*breach',
    r'\byour account\b.*\block', r'\bblock.*your account\b',
]

