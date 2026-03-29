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

SENSITIVE_PATTERNS = [
    r'\bpassword\b', r'\bcredit card\b', r'\bssn\b', r'\bsocial security\b',
    r'\bbank account\b', r'\brouting number\b', r'\bpin\b',
    r'\bdate of birth\b', r'\bmother.*maiden\b', r'\bsecurity question\b',
    r'\bpan card\b', r'\baadhar\b', r'\bkyc\b', r'\bnetbanking\b',
    r'\bupi\b', r'\bifsc\b', r'\botp\b', r'\bcvv\b',
]

FINANCIAL_LURE_PATTERNS = [
    r'\bwon\b.*\bprize\b', r'\blottery\b', r'\bmillion\b.*\bdollar',
    r'\brefund\b', r'\btax.*return\b', r'\bclaim.*reward',
    r'\bfree.*gift\b', r'\bcongratulation', r'\binheritance\b',
    r'\bwire.*transfer\b', r'\bwestern union\b', r'\bmoneygram\b',
]

