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

DECEPTIVE_PATTERNS = [
    r'\bclick here\b', r'\bclick.*link\b', r'\bdo not.*ignore\b',
    r'\bdo not.*delete\b', r'\bnever share\b', r'\bbelow.*link\b',
    r'\blogin.*here\b', r'\bsign.*in.*here\b', r'\bupdate.*now\b',
]

SPOOFING_PATTERNS = [
    r'\bpaypal\b', r'\bamazon\b', r'\bnetflix\b', r'\bapple\b',
    r'\bmicrosoft\b', r'\bgoogle\b', r'\bfacebook\b', r'\binstagram\b',
    r'\bsbi\b', r'\bhdfc\b', r'\bicici\b', r'\baxis bank\b',
    r'\bincome tax\b', r'\birdai\b', r'\bsebi\b',
]

SUSPICIOUS_URL_PATTERNS = [
    r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP-based URL
    r'http://',  # Non-HTTPS
    r'secure.*login.*\.(?!com\b|org\b|gov\b)',
    r'account.*verify.*\.',
    r'update.*info.*\.',
    r'\.tk\b', r'\.ml\b', r'\.ga\b', r'\.cf\b', r'\.gq\b',  # Free TLDs
    r'bit\.ly', r'tinyurl', r'goo\.gl', r't\.co',  # URL shorteners
]

GENERIC_GREETING = [
    r'\bdear customer\b', r'\bdear user\b', r'\bdear member\b',
    r'\bdear account holder\b', r'\bvalued customer\b',
    r'\bto whom it may concern\b',
]

# ─── Feature Engineering ─────────────────────────────────────────────────────
class SecurityFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts hand-crafted security features from email text."""

    def _count_urls(self, text):
        return len(re.findall(r'https?://\S+', text))

    def _count_suspicious_urls(self, text):
        count = 0
        for p in SUSPICIOUS_URL_PATTERNS:
            count += len(re.findall(p, text, re.I))
        return count

    def _count_pattern_group(self, text, patterns):
        return sum(1 for p in patterns if re.search(p, text, re.I))

    def _html_indicator(self, text):
        html_tags = len(re.findall(r'<[a-z][^>]*>', text, re.I))
        return min(html_tags, 20)

    def _special_char_ratio(self, text):
        if not text: return 0
        specials = sum(1 for c in text if c in '!@#$%^&*()_+[]{}|;:,.<>?')
        return specials / max(len(text), 1)

    def _caps_ratio(self, text):
        if not text: return 0
        letters = [c for c in text if c.isalpha()]
        if not letters: return 0
        return sum(1 for c in letters if c.isupper()) / len(letters)

    def _word_count(self, text):
        return len(text.split())

    def _avg_word_len(self, text):
        words = text.split()
        if not words: return 0
        return sum(len(w) for w in words) / len(words)

    def _exclamation_count(self, text):
        return min(text.count('!'), 10)

    def _question_count(self, text):
        return min(text.count('?'), 10)

    def _has_unsubscribe(self, text):
        return 1 if re.search(r'\bunsubscribe\b', text, re.I) else 0

    def _has_dear_name(self, text):
        # Legitimate emails often address by name
        return 1 if re.search(r'\bdear\s+[A-Z][a-z]+\b', text) else 0

    def fit(self, X, y=None): return self

