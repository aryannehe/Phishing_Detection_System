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

    def transform(self, X):
        features = []
        for text in X:
            text = text or ''
            f = [
                self._count_urls(text),
                self._count_suspicious_urls(text),
                self._count_pattern_group(text, URGENT_PATTERNS),
                self._count_pattern_group(text, SENSITIVE_PATTERNS),
                self._count_pattern_group(text, FINANCIAL_LURE_PATTERNS),
                self._count_pattern_group(text, DECEPTIVE_PATTERNS),
                self._count_pattern_group(text, SPOOFING_PATTERNS),
                self._count_pattern_group(text, GENERIC_GREETING),
                self._html_indicator(text),
                self._special_char_ratio(text),
                self._caps_ratio(text),
                self._word_count(text),
                self._avg_word_len(text),
                self._exclamation_count(text),
                self._question_count(text),
                self._has_unsubscribe(text),
                self._has_dear_name(text),
                len(text),
            ]
            features.append(f)
        return np.array(features, dtype=float)


def preprocess(text):
    """Clean and normalize email text."""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)       # strip HTML
    text = re.sub(r'https?://\S+', ' URL ', text)  # replace URLs
    text = re.sub(r'\S+@\S+', ' EMAIL ', text) # replace emails
    text = re.sub(r'\d{10,}', ' PHONE ', text)  # long numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─── Dataset ─────────────────────────────────────────────────────────────────
PHISHING_EMAILS = [
    # Account suspension / credential theft
    ("URGENT: Verify your account now", "Dear Customer, Your account has been suspended due to unusual activity. Click here immediately to verify your identity and restore access: http://secure-account-verify-login.com/restore?id=123. Failure to act within 24 hours will result in permanent account closure. Do not ignore this message."),
    ("Action Required: Update Your Banking Information", "Dear Valued Customer, We have detected suspicious activity on your SBI NetBanking account. Your account will be blocked unless you update your information immediately. Visit http://sbi-secure-update.tk/login to confirm your details. Provide your PAN card, Aadhaar number, and OTP for verification."),
    ("Your PayPal Account is Limited", "Dear PayPal User, We've noticed some unusual activity on your account. Your account access has been limited. To restore full access, please verify your credit card and confirm your date of birth at http://paypal-verify.ml/confirm. This is urgent - act now!"),
    ("Microsoft Security Alert", "Warning! Your Microsoft account has been compromised. Sign in here immediately to secure your account: http://192.168.1.1/microsoft-secure. Enter your password and security question answers to prevent unauthorized access."),
    ("Amazon: Your order has been cancelled", "Dear Customer, Your recent Amazon order #928374 has been flagged for suspicious activity. To prevent account suspension, verify your billing information now: http://amzon-secure.ga/verify. Enter your credit card number and CVV."),
    ("IRS Tax Refund Notice", "Congratulations! You are eligible for a tax refund of $3,847. To claim your refund immediately, provide your social security number and bank account routing number at http://irs-refund-claim.cf/claim. This offer expires in 24 hours."),
    ("HDFC Bank: KYC Update Required", "Dear Account Holder, Your HDFC Bank account KYC is incomplete. Your account will be frozen within 48 hours. Update your KYC details including Aadhaar, PAN, and date of birth at http://hdfc-kyc-update.tk/form. Enter your IFSC code and netbanking password."),
    ("Netflix: Payment Failed - Update Now", "Dear Netflix Member, Your payment method has expired and your account will be suspended. Update your credit card details immediately at http://netflix-billing.ml/update. Enter your full card number, expiry date, and CVV to continue your subscription."),
    ("URGENT: Your Apple ID has been locked", "Your Apple ID has been locked due to too many failed login attempts. To unlock your account, click here: http://apple-id-unlock.cf/restore. Provide your Apple ID password, date of birth, and mother's maiden name to verify your identity."),
    ("Lottery Winner Notification", "CONGRATULATIONS! You have won $5,000,000 in the International Lottery. To claim your prize, send your full name, bank account number, and a processing fee of $500 via Western Union to lottery@prize-claim.ml. This is a limited time offer!"),
    ("ICICI Bank: Unusual Login Detected", "Dear Customer, We detected an unusual login to your ICICI Bank account from an unknown device. Your account is at risk. Verify your identity immediately at http://icici-secure-login.ga/verify. Enter your debit card PIN and OTP sent to your registered mobile."),
    ("Security Alert: Suspicious Sign-in", "Warning! Someone tried to sign in to your Google account from Russia. If this wasn't you, click here to secure your account: http://google-security.tk/protect. Enter your password and recovery email to confirm your identity."),
    ("Free Gift Card - Claim Now", "You have been selected to receive a FREE $500 Amazon Gift Card! Click the link below to claim your gift: http://free-gifts.ml/amazon. Act now - only 10 cards remaining! No purchase necessary. Enter your name, email, and phone number."),
    ("Income Tax Department: Notice", "This is to inform you that you have an outstanding tax liability of Rs. 45,000. Failure to pay within 24 hours will result in legal action. Pay now at http://incometax-notice.tk/pay and provide your PAN card details to avoid arrest warrant."),
    ("Axis Bank: Card Blocked", "Dear Cardholder, Your Axis Bank credit card ending in 4521 has been blocked due to suspicious transactions. Unblock your card immediately at http://axis-unblock.ml/card. Enter your full card number, PIN, and OTP to restore services."),
    ("UPI Fraud Alert - Action Required", "Your UPI ID has been used for an unauthorized transaction of Rs. 9,999. To reverse this transaction, provide your UPI PIN and linked bank account details at http://upi-fraud-reverse.ga/claim. Hurry - this link expires in 1 hour."),
    ("Verify Your Identity - Account at Risk", "To whom it may concern, Our system has flagged your account for unusual behavior. To prevent permanent closure, verify your identity immediately. Click here and provide your password, date of birth, and security question: http://verify-now.cf/secure"),
    ("SEBI: Your Demat Account", "Dear Investor, SEBI has detected irregularities in your Demat account. Your account will be frozen unless you verify your identity within 24 hours. Provide your PAN, Aadhaar, and bank account number at http://sebi-verify.tk/demat"),
    ("Facebook: Someone accessed your account", "Warning: Someone logged into your Facebook account from a new device. If this wasn't you, click here immediately: http://facebook-secure.ml/protect. Enter your password to secure your account and prevent unauthorized access."),
    ("Claim Your Inheritance", "Dear Friend, I am Mr. James Smith, attorney representing the late Dr. Robert Brown who died leaving $8.5 million. You share the same surname and are the beneficiary. Send your bank account details and $200 processing fee via Western Union to claim this inheritance."),
