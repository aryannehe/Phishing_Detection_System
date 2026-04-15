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
        'id': 'suspicious_url',
        'label': 'Suspicious or malformed URL',
        'patterns': [
            r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
            r'http://',
            r'bit\.ly', r'tinyurl', r'goo\.gl',
            r'\.tk\b', r'\.ml\b', r'\.ga\b', r'\.cf\b', r'\.gq\b',
            r'secure.*login.*\.',
            r'account.*verify.*\.',
            r'verify.*account.*\.',
        ],
    },
    {
        'id': 'sensitive_data',
        'label': 'Request for sensitive / personal information',
        'patterns': [
            r'\bpassword\b', r'\bcredit card\b', r'\bssn\b',
            r'\bsocial security\b', r'\bbank account\b', r'\bpin\b',
            r'\bcvv\b', r'\botp\b', r'\bpan card\b', r'\baadhaar\b',
            r'\bkyc\b', r'\bifsc\b', r'\bmother.*maiden\b',
            r'\bdate of birth\b', r'\bnetbanking\b',
        ],
    },
    {
        'id': 'financial_lure',
        'label': 'Financial lure / prize / lottery',
        'patterns': [
            r'\blottery\b', r'\bwon\b.*\bprize\b', r'\bcongratulation',
            r'\bmillion.*dollar', r'\binheritance\b', r'\brefund\b',
            r'\btax.*return\b', r'\bclaim.*reward', r'\bfree.*gift\b',
            r'\bwire.*transfer\b', r'\bwestern union\b',
        ],
    },
    {
        'id': 'brand_impersonation',
        'label': 'Brand / institution impersonation',
        'patterns': [
            r'\bpaypal\b', r'\bamazon\b', r'\bnetflix\b',
            r'\bapple\b', r'\bmicrosoft\b', r'\bgoogle\b',
            r'\bfacebook\b', r'\bsbi\b', r'\bhdfc\b', r'\bicici\b',
            r'\baxis bank\b', r'\bincome tax\b', r'\birdai\b',
            r'\bsebi\b', r'\birdai\b',
        ],
    },
    {
        'id': 'generic_greeting',
        'label': 'Generic / impersonal salutation',
        'patterns': [
            r'\bdear customer\b', r'\bdear user\b',
            r'\bdear member\b', r'\bdear account holder\b',
            r'\bvalued customer\b', r'\bto whom it may concern\b',
        ],
    },
    {
        'id': 'upi_fraud',
        'label': 'UPI / digital payment fraud indicators',
        'patterns': [
            r'\bupi\b', r'\bupi.*pin\b', r'\bgooglepay\b',
            r'\bphonepay\b', r'\bpaytm\b', r'\bbhim\b',
        ],
    },
    {
        'id': 'excessive_caps',
        'label': 'Excessive use of capital letters',
        'check': lambda text: (
            sum(1 for c in text if c.isupper()) / max(len([c for c in text if c.isalpha()]), 1)
        ) > 0.4 and len(text) > 30,
    },
    {
        'id': 'many_exclamations',
        'label': 'Excessive exclamation marks',
        'check': lambda text: text.count('!') >= 3,
    },
]

def detect_indicators(text):
    found = []
    text_lower = text.lower()
    for rule in INDICATOR_RULES:
        triggered = False
        if 'patterns' in rule:
            for pattern in rule['patterns']:
                if re.search(pattern, text_lower):
                    triggered = True
                    break
        elif 'check' in rule:
            try:
                triggered = rule['check'](text)
            except:
                pass
        if triggered:
            found.append(rule['label'])
    return found


def recommended_actions(classification, indicators):
    if classification == 'Phishing':
        actions = [
            "Do not click any links or download attachments in this email.",
            "Delete this email immediately from your inbox and trash.",
            "Do not reply to or forward this email.",
            "Never provide personal, financial, or login information via email.",
            "Report this email to your email provider as phishing/spam.",
        ]
        if 'Request for sensitive / personal information' in indicators:
            actions.append("If you already shared information, contact your bank or relevant institution immediately.")
        if 'Brand / institution impersonation' in indicators:
            actions.append("Contact the impersonated organization directly using their official website or phone number.")
        if 'UPI / digital payment fraud indicators' in indicators:
            actions.append("Alert your bank about potential UPI fraud attempts.")
        if 'Suspicious or malformed URL' in indicators:
            actions.append("If you clicked any link, run a security scan on your device immediately.")
    else:
        actions = [
            "This email appears legitimate based on our analysis.",
            "Always verify sender addresses even in legitimate-looking emails.",
            "Exercise caution before clicking links even in trusted emails.",
        ]
    return actions


def preprocess(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\d{10,}', ' PHONE ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_file('index.html')


