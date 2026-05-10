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
    ("Your Package Could Not Be Delivered", "Dear Customer, We were unable to deliver your package. A re-delivery fee of Rs. 49 is required. Pay now at http://delivery-fee.tk/pay and confirm your address and credit card details to reschedule your delivery."),
    ("IRDAI: Insurance Policy Suspended", "Dear Policyholder, Your insurance policy has been suspended due to KYC non-compliance. Restore your policy immediately by submitting your Aadhaar, PAN, and bank details at http://irdai-kyc.ml/restore. Failure to comply will result in policy termination."),
    ("COVID Relief Fund - Apply Now", "You are eligible for Rs. 50,000 COVID relief fund from the government. Apply now at http://covid-relief.tk/apply. Submit your Aadhaar, PAN, bank account number, and OTP to receive the funds directly in your account within 24 hours."),
    ("Your Email Account Will Be Closed", "Dear Email User, Your email account storage is full and will be permanently deleted in 24 hours unless you verify your account. Click here to verify: http://email-verify.ga/confirm. Enter your email password and recovery phone number immediately."),
    ("HR Department: Salary Revision", "Dear Employee, We are processing salary revisions. To receive your revised salary, update your bank account details at http://hr-salary.ml/update. Provide your account number, IFSC code, and employee ID. This is urgent - deadline is today."),
    ("Prize Winner: You Won iPhone 15", "CONGRATULATIONS! You have been randomly selected to WIN an iPhone 15 Pro! To claim your prize, pay a shipping fee of $50 at http://prize-claim.cf/iphone. Enter your credit card details and home address. Offer valid for 24 hours only!"),
    ("Urgent: PayPal Money Waiting", "Dear User, Someone has sent you $2,500 via PayPal. To receive this money, verify your account at http://paypal-receive.tk/money. Enter your PayPal login, credit card, and bank account number. The funds will expire if not claimed within 48 hours."),
    ("Bank Merger Notice - Action Required", "Dear Valued Customer, Due to the upcoming bank merger, all customers must re-verify their KYC. Visit http://bank-merger-kyc.ml/verify and submit your Aadhaar, PAN, photo, and signature. Accounts not verified will be closed permanently."),
    ("Suspicious Transaction Detected", "Alert! A transaction of Rs. 25,000 was initiated from your account. If you did not authorize this, click here immediately to cancel: http://cancel-transaction.ga/stop. Provide your debit card PIN and OTP to reverse this transaction within 30 minutes."),
    ("Apple iTunes Billing", "Your Apple account has been charged $299 for iTunes subscription renewal. If you did not authorize this charge, dispute it immediately at http://apple-billing-dispute.cf/cancel. Enter your Apple ID, password, and credit card to get a refund."),
    # Legitimate-looking but with red flags
    ("Account Security Update", "Dear account holder, this is a final warning that your account security is outdated. You must update your security settings and password immediately at http://bit.ly/update-security. Do not ignore this urgent security notice."),
    ("Your Subscription Has Expired", "Dear Customer, your premium subscription has expired. To continue enjoying our services, renew now by clicking http://tinyurl.com/renew-now and entering your credit card number, CVV, and billing address. Act fast - offer expires tonight!"),
]

LEGITIMATE_EMAILS = [
    # Professional / business emails
    ("Meeting Tomorrow at 2 PM", "Hi Sarah, just wanted to confirm our meeting tomorrow at 2 PM in Conference Room B. The agenda will cover Q3 performance metrics and the new product roadmap. Please bring your laptop. Let me know if you need to reschedule. Best regards, John"),
    ("Project Update - Development Sprint", "Hi team, quick update on the current sprint. We've completed the user authentication module and are 70% through the payment integration. The estimated completion date is Friday. Please update your tasks in Jira before the standup. Thanks, Dev Team"),
    ("Quarterly Report Available", "Dear Shareholders, the Q3 2024 financial report is now available on our investor relations portal. Please log in to your registered account at investors.company.com to access the full report, earnings call transcript, and analyst presentations. Regards, Investor Relations"),
    ("Welcome to Your New Role", "Dear Michael, on behalf of the entire team, welcome aboard! Your first day is Monday, September 4th. Please bring two forms of government-issued ID for HR processing. Your manager, Lisa Chen, will be in touch to discuss your onboarding schedule. We're excited to have you join us!"),
    ("Invoice #INV-2024-0892 Attached", "Hi, please find attached invoice #INV-2024-0892 for the consulting services provided in August. The total amount due is $4,500, payable within 30 days. Bank transfer details are on the invoice. Feel free to reach out if you have any questions. Thank you for your business."),
    ("Team Lunch This Friday", "Hey everyone! We're doing a team lunch this Friday at Olive Garden (the one on Main Street) at 12:30 PM. RSVP by Thursday so we can make reservations. Looking forward to seeing everyone! - Rachel"),
    ("Software Deployment Notification", "This is an automated notification that a new software update will be deployed to your system this weekend during the maintenance window (Saturday 2-4 AM). There may be brief service interruptions. No action is required on your part. Contact IT helpdesk if you experience issues after the maintenance window."),
    ("Conference Registration Confirmed", "Dear Dr. Patel, thank you for registering for the Annual Data Science Conference 2024. Your registration ID is DS2024-8823. The conference will be held at the Marriott Downtown from October 15-17. Your badge and session details will be emailed one week before the event."),
    ("Monthly Newsletter - October 2024", "Hello valued subscriber, here's what's new this month at TechCorp. We've launched three new features based on your feedback, expanded our customer support to 24/7, and published two new case studies. Read more on our blog. To unsubscribe from these emails, click here."),
    ("Feedback Request: Recent Purchase", "Hi there, thank you for your recent purchase from BookStore Online. We hope you're enjoying your books! If you have a moment, we'd love to hear your feedback. Please rate your experience using the link below. Your input helps us serve you better. - Customer Experience Team"),
    ("Annual Performance Review Scheduled", "Dear Employee, your annual performance review has been scheduled for October 28th at 3:00 PM with your manager. Please complete the self-assessment form on the HR portal before the meeting. The form will be available for 10 days. Contact HR if you have any questions."),
    ("Library Book Due Reminder", "Dear Library Member, this is a friendly reminder that the following books are due in 3 days: 'Clean Code' by Robert Martin and 'Design Patterns' by GoF. You can renew them online at library.edu/renew or at the front desk. Late fees are $0.25 per day."),
    ("Flight Confirmation - AA1234", "Dear Passenger, your flight booking is confirmed. Flight AA1234, departing October 20 at 8:45 AM from JFK to LAX. Seat 14C (Economy, aisle). Check-in opens 24 hours before departure. Manage your booking at aa.com. Have a pleasant journey!"),
    ("Gym Membership Renewal", "Hi Alex, your gym membership expires on November 1st. Renew for another year and save 15% with our loyalty discount. Visit your member portal at fitnessclub.com/renew or stop by the front desk. Your current member ID is 84721. We hope to continue seeing you!"),
    ("Study Group Meeting - Thursday", "Hey everyone, our study group for the Machine Learning exam is meeting Thursday at 6 PM in the library, Room 204. We'll cover chapters 8-12. Please bring your notes and the textbook. Priya will bring the past exam papers. See you there!"),
    ("Order Shipped - Order #ORD-2024-50391", "Hi there, great news! Your order #ORD-2024-50391 has been shipped. Estimated delivery: October 18-20. You can track your package using tracking number 1Z999AA10123456784 on UPS.com. If you have questions about your order, contact us at support@shop.com"),
    ("Dentist Appointment Reminder", "Hi, this is a reminder of your dental appointment with Dr. Williams on Tuesday, October 22 at 10:00 AM. Please arrive 15 minutes early to complete paperwork. Call 555-123-4567 to reschedule if needed. We look forward to seeing you!"),
    ("Python Workshop Materials", "Hi participants, thank you for attending yesterday's Python workshop! As promised, here are the slide decks, code samples, and additional resources for further learning. The GitHub repository is available at github.com/workshop/python-intro. Let me know if you have questions."),
    ("Parent-Teacher Conference Schedule", "Dear Parent/Guardian, parent-teacher conferences are scheduled for November 8th from 4-7 PM. Please use the online scheduler at school.edu/conferences to book your 15-minute slot with your child's teachers. Slots are available on a first-come, first-served basis."),
    ("Payroll Processing Notification", "Dear Team, payroll for October will be processed on October 31st. Salaries will be credited to your accounts by November 1st. Please submit any reimbursement claims by October 25th. For payroll queries, contact payroll@company.com or ext. 4521."),
    ("AWS Cost Optimization Report", "Hi, your monthly AWS cost report is ready. This month's spend: $2,847 (down 12% from last month). Top services: EC2 ($1,200), RDS ($680), S3 ($340). We identified 3 optimization opportunities that could save an estimated $450/month. Review the full report in your AWS Console."),
    ("New Research Paper Published", "Dear Subscriber, a new paper matching your interests has been published: 'Advances in Natural Language Processing for Low-Resource Languages' by researchers at MIT. Published in Nature Machine Intelligence. Access the paper at nature.com/articles/NLP2024 with your institutional login."),
    ("Community Volunteer Opportunity", "Hi neighbors, we're organizing a neighborhood cleanup event on Saturday, October 26 from 9 AM - 12 PM. We'll be cleaning up the park and planting new trees. All supplies will be provided. Lunch will be served after! Sign up at community.org/volunteer. Hope to see you there!"),
    ("Subscription Renewal Notice", "Hi there, your annual subscription to Adobe Creative Cloud will renew on November 15 for $599.88. No action is needed if you'd like to continue your subscription. To cancel or modify your plan, visit your account at adobe.com/account. Thank you for being a valued customer."),
    ("Code Review Request", "Hi, I've opened a pull request for the new user dashboard feature (PR #384). Please review when you get a chance. The changes include responsive layout updates, new chart components, and improved accessibility. Test instructions are in the PR description. Thanks!"),
    ("Health Insurance Open Enrollment", "Dear Employee, the annual health insurance open enrollment period begins November 1st and ends November 30th. You can review and update your coverage options on the HR benefits portal. Attend one of the informational webinars to learn about this year's plan changes. No action is needed to keep your current plan."),
    ("Book Club Next Meeting", "Hi book club members, our next meeting is Saturday, November 2 at 7 PM at Maria's house. We'll be discussing 'The Midnight Library' by Matt Haig. Please read chapters 15-22 before the meeting. Maria will provide snacks. Let her know if you can't make it."),
    ("Server Maintenance Complete", "This is to confirm that the scheduled server maintenance has been completed successfully. All services are now running normally. The maintenance involved OS updates, security patches, and database optimization. If you notice any issues, please contact the IT helpdesk at support@company.com or ext. 5000."),
    ("Internship Offer Letter", "Dear Priya, we are pleased to offer you a summer internship position as a Software Engineering Intern at TechCorp. The internship will be from June 1 to August 31, 2025. The stipend is $2,500/month. Please review and sign the attached offer letter and return it by November 15th."),
    ("Recipe Newsletter - This Week's Picks", "Hi food lover, this week we're featuring autumn comfort foods! Check out our top recipes: pumpkin soup, apple cinnamon muffins, and butternut squash risotto. All recipes are on our website with step-by-step instructions and video guides. Enjoy your cooking! To unsubscribe, click here."),
    ("Department Budget Review", "Hi managers, Q3 budget review meeting is scheduled for October 30 at 2 PM (hybrid format). Please prepare a brief summary of your department's spending vs. budget for Q3 and your Q4 projections. Submit your slides to finance@company.com by October 28. Join via Teams link shared separately."),
    ("Alumni Association Reunion", "Dear Class of 2018, your 7-year reunion is happening on December 14th at the Grand Ballroom, University Campus. Registration is open at alumni.university.edu/reunion2025. Early bird rate of $75 available until November 1st. Reconnect with classmates and faculty!"),
]

def build_dataset():
    rows = []
    for subject, body in PHISHING_EMAILS:
        rows.append({'text': subject + ' ' + body, 'label': 1})
    for subject, body in LEGITIMATE_EMAILS:
        rows.append({'text': subject + ' ' + body, 'label': 0})

    # Augment with variations
    augmented = []
    for row in rows[:]:
        text = row['text']
        label = row['label']
        # Uppercase version
        augmented.append({'text': text.upper(), 'label': label})
        # Add extra punctuation for phishing
        if label == 1:
            augmented.append({'text': text + '!!!', 'label': label})
            augmented.append({'text': 'IMPORTANT: ' + text, 'label': label})
        else:
            augmented.append({'text': 'Hi, ' + text, 'label': label})

    rows.extend(augmented)
    df = pd.DataFrame(rows)
    df['text_clean'] = df['text'].apply(preprocess)
    return df


class CombinedFeatures(BaseEstimator, TransformerMixin):
    """Combine TF-IDF and security features, both normalized."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            max_features=8000,
            sublinear_tf=True,
            min_df=1,
        )
        self.tfidf_char = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            max_features=4000,
            sublinear_tf=True,
            min_df=1,
        )
        self.sec = SecurityFeatureExtractor()
        self.sec_scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        self.tfidf_char.fit(X)
        sec_feats = self.sec.transform(X)
        self.sec_scaler.fit(sec_feats)
        return self

    def transform(self, X):
        from scipy.sparse import hstack, csr_matrix
        t1 = self.tfidf.transform(X)
        t2 = self.tfidf_char.transform(X)
        sec = self.sec.transform(X)
        sec_scaled = self.sec_scaler.transform(sec)
        return hstack([t1, t2, csr_matrix(sec_scaled)])


def train():
    print("Building dataset...")
    df = build_dataset()
    print(f"Total samples: {len(df)} (phishing: {df['label'].sum()}, legit: {(df['label']==0).sum()})")

    X = df['text_clean'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Extracting features...")
    combiner = CombinedFeatures()
    combiner.fit(X_train)
    X_train_feat = combiner.transform(X_train)
    X_test_feat = combiner.transform(X_test)

    print("Training ensemble model...")
    lr = LogisticRegression(C=5.0, max_iter=1000, class_weight='balanced', solver='lbfgs')
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42, n_jobs=4)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
        voting='soft',
    )
    ensemble.fit(X_train_feat, y_train)

    y_pred = ensemble.predict(X_test_feat)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    # Save artifacts
    model_data = {
        'combiner': combiner,
        'model': ensemble,
        'version': '2.0',
    }
    with open('phish_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("\nModel saved to phish_model.pkl")
    return acc


if __name__ == '__main__':
    train()
