# ==========================================
#  Health Myth/Fact Detection
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Step 1️: Load and Merge Datasets
# ==========================================
try:
    df1 = pd.read_csv('health_myths_facts_2025.csv')
    df2 = pd.read_csv('web_scraped_health_myths_2025.csv')
    df = pd.concat([df1, df2], ignore_index=True)
except Exception as e:
    print(f"⚠️ Error loading files: {e}")
    raise

# Clean and prepare
df = df.drop_duplicates(subset=['text']).dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(int)

# ==========================================
# Step 2️: TF-IDF Vectorization
# ==========================================
vectorizer = TfidfVectorizer(max_features=700, stop_words='english', ngram_range=(1,1))
X = vectorizer.fit_transform(df['text'])
y = df['label']

# ==========================================
# Step 3️: Train-Test Split 
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
noise_ratio = 0.05
noise_indices = np.random.choice(len(y_train), size=int(noise_ratio * len(y_train)), replace=False)
y_train_noisy = y_train.copy().to_numpy()
y_train_noisy[noise_indices] = 1 - y_train_noisy[noise_indices]

# ==========================================
# Step 4️: Train Random Forest Model
# ==========================================
clf = RandomForestClassifier(
    n_estimators=90,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    class_weight='balanced'
)
clf.fit(X_train, y_train_noisy)

# ==========================================
# Step 5️: Prediction + Fact Retrieval
# ==========================================
def predict_and_debunk(input_text: str):
    input_vec = vectorizer.transform([input_text])
    pred = clf.predict(input_vec)[0]
    prob = clf.predict_proba(input_vec)[0].max()

    if pred == 1:
        sim_scores = cosine_similarity(input_vec, vectorizer.transform(df['text']))[0]
        best_idx = np.argmax(sim_scores)
        fact = df.iloc[best_idx]['fact'] if 'fact' in df.columns else "Fact not available in dataset."
        return {'prediction': 'Myth', 'confidence': prob, 'fact': fact}
    else:
        return {'prediction': 'Fact', 'confidence': prob, 'fact': 'Evidence-based health knowledge.'}

# ==========================================
# Step 6️: User I/O Interaction
# ==========================================
print("\n--- 🧠 Health Myth/Fact Classifier ---")
print("Enter a health statement to check if it's a Myth or Fact.")
print("Type 'quit' to exit.\n")

while True:
    text = input("Your statement: ")
    if text.lower() == 'quit':
        print("Exiting...")
        break
    if not text.strip():
        print("Please enter a valid statement.\n")
        continue

    result = predict_and_debunk(text)
    print(f"\nPrediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
    print(f"Fact: {result['fact']}\n")
