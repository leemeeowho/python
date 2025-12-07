import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

def extract_context_and_label(line):
    pattern = r'(.*?)(\bза`мок\b|\bзамо`к\b)(.*?)'
    match = re.search(pattern, line, re.IGNORECASE)
    if match:
        before, full_word, after = match.groups()
        if 'за`мок' in full_word.lower():
            label = 'noun_castle' # зАмок - замок (крепость)
        elif 'замо`к' in full_word.lower():
            label = 'noun_lock' # замОк - замок (устройство)
        else:
            return None, None # Если не удалось определить

        context = (before.strip() + " [ЗАМОК] " + after.strip()).strip()
        context_clean = re.sub(r'[`]', '', context)
        return context_clean, label
    return None, None

try:
    with open('замок.test', 'r', encoding='utf-8') as f:
        raw_data = f.read()
except FileNotFoundError:
    exit()

contexts = []
labels = []
original_sentences = []

for line in raw_data.strip().split('\n'):
    line = line.strip()
    if line:
        context, label = extract_context_and_label(line)
        if context and label:
            contexts.append(context)
            labels.append(label)
            original_sentences.append(line)

df = pd.DataFrame({
    'original_sentence': original_sentences,
    'context': contexts,
    'label': labels
})

vectorizer = CountVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=1000
)

X = vectorizer.fit_transform(df['context'])
y = df['label']

svm_model = LinearSVC(random_state=42)
svm_model.fit(X, y)

y_pred = svm_model.predict(X)

accuracy = accuracy_score(y, y_pred)

final_dataset = df[['original_sentence', 'label']].copy()
df.to_csv('final_dataset.csv', index=False)