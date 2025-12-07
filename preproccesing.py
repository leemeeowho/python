import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка данных
df = pd.read_csv('reviews.csv', sep='\t', on_bad_lines='skip', encoding='utf-8')

# Загрузка стоп-слов из файла russian.txt
with open('russian.txt', 'r', encoding='utf-8') as f:
    stop_words = set(word.strip().lower() for word in f.readlines() if word.strip())

# Функции очистки
def remove_punctuation(text):
    if isinstance(text, str):
        return re.sub(r'[^\w\s]', '', text)
    return text

def remove_stopwords(text):
    if isinstance(text, str) and text.strip():
        words = text.split()
        filtered = [word for word in words if word not in stop_words and len(word) > 1]
        return ' '.join(filtered)
    return text

def remove_digits(text):
    if isinstance(text, str):
        return re.sub(r'\d+', '', text)
    return text

def remove_emojis(text):
    if isinstance(text, str):
        emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" 
                                   u"\U0001F1E0-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251" "]+",
                                   flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    return text

def remove_urls(text):
    if isinstance(text, str):
        url_pattern = r'https?://\S+|www\.\S+|http?://\S+'
        return re.sub(url_pattern, '', text)
    return text

def remove_html(text):
    if isinstance(text, str):
        html_pattern = r'<[^>]+>'
        return re.sub(html_pattern, '', text)
    return text

def clean_spaces(text):
    if isinstance(text, str):
        return re.sub(r'\s+', ' ', text).strip()
    return text

# Основная обработка
df['review'] = df['review'].astype(str).str.lower()
df['review'] = df['review'].apply(remove_punctuation)
df['review'] = df['review'].apply(remove_stopwords)
df['review'] = df['review'].apply(remove_digits)
df['review'] = df['review'].apply(remove_emojis)
df['review'] = df['review'].apply(remove_urls)
df['review'] = df['review'].apply(remove_html)

# Удаление высокочастотных слов
all_words = []
for text in df['review'].dropna():
    words = text.split()
    all_words.extend(words)

if all_words:
    word_counts = Counter(all_words)
    top_10_words = [word for word, _ in word_counts.most_common(10)]
    df['review'] = df['review'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in top_10_words]))

# TF-IDF анализ и удаление редких/частых слов
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df['review'].fillna(''))

feature_names = vectorizer.get_feature_names_out()
idf_values = vectorizer.idf_
word_idf = list(zip(feature_names, idf_values))
word_idf_sorted = sorted(word_idf, key=lambda x: x[1])

rare_words = set([word for word, idf in word_idf_sorted[:20]])
common_words = set([word for word, idf in word_idf_sorted[-20:]])
words_to_remove = rare_words.union(common_words)

df['review'] = df['review'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in words_to_remove]))
df['review'] = df['review'].apply(clean_spaces)

# Сохранение результата
df.to_csv('processed_reviews.csv', index=False, encoding='utf-8')