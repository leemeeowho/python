# Загрузка данных
import matplotlib.pyplot as plt
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

df = pd.read_csv('processed_reviews.csv')
def tokens_to_text(tokens):
    if isinstance(tokens, str):
        try:
            token_list = eval(tokens)
            return ' '.join(token_list) if isinstance(token_list, list) else tokens
        except:
            return tokens
    elif isinstance(tokens, list):
        return ' '.join(tokens)
    else:
        return str(tokens)

df['text_for_modeling'] = df['tokens'].apply(tokens_to_text)
df = df[df['text_for_modeling'].str.strip() != '']
df = df[~df['review'].isna()]
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

topic_model = BERTopic(
    language="multilingual",
    embedding_model=embedding_model,
    min_topic_size=10,
    nr_topics='auto',
    calculate_probabilities=False,
    verbose=False,
    top_n_words=8
)

texts = df['text_for_modeling'].tolist()
if len(texts) > 0:
    topics, probs = topic_model.fit_transform(texts)
    if len(set(topics)) > 2:
        try:
            topic_model.reduce_topics(texts, nr_topics=min(8, len(set(topics)) - 1))
        except:
            pass
else:
    topics = [0] * len(df)

topic_info = topic_model.get_topic_info()

plt.figure(figsize=(12, 6))
valid_topics = topic_info[topic_info['Topic'] != -1]
if len(valid_topics) > 0:
    bars = plt.bar(valid_topics['Topic'].astype(str), valid_topics['Count'], color='skyblue')
    plt.title('Распределение отзывов по темам', fontsize=14)
    plt.xlabel('Номер темы', fontsize=12)
    plt.ylabel('Количество отзывов', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1, f'{int(height)}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('topic_distribution.png', dpi=150)
else:
    print("Недостаточно тем для визуализации")

df['topic'] = topics

for topic_id in valid_topics['Topic'].tolist():
    if topic_id == -1:
        continue

    # Фильтруем отзывы по текущей теме
    topic_df = df[df['topic'] == topic_id].copy()

    # Получаем ключевые слова для темы
    topic_words = topic_model.get_topic(topic_id)
    if topic_words:
        keywords = ", ".join([word for word, _ in topic_words[:6]])
        for i, (_, row) in enumerate(topic_df.head(2).iterrows()):
            # Обрабатываем возможные NaN значения
            review_text = str(row['review']).strip()
            if len(review_text) > 120:
                review_text = review_text[:120] + '...'
            sentiment = str(row['sentiment']).strip() if 'sentiment' in row else 'unknown'
            print(f"  {i + 1}. [{sentiment}] {review_text}")

# Сохранение результатов
df.to_csv('clustered_reviews_with_topics.csv', index=False)