# Загрузка данных
import pandas as pd

df = pd.read_csv('processed_reviews.csv')
print(df.head())
# Обучение модели с использованием колонок tokens и sentiment
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


# Подготовка текстов из колонки tokens
# Проверяем тип данных и конвертируем в строку при необходимости
def tokens_to_text(tokens):
    if isinstance(tokens, str):
        try:
            # Пытаемся преобразовать строку в список
            token_list = eval(tokens)
            return ' '.join(token_list) if isinstance(token_list, list) else tokens
        except:
            return tokens
    elif isinstance(tokens, list):
        return ' '.join(tokens)
    else:
        return str(tokens)


# Применяем преобразование и удаляем пустые значения
df['text_for_modeling'] = df['tokens'].apply(tokens_to_text)
df = df[df['text_for_modeling'].str.strip() != '']
df = df[~df['review'].isna()]

# Загрузка модели для русского языка
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Настройка и обучение BERTopic
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
    # Пытаемся уменьшить количество тем, но только если их больше 1
    if len(set(topics)) > 2:  # Кроме выбросов
        try:
            topic_model.reduce_topics(texts, nr_topics=min(8, len(set(topics)) - 1))
        except:
            pass
else:
    print("Недостаточно данных для обучения модели")
    topics = [0] * len(df)

# Анализ тем с учетом тональности
topic_info = topic_model.get_topic_info()
print("Найдено тематических кластеров:", len(topic_info[topic_info['Topic'] != -1]))

# Визуализация распределения тем
plt.figure(figsize=(12, 6))
valid_topics = topic_info[topic_info['Topic'] != -1]
if len(valid_topics) > 0:
    bars = plt.bar(valid_topics['Topic'].astype(str), valid_topics['Count'], color='skyblue')
    plt.title('Распределение отзывов по темам', fontsize=14)
    plt.xlabel('Номер темы', fontsize=12)
    plt.ylabel('Количество отзывов', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # Добавление значений над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1, f'{int(height)}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('topic_distribution.png', dpi=150)
    print("График распределения тем сохранен в 'topic_distribution.png'")
else:
    print("Недостаточно тем для визуализации")

# Примеры отзывов для каждой темы с указанием тональности
print("\nПримеры отзывов по темам с тональностью:")
df['topic'] = topics

# Группируем по темам и показываем примеры
for topic_id in valid_topics['Topic'].tolist():
    if topic_id == -1:
        continue

    # Фильтруем отзывы по текущей теме
    topic_df = df[df['topic'] == topic_id].copy()

    # Получаем ключевые слова для темы
    topic_words = topic_model.get_topic(topic_id)
    if topic_words:
        keywords = ", ".join([word for word, _ in topic_words[:6]])
        print(f"\nТема {topic_id} - Ключевые слова: {keywords}")

        # Показываем распределение тональности в теме
        sentiment_dist = topic_df['sentiment'].value_counts()
        print(f"Распределение тональности: {sentiment_dist.to_dict()}")

        # Примеры отзывов (не более 2)
        print("Примеры отзывов:")
        for i, (_, row) in enumerate(topic_df.head(2).iterrows()):
            # Обрабатываем возможные NaN значения
            review_text = str(row['review']).strip()
            if len(review_text) > 120:
                review_text = review_text[:120] + '...'
            sentiment = str(row['sentiment']).strip() if 'sentiment' in row else 'unknown'
            print(f"  {i + 1}. [{sentiment}] {review_text}")

# Сохранение результатов
df.to_csv('clustered_reviews_with_topics.csv', index=False)
print("\nРезультаты сохранены в 'clustered_reviews_with_topics.csv'")