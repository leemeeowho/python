import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

df = pd.read_csv('processed_reviews.csv')
df = df.dropna(subset=['review'])
df['review'] = df['review'].astype(str).str.strip()
df = df[df['review'] != '']
df = df[df['review'].str.len() > 10]
texts = df['review'].tolist()

embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
topic_model = BERTopic(
    language="multilingual",
    embedding_model=embedding_model,
    min_topic_size=15,
    nr_topics='auto',
    calculate_probabilities=False,
    verbose=True,
    top_n_words=10
)

topics, probs = topic_model.fit_transform(texts)
topic_model.reduce_topics(texts, nr_topics=10)
topic_info = topic_model.get_topic_info()

for topic_id in topic_info[topic_info['Topic'] != -1]['Topic'].head(8).tolist():
    if topic_id == -1:
        continue

    topic_words = topic_model.get_topic(topic_id)
    if topic_words:
        words_str = ", ".join([word for word, _ in topic_words[:8]])

        topic_docs = df.iloc[[i for i, t in enumerate(topics) if t == topic_id]]['review'].head(3).tolist()
        if topic_docs:
            print("Примеры отзывов:")
            for i, doc in enumerate(topic_docs[:3]):
                clean_doc = str(doc).replace('\n', ' ').strip()
                print(f"  • {clean_doc[:120]}..." if len(clean_doc) > 120 else f"  • {clean_doc}")
    else:
        print("Не удалось определить ключевые слова")

plt.figure(figsize=(14, 8))
valid_topics = topic_info[topic_info['Topic'] != -1].head(10)
bars = plt.bar(valid_topics['Topic'].astype(str), valid_topics['Count'],
               color=plt.cm.viridis(np.linspace(0, 1, len(valid_topics))))
plt.title('Распределение отзывов по тематическим кластерам', fontsize=16, fontweight='bold')
plt.xlabel('Номер темы', fontsize=14)
plt.ylabel('Количество отзывов', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Добавляем значения над столбцами
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('topic_distribution.png', dpi=300, bbox_inches='tight')

try:
    fig = topic_model.visualize_topics()
    fig.write_html("topic_visualization.html", include_plotlyjs='cdn')
except Exception as e:
    print("Ошибка")

try:
    fig = topic_model.visualize_heatmap()
    fig.write_html("topic_similarity.html", include_plotlyjs='cdn')
    print("Визуализация сохранена в файл ")
except Exception as e:
    print("Ошибка")

try:
    fig = topic_model.visualize_barchart(top_n_topics=8, n_words=10, width=1000, height=800)
    fig.write_html("topic_word_barchart.html", include_plotlyjs='cdn')
except Exception as e:
    print("Ошибка")

df['topic'] = topics
topic_names = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
df['topic_name'] = df['topic'].map(topic_names)

df.to_csv('clustered_reviews_with_topics.csv', index=False, encoding='utf-8')

topic_summary = []
for topic_id in topic_info[topic_info['Topic'] != -1]['Topic'].head(10):
    words = topic_model.get_topic(topic_id)
    if words:
        words_str = ", ".join([word for word, _ in words[:10]])
        topic_summary.append({
            'topic_id': topic_id,
            'topic_name': topic_names.get(topic_id, f"Тема {topic_id}"),
            'count': topic_info[topic_info['Topic'] == topic_id]['Count'].values[0],
            'keywords': words_str
        })

pd.DataFrame(topic_summary).to_csv('topic_summary.csv', index=False, encoding='utf-8')
for topic in topic_summary[:5]:
    print(f"- {topic['topic_name']}: {topic['count']} отзывов")