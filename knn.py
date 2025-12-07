import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка данных
df = pd.read_csv('processed_reviews.csv')
df['review'] = df['review'].fillna('').astype(str)
df = df[df['review'].str.strip() != '']

# Преобразование текста в векторы
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.95)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['review'])

# Кластеризация K-means
optimal_k = 5  # Фиксированное количество кластеров
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(tfidf_matrix)
df['cluster'] = kmeans.labels_

# Визуализация с помощью PCA
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# Создание графика
plt.figure(figsize=(10, 8))
vis_df = pd.DataFrame({
    'x': pca_result[:, 0],
    'y': pca_result[:, 1],
    'cluster': df['cluster']
})

sns.scatterplot(
    data=vis_df,
    x='x',
    y='y',
    hue='cluster',
    palette='viridis',
    alpha=0.7,
    s=80
)

plt.title(f'Кластеризация отзывов (K={optimal_k})', fontsize=14)
plt.xlabel('PCA компонент 1', fontsize=12)
plt.ylabel('PCA компонент 2', fontsize=12)
plt.legend(title='Кластер')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('clusters_visualization.png', dpi=150)

# Распределение по кластерам
cluster_sizes = df['cluster'].value_counts().sort_index()
for cluster_id, size in cluster_sizes.items():
    percentage = (size / len(df)) * 100

# Сохранение результатов
df.to_csv('clustered_reviews.csv', index=False)