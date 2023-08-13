import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


def prepros_text(text):
    stemmer = SnowballStemmer("russian")
    stop_words = set(stopwords.words('russian'))
    words = nltk.word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)


phrases = [

]

cluster_phrasers = [prepros_text(phrase) for phrase in phrases]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cluster_phrasers)

num_cluster = 174
kmeans = KMeans(n_clusters=num_cluster, random_state=42)
kmeans.fit(X)

for cluster_num in range(num_cluster):
    cluster_indices = np.where(kmeans.labels_ == cluster_num)[0]
    print(f"{cluster_num + 1}:")
    for idx in cluster_indices:
        print(f"{phrases[idx]}")
    print()
