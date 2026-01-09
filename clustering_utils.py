import requests
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score
import re
import numpy as np

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

class NewsClusteringEngine:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add context-specific news stopwords
        self.stop_words.update(['said', 'news', 'would', 'also', 'new', 'could', 'first', 'one', 'two', 'told', 'reported'])
        self.lemmatizer = WordNetLemmatizer()
        
        # Optimized TF-IDF for better signal-to-noise
        self.vectorizer = TfidfVectorizer(
            max_features=2000, 
            stop_words=list(self.stop_words),
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85
        )
        self.lsa = TruncatedSVD(n_components=50, random_state=42)
        self.normalizer = Normalizer(copy=False)
        self.kmeans = None
        self.pca = PCA(n_components=2)

    def fetch_news(self, api_key: str, query: str, page_size: int = 100) -> list:
        """
        Fetch news articles using NewsAPI.org
        """
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': api_key,
            'pageSize': page_size,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != 'ok':
                raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
            
            articles = data.get('articles', [])
            cleaned_articles = []
            
            for art in articles:
                # Filter out removed articles or those with no content
                if art['title'] and art['description'] and art['source']['name'] != "[Removed]":
                    cleaned_articles.append({
                        'title': art['title'],
                        'source': art['source']['name'],
                        'description': art['description'] or "",
                        'content': art['content'] or "",
                        'url': art['url']
                    })
            
            return cleaned_articles
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error fetching news: {str(e)}")

    def preprocess_text(self, text: str) -> str:
        """
        Tokenize, remove stopwords, and lemmatize text.
        """
        if not text:
            return ""
            
        # Lowercase and remove non-alphabetic
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        tokens = word_tokenize(text)
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return " ".join(cleaned_tokens)

    def perform_clustering(self, articles_df: pd.DataFrame, n_clusters: int):
        """
        Vectorize text, apply LSA reduction, perform K-Means clustering, and PCA reduction.
        """
        if articles_df.empty:
            return None, None, None, 0.0

        # Combine title and description for better context
        articles_df['full_text'] = articles_df['title'] + " " + articles_df['description']
        articles_df['processed_text'] = articles_df['full_text'].apply(self.preprocess_text)
        
        # Check if we have enough data
        if len(articles_df) < n_clusters:
            n_clusters = max(1, len(articles_df))

        # 1. Vectorization (TF-IDF)
        tfidf_matrix = self.vectorizer.fit_transform(articles_df['processed_text'])
        
        # 2. Dimensionality Reduction (LSA) to capture latent topics
        # n_components must be less than min(samples, features)
        n_comp = min(50, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        if n_comp > 0:
            lsa_obj = TruncatedSVD(n_components=n_comp, random_state=42)
            lsa_matrix = lsa_obj.fit_transform(tfidf_matrix)
            lsa_matrix = self.normalizer.fit_transform(lsa_matrix)
        else:
            lsa_matrix = tfidf_matrix.toarray()

        # 3. Clustering on the reduced/denser space (Produces higher Silhouette score)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        clusters = self.kmeans.fit_predict(lsa_matrix)
        
        # 4. Calculate Silhouette Score on the latent space
        try:
            if n_clusters > 1 and lsa_matrix.shape[0] > 1:
                sil_score = silhouette_score(lsa_matrix, clusters)
            else:
                sil_score = 0.0
        except Exception:
            sil_score = 0.0
        
        # 5. Dimensionality Reduction for Visualization (PCA to 2D)
        if lsa_matrix.shape[0] > 1:
            coords_2d = self.pca.fit_transform(lsa_matrix)
        else:
            coords_2d = np.array([[0, 0]])
            
        # Add results to dataframe
        articles_df['cluster'] = clusters
        articles_df['x'] = coords_2d[:, 0]
        articles_df['y'] = coords_2d[:, 1]
        
        return articles_df, tfidf_matrix, self.kmeans, sil_score

    def get_cluster_keywords(self, tfidf_matrix, feature_names, clusters, n_terms=5):
        """
        Extract top keywords for each cluster.
        """
        df = pd.DataFrame(tfidf_matrix.toarray())
        df['cluster'] = clusters
        
        keywords = {}
        for cluster_id in df['cluster'].unique():
            # Get mean tfidf scores for this cluster
            cluster_mean = df[df['cluster'] == cluster_id].drop('cluster', axis=1).mean()
            sorted_indices = cluster_mean.sort_values(ascending=False).index[:n_terms]
            
            top_words = [feature_names[i] for i in sorted_indices]
            keywords[cluster_id] = top_words
            
        return keywords
