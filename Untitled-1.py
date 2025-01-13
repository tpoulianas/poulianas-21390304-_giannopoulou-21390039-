# -*- coding: utf-8 -*-
"""
Enhanced Information Retrieval System with Evaluation Metrics
This script improves upon the original to include evaluation metrics for the system's performance.
"""

# Required Libraries
import json
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

""" ----------- Data Collection (Wikipedia API) ----------- """
def fetch_wikipedia_articles(query, article_count):
    """
    Fetches articles from Wikipedia using a search query and a specified number of results.
    """
    base_url = "https://en.wikipedia.org/w/api.php"
    articles = []

    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': query,
        'srlimit': article_count
    }

    # Web Crawler
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        results = response.json().get('query', {}).get('search', [])
        for result in results:
            title = result['title']
            content = fetch_article_content(title)
            if content:
                articles.append({
                    'title': title,
                    'content': preprocess_text(content)
                })

    # Save results as JSON
    with open(f'{query}_wikipedia_data.json', 'w', encoding='utf-8') as file:
        json.dump(articles, file, ensure_ascii=False, indent=2)

    return articles, create_inverted_index(articles)

def fetch_article_content(title):
    """
    Fetches the full content of a Wikipedia article given its title.
    """
    base_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    response = requests.get(base_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join(p.text for p in paragraphs if p.text.strip())
        return content
    return None

""" ----------- Text Preprocessing ----------- """
def preprocess_text(raw_text):
    """
    Processes raw text: tokenization, stop-word removal, lemmatization, and stemming.
    """
    # Tokenization
    tokens = word_tokenize(raw_text)

    # Stop-word removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [t for t in tokens if t.isalnum() and t.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]

    return stemmed_tokens

""" ----------- Inverted Index Creation ----------- """
def create_inverted_index(documents):
    """
    Builds an inverted index for efficient retrieval.
    """
    index = defaultdict(set)
    for doc_id, doc in enumerate(documents):
        for word in doc.get('content', []):
            index[word].add(doc_id)
    return index

""" ----------- Search Engine ----------- """
def execute_search():
    """
    Prompts user to perform dynamic searches.
    """
    while True:
        query = input("Enter your search query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting search engine. Goodbye!")
            break

        article_count = int(input("Enter number of articles to fetch: ").strip())
        documents, index = fetch_wikipedia_articles(query=query, article_count=article_count)

        retrieval_method = input("Choose retrieval model (BOOLEAN, VSM, BM25): ").upper()
        if retrieval_method not in ['BOOLEAN', 'VSM', 'BM25']:
            print("Invalid choice. Please select BOOLEAN, VSM, or BM25.")
            continue

        results = retrieve_documents(query, retrieval_method, documents, index)
        if results:
            display_search_results(results)

            # Evaluation Section
            print("\nEvaluating system performance...")
            evaluate_system(documents, results, query)
        else:
            print("No relevant documents found.")

""" ----------- Retrieval Models ----------- """
def retrieve_documents(query, method, documents, index):
    """
    Handles retrieval based on the selected method.
    """
    if method == 'BOOLEAN':
        return boolean_model(query, index, documents)
    elif method == 'VSM':
        return vector_space_model(query, documents)
    elif method == 'BM25':
        return bm25_model(query, documents)

def boolean_model(query, index, docs):
    """
    Implements Boolean retrieval model.
    """
    terms = query.split()
    matching_docs = set(range(len(docs)))
    for term in terms:
        if term in index:
            matching_docs &= index[term]
    return [docs[i] for i in matching_docs]

def vector_space_model(query, docs):
    """
    Implements Vector Space Model retrieval.
    """
    vectorizer = TfidfVectorizer()
    contents = [" ".join(doc['content']) for doc in docs]
    tfidf_matrix = vectorizer.fit_transform([query] + contents)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_indices = np.argsort(scores)[::-1]
    return [docs[i] for i in ranked_indices if scores[i] > 0]

def bm25_model(query, docs, k1=1.5, b=0.75):
    """
    Implements a simplified BM25 model.
    """
    vectorizer = TfidfVectorizer()
    contents = [" ".join(doc['content']) for doc in docs]
    tfidf_matrix = vectorizer.fit_transform([query] + contents)
    doc_lengths = np.array([len(c.split()) for c in contents])
    avg_length = np.mean(doc_lengths)

    scores = []
    for idx, content in enumerate(contents):
        score = 0
        for term in query.split():
            term_freq = content.split().count(term)
            idf = vectorizer.idf_[vectorizer.vocabulary_.get(term, 0)]
            score += idf * ((term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * (doc_lengths[idx] / avg_length))))
        scores.append((idx, score))

    ranked_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [docs[i] for i, _ in ranked_scores if _ > 0]

""" ----------- Display Results ----------- """
def display_search_results(results):
    """
    Outputs search results in a user-friendly format.
    """
    for idx, result in enumerate(results, 1):
        print(f"{idx}. Title: {result['title']}")
        print(f"Preview: {' '.join(result['content'][:30])}...\n")

""" ----------- Evaluation Metrics ----------- """
def evaluate_system(documents, results, query):
    """
    Evaluate system using Precision, Recall, F1-Score, and MAP.
    """
    # Ground truth: Assume first half of documents are relevant (for demo purposes)
    true_relevance = [1 if i < len(documents) // 2 else 0 for i in range(len(documents))]
    retrieved_relevance = [1 if doc in results else 0 for doc in documents]

    precision = precision_score(true_relevance, retrieved_relevance)
    recall = recall_score(true_relevance, retrieved_relevance)
    f1 = f1_score(true_relevance, retrieved_relevance)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    def mean_average_precision(true_relevance, retrieved_relevance):
        relevant = 0
        avg_precision = 0
        for i, val in enumerate(retrieved_relevance):
            if val == 1 and true_relevance[i] == 1:
                relevant += 1
                avg_precision += relevant / (i + 1)
        return avg_precision / sum(true_relevance)

    map_score = mean_average_precision(true_relevance, retrieved_relevance)
    print(f"MAP: {map_score:.2f}")

# Main Execution
if __name__ == "__main__":
    execute_search()
