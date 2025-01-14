# -*- coding: utf-8 -*-
"""

@authors: Tilemachos Poulianas 21390304
	  Dionysia Giannopoulou 21390039
    
"""

# Required Libraries
import json
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import math
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

""" ----------- Data Collection (Wikipedia API) ----------- """
def fetch_wikipedia_articles(query, article_count):
    """
    Fetches articles from Wikipedia using a search query and a specified number of results.
    Automatically saves the articles to a JSON file named after the query.
    """
    # Base URL for Wikipedia API
    base_url = "https://en.wikipedia.org/w/api.php"
    articles = []

    # Parameters for the API request
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': query,
        'srlimit': article_count
    }

    # Use Wikipedia API to fetch articles
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        results = response.json().get('query', {}).get('search', [])
        for result in results:
            title = result['title']
            content = fetch_article_content(title)
            if content:
                articles.append({
                    'title': title,
                    'content': preprocess_text(content)[:500],  # Limit to first 500 words
                    'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"  # Create URL
                })

    # Save the fetched articles to a JSON file
    filename = f"{query.replace(' ', '_')}.json"
    if articles:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)
        print(f"Articles saved to '{filename}'.")
    else:
        print("No articles fetched.")

    return articles

def fetch_article_content(title):
    """
    Fetches the full content of a Wikipedia article given its title using the API.
    """
    # Base URL for fetching article content
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'extracts',
        'exintro': True,
        'titles': title
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        pages = data['query']['pages']
        page = next(iter(pages.values()))
        content = page.get('extract', '')
        if content:
            content = clean_html(content)  # Clean HTML tags
        return content
    return None

""" ----------- Text Preprocessing ----------- """
def preprocess_text(raw_text, use_stemming=False, use_lemmatization=False):
    """
    Processes raw text: tokenization, stop-word removal, and optional stemming/lemmatization.
    """
    # Load stop words for filtering
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(raw_text)  # Tokenize the raw text
    cleaned_tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words]

    # Apply stemming if enabled
    if use_stemming:
        stemmer = PorterStemmer()
        cleaned_tokens = [stemmer.stem(token) for token in cleaned_tokens]

    # Apply lemmatization if enabled
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]

    return ' '.join(cleaned_tokens)

# Clean HTML
def clean_html(raw_html):
    """
    Removes HTML tags from raw HTML content.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text()

""" ----------- Indexing ----------- """
def create_inverted_index(docs):
    """
    Creates an inverted index from the document collection.
    """
    # Initialize an empty inverted index using defaultdict
    inverted_index = defaultdict(list)
    for doc_id, doc in enumerate(docs):
        terms = doc['content'].split()  # Split document content into terms
        for term in set(terms):
            inverted_index[term].append(doc_id)  # Map each term to the document ID
    return inverted_index

""" ----------- Retrieval Models ----------- """
def boolean_search(query, docs, inverted_index):
    """
    Implements Boolean retrieval model with improved handling of NOT.
    """
    query_terms = query.split()

    # Initialize an empty set for the final result
    matching_docs = set(range(len(docs)))

    # Initialize a flag to track whether the next term should be negated
    negate_next = False

    # Initialize the logical operator to None
    operator = None

    for term in query_terms:
        # Handle negation
        if term == 'NOT':
            negate_next = True
            continue

        # Check if the term is a logical operator
        if term in ['AND', 'OR']:
            operator = term
            continue  # Skip logical operators

        term_str = str(term)  # Convert term to a string

        # Perform Boolean operations (AND, OR, NOT)
        if term_str in inverted_index:
            term_results = set(inverted_index[term_str])
            if negate_next:
                matching_docs -= term_results
                negate_next = False
            else:
                # Apply logical operator
                if operator == 'AND':
                    matching_docs = matching_docs.intersection(term_results)
                elif operator == 'OR':
                    matching_docs = matching_docs.union(term_results)
                else:
                    matching_docs = term_results

    # Return the documents matching the final result set
    return [docs[doc_id] for doc_id in matching_docs]


def vsm_search(query, docs):
    """
    Implements Vector Space Model retrieval.
    """
    # Convert documents into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    doc_texts = [doc['content'] for doc in docs]
    tfidf_matrix = vectorizer.fit_transform(doc_texts)

    # Convert the query into a TF-IDF vector
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Rank documents based on cosine similarity
    ranked_indices = cosine_similarities.argsort()[::-1]
    ranked_docs = [docs[idx] for idx in ranked_indices if cosine_similarities[idx] > 0]

    return ranked_docs

def bm25(query, docs, k1=1.5, b=0.75):
    """
    Implements BM25 model.
    """
    # Calculate average document length for BM25
    doc_lengths = [len(doc['content'].split()) for doc in docs]
    avg_doc_length = sum(doc_lengths) / len(docs)

    query_tokens = query.lower().split()  # Tokenize the query
    scores = defaultdict(int)

    for idx, doc in enumerate(docs):
        doc_tokens = doc['content'].split()
        doc_term_freq = defaultdict(int)
        for token in doc_tokens:
            doc_term_freq[token] += 1

        for token in query_tokens:
            # Calculate BM25 score components
            idf = math.log((len(docs) - sum([1 for doc in docs if token in doc['content']]) + 0.5) / (sum([1 for doc in docs if token in doc['content']]) + 0.5) + 1.0)
            tf = doc_term_freq[token] if token in doc_term_freq else 0
            length_factor = (doc_lengths[idx] / avg_doc_length)
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length_factor))
            scores[idx] += score

    # Rank documents based on BM25 scores
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[i] for i, _ in ranked_docs if _ > 0]

""" ----------- Evaluation Metrics ----------- """
def evaluate_system(documents, results, query):
    """
    Evaluate system using Precision, Recall, F1-Score, and MAP.
    Dynamically limits the number of relevant documents to half the total retrieved documents.
    """
    # Identify relevant documents based on the query
    query_tokens = set(query.lower().split())
    true_relevance = [1 if any(token in doc['content'] for token in query_tokens) else 0 for doc in documents]
    retrieved_relevance = [1 if doc in results else 0 for doc in documents]

    # Limit relevant documents to a specific number (e.g., half of retrieved)
    max_relevant = len(results) // 2
    true_relevance = [1 if idx < max_relevant else 0 for idx, val in enumerate(true_relevance)]

    # Calculate evaluation metrics
    precision = precision_score(true_relevance, retrieved_relevance, zero_division=0)
    recall = recall_score(true_relevance, retrieved_relevance, zero_division=0)
    f1 = f1_score(true_relevance, retrieved_relevance, zero_division=0)

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
        return avg_precision / sum(true_relevance) if sum(true_relevance) > 0 else 0

    map_score = mean_average_precision(true_relevance, retrieved_relevance)
    print(f"MAP: {map_score:.2f}")

""" ----------- Display Results ----------- """
def display_search_results(results):
    """
    Outputs search results in a user-friendly format.
    """
    if results:
        for idx, result in enumerate(results, 1):
            print(f"{idx}. Title: {result['title']}")
            print(f"Preview: {' '.join(result['content'].split()[:30])}...\n")
    else:
        print("No relevant documents found.")

""" ----------- Main Search Engine Execution ----------- """
def execute_search():
    """
    Prompts user to perform dynamic searches.
    """
    while True:
        # Prompt user for query input
        query = input("Enter your search query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting search engine. Goodbye!")
            break

        try:
            # Prompt user for the number of articles to fetch
            article_count = int(input("Enter number of articles to fetch: ").strip())
        except ValueError:
            print("Please enter a valid number for the article count.")
            continue

        # Fetch documents from Wikipedia
        documents = fetch_wikipedia_articles(query=query, article_count=article_count)

        if not documents:
            print("No articles found. Try a different query.")
            continue

        if len(documents) < article_count:
            print(f"Warning: Only {len(documents)} articles fetched instead of {article_count}.")

        print(f"Fetched {len(documents)} documents.")

        # Create inverted index for the fetched documents
        inverted_index = create_inverted_index(documents)

        # Prompt user to choose a retrieval model
        retrieval_method = input("Choose retrieval model (BOOLEAN, VSM, BM25): ").upper()
        if retrieval_method == 'BOOLEAN':
            results = boolean_search(query, documents, inverted_index)
        elif retrieval_method == 'VSM':
            results = vsm_search(query, documents)
        elif retrieval_method == 'BM25':
            results = bm25(query, documents)
        else:
            print("Invalid choice. Please select BOOLEAN, VSM, or BM25.")
            continue

        # Display the search results to the user
        display_search_results(results)

        # Evaluate system performance based on results
        evaluate_system(documents, results, query)

# Main Execution
if __name__ == "__main__":
    execute_search()
