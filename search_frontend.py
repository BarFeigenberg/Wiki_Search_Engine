from flask import Flask, request, jsonify
from collections import Counter, defaultdict
import math
import pickle
import re
import os
from google.cloud import storage
from contextlib import closing
from inverted_index_gcp import InvertedIndex, MultiFileReader


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
# ==============================================================================
# 1. CONSTANTS & CONFIG
# ==============================================================================
BUCKET_NAME = "raz439"
N_CORPUS = 6348910
TUPLE_SIZE = 6
# BM25+ Hyperparameters
k1 = 1.5
b = 0.75
delta = 1.0  # BM25+ improvement
# Signal Weights
W_BM25 = 0.75
W_PAGERANK = 0.25
# ==============================================================================
# 2. TOKENIZATION LOGIC
# ==============================================================================
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
STOPWORDS = frozenset([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
    'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
    'category', 'references', 'also', 'external', 'links', 'coronawarning',
    'wikipedia', 'google', 'facebook', 'twitter'
])


def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in STOPWORDS]


# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def get_posting_list(inverted_index, w):
    if w not in inverted_index.posting_locs:
        return []
    posting_locs = inverted_index.posting_locs[w]
    if isinstance(posting_locs, tuple):
        posting_locs = [posting_locs]
    try:
        reader = MultiFileReader("postings_gcp", BUCKET_NAME)
    except TypeError:
        reader = MultiFileReader()
    with closing(reader) as reader:
        n_bytes = inverted_index.df[w] * TUPLE_SIZE
        try:
            b = reader.read(posting_locs, n_bytes)
        except TypeError:
            b = reader.read(posting_locs)
        posting_list = []
        for i in range(inverted_index.df[w]):
            chunk = b[i * TUPLE_SIZE: (i + 1) * TUPLE_SIZE]
            doc_id = int.from_bytes(chunk[0:4], 'big')
            tf = int.from_bytes(chunk[4:6], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def load_pickle_from_bucket(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    return pickle.loads(blob.download_as_string())


def normalize_dict(d):
    if not d:
        return {}
    max_val = max(d.values())
    min_val = min(d.values())
    if max_val == min_val:
        return {k: 1.0 for k in d}
    return {k: (v - min_val) / (max_val - min_val) for k, v in d.items()}


# ==============================================================================
# 4. SERVER SETUP & LOADING
# ==============================================================================
print("Loading Index and Dictionaries...")
# 1. Load Body Index
try:
    index_body = InvertedIndex.read_index("postings_gcp", "index", BUCKET_NAME)
except Exception as e:
    print(f"Warning: Could not load index_body. Error: {e}")
    index_body = None
# 2. Load Title Dictionary
try:
    id2title = load_pickle_from_bucket(BUCKET_NAME, "id2title.pkl")
except Exception as e:
    print(f"Warning: Could not load id2title. Error: {e}")
    id2title = {}
# 3. Load Document Norms (CRITICAL for BM25!)
try:
    doc_norms = load_pickle_from_bucket(BUCKET_NAME, "doc_norms.pkl")
except Exception as e:
    print(f"Warning: Could not load doc_norms. Error: {e}")
    doc_norms = {}
# 4. Load PageRank (Optional but recommended)
try:
    pagerank = load_pickle_from_bucket(BUCKET_NAME, "pagerank.pkl")
    pagerank_normalized = normalize_dict(pagerank)
    print(f"PageRank loaded: {len(pagerank)} documents")
except Exception as e:
    print(f"Info: PageRank not available. Using BM25+ only. Error: {e}")
    pagerank = {}
    pagerank_normalized = {}
# Calculate AVGDL
AVGDL = sum(doc_norms.values()) / len(doc_norms) if doc_norms else 1
print("Server is ready.")


# ==============================================================================
# 5. SEARCH ENDPOINT (BM25+ with optional PageRank)
# ==============================================================================
@app.route("/search")
def search():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])
    # A. Tokenize query
    query_tokens = tokenize(query)
    query_tokens = [t for t in query_tokens if t in index_body.df]
    if not query_tokens:
        return jsonify([])
    bm25_scores = Counter()
    # B. Calculate BM25+ scores
    for term in query_tokens:
        df = index_body.df[term]

        # IDF calculation
        numerator = N_CORPUS - df + 0.5
        denominator = df + 0.5
        idf = math.log(1 + (numerator / denominator))
        posting_list = get_posting_list(index_body, term)
        for doc_id, tf in posting_list:
            doc_len = doc_norms.get(doc_id, AVGDL)

            # BM25+ formula
            num = tf * (k1 + 1)
            den = tf + k1 * (1 - b + b * (doc_len / AVGDL))
            score = idf * ((num / den) + delta)

            bm25_scores[doc_id] += score
    # C. Normalize BM25 scores to [0, 1]
    if bm25_scores:
        max_bm25 = max(bm25_scores.values())
        if max_bm25 > 0:
            for doc_id in bm25_scores:
                bm25_scores[doc_id] /= max_bm25
    # D. Combine with PageRank (if available)
    final_scores = Counter()

    # Use PageRank if available, otherwise pure BM25+
    use_pagerank = bool(pagerank_normalized)

    for doc_id, bm25_score in bm25_scores.items():
        if use_pagerank:
            pr_score = pagerank_normalized.get(doc_id, 0)
            final_scores[doc_id] = W_BM25 * bm25_score + W_PAGERANK * pr_score
        else:
            final_scores[doc_id] = bm25_score
    # E. Get top 100 results
    sorted_results = final_scores.most_common(100)
    # F. Format results
    res = [(str(doc_id), id2title.get(doc_id, str(doc_id)))
           for doc_id, score in sorted_results]
    return jsonify(res)


def run(**options):
    app.run(**options)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
# @app.route("/search_body")
# def search_body():
#     ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
#         SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
#         staff-provided tokenizer from Assignment 3 (GCP part) to do the
#         tokenization and remove stopwords.
#
#         To issue a query navigate to a URL like:
#          http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of up to 100 search results, ordered from best to worst where each
#         element is a tuple (wiki_id, title).
#     '''
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#       return jsonify(res)
#     # BEGIN SOLUTION
#
#     # END SOLUTION
#     return jsonify(res)
#
# @app.route("/search_title")
# def search_title():
#     ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
#         IN THE TITLE of articles, ordered in descending order of the NUMBER OF
#         DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
#         USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
#         tokenization and remove stopwords. For example, a document
#         with a title that matches two distinct query words will be ranked before a
#         document with a title that matches only one distinct query word,
#         regardless of the number of times the term appeared in the title (or
#         query).
#
#         Test this by navigating to the a URL like:
#          http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of ALL (not just top 100) search results, ordered from best to
#         worst where each element is a tuple (wiki_id, title).
#     '''
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#       return jsonify(res)
#     # BEGIN SOLUTION
#
#     # END SOLUTION
#     return jsonify(res)
#
# @app.route("/search_anchor")
# def search_anchor():
#     ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
#         IN THE ANCHOR TEXT of articles, ordered in descending order of the
#         NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
#         DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
#         3 (GCP part) to do the tokenization and remove stopwords. For example,
#         a document with a anchor text that matches two distinct query words will
#         be ranked before a document with anchor text that matches only one
#         distinct query word, regardless of the number of times the term appeared
#         in the anchor text (or query).
#
#         Test this by navigating to the a URL like:
#          http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of ALL (not just top 100) search results, ordered from best to
#         worst where each element is a tuple (wiki_id, title).
#     '''
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#       return jsonify(res)
#     # BEGIN SOLUTION
#
#     # END SOLUTION
#     return jsonify(res)
#
# @app.route("/get_pagerank", methods=['POST'])
# def get_pagerank():
#     ''' Returns PageRank values for a list of provided wiki article IDs.
#
#         Test this by issuing a POST request to a URL like:
#           http://YOUR_SERVER_DOMAIN/get_pagerank
#         with a json payload of the list of article ids. In python do:
#           import requests
#           requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
#         As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of floats:
#           list of PageRank scores that correrspond to the provided article IDs.
#     '''
#     res = []
#     wiki_ids = request.get_json()
#     if len(wiki_ids) == 0:
#       return jsonify(res)
#     # BEGIN SOLUTION
#
#     # END SOLUTION
#     return jsonify(res)
#
# @app.route("/get_pageview", methods=['POST'])
# def get_pageview():
#     ''' Returns the number of page views that each of the provide wiki articles
#         had in August 2021.
#
#         Test this by issuing a POST request to a URL like:
#           http://YOUR_SERVER_DOMAIN/get_pageview
#         with a json payload of the list of article ids. In python do:
#           import requests
#           requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
#         As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of ints:
#           list of page view numbers from August 2021 that correrspond to the
#           provided list article IDs.
#     '''
#     res = []
#     wiki_ids = request.get_json()
#     if len(wiki_ids) == 0:
#       return jsonify(res)
#     # BEGIN SOLUTION
#
#     # END SOLUTION
#     return jsonify(res)
#
# def run(**options):
#     app.run(**options)
#
# if __name__ == '__main__':
#     # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
#     app.run(host='0.0.0.0', port=8080, debug=True)
