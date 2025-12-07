import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import gensim
import nltk
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from nltk.corpus import stopwords, wordnet
from nltk.metrics import edit_distance
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
MODEL = None
SEMANTIC = None
DEFAULT_TOP_K = 5


def ensure_nltk():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def load_model(model_path: Path):
    with model_path.open("rb") as f:
        return pickle.load(f)


def load_semantic(sem_model_path: Path, sem_faiss_path: Path, sem_docids_path: Path):
    if not sem_model_path.exists() or not sem_faiss_path.exists() or not sem_docids_path.exists():
        return None
    w2v = gensim.models.Word2Vec.load(str(sem_model_path))
    faiss_index = faiss.read_index(str(sem_faiss_path))
    with sem_docids_path.open("rb") as f:
        doc_ids = pickle.load(f)
    return {"w2v": w2v, "index": faiss_index, "doc_ids": doc_ids}


def expand_query(query: str) -> str:
    ensure_nltk()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in word_tokenize(query.lower()) if t.isalnum() and t not in stop_words]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]

    synonyms = set()
    for lemma in lemmas:
        for syn in wordnet.synsets(lemma):
            for lemma_obj in syn.lemmas():
                raw = lemma_obj.name().replace("_", " ")
                for token in raw.split():
                    if token.isalpha() and token not in stop_words:
                        synonyms.add(lemmatizer.lemmatize(token))
    expanded = lemmas + list(synonyms)
    return " ".join(expanded) if expanded else query


def suggest_spellings(query: str, vocabulary: Dict[str, int]) -> Dict[str, str]:
    # Simple edit-distance suggestion against vectorizer vocabulary
    vocab_terms = list(vocabulary.keys())
    suggestions = {}
    for token in word_tokenize(query.lower()):
        if not token.isalpha():
            continue
        if token in vocabulary:
            continue
        best_term = None
        best_distance = 3  # small radius to avoid noisy matches
        for term in vocab_terms:
            dist = edit_distance(token, term)
            if dist < best_distance:
                best_distance = dist
                best_term = term
                if dist == 1:
                    break
        if best_term:
            suggestions[token] = best_term
    return suggestions


def apply_corrections(query: str, suggestions: Dict[str, str]) -> str:
    tokens = word_tokenize(query)
    corrected = []
    for tok in tokens:
        key = tok.lower()
        corrected.append(suggestions.get(key, tok))
    return " ".join(corrected)


def rank_query(model, query_text: str, top_k: int) -> List[Tuple[str, float]]:
    vectorizer = model["vectorizer"]
    tfidf_matrix = model["tfidf_matrix"]
    doc_ids = model["doc_ids"]

    suggestions = suggest_spellings(query_text, vectorizer.vocabulary_)
    corrected = apply_corrections(query_text, suggestions)
    expanded = expand_query(corrected)
    query_vec = vectorizer.transform([expanded])
    scores = linear_kernel(query_vec, tfidf_matrix).flatten()
    ranked_indices = scores.argsort()[::-1][:top_k]
    return [(doc_ids[i], float(scores[i])) for i in ranked_indices]


def rank_query_semantic(model, semantic, query_text: str, top_k: int) -> List[Tuple[str, float]]:
    vectorizer = model["vectorizer"]
    vocab = vectorizer.vocabulary_
    w2v = semantic["w2v"]
    faiss_index = semantic["index"]
    doc_ids = semantic["doc_ids"]

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = [lemmatizer.lemmatize(t) for t in word_tokenize(query_text.lower()) if t.isalpha() and t not in stop_words]
    synonyms = []
    for tok in tokens:
        for syn in wordnet.synsets(tok):
            for lemma in syn.lemmas():
                cand = lemma.name().replace("_", " ")
                synonyms.extend([lemmatizer.lemmatize(x) for x in cand.split() if x.isalpha()])
    semantic_tokens = tokens + synonyms
    word_vecs = [w2v.wv[t] for t in semantic_tokens if t in w2v.wv]
    if not word_vecs:
        return []
    query_vec = np.mean(word_vecs, axis=0).astype(np.float32)
    query_vec = query_vec.reshape(1, -1)
    faiss.normalize_L2(query_vec)
    scores, indices = faiss_index.search(query_vec, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        results.append((doc_ids[int(idx)], float(score)))
    return results


def rank_queries_csv(model, queries_csv: Path, output_csv: Path, top_k: int):
    df = pd.read_csv(queries_csv)
    rows = []
    for _, row in df.iterrows():
        qid = row["query_id"]
        qtext = row["query_text"]
        if SEMANTIC:
            results = rank_query_semantic(model, SEMANTIC, qtext, top_k)
            if not results:
                results = rank_query(model, qtext, top_k)
        else:
            results = rank_query(model, qtext, top_k)
        for rank, (doc_id, score) in enumerate(results, start=1):
            rows.append({
                "query_id": qid,
                "rank": rank,
                "document_id": doc_id,
                "score": score,
            })
    out_df = pd.DataFrame(rows, columns=["query_id", "rank", "document_id", "score"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)


@app.route("/query", methods=["POST"])
def query_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    query_text = data.get("query_text")
    top_k = int(data.get("top_k", DEFAULT_TOP_K))
    if not query_text:
        return jsonify({"error": "query_text is required"}), 400
    # Spell suggestions against TF-IDF vocabulary
    suggestions = suggest_spellings(query_text, MODEL["vectorizer"].vocabulary_)
    corrected = apply_corrections(query_text, suggestions)

    # Choose semantic if enabled, fallback to TF-IDF
    if SEMANTIC:
        results = rank_query_semantic(MODEL, SEMANTIC, corrected, top_k)
        if not results:
            results = rank_query(MODEL, corrected, top_k)
    else:
        results = rank_query(MODEL, corrected, top_k)

    return jsonify({
        "query_text": query_text,
        "corrected_query": corrected,
        "suggestions": suggestions,
        "results": [{"document_id": doc, "score": score} for doc, score in results],
    })


def main():
    parser = argparse.ArgumentParser(description="Query processor with Flask and batch CSV support")
    parser.add_argument("--model", default="data/tfidf_model.pkl", help="Path to TF-IDF model pickle")
    parser.add_argument("--queries", default="data/queries.csv", help="Path to queries CSV")
    parser.add_argument("--output", default="data/ranked_results.csv", help="Path for ranked results CSV")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--serve", action="store_true", help="Start Flask server instead of batch scoring")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--semantic-index", default="data/semantic/faiss.index", help="Path to FAISS semantic index")
    parser.add_argument("--semantic-model", default="data/semantic/word2vec.model", help="Path to Word2Vec model")
    parser.add_argument("--semantic-docids", default="data/semantic/semantic_doc_ids.pkl", help="Path to semantic doc_ids pickle")
    parser.add_argument("--use-semantic", action="store_true", help="Use semantic kNN (Word2Vec + FAISS) if available")
    args = parser.parse_args()

    global MODEL, DEFAULT_TOP_K
    DEFAULT_TOP_K = args.top_k
    MODEL = load_model(Path(args.model))
    if args.use_semantic:
        sem = load_semantic(Path(args.semantic_model), Path(args.semantic_index), Path(args.semantic_docids))
        if sem:
            global SEMANTIC
            SEMANTIC = sem

    if args.serve:
        app.run(host=args.host, port=args.port, debug=False)
    else:
        rank_queries_csv(MODEL, Path(args.queries), Path(args.output), args.top_k)


if __name__ == "__main__":
    main()
