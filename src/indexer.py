import argparse
import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple

import faiss
import gensim
import nltk
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class LemmaTokenizer:
    """Pickle-safe tokenizer that cleans, tokenizes, and lemmatizes."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def __call__(self, text: str):
        text = re.sub(r"[^A-Za-z0-9\s]+", " ", text)
        tokens = word_tokenize(text.lower())
        return [self.lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in self.stop_words]


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


def train_word2vec(tokenized_docs: List[List[str]], vector_size: int = 100) -> gensim.models.Word2Vec:
    # Train a lightweight CBOW model for semantic vectors
    return gensim.models.Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        sg=0,
        epochs=10,
    )


def build_faiss_index(doc_vectors: np.ndarray) -> faiss.Index:
    # Normalize for cosine similarity and index with inner product
    faiss.normalize_L2(doc_vectors)
    index = faiss.IndexFlatIP(doc_vectors.shape[1])
    index.add(doc_vectors)
    return index


def average_doc_vectors(model: gensim.models.Word2Vec, tokenized_docs: List[List[str]]) -> np.ndarray:
    vectors = []
    for tokens in tokenized_docs:
        word_vecs = [model.wv[t] for t in tokens if t in model.wv]
        if word_vecs:
            vec = np.mean(word_vecs, axis=0)
        else:
            vec = np.zeros(model.vector_size, dtype=np.float32)
        vectors.append(vec.astype(np.float32))
    return np.vstack(vectors)


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def build_index(
    html_dir: Path,
    index_out: Path,
    model_out: Path,
    semantic_model_out: Path | None = None,
    semantic_index_out: Path | None = None,
    build_semantic: bool = False,
    vector_size: int = 100,
):
    ensure_nltk()
    files = sorted(html_dir.glob("*.html"))
    if not files:
        raise FileNotFoundError(f"No HTML files found in {html_dir}")

    documents: list[str] = []
    doc_ids: list[str] = []
    for fp in files:
        html = fp.read_text(encoding="utf-8", errors="ignore")
        documents.append(extract_text(html))
        doc_ids.append(fp.stem)

    tokenizer = LemmaTokenizer()
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        preprocessor=None,
        token_pattern=None,
        lowercase=True,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    inverted = {}
    for term, idx in vectorizer.vocabulary_.items():
        column = tfidf_matrix[:, idx].tocoo()
        postings = [(doc_ids[row], float(val)) for row, val in zip(column.row, column.data)]
        postings.sort(key=lambda x: x[1], reverse=True)
        inverted[term] = postings

    index_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    with index_out.open("w", encoding="utf-8") as f:
        json.dump(inverted, f, ensure_ascii=False, indent=2)

    with model_out.open("wb") as f:
        pickle.dump({"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix, "doc_ids": doc_ids}, f)

    # Optional semantic index (Word2Vec + FAISS)
    if build_semantic:
        if semantic_model_out is None or semantic_index_out is None:
            raise ValueError("Semantic outputs must be provided when build_semantic=True")

        tokenized_docs = [tokenizer(doc) for doc in documents]
        w2v = train_word2vec(tokenized_docs, vector_size=vector_size)
        doc_vectors = average_doc_vectors(w2v, tokenized_docs)
        faiss_index = build_faiss_index(doc_vectors)

        semantic_model_out.parent.mkdir(parents=True, exist_ok=True)
        semantic_index_out.parent.mkdir(parents=True, exist_ok=True)

        w2v.save(str(semantic_model_out))
        faiss.write_index(faiss_index, str(semantic_index_out))

        # Store doc_ids alongside semantic artifacts for lookup
        with (semantic_model_out.parent / "semantic_doc_ids.pkl").open("wb") as f:
            pickle.dump(doc_ids, f)


def main():
    parser = argparse.ArgumentParser(description="Build TF-IDF index from HTML corpus")
    parser.add_argument("--html-dir", default="data/raw_html")
    parser.add_argument("--index-out", default="data/index.json")
    parser.add_argument("--model-out", default="data/tfidf_model.pkl")
    parser.add_argument("--semantic-model-out", default="data/semantic/word2vec.model", help="Path to save Word2Vec model")
    parser.add_argument("--semantic-index-out", default="data/semantic/faiss.index", help="Path to save FAISS index")
    parser.add_argument("--semantic", action="store_true", help="Build semantic embedding index (Word2Vec + FAISS)")
    parser.add_argument("--vector-size", type=int, default=100, help="Embedding dimensionality for Word2Vec")
    args = parser.parse_args()

    build_index(
        Path(args.html_dir),
        Path(args.index_out),
        Path(args.model_out),
        Path(args.semantic_model_out),
        Path(args.semantic_index_out),
        build_semantic=args.semantic,
        vector_size=args.vector_size,
    )


if __name__ == "__main__":
    main()
