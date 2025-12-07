# CogniSearch (Wikipedia-only IR)

Python 3.12+ stack: Scrapy crawler, scikit-learn TF-IDF indexer, optional Word2Vec+FAISS semantic index, and Flask query processor with WordNet-based query expansion plus spelling suggestions. Domain is restricted to `en.wikipedia.org` with the seed at `https://en.wikipedia.org/wiki/Information_retrieval`.

## Setup
1. Create and activate a virtual env:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -U pip
   pip install -r requirements.txt
   ```

## Directory Layout
- `data/raw_html/` crawled HTML files (md5-hashed filenames)
- `data/queries.csv` input queries (Q01-Q05)
- `data/index.json` inverted index output
- `data/ranked_results.csv` ranked output
- `src/crawler/spiders/wiki_spider.py` Scrapy spider
- `src/indexer.py` TF-IDF index builder (optional Word2Vec + FAISS semantic index)
- `src/processor.py` batch + Flask query processor (WordNet expansion + spelling suggestions + optional semantic kNN)
- `src/artifact_generator.py` pipeline runner for Mode A/B (semantic build + scrapyd scheduling optional)
- `notebooks/Project_Report.ipynb` report skeleton

## Execution Modes (artifact_generator)
- Mode A (report/minimal): crawl 10 pages, depth 1, top-K=3
- Mode B (submission/expansive): crawl 100 pages, depth 3, top-K=10

Run pipeline end-to-end (defaults to Mode A):
```bash
python -m src.artifact_generator --mode A --clean
```
Run expansive submission artifacts:
```bash
python -m src.artifact_generator --mode B --clean
```
Add semantic index build (Word2Vec + FAISS) and/or distributed crawl via scrapyd:
```bash
python -m src.artifact_generator --mode B --clean --semantic --use-scrapyd --scrapyd-url http://localhost:6800
```

## Individual Steps
### Crawl (Scrapy)
```bash
python -m scrapy runspider src/crawler/spiders/wiki_spider.py \
  -a seed_url=https://en.wikipedia.org/wiki/Information_retrieval \
  -a output_dir=data/raw_html \
  -a max_pages=10 -a max_depth=1 \
  -a autothrottle_enabled=True -a ignore_robots=True
```

### Index (TF-IDF)
```bash
python -m src.indexer --html-dir data/raw_html --index-out data/index.json --model-out data/tfidf_model.pkl
```

### Index (Semantic: Word2Vec + FAISS)
```bash
python -m src.indexer --html-dir data/raw_html \
  --index-out data/index.json --model-out data/tfidf_model.pkl \
  --semantic --semantic-model-out data/semantic/word2vec.model \
  --semantic-index-out data/semantic/faiss.index --vector-size 100
```

### Process Queries (batch CSV)
```bash
python -m src.processor --model data/tfidf_model.pkl --queries data/queries.csv --output data/ranked_results.csv --top-k 5 --use-semantic
```

### Serve Queries (Flask API)
```bash
python -m src.processor --model data/tfidf_model.pkl --serve --top-k 5 --port 5000 --use-semantic
```
POST a query:
```bash
curl -X POST http://localhost:5000/query -H "Content-Type: application/json" \
  -d '{"query_text": "vector space model", "top_k": 5}'

Response includes `corrected_query` and `suggestions` when spell-fixes apply; semantic kNN is used when available, with TF-IDF fallback.
```

## Notes
- Robot rules are ignored by default for testing; adjust `ignore_robots` if needed.
- NLTK data is downloaded on-demand (wordnet, stopwords, punkt, omw-1.4).
- Filenames are md5(url).html to keep deterministic doc IDs.
- Crawl and submissions stay within `en.wikipedia.org`.
