# CogniSearch — Wikipedia Information Retrieval System

A reproducible information-retrieval project that crawls a bounded set of English Wikipedia pages, builds a TF-IDF index, optionally adds Word2Vec/FAISS semantic retrieval, and serves ranked queries through a Flask API.

## Capabilities

- Domain-restricted Scrapy crawler with page/depth limits and auto-throttling
- TF-IDF document indexing and ranked retrieval
- Query expansion and spelling suggestions using NLTK
- Optional Word2Vec embeddings and FAISS nearest-neighbor search
- Batch CSV evaluation artifacts and an interactive Flask endpoint
- Deterministic document identifiers derived from source URLs

## Responsible crawling defaults

The crawler now obeys `robots.txt` by default, stays within `en.wikipedia.org`, filters non-article namespaces, and uses bounded page counts plus auto-throttling. Before crawling any other site, review its terms, robots policy, rate limits, and data license.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the small, report-oriented pipeline:

```bash
python -m src.artifact_generator --mode A --clean
```

Run the larger checked configuration:

```bash
python -m src.artifact_generator --mode B --clean
```

Add the optional semantic index:

```bash
python -m src.artifact_generator --mode B --clean --semantic
```

## Individual stages

```bash
python -m scrapy runspider src/crawler/spiders/wiki_spider.py \
  -a seed_url=https://en.wikipedia.org/wiki/Information_retrieval \
  -a output_dir=data/raw_html -a max_pages=10 -a max_depth=1 \
  -a autothrottle_enabled=True -a ignore_robots=False

python -m src.indexer \
  --html-dir data/raw_html \
  --index-out data/index.json \
  --model-out data/tfidf_model.pkl

python -m src.processor \
  --model data/tfidf_model.pkl \
  --queries data/queries.csv \
  --output data/ranked_results.csv \
  --top-k 5
```

Serve the query API:

```bash
python -m src.processor --model data/tfidf_model.pkl --serve --top-k 5 --port 5000
```

## Repository map

- `src/crawler/`: crawling and article filtering
- `src/indexer.py`: TF-IDF and optional semantic indexes
- `src/processor.py`: query processing, ranking, and API
- `src/artifact_generator.py`: end-to-end pipeline orchestration
- `data/`: checked-in sample crawl and generated retrieval artifacts
- `notebooks/Project_Report.ipynb`: project report notebook

## Evaluation and limitations

The committed ranked results are a reproducibility snapshot, not evidence of general search quality. A stronger evaluation should use relevance judgments and report metrics such as Precision@k, Recall@k, MAP, or nDCG. Results depend on the crawl frontier, page versions, tokenization, expansion rules, and optional semantic settings. Pickled and FAISS artifacts should only be loaded from trusted sources.

## License

Code is available under the [MIT License](LICENSE). Wikipedia content is governed by its own licensing and attribution requirements.
