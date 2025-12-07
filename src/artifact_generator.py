import argparse
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from scrapy.crawler import CrawlerProcess

from src.crawler.spiders.wiki_spider import WikiSpider
from src.indexer import build_index
from src.processor import load_model, rank_queries_csv

SEED_URL = "https://en.wikipedia.org/wiki/Information_retrieval"
RAW_HTML_DIR = Path("data/raw_html")
INDEX_PATH = Path("data/index.json")
MODEL_PATH = Path("data/tfidf_model.pkl")
SEM_MODEL_PATH = Path("data/semantic/word2vec.model")
SEM_FAISS_PATH = Path("data/semantic/faiss.index")
SEM_DOCIDS_PATH = Path("data/semantic/semantic_doc_ids.pkl")
QUERIES_PATH = Path("data/queries.csv")
RANKED_RESULTS_PATH = Path("data/ranked_results.csv")

MODES = {
    "A": {"pages": 10, "depth": 1, "top_k": 3},
    "B": {"pages": 100, "depth": 3, "top_k": 10},
}


def run_crawl(seed_url: str, max_pages: int, max_depth: int):
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    process = CrawlerProcess(
        settings={
            "LOG_LEVEL": "INFO",
            "ROBOTSTXT_OBEY": False,
            "AUTOTHROTTLE_ENABLED": True,
            "DEPTH_LIMIT": max_depth,
            "CLOSESPIDER_PAGECOUNT": max_pages,
        }
    )
    process.crawl(
        WikiSpider,
        seed_url=seed_url,
        output_dir=str(RAW_HTML_DIR),
        max_pages=max_pages,
        max_depth=max_depth,
        autothrottle_enabled=True,
        ignore_robots=True,
    )
    process.start()


def run_crawl_scrapyd(seed_url: str, max_pages: int, max_depth: int, scrapyd_url: str, project: str = "default"):
    data = {
        "project": project,
        "spider": "wiki_spider",
        "seed_url": seed_url,
        "max_pages": str(max_pages),
        "max_depth": str(max_depth),
    }
    body = urllib.parse.urlencode(data).encode()
    schedule_url = urllib.parse.urljoin(scrapyd_url, "/schedule.json")
    try:
        with urllib.request.urlopen(schedule_url, data=body, timeout=10) as resp:
            return resp.read()
    except Exception as exc:  # pragma: no cover - best-effort fallback
        print(f"[scrapyd] Failed to schedule job ({exc}); falling back to local crawl.")
        run_crawl(seed_url, max_pages, max_depth)


def generate_artifacts(mode: str, seed_url: str, clean: bool, semantic: bool, use_scrapyd: bool, scrapyd_url: str):
    if mode not in MODES:
        raise ValueError(f"Mode must be one of {list(MODES.keys())}")
    config = MODES[mode]

    if clean and RAW_HTML_DIR.exists():
        shutil.rmtree(RAW_HTML_DIR)
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

    if use_scrapyd:
        run_crawl_scrapyd(seed_url, config["pages"], config["depth"], scrapyd_url)
    else:
        run_crawl(seed_url, config["pages"], config["depth"])
    build_index(
        RAW_HTML_DIR,
        INDEX_PATH,
        MODEL_PATH,
        semantic_model_out=SEM_MODEL_PATH,
        semantic_index_out=SEM_FAISS_PATH,
        build_semantic=semantic,
    )

    model = load_model(MODEL_PATH)
    rank_queries_csv(model, QUERIES_PATH, RANKED_RESULTS_PATH, config["top_k"])


def main():
    parser = argparse.ArgumentParser(description="Generate artifacts for CogniSearch")
    parser.add_argument("--mode", choices=["A", "B"], default="A", help="A=Minimal (report), B=Expansive (submission)")
    parser.add_argument("--seed", default=SEED_URL, help="Seed URL within en.wikipedia.org")
    parser.add_argument("--clean", action="store_true", help="Remove existing raw_html before crawling")
    parser.add_argument("--semantic", action="store_true", help="Build semantic Word2Vec + FAISS index")
    parser.add_argument("--use-scrapyd", action="store_true", help="Schedule crawl via scrapyd (distributed)")
    parser.add_argument("--scrapyd-url", default="http://localhost:6800", help="Scrapyd base URL")
    args = parser.parse_args()

    generate_artifacts(args.mode, args.seed, args.clean, args.semantic, args.use_scrapyd, args.scrapyd_url)


if __name__ == "__main__":
    main()
