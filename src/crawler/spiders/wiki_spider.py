import hashlib
import os
from urllib.parse import urljoin, urlparse

import scrapy


class WikiSpider(scrapy.Spider):
    name = "wiki_spider"
    allowed_domains = ["en.wikipedia.org"]

    def __init__(
        self,
        seed_url: str,
        output_dir: str = "data/raw_html",
        max_pages: int = 10,
        max_depth: int = 1,
        concurrent_requests: int = 8,
        autothrottle_enabled: bool = True,
        autothrottle_start_delay: float = 0.5,
        autothrottle_max_delay: float = 5.0,
        ignore_robots: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.start_urls = [seed_url]
        self.output_dir = output_dir
        self.seen_urls: set[str] = set()
        self.max_pages = max_pages

        self.custom_settings = {
            "ROBOTSTXT_OBEY": not ignore_robots,
            "AUTOTHROTTLE_ENABLED": autothrottle_enabled,
            "AUTOTHROTTLE_START_DELAY": autothrottle_start_delay,
            "AUTOTHROTTLE_MAX_DELAY": autothrottle_max_delay,
            "CONCURRENT_REQUESTS": concurrent_requests,
            "DEPTH_LIMIT": max_depth,
            "CLOSESPIDER_PAGECOUNT": max_pages,
            "FEED_EXPORT_ENCODING": "utf-8",
        }

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        url = response.url
        if url in self.seen_urls:
            return
        self.seen_urls.add(url)

        self._save_html(response)

        for href in response.css("a::attr(href)").getall():
            next_url = urljoin(url, href)
            if self._is_valid_article(next_url) and next_url not in self.seen_urls:
                yield scrapy.Request(next_url, callback=self.parse)

    def _is_valid_article(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        if parsed.netloc != "en.wikipedia.org":
            return False
        if not parsed.path.startswith("/wiki/"):
            return False
        title = parsed.path.split("/wiki/")[-1]
        if ":" in title:
            return False
        return True

    def _save_html(self, response):
        os.makedirs(self.output_dir, exist_ok=True)
        url = response.url
        doc_id = hashlib.md5(url.encode("utf-8")).hexdigest()
        filename = f"{doc_id}.html"
        path = os.path.join(self.output_dir, filename)
        with open(path, "wb") as f:
            f.write(response.body)
        self.logger.info("Saved %s", path)
