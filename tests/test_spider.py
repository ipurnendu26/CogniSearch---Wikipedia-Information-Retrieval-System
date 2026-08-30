from src.crawler.spiders.wiki_spider import WikiSpider


def make_spider():
    return WikiSpider(seed_url="https://en.wikipedia.org/wiki/Information_retrieval")


def test_robots_are_obeyed_by_default():
    spider = make_spider()
    assert spider.custom_settings["ROBOTSTXT_OBEY"] is True


def test_article_filter_restricts_domain_and_namespace():
    spider = make_spider()
    assert spider._is_valid_article("https://en.wikipedia.org/wiki/Search_engine")
    assert not spider._is_valid_article("https://example.com/wiki/Search_engine")
    assert not spider._is_valid_article("https://en.wikipedia.org/wiki/File:Example.png")
