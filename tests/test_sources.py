from concurrent.futures import ThreadPoolExecutor

from conftest import fake_paper
from jarvis.sources import SourceRegistry, finalize_citations, render_item


def test_same_paper_from_two_apis_is_one_source():
    reg = SourceRegistry()
    a = reg.add("paper", fake_paper(1, pdf_url=None))
    # Semantic Scholar reports the arXiv id as `id`, has a pdf link arXiv entry lacked
    b = reg.add("paper", {"source": "semantic_scholar", "id": "2401.00001", "title": "Paper Number 1",
                          "url": "https://www.semanticscholar.org/paper/x", "pdf_url": "https://x/p.pdf"})
    assert a == b == "S1"
    assert reg.get("S1")["item"]["pdf_url"] == "https://x/p.pdf"  # enriched, not overwritten


def test_title_dedup_and_distinct_ids():
    reg = SourceRegistry()
    s1 = reg.add("paper", {"source": "openalex", "id": "https://doi.org/10.1145/abc", "title": "RAGAS: Automated Eval"})
    s2 = reg.add("paper", {"source": "crossref", "id": "10.1145/ABC", "title": "different title"})
    s3 = reg.add("web", {"title": "Blog", "url": "https://www.example.com/post/"})
    s4 = reg.add("web", {"title": "Blog again", "url": "http://example.com/post"})
    assert s1 == s2  # same DOI, case-insensitive
    assert s3 == s4  # same URL modulo scheme/www/trailing slash
    assert s1 != s3


def test_registry_seeded_from_state_continues_numbering():
    reg = SourceRegistry()
    reg.add("paper", fake_paper(1))
    reg.add("paper", fake_paper(2))
    reg2 = SourceRegistry(reg.to_dict())
    assert reg2.add("paper", fake_paper(2)) == "S2"
    assert reg2.add("paper", fake_paper(3)) == "S3"


def test_registry_is_thread_safe():
    reg = SourceRegistry()
    with ThreadPoolExecutor(8) as ex:
        ids = list(ex.map(lambda i: reg.add("paper", fake_paper(i % 20)), range(200)))
    assert len(set(ids)) == 20
    assert len(reg.items) == 20


def test_finalize_citations_renumbers_in_order_of_use():
    sources = SourceRegistry()
    for i in range(1, 5):
        sources.add("paper", fake_paper(i))
    text, cited, stats = finalize_citations("A [S3]. B [S1, S3]. C [S2][S3].", sources.to_dict())
    assert text.startswith("A [1]. B [2][1]. C [3][1].")
    assert [c["sid"] for c in cited] == ["S3", "S1", "S2"]
    assert "## Sources" in text and "- [1] Paper number 3 (2024) — https://arxiv.org/abs/2401.00003" in text
    assert stats == {"cited": 3, "invalid_tags": 0, "retrieved": 4}


def test_finalize_citations_drops_invented_tags_and_model_sources_section():
    sources = SourceRegistry()
    sources.add("paper", fake_paper(1))
    report = "Claim [S1]. Made-up [S42].\n\n## References\n[1] Some invented paper"
    text, cited, stats = finalize_citations(report, sources.to_dict())
    assert "[S42]" not in text and "invented paper" not in text
    assert "Made-up." in text
    assert stats["invalid_tags"] == 1 and stats["cited"] == 1


def test_no_tags_means_no_sources_section():
    text, cited, stats = finalize_citations("Nothing cited here.", {})
    assert text == "Nothing cited here." and cited == [] and stats["cited"] == 0


def test_render_item_kinds():
    assert render_item("S1", "paper", fake_paper(1, cited_by=12)).startswith("[S1] Paper number 1 (2024, cited 12×)")
    assert "★42" in render_item("S2", "code", {"title": "a/b", "stars": 42, "url": "u"})
    assert render_item("S3", "community", {"source": "hackernews", "title": "t", "url": "u"}).startswith("[S3] hackernews: t")
