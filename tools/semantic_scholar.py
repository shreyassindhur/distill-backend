"""
Semantic Scholar API integration for Distill.
Fetches real peer-reviewed papers with verified metadata.
Free tier: 100 requests/5 min, no API key needed for basic use.
"""

import requests
import time
from typing import Optional

BASE_URL = "https://api.semanticscholar.org/graph/v1"

HEADERS = {
    "User-Agent": "Distill-Research-Assistant/1.0",
}

FIELDS = (
    "paperId,title,abstract,year,authors,venue,publicationVenue,"
    "externalIds,citationCount,referenceCount,openAccessPdf,"
    "publicationTypes,journal"
)


def search_papers(
    query: str,
    limit: int = 10,
    min_citations: int = 0,
    year_from: Optional[int] = None,
) -> list[dict]:
    """
    Search Semantic Scholar for real academic papers.
    Returns list of paper dicts with verified metadata.
    """
    try:
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": FIELDS,
        }
        if year_from:
            params["year"] = f"{year_from}-"

        resp = requests.get(
            f"{BASE_URL}/paper/search",
            params=params,
            headers=HEADERS,
            timeout=10,
        )

        if resp.status_code == 429:
            # Rate limited — wait and retry once
            time.sleep(3)
            resp = requests.get(
                f"{BASE_URL}/paper/search",
                params=params,
                headers=HEADERS,
                timeout=10,
            )

        if resp.status_code != 200:
            print(f"[Semantic Scholar] Search failed: {resp.status_code}")
            return []

        data = resp.json()
        papers = data.get("data", [])

        results = []
        for p in papers:
            if not p.get("abstract"):
                continue  # skip papers with no abstract
            if p.get("citationCount", 0) < min_citations:
                continue

            # Build clean paper object
            authors = p.get("authors", [])
            author_str = _format_authors(authors)

            doi = (p.get("externalIds") or {}).get("DOI", "")
            arxiv = (p.get("externalIds") or {}).get("ArXiv", "")

            # Build URL — prefer DOI, then ArXiv, then S2 page
            if doi:
                url = f"https://doi.org/{doi}"
            elif arxiv:
                url = f"https://arxiv.org/abs/{arxiv}"
            else:
                pid = p.get("paperId", "")
                url = f"https://www.semanticscholar.org/paper/{pid}"

            # Open access PDF if available
            oa = p.get("openAccessPdf") or {}
            pdf_url = oa.get("url", "")

            venue = (
                p.get("venue")
                or (p.get("publicationVenue") or {}).get("name", "")
                or (p.get("journal") or {}).get("name", "")
                or "Academic Publication"
            )

            results.append({
                "title":          p.get("title", "Untitled"),
                "authors":        author_str,
                "year":           p.get("year"),
                "venue":          venue,
                "abstract":       p.get("abstract", "")[:800],
                "citation_count": p.get("citationCount", 0),
                "doi":            doi,
                "url":            url,
                "pdf_url":        pdf_url,
                "paper_id":       p.get("paperId", ""),
                "source":         "semantic_scholar",
            })

        # Sort by citation count descending — most cited papers first
        results.sort(key=lambda x: x["citation_count"], reverse=True)
        return results

    except requests.exceptions.Timeout:
        print("[Semantic Scholar] Request timed out")
        return []
    except Exception as e:
        print(f"[Semantic Scholar] Error: {e}")
        return []


def get_paper_details(paper_id: str) -> Optional[dict]:
    """Fetch full details for a specific paper by ID."""
    try:
        resp = requests.get(
            f"{BASE_URL}/paper/{paper_id}",
            params={"fields": FIELDS},
            headers=HEADERS,
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def search_multi_query(topic: str, limit_per_query: int = 6) -> list[dict]:
    """
    Run multiple searches to get diverse, high-quality sources.
    Combines results and deduplicates.
    """
    # Generate search variants to get diverse results
    queries = [
        topic,
        f"{topic} survey",
        f"{topic} systematic review",
        f"{topic} empirical study",
    ]

    seen_ids = set()
    all_results = []

    for query in queries:
        papers = search_papers(query, limit=limit_per_query, min_citations=0)
        for p in papers:
            pid = p["paper_id"]
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_results.append(p)
        # Small delay between requests to respect rate limits
        time.sleep(0.5)

    # Sort by citation count and return top results
    all_results.sort(key=lambda x: x["citation_count"], reverse=True)
    return all_results[:20]


def format_for_synthesis(papers: list[dict]) -> list[dict]:
    """
    Convert Semantic Scholar papers to the format expected
    by the synthesis agent (same structure as Tavily results).
    """
    formatted = []
    for p in papers:
        authors = p.get("authors", "Unknown Authors")
        year = p.get("year", "n.d.")
        venue = p.get("venue", "Academic Publication")
        title = p.get("title", "Untitled")
        citations = p.get("citation_count", 0)
        url = p.get("url", "")

        # Format content as the agent expects
        content = f"""Title: {title}
Authors: {authors}
Year: {year}
Published in: {venue}
Citation count: {citations}
Abstract: {p.get("abstract", "No abstract available.")}"""

        # Add citation quality signal
        if citations > 500:
            quality = "Highly cited foundational paper"
        elif citations > 100:
            quality = "Well-cited established research"
        elif citations > 20:
            quality = "Moderately cited recent research"
        else:
            quality = "Recent or emerging research"

        content += f"\nSource quality: {quality}"

        formatted.append({
            "title":  f"[Academic Paper] {title} ({authors}, {year})",
            "url":    url,
            "content": content,
            "year":   year,
            "authors": authors,
            "venue":  venue,
            "citation_count": citations,
            "doi":    p.get("doi", ""),
            "pdf_url": p.get("pdf_url", ""),
            "source": "semantic_scholar",
        })

    return formatted


def _format_authors(authors: list[dict]) -> str:
    """Format author list for citation."""
    if not authors:
        return "Unknown Authors"
    names = [a.get("name", "") for a in authors if a.get("name")]
    if len(names) == 0:
        return "Unknown Authors"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    if len(names) <= 4:
        return ", ".join(names[:-1]) + f", and {names[-1]}"
    # More than 4 authors — use "et al."
    return f"{names[0]} et al."


def build_ieee_reference(paper: dict, ref_num: int) -> str:
    """
    Build a properly formatted IEEE reference string.
    Format: [N] A. Author, "Title," Venue, year. doi: xxx
    """
    authors = paper.get("authors", "Unknown Authors")
    title = paper.get("title", "Untitled")
    venue = paper.get("venue", "")
    year = paper.get("year", "n.d.")
    doi = paper.get("doi", "")
    url = paper.get("url", "")

    # IEEE author format: First initial. Last name
    ref = f"[{ref_num}] {authors}, \"{title},\""
    if venue:
        ref += f" {venue},"
    ref += f" {year}."
    if doi:
        ref += f" doi: {doi}."
    elif url:
        ref += f" [Online]. Available: {url}"

    return ref