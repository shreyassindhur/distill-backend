from agents.search_agent import run_search_agent
from agents.synthesis_agent import run_synthesis_agent


def parse_followups(report: str):
    main_report = report
    questions = []
    contested = ""

    if "## Contested Claims" in report:
        parts = report.split("## Contested Claims")
        main_report = parts[0].strip()
        rest = parts[1].strip()
        if "## Follow-up Questions" in rest:
            contested_part, followup_part = rest.split("## Follow-up Questions")
            contested = contested_part.strip()
            for line in followup_part.split('\n'):
                line = line.strip()
                if line.startswith('- ') or line.startswith('* '):
                    questions.append(line[2:].strip())
        else:
            contested = rest.strip()
    elif "## Follow-up Questions" in report:
        parts = report.split("## Follow-up Questions")
        main_report = parts[0].strip()
        for line in parts[1].split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                questions.append(line[2:].strip())

    return main_report, questions[:3], contested


def run_research(topic: str, depth: str = "normal", tone: str = "default") -> dict:
    print(f"[Distill] Researching: {topic}")
    search_results = run_search_agent(topic, depth=depth)
    clean = [r for r in search_results if "error" not in r and r.get("content", "").strip()]
    if len(clean) < 2:
        raise Exception("Couldn't find enough reliable sources. Try a more specific term.")
    raw = run_synthesis_agent(topic, clean, mode="topic", tone=tone)
    report, followups, contested = parse_followups(raw)
    return {
        "topic": topic,
        "sources_found": len(clean),
        "report": report,
        "followups": followups,
        "contested": contested,
        "mode": "topic"
    }


def run_url_research(url: str, tone: str = "default") -> dict:
    from tools.url_reader import read_url
    print(f"[Distill] Reading URL: {url}")
    url_content = read_url(url)
    if "error" in url_content:
        raise Exception(f"Could not read URL: {url_content['error']}")

    # Fetch supporting web sources for cross-reference
    topic_for_search = url_content.get("title", "") or url
    supporting = run_search_agent(topic_for_search[:200], depth="quick")
    clean_supporting = [
        r for r in supporting
        if "error" not in r and r.get("content", "").strip()
    ]

    combined = [{
        "title": "SOURCE ARTICLE",
        "url": url,
        "content": url_content["content"][:4000]
    }] + clean_supporting[:6]

    total = len(combined)
    raw = run_synthesis_agent(f"Article Analysis: {url}", combined, mode="url", tone=tone)
    report, followups, contested = parse_followups(raw)
    return {
        "topic": f"URL: {url[:60]}",
        "sources_found": total,
        "report": report,
        "followups": followups,
        "contested": contested,
        "mode": "url"
    }


def run_pdf_research(uploaded_file, tone: str = "default") -> dict:
    from tools.pdf_reader import read_pdf
    file_name = (
        getattr(uploaded_file, "filename", None)
        or getattr(uploaded_file, "name", "document.pdf")
    )
    print(f"[Distill] Reading PDF: {file_name}")
    pdf_content = read_pdf(uploaded_file)
    if "error" in pdf_content:
        raise Exception(f"Could not read PDF: {pdf_content['error']}")

    # Fetch supporting web sources for context
    supporting = run_search_agent(pdf_content["content"][:200], depth="quick")
    clean_supporting = [
        r for r in supporting
        if "error" not in r and r.get("content", "").strip()
    ]

    combined = [{
        "title": f"Uploaded PDF: {file_name}",
        "url": "Uploaded document",
        "content": pdf_content["content"][:4000]
    }] + clean_supporting[:6]

    total = len(combined)
    raw = run_synthesis_agent(f"Analysis of: {file_name}", combined, mode="pdf", tone=tone)
    report, followups, contested = parse_followups(raw)
    return {
        "topic": f"PDF: {file_name[:40]}",
        "sources_found": total,
        "report": report,
        "followups": followups,
        "contested": contested,
        "mode": "pdf"
    }


def run_analyze(tone: str = "default", url: str = "", uploaded_file=None) -> dict:
    combined = []
    source_label = ""
    search_query = ""

    if url and url.startswith("http"):
        from tools.url_reader import read_url
        print(f"[Distill] Reading URL: {url}")
        content = read_url(url)
        if "error" in content:
            raise Exception(f"Could not read URL: {content['error']}")
        combined.append({"title": "SOURCE ARTICLE", "url": url, "content": content["content"][:4000]})
        source_label = url[:60]
        search_query = content.get("title", "") or url

    if uploaded_file:
        from tools.pdf_reader import read_pdf
        file_name = getattr(uploaded_file, "filename", None) or getattr(uploaded_file, "name", "document.pdf")
        print(f"[Distill] Reading PDF: {file_name}")
        content = read_pdf(uploaded_file)
        if "error" in content:
            raise Exception(f"Could not read PDF: {content['error']}")
        combined.append({"title": f"Uploaded PDF: {file_name}", "url": "Uploaded document", "content": content["content"][:4000]})
        if not source_label:
            source_label = f"PDF: {file_name[:40]}"
            search_query = content["content"][:200]
        else:
            search_query = content.get("title", f"{file_name} {content['content'][:100]}")

    if not combined:
        raise Exception("Provide a URL or upload a file to analyze.")

    # Fetch supporting web sources for cross-reference
    if search_query:
        supporting = run_search_agent(search_query[:200], depth="quick")
        clean_supporting = [
            r for r in supporting
            if "error" not in r and r.get("content", "").strip()
        ]
        combined += clean_supporting[:6]

    label = f"Analysis: {source_label}"
    raw = run_synthesis_agent(label, combined, mode="url", tone=tone)
    report, followups, contested = parse_followups(raw)
    return {
        "topic": label,
        "sources_found": len(combined),
        "report": report,
        "followups": followups,
        "contested": contested,
        "mode": "analyze"
    }


def run_comparison_research(topic_a: str, topic_b: str, depth: str = "normal") -> dict:
    print(f"[Distill] Comparing: {topic_a} vs {topic_b}")
    results_a = run_search_agent(topic_a, depth=depth)
    results_b = run_search_agent(topic_b, depth=depth)
    clean_a = [r for r in results_a if "error" not in r and r.get("content", "").strip()]
    clean_b = [r for r in results_b if "error" not in r and r.get("content", "").strip()]
    if len(clean_a) < 1 or len(clean_b) < 1:
        raise Exception("Couldn't find enough sources for one or both topics.")
    for r in clean_a:
        r["title"] = f"[{topic_a}] {r.get('title', '')}"
    for r in clean_b:
        r["title"] = f"[{topic_b}] {r.get('title', '')}"
    combined = clean_a + clean_b
    raw = run_synthesis_agent(f"{topic_a} vs {topic_b}", combined, mode="comparison")
    report, followups, contested = parse_followups(raw)
    return {
        "topic": f"{topic_a} vs {topic_b}",
        "sources_found": len(combined),
        "report": report,
        "followups": followups,
        "contested": contested,
        "mode": "comparison"
    }


def run_write_paper(topic: str, depth: str = "normal") -> dict:
    """
    Write a full IEEE research paper.
    Uses Semantic Scholar for real academic citations + Tavily for current web context.
    """
    from tools.semantic_scholar import search_multi_query, format_for_synthesis

    print(f"[Distill] Writing IEEE paper on: {topic}")

    # Step 1 — Fetch real academic papers from Semantic Scholar
    print("[Distill] Querying Semantic Scholar for academic sources...")
    academic_papers = search_multi_query(topic, limit_per_query=6)
    academic_sources = format_for_synthesis(academic_papers)
    print(f"[Distill] Found {len(academic_sources)} academic papers")

    # Step 2 — Fetch current web context from Tavily
    print("[Distill] Fetching current web context...")
    web_results = run_search_agent(topic, depth="normal")
    web_clean = [
        r for r in web_results
        if "error" not in r and r.get("content", "").strip()
    ]

    # Also search for related academic angle
    web_extra = run_search_agent(f"{topic} research findings 2023 2024", depth="quick")
    web_extra_clean = [
        r for r in web_extra
        if "error" not in r and r.get("content", "").strip()
    ]

    # Step 3 — Combine: academic papers first (higher quality), web context second
    # Academic papers get priority in ordering — model sees them first
    combined = academic_sources + web_clean + web_extra_clean

    # Deduplicate by URL
    seen_urls = set()
    deduped = []
    for r in combined:
        url = r.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            deduped.append(r)

    if len(deduped) < 3:
        raise Exception(
            "Couldn't find enough sources to write a well-supported paper. "
            "Try a more specific topic."
        )

    print(f"[Distill] Total sources for paper: {len(deduped)} "
          f"({len(academic_sources)} academic, {len(web_clean + web_extra_clean)} web)")

    # Step 4 — Generate paper with academic-focused synthesis
    raw = run_synthesis_agent(
        topic,
        deduped[:15],  # cap at 15 sources for quality
        mode="write_paper",
        tone="academic"
    )

    return {
        "topic": topic,
        "sources_found": len(deduped),
        "academic_sources": len(academic_sources),
        "report": raw,
        "followups": [],
        "contested": "",
        "mode": "write_paper"
    }


def run_improve_paper(uploaded_file) -> dict:
    """Read uploaded paper and provide specific improvements with current sources."""
    from tools.pdf_reader import read_pdf
    from tools.semantic_scholar import search_papers, format_for_synthesis

    file_name = (
        getattr(uploaded_file, "filename", None)
        or getattr(uploaded_file, "name", "paper.pdf")
    )
    print(f"[Distill] Improving paper: {file_name}")

    pdf_content = read_pdf(uploaded_file)
    if "error" in pdf_content:
        raise Exception(f"Could not read PDF: {pdf_content['error']}")

    paper_text = pdf_content["content"]

    # Extract topic from beginning of paper for targeted searches
    search_query = paper_text[:400].replace('\n', ' ').strip()

    # Get academic sources related to the paper's topic
    print("[Distill] Finding related academic sources...")
    academic_papers = search_papers(search_query, limit=8, min_citations=0)
    academic_sources = format_for_synthesis(academic_papers)

    # Also get current web sources
    web_results = run_search_agent(search_query[:200], depth="normal")
    web_clean = [r for r in web_results if "error" not in r and r.get("content", "").strip()]

    # Paper being reviewed goes first so agent reads it fully
    combined = [{
        "title": f"PAPER BEING REVIEWED: {file_name}",
        "url": "Uploaded paper",
        "content": paper_text[:3500]
    }] + academic_sources[:6] + web_clean[:4]

    raw = run_synthesis_agent(
        f"Review and improve: {file_name}",
        combined,
        mode="improve_paper",
        tone="academic"
    )

    return {
        "topic": f"Paper Review: {file_name[:40]}",
        "sources_found": len(academic_sources) + len(web_clean),
        "report": raw,
        "followups": [],
        "contested": "",
        "mode": "improve_paper"
    }