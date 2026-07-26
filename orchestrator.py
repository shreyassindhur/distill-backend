import asyncio
import re
from dataclasses import dataclass, field

from agents.search_agent import run_search_agent
from agents.synthesis_agent import run_synthesis_agent


@dataclass
class ResearchContext:
    query: str = ""
    mode: str = "topic"
    tone: str = "default"
    sources: list = field(default_factory=list)
    report: str = ""
    followups: list = field(default_factory=list)
    contested: str = ""
    source_urls: set = field(default_factory=set)
    state: dict = field(default_factory=dict)
    error: str | None = None


# ── Utilities ───────────────────────────────────────────────────────────────


def validate_citations(report: str, source_urls: set) -> str:
    def _check(m):
        text = m.group(1)
        url = m.group(2).rstrip(")")
        if url in source_urls:
            return m.group(0)
        return text
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _check, report)


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


# ── Async helpers (the "agents") ────────────────────────────────────────────


async def _search(ctx: ResearchContext, query: str | None = None, depth: str = "normal"):
    q = query or ctx.query
    results = await asyncio.to_thread(run_search_agent, q, depth)
    clean = [r for r in results if "error" not in r and r.get("content", "").strip()]
    ctx.sources.extend(clean)
    ctx.source_urls.update(r.get("url", "") for r in clean if r.get("url"))


async def _academic_search(ctx: ResearchContext, query: str | None = None):
    from tools.semantic_scholar import search_multi_query, format_for_synthesis
    q = query or ctx.query
    papers = await asyncio.to_thread(search_multi_query, q, 6)
    sources = await asyncio.to_thread(format_for_synthesis, papers)
    ctx.state["academic_sources"] = sources
    ctx.sources.extend(sources)
    ctx.source_urls.update(r.get("url", "") for r in sources if r.get("url"))


async def _extract(ctx: ResearchContext, url: str = "", uploaded_file=None):
    extracted = []
    if url and url.startswith("http"):
        from tools.url_reader import read_url
        content = await asyncio.to_thread(read_url, url)
        if "error" in content:
            raise Exception(f"Could not read URL: {content['error']}")
        extracted.append({
            "title": content.get("title", url)[:80],
            "url": url,
            "content": content["content"][:4000]
        })
        ctx.state["search_query"] = content.get("title", "") or url
    if uploaded_file:
        from tools.pdf_reader import read_pdf
        file_name = (getattr(uploaded_file, "filename", None)
                     or getattr(uploaded_file, "name", "document.pdf"))
        content = await asyncio.to_thread(read_pdf, uploaded_file)
        if "error" in content:
            raise Exception(f"Could not read PDF: {content['error']}")
        extracted.append({
            "title": f"Uploaded PDF: {file_name}",
            "url": "Uploaded document",
            "content": content["content"][:4000]
        })
        if not ctx.state.get("search_query"):
            ctx.state["search_query"] = content["content"][:200]
        else:
            ctx.state["search_query"] = content.get("title", f"{file_name} {content['content'][:100]}")
        ctx.state["file_name"] = file_name
    ctx.state["extracted"] = extracted
    ctx.sources = extracted + ctx.sources
    ctx.source_urls.update(r.get("url", "") for r in extracted if r.get("url"))


async def _draft(ctx: ResearchContext, topic_label: str | None = None):
    topic = topic_label or ctx.query
    raw = await asyncio.to_thread(run_synthesis_agent, topic, ctx.sources, ctx.mode, ctx.tone)
    raw = validate_citations(raw, ctx.source_urls)
    ctx.report, ctx.followups, ctx.contested = parse_followups(raw)


async def _review_and_research(ctx: ResearchContext):
    word_count = len(ctx.report.split())
    unclear_count = ctx.report.lower().count("unclear") + ctx.report.lower().count("limited evidence")
    needs_improvement = word_count < 600 or unclear_count > 3
    if not needs_improvement:
        return

    if ctx.followups:
        tasks = [asyncio.to_thread(run_search_agent, q, "quick") for q in ctx.followups]
        all_results = await asyncio.gather(*tasks)
    else:
        results = await asyncio.to_thread(
            run_search_agent, f"{ctx.query} statistics data findings 2024 2025", "quick"
        )
        all_results = [results]

    new_sources = []
    for results in all_results:
        for r in results:
            if ("error" not in r and r.get("content", "").strip()
                    and r.get("url", "") not in ctx.source_urls):
                new_sources.append(r)
                ctx.source_urls.add(r.get("url", ""))

    if new_sources:
        ctx.sources.extend(new_sources[:4])


# ── Mode pipelines ─────────────────────────────────────────────────────────


async def run_research(topic: str, depth: str = "normal", tone: str = "default") -> dict:
    ctx = ResearchContext(query=topic, mode="topic", tone=tone)

    await asyncio.gather(_search(ctx, depth=depth), _academic_search(ctx))

    clean = [r for r in ctx.sources if "error" not in r and r.get("content", "").strip()]
    if len(clean) < 2:
        raise Exception("Couldn't find enough reliable sources. Try a more specific term.")
    ctx.sources = clean

    await _draft(ctx)
    await _review_and_research(ctx)
    if ctx.state.get("new_sources") or len(ctx.sources) > len(clean):
        await _draft(ctx)

    return {"topic": topic, "sources_found": len(ctx.sources), "report": ctx.report,
            "followups": ctx.followups, "contested": ctx.contested, "mode": "topic"}


async def run_url_research(url: str, tone: str = "default") -> dict:
    ctx = ResearchContext(query=url, mode="url", tone=tone)

    await asyncio.gather(_extract(ctx, url=url), _search(ctx, depth="quick"))

    extracted = ctx.state.get("extracted", [])
    supporting = [r for r in ctx.sources if "error" not in r and r.get("content", "").strip()
                  and r.get("url") not in {e.get("url") for e in extracted}][:6]
    ctx.sources = extracted + supporting
    ctx.source_urls = {r.get("url", "") for r in ctx.sources if r.get("url")}

    await _draft(ctx, topic_label=f"Article Analysis: {url}")
    return {"topic": f"URL: {url[:60]}", "sources_found": len(ctx.sources), "report": ctx.report,
            "followups": ctx.followups, "contested": ctx.contested, "mode": "url"}


async def run_pdf_research(uploaded_file, tone: str = "default") -> dict:
    file_name = (getattr(uploaded_file, "filename", None)
                 or getattr(uploaded_file, "name", "document.pdf"))
    ctx = ResearchContext(query=f"Analysis of: {file_name}", mode="pdf", tone=tone)

    await _extract(ctx, uploaded_file=uploaded_file)
    await _search(ctx, query=ctx.state.get("search_query", file_name)[:200], depth="quick")

    extracted = ctx.state.get("extracted", [])
    supporting = [r for r in ctx.sources if r.get("url") != "Uploaded document"
                  if "error" not in r and r.get("content", "").strip()][:6]
    ctx.sources = extracted + supporting
    ctx.source_urls = {r.get("url", "") for r in ctx.sources if r.get("url")}

    await _draft(ctx)
    return {"topic": f"PDF: {file_name[:40]}", "sources_found": len(ctx.sources), "report": ctx.report,
            "followups": ctx.followups, "contested": ctx.contested, "mode": "pdf"}


async def run_analyze(tone: str = "default", url: str = "", uploaded_file=None) -> dict:
    if not url and not uploaded_file:
        raise Exception("Provide a URL or upload a file to analyze.")
    ctx = ResearchContext(mode="url", tone=tone)

    await asyncio.gather(
        _extract(ctx, url=url, uploaded_file=uploaded_file),
        _search(ctx, query=(url or "Uploaded document")[:200], depth="quick"),
    )

    extracted = ctx.state.get("extracted", [])
    supporting = [r for r in ctx.sources if "error" not in r and r.get("content", "").strip()
                  and r.get("url") not in {e.get("url") for e in extracted}][:6]
    ctx.sources = extracted + supporting
    ctx.source_urls = {r.get("url", "") for r in ctx.sources if r.get("url")}

    label = f"Analysis: {url[:60] if url else ctx.state.get('file_name', 'document')[:40]}"
    await _draft(ctx, topic_label=label)
    return {"topic": label, "sources_found": len(ctx.sources), "report": ctx.report,
            "followups": ctx.followups, "contested": ctx.contested, "mode": "analyze"}


async def run_comparison_research(topic_a: str, topic_b: str, depth: str = "normal") -> dict:
    ctx_a = ResearchContext(query=topic_a)
    ctx_b = ResearchContext(query=topic_b)

    await asyncio.gather(_search(ctx_a, depth=depth), _search(ctx_b, depth=depth))

    clean_a = [r for r in ctx_a.sources if "error" not in r and r.get("content", "").strip()]
    clean_b = [r for r in ctx_b.sources if "error" not in r and r.get("content", "").strip()]
    if len(clean_a) < 1 or len(clean_b) < 1:
        raise Exception("Couldn't find enough sources for one or both topics.")

    for r in clean_a:
        r["title"] = f"[{topic_a}] {r.get('title', '')}"
    for r in clean_b:
        r["title"] = f"[{topic_b}] {r.get('title', '')}"

    ctx = ResearchContext(query=f"{topic_a} vs {topic_b}", mode="comparison",
                          sources=clean_a + clean_b)
    ctx.source_urls = {r.get("url", "") for r in ctx.sources if r.get("url")}

    await _draft(ctx)
    return {"topic": f"{topic_a} vs {topic_b}", "sources_found": len(ctx.sources),
            "report": ctx.report, "followups": ctx.followups, "contested": ctx.contested,
            "mode": "comparison"}


async def run_write_paper(topic: str, depth: str = "normal") -> dict:
    from tools.semantic_scholar import search_multi_query, format_for_synthesis

    async def _academic():
        papers = await asyncio.to_thread(search_multi_query, topic, 6)
        sources = await asyncio.to_thread(format_for_synthesis, papers)
        return sources

    async def _web():
        results = await asyncio.to_thread(run_search_agent, topic, "normal")
        return [r for r in results if "error" not in r and r.get("content", "").strip()]

    async def _web_extra():
        results = await asyncio.to_thread(
            run_search_agent, f"{topic} research findings 2023 2024", "quick")
        return [r for r in results if "error" not in r and r.get("content", "").strip()]

    academic, web, web_extra = await asyncio.gather(_academic(), _web(), _web_extra())

    seen = set()
    combined = []
    for r in academic + web + web_extra:
        if r.get("url", "") not in seen:
            seen.add(r.get("url", ""))
            combined.append(r)

    if len(combined) < 3:
        raise Exception("Couldn't find enough sources. Try a more specific topic.")

    print(f"[Distill] Paper sources: {len(academic)} academic, {len(web + web_extra)} web")

    raw = await asyncio.to_thread(run_synthesis_agent, topic, combined[:15], "write_paper", "academic")
    return {"topic": topic, "sources_found": len(combined), "academic_sources": len(academic),
            "report": raw, "followups": [], "contested": "", "mode": "write_paper"}


async def run_improve_paper(uploaded_file) -> dict:
    from tools.pdf_reader import read_pdf
    from tools.semantic_scholar import search_papers, format_for_synthesis

    file_name = (getattr(uploaded_file, "filename", None)
                 or getattr(uploaded_file, "name", "paper.pdf"))

    pdf_content = await asyncio.to_thread(read_pdf, uploaded_file)
    if "error" in pdf_content:
        raise Exception(f"Could not read PDF: {pdf_content['error']}")

    search_query = pdf_content["content"][:400].replace('\n', ' ').strip()

    async def _academic():
        papers = await asyncio.to_thread(search_papers, search_query, 8, 0)
        return await asyncio.to_thread(format_for_synthesis, papers)

    async def _web():
        results = await asyncio.to_thread(run_search_agent, search_query[:200], "normal")
        return [r for r in results if "error" not in r and r.get("content", "").strip()]

    academic_sources, web_clean = await asyncio.gather(_academic(), _web())

    combined = [{"title": f"PAPER BEING REVIEWED: {file_name}", "url": "Uploaded paper",
                  "content": pdf_content["content"][:3500]}] + academic_sources[:6] + web_clean[:4]

    raw = await asyncio.to_thread(run_synthesis_agent, f"Review and improve: {file_name}",
                                   combined, "improve_paper", "academic")
    return {"topic": f"Paper Review: {file_name[:40]}",
            "sources_found": len(academic_sources) + len(web_clean),
            "report": raw, "followups": [], "contested": "", "mode": "improve_paper"}


# ── Sync wrappers for tests ────────────────────────────────────────────────


run_research_sync = lambda t, d="normal", tn="default": asyncio.run(run_research(t, d, tn))
run_url_research_sync = lambda u, tn="default": asyncio.run(run_url_research(u, tn))
run_pdf_research_sync = lambda f, tn="default": asyncio.run(run_pdf_research(f, tn))
run_comparison_research_sync = lambda a, b, d="normal": asyncio.run(run_comparison_research(a, b, d))
run_write_paper_sync = lambda t, d="normal": asyncio.run(run_write_paper(t, d))
run_improve_paper_sync = lambda f: asyncio.run(run_improve_paper(f))
run_analyze_sync = lambda tn="default", u="", fl=None: asyncio.run(run_analyze(tn, u, fl))
