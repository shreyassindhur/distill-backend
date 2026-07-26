import asyncio
import re

from agents.base import Agent, ResearchContext
from agents.search_agent import run_search_agent
from agents.synthesis_agent import run_synthesis_agent


# ── Utility functions (shared, not agents) ─────────────────────────────────


def validate_citations(report: str, source_urls: set) -> str:
    """Remove citations whose URLs don't appear in the actual source list."""
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


# ── Agents ─────────────────────────────────────────────────────────────────


class SearchAgent(Agent):
    """Performs web search for a given query."""
    def __init__(self, query: str | None = None, depth: str = "normal"):
        super().__init__("SearchAgent")
        self._query = query
        self._depth = depth

    async def run(self, ctx: ResearchContext) -> ResearchContext:
        query = self._query or ctx.query
        print(f"[SearchAgent] depth={self._depth}: {query[:80]}")
        results = await asyncio.to_thread(run_search_agent, query, self._depth)
        clean = [r for r in results if "error" not in r and r.get("content", "").strip()]
        ctx.state["search_results"] = clean
        ctx.sources.extend(clean)
        ctx.source_urls.update(r.get("url", "") for r in clean if r.get("url"))
        return ctx


class AcademicSearchAgent(Agent):
    """Performs academic paper search via Semantic Scholar."""
    def __init__(self, query: str | None = None):
        super().__init__("AcademicSearchAgent")
        self._query = query

    async def run(self, ctx: ResearchContext) -> ResearchContext:
        from tools.semantic_scholar import search_multi_query, format_for_synthesis
        query = self._query or ctx.query
        print(f"[AcademicSearchAgent] searching: {query[:80]}")
        papers = await asyncio.to_thread(search_multi_query, query, 6)
        sources = await asyncio.to_thread(format_for_synthesis, papers)
        ctx.state["academic_sources"] = sources
        ctx.sources.extend(sources)
        ctx.source_urls.update(r.get("url", "") for r in sources if r.get("url"))
        return ctx


class ExtractAgent(Agent):
    """Extracts content from a URL or uploaded file."""
    def __init__(self, url: str = "", uploaded_file=None):
        super().__init__("ExtractAgent")
        self._url = url
        self._file = uploaded_file

    async def run(self, ctx: ResearchContext) -> ResearchContext:
        extracted = []
        if self._url and self._url.startswith("http"):
            from tools.url_reader import read_url
            print(f"[ExtractAgent] reading URL: {self._url[:80]}")
            content = await asyncio.to_thread(read_url, self._url)
            if "error" in content:
                raise Exception(f"Could not read URL: {content['error']}")
            extracted.append({
                "title": content.get("title", self._url)[:80],
                "url": self._url,
                "content": content["content"][:4000]
            })
            ctx.state["extract_title"] = content.get("title", "")
            ctx.state["search_query"] = content.get("title", "") or self._url

        if self._file:
            from tools.pdf_reader import read_pdf
            file_name = (
                getattr(self._file, "filename", None)
                or getattr(self._file, "name", "document.pdf")
            )
            print(f"[ExtractAgent] reading PDF: {file_name}")
            content = await asyncio.to_thread(read_pdf, self._file)
            if "error" in content:
                raise Exception(f"Could not read PDF: {content['error']}")
            extracted.append({
                "title": f"Uploaded PDF: {file_name}",
                "url": "Uploaded document",
                "content": content["content"][:4000]
            })
            if not ctx.state.get("search_query"):
                ctx.state["search_query"] = content["content"][:200]
                ctx.state["source_label"] = f"PDF: {file_name[:40]}"
            else:
                ctx.state["search_query"] = content.get("title", f"{file_name} {content['content'][:100]}")
            ctx.state["source_label"] = ctx.state.get("source_label") or self._url[:60]
            ctx.state["file_name"] = file_name

        ctx.state["extracted"] = extracted
        ctx.sources = extracted + ctx.sources  # primary source first
        ctx.source_urls.update(r.get("url", "") for r in extracted if r.get("url"))
        return ctx


class DraftAgent(Agent):
    """Synthesizes sources into a report using the LLM."""
    def __init__(self, topic_label: str | None = None):
        super().__init__("DraftAgent")
        self._topic_label = topic_label

    async def run(self, ctx: ResearchContext) -> ResearchContext:
        topic = self._topic_label or ctx.query
        print(f"[DraftAgent] synthesizing {len(ctx.sources)} sources")
        raw = await asyncio.to_thread(
            run_synthesis_agent, topic, ctx.sources, ctx.mode, ctx.tone
        )
        raw = validate_citations(raw, ctx.source_urls)
        report, followups, contested = parse_followups(raw)
        ctx.report = report
        ctx.followups = followups
        ctx.contested = contested
        ctx.state["raw_report"] = raw
        return ctx


class ReviewAgent(Agent):
    """Evaluates report quality and determines if re-search is needed."""
    def __init__(self, min_words: int = 600):
        super().__init__("ReviewAgent")
        self._min_words = min_words

    async def run(self, ctx: ResearchContext) -> ResearchContext:
        word_count = len(ctx.report.split())
        unclear_count = (
            ctx.report.lower().count("unclear")
            + ctx.report.lower().count("limited evidence")
        )
        needs_improvement = word_count < self._min_words or unclear_count > 3
        ctx.state["word_count"] = word_count
        ctx.state["unclear_count"] = unclear_count
        ctx.state["needs_improvement"] = needs_improvement
        if needs_improvement:
            print(f"[ReviewAgent] thin ({word_count}w, {unclear_count} unclear)")
        else:
            print(f"[ReviewAgent] quality OK ({word_count}w)")
        return ctx


class ResearchAgent(Agent):
    """Performs targeted follow-up searches for thin reports."""
    def __init__(self):
        super().__init__("ResearchAgent")

    async def run(self, ctx: ResearchContext) -> ResearchContext:
        if not ctx.state.get("needs_improvement"):
            return ctx
        if ctx.followups:
            # Parallel search per follow-up question
            print(f"[ResearchAgent] {len(ctx.followups)} follow-up queries in parallel")
            tasks = [
                asyncio.to_thread(run_search_agent, q, "quick")
                for q in ctx.followups
            ]
            all_results = await asyncio.gather(*tasks)
        else:
            # Generic fallback search
            print("[ResearchAgent] generic fallback search")
            results = await asyncio.to_thread(
                run_search_agent, f"{ctx.query} statistics data findings 2024 2025", "quick"
            )
            all_results = [results]

        new_sources = []
        for results in all_results:
            for r in results:
                if (
                    "error" not in r
                    and r.get("content", "").strip()
                    and r.get("url", "") not in ctx.source_urls
                ):
                    new_sources.append(r)
                    ctx.source_urls.add(r.get("url", ""))

        ctx.state["new_sources"] = new_sources
        if new_sources:
            ctx.sources.extend(new_sources[:4])
            print(f"[ResearchAgent] added {min(len(new_sources), 4)} new sources")
        return ctx


# ── Mode pipelines (agent DAGs) ────────────────────────────────────────────


async def run_research(topic: str, depth: str = "normal", tone: str = "default") -> dict:
    ctx = ResearchContext(query=topic, mode="topic", tone=tone)

    # Stage 1: Parallel web + academic search
    agents = [
        SearchAgent(depth=depth),
        AcademicSearchAgent(),
    ]
    await asyncio.gather(*[a.run(ctx) for a in agents])

    clean = [r for r in ctx.sources if "error" not in r and r.get("content", "").strip()]
    if len(clean) < 2:
        raise Exception("Couldn't find enough reliable sources. Try a more specific term.")
    ctx.sources = clean

    # Stage 2: Draft
    await DraftAgent().run(ctx)

    # Stage 3: Review + Research in parallel
    await ReviewAgent().run(ctx)
    await ResearchAgent().run(ctx)

    # Stage 4: Re-draft if new sources were found
    new_sources = ctx.state.get("new_sources", [])
    if new_sources:
        ctx.mode = "topic"
        await DraftAgent().run(ctx)

    return {
        "topic": topic,
        "sources_found": len(ctx.sources),
        "report": ctx.report,
        "followups": ctx.followups,
        "contested": ctx.contested,
        "mode": "topic"
    }


async def run_url_research(url: str, tone: str = "default") -> dict:
    ctx = ResearchContext(query=f"Article Analysis: {url}", mode="url", tone=tone)

    # Stage 1: Parallel extract + search
    extract = ExtractAgent(url=url)
    search = SearchAgent(query=url, depth="quick")
    await asyncio.gather(extract.run(ctx), search.run(ctx))

    combined = ctx.state.get("extracted", []) + [
        r for r in ctx.sources
        if "error" not in r and r.get("content", "").strip()
    ][:6]
    ctx.sources = combined
    ctx.source_urls = {r.get("url", "") for r in combined if r.get("url")}

    # Stage 2: Draft
    await DraftAgent(topic_label=f"Article Analysis: {url}").run(ctx)

    return {
        "topic": f"URL: {url[:60]}",
        "sources_found": len(ctx.sources),
        "report": ctx.report,
        "followups": ctx.followups,
        "contested": ctx.contested,
        "mode": "url"
    }


async def run_pdf_research(uploaded_file, tone: str = "default") -> dict:
    file_name = (
        getattr(uploaded_file, "filename", None)
        or getattr(uploaded_file, "name", "document.pdf")
    )
    ctx = ResearchContext(query=f"Analysis of: {file_name}", mode="pdf", tone=tone)

    # Stage 1a: Extract PDF content (search query depends on it)
    extract = ExtractAgent(uploaded_file=uploaded_file)
    await extract.run(ctx)
    extracted = ctx.state.get("extracted", [])

    # Stage 1b: Supporting search (now we know the content)
    search_query = ctx.state.get("search_query", file_name)
    search = SearchAgent(query=search_query[:200], depth="quick")
    await search.run(ctx)

    supporting = [
        r for r in ctx.sources if r.get("url") != "Uploaded document"
        if "error" not in r and r.get("content", "").strip()
    ][:6]
    ctx.sources = extracted + supporting
    ctx.source_urls = {r.get("url", "") for r in ctx.sources if r.get("url")}

    # Stage 2: Draft
    await DraftAgent().run(ctx)

    return {
        "topic": f"PDF: {file_name[:40]}",
        "sources_found": len(ctx.sources),
        "report": ctx.report,
        "followups": ctx.followups,
        "contested": ctx.contested,
        "mode": "pdf"
    }


async def run_analyze(tone: str = "default", url: str = "", uploaded_file=None) -> dict:
    ctx = ResearchContext(query="", mode="url", tone=tone)

    if not url and not uploaded_file:
        raise Exception("Provide a URL or upload a file to analyze.")

    # Stage 1: Parallel extraction + supporting search
    extract = ExtractAgent(url=url, uploaded_file=uploaded_file)
    search_query = url if url else "Uploaded document"
    web_search = SearchAgent(query=search_query[:200], depth="quick")
    await asyncio.gather(extract.run(ctx), web_search.run(ctx))

    extracted = ctx.state.get("extracted", [])
    supporting = [
        r for r in ctx.sources
        if "error" not in r and r.get("content", "").strip()
        and r.get("url", "") not in {e.get("url", "") for e in extracted}
    ][:6]
    ctx.sources = extracted + supporting
    ctx.source_urls = {r.get("url", "") for r in ctx.sources if r.get("url")}

    label = f"Analysis: {url[:60] if url else ctx.state.get('file_name', 'document')[:40]}"

    # Stage 2: Draft
    await DraftAgent(topic_label=label).run(ctx)

    return {
        "topic": label,
        "sources_found": len(ctx.sources),
        "report": ctx.report,
        "followups": ctx.followups,
        "contested": ctx.contested,
        "mode": "analyze"
    }


async def run_comparison_research(topic_a: str, topic_b: str, depth: str = "normal") -> dict:
    ctx = ResearchContext(query=f"{topic_a} vs {topic_b}", mode="comparison")

    # Stage 1: Parallel search for both topics
    search_a = SearchAgent(query=topic_a, depth=depth)
    search_b = SearchAgent(query=topic_b, depth=depth)
    await asyncio.gather(search_a.run(ctx), search_b.run(ctx))

    # Separate results by topic
    all_results = ctx.state.get("search_results", [])
    # Since both searches wrote to ctx.sources, we need to distinguish them
    # We'll re-run with separate contexts to keep sources clean
    ctx_a = ResearchContext(query=topic_a)
    ctx_b = ResearchContext(query=topic_b)
    await asyncio.gather(
        SearchAgent(query=topic_a, depth=depth).run(ctx_a),
        SearchAgent(query=topic_b, depth=depth).run(ctx_b),
    )

    clean_a = [r for r in ctx_a.sources if "error" not in r and r.get("content", "").strip()]
    clean_b = [r for r in ctx_b.sources if "error" not in r and r.get("content", "").strip()]
    if len(clean_a) < 1 or len(clean_b) < 1:
        raise Exception("Couldn't find enough sources for one or both topics.")

    for r in clean_a:
        r["title"] = f"[{topic_a}] {r.get('title', '')}"
    for r in clean_b:
        r["title"] = f"[{topic_b}] {r.get('title', '')}"

    ctx.sources = clean_a + clean_b
    ctx.source_urls = {r.get("url", "") for r in ctx.sources if r.get("url")}

    # Stage 2: Draft
    await DraftAgent().run(ctx)

    return {
        "topic": f"{topic_a} vs {topic_b}",
        "sources_found": len(ctx.sources),
        "report": ctx.report,
        "followups": ctx.followups,
        "contested": ctx.contested,
        "mode": "comparison"
    }


async def run_write_paper(topic: str, depth: str = "normal") -> dict:
    from tools.semantic_scholar import search_multi_query, format_for_synthesis

    ctx = ResearchContext(query=topic, mode="write_paper", tone="academic")
    print(f"[Distill] Writing IEEE paper on: {topic}")

    # Stage 1: Parallel academic + web search
    async def _academic():
        print("[Distill] Querying Semantic Scholar for academic sources...")
        papers = await asyncio.to_thread(search_multi_query, topic, 6)
        sources = await asyncio.to_thread(format_for_synthesis, papers)
        ctx.state["academic_sources"] = sources
        print(f"[Distill] Found {len(sources)} academic papers")
        return sources

    async def _web():
        print("[Distill] Fetching current web context...")
        results = await asyncio.to_thread(run_search_agent, topic, "normal")
        return [r for r in results if "error" not in r and r.get("content", "").strip()]

    async def _web_extra():
        results = await asyncio.to_thread(
            run_search_agent, f"{topic} research findings 2023 2024", "quick"
        )
        return [r for r in results if "error" not in r and r.get("content", "").strip()]

    academic_sources, web_clean, web_extra_clean = await asyncio.gather(
        _academic(), _web(), _web_extra()
    )

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

    ctx.sources = deduped[:15]
    ctx.source_urls = {r.get("url", "") for r in ctx.sources if r.get("url")}

    print(f"[Distill] Total sources for paper: {len(ctx.sources)} "
          f"({len(academic_sources)} academic, {len(web_clean + web_extra_clean)} web)")

    # Stage 2: Draft
    raw = await asyncio.to_thread(
        run_synthesis_agent, topic, ctx.sources, "write_paper", "academic"
    )

    return {
        "topic": topic,
        "sources_found": len(ctx.sources),
        "academic_sources": len(academic_sources),
        "report": raw,
        "followups": [],
        "contested": "",
        "mode": "write_paper"
    }


async def run_improve_paper(uploaded_file) -> dict:
    from tools.pdf_reader import read_pdf
    from tools.semantic_scholar import search_papers, format_for_synthesis
    from agents.synthesis_agent import run_synthesis_agent

    file_name = (
        getattr(uploaded_file, "filename", None)
        or getattr(uploaded_file, "name", "paper.pdf")
    )
    print(f"[Distill] Improving paper: {file_name}")

    pdf_content = await asyncio.to_thread(read_pdf, uploaded_file)
    if "error" in pdf_content:
        raise Exception(f"Could not read PDF: {pdf_content['error']}")

    paper_text = pdf_content["content"]
    search_query = paper_text[:400].replace('\n', ' ').strip()

    # Stage 1: Parallel academic + web search
    async def _academic():
        print("[Distill] Finding related academic sources...")
        papers = await asyncio.to_thread(search_papers, search_query, 8, 0)
        return await asyncio.to_thread(format_for_synthesis, papers)

    async def _web():
        results = await asyncio.to_thread(run_search_agent, search_query[:200], "normal")
        return [r for r in results if "error" not in r and r.get("content", "").strip()]

    academic_sources, web_clean = await asyncio.gather(_academic(), _web())

    combined = [{
        "title": f"PAPER BEING REVIEWED: {file_name}",
        "url": "Uploaded paper",
        "content": paper_text[:3500]
    }] + academic_sources[:6] + web_clean[:4]

    raw = await asyncio.to_thread(
        run_synthesis_agent,
        f"Review and improve: {file_name}",
        combined,
        "improve_paper",
        "academic"
    )

    return {
        "topic": f"Paper Review: {file_name[:40]}",
        "sources_found": len(academic_sources) + len(web_clean),
        "report": raw,
        "followups": [],
        "contested": "",
        "mode": "improve_paper"
    }


# ── Legacy sync wrappers (used by tests) ───────────────────────────────────


def run_research_sync(topic: str, depth: str = "normal", tone: str = "default") -> dict:
    return asyncio.run(run_research(topic, depth, tone))


def run_url_research_sync(url: str, tone: str = "default") -> dict:
    return asyncio.run(run_url_research(url, tone))


def run_pdf_research_sync(uploaded_file, tone: str = "default") -> dict:
    return asyncio.run(run_pdf_research(uploaded_file, tone))


def run_comparison_research_sync(topic_a: str, topic_b: str, depth: str = "normal") -> dict:
    return asyncio.run(run_comparison_research(topic_a, topic_b, depth))


def run_write_paper_sync(topic: str, depth: str = "normal") -> dict:
    return asyncio.run(run_write_paper(topic, depth))


def run_improve_paper_sync(uploaded_file) -> dict:
    return asyncio.run(run_improve_paper(uploaded_file))


def run_analyze_sync(tone: str = "default", url: str = "", uploaded_file=None) -> dict:
    return asyncio.run(run_analyze(tone, url, uploaded_file))
