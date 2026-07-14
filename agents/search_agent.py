import os
from groq import Groq
from tools.web_search import search_web
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_queries(topic: str, count: int = 5) -> list[str]:
    """Use Groq to generate diverse search queries for a topic."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Generate exactly {count} diverse search queries for researching the given topic. "
                        "Each query should target a different angle: overview, recent news, data/statistics, "
                        "expert opinion, and criticism or counterarguments. "
                        "Return ONLY the queries, one per line, no numbering, no extra text."
                    )
                },
                {"role": "user", "content": f"Topic: {topic}"}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()
        queries = [q.strip() for q in raw.split("\n") if q.strip()]
        return queries[:count]
    except Exception as e:
        print(f"[Search Agent] Query generation failed: {e}")
        return [topic]


def run_search_agent(topic: str, depth: str = "normal") -> list:
    """
    Search the web for a topic.

    depth="quick"  — 2 queries, 2 results each  → ~4 sources, fast
    depth="normal" — 4 queries, 3 results each  → ~12 sources, thorough
    """
    if depth == "quick":
        num_queries = 2
        results_per_query = 2
    else:
        num_queries = 4
        results_per_query = 3

    print(f"[Search Agent] depth={depth} → {num_queries} queries × {results_per_query} results")

    queries = generate_queries(topic, count=num_queries)
    print(f"[Search Agent] Queries: {queries}")

    search_results = []
    seen_urls = set()

    for query in queries:
        try:
            results = search_web(query, max_results=results_per_query)
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    search_results.append(r)
        except Exception as e:
            print(f"[Search Agent] Search failed for '{query}': {e}")
            continue

    print(f"[Search Agent] Total unique results: {len(search_results)}")
    return search_results