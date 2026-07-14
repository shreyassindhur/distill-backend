import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_web(query: str, max_results: int = 2) -> list:
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            include_raw_content=False
        )
        results = []
        for r in response.get("results", []):
            content = r.get("content", "").strip()
            if content:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": content[:500]
                })
        return results
    except Exception as e:
        return [{"error": str(e)}]