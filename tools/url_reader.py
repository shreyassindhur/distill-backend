import trafilatura
import requests

def read_url(url: str) -> dict:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True
        )
        
        if not text or len(text.strip()) < 100:
            # fallback — try without precision mode
            text = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=False,
                no_fallback=True
            )

        if not text or len(text.strip()) < 100:
            return {"error": "Could not extract readable content from this URL. The site may be paywalled or JavaScript-rendered."}

        return {
            "url": url,
            "content": text[:4000],
            "success": True
        }
    except Exception as e:
        return {"error": str(e)}