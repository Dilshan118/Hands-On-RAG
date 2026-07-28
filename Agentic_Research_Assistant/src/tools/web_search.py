import re
import urllib.parse
import urllib.request
from typing import List, Dict, Any

def _fallback_http_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Direct HTTP fallback scraper for DuckDuckGo HTML if library API encounters issues."""
    results = []
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )
        html = urllib.request.urlopen(req, timeout=5).read().decode("utf-8")
        
        # Simple regex extraction for result titles and links
        links = re.findall(r'<a class="result__url" href="([^"]+)".*?>\s*(.*?)\s*</a>', html)
        snippets = re.findall(r'<a class="result__snippet".*?>(.*?)</a>', html, re.DOTALL)

        for i in range(min(max_results, len(links))):
            href, raw_title = links[i]
            # Clean HTML tags
            clean_title = re.sub(r'<[^>]+>', '', raw_title).strip()
            clean_snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else "Live web result snippet."
            
            # Clean URL redirection
            if "uddg=" in href:
                href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
                
            results.append({
                "query": query,
                "title": clean_title if clean_title else "Web Source",
                "url": href,
                "snippet": clean_snippet
            })
    except Exception:
        pass
    return results

def search_web(queries: List[str], max_results_per_query: int = 3) -> List[Dict[str, Any]]:
    """
    Executes live web search using DDGS library with fallback to direct HTTP scraping.
    """
    all_results = []
    seen_urls = set()
    
    # Try primary ddgs library
    try:
        from ddgs import DDGS
        ddgs_client = DDGS()
        for query in queries:
            try:
                raw_res = list(ddgs_client.text(query, max_results=max_results_per_query))
                for r in raw_res:
                    url = r.get("href", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append({
                            "query": query,
                            "title": r.get("title", ""),
                            "url": url,
                            "snippet": r.get("body", "")
                        })
            except Exception:
                # Direct HTTP fallback for query
                fb_res = _fallback_http_search(query, max_results=max_results_per_query)
                for r in fb_res:
                    if r["url"] not in seen_urls:
                        seen_urls.add(r["url"])
                        all_results.append(r)
    except Exception:
        # Fallback to direct HTTP scraping if library fails
        for query in queries:
            fb_res = _fallback_http_search(query, max_results=max_results_per_query)
            for r in fb_res:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
                    
    return all_results
