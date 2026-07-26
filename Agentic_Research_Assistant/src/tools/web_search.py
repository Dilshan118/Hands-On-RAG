from typing import List, Dict, Any
from duckduckgo_search import DDGS

def search_web(queries: List[str], max_results_per_query: int = 3) -> List[Dict[str, Any]]:
    """
    Executes live DuckDuckGo web searches for a list of queries.
    Returns deduplicated list of search result dictionaries containing title, href, and body.
    """
    all_results = []
    seen_urls = set()
    
    ddgs = DDGS()
    for query in queries:
        try:
            results = ddgs.text(query, max_results=max_results_per_query)
            for r in results:
                url = r.get("href", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        "query": query,
                        "title": r.get("title", ""),
                        "url": url,
                        "snippet": r.get("body", "")
                    })
        except Exception as e:
            all_results.append({
                "query": query,
                "title": "Search Error",
                "url": "",
                "snippet": f"Web search encountered an issue for '{query}': {str(e)}"
            })
            
    return all_results
