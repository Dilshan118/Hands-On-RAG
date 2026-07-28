"""
web_search.py — Concurrent Multi-Threaded Web Search Engine
=============================================================

📚 BEGINNER EXPLANATION — What Does This File Do?
    This is the system's connection to the live internet. When the Research Agent
    needs to find current information about a topic, it calls the `search_web()`
    function in this file with a list of search queries.

    Instead of searching one query at a time (which is slow), this file searches
    ALL queries SIMULTANEOUSLY using Python threads. This is called
    "concurrent execution" or "parallel processing."

🏗️ DESIGN PATTERN — Thread Pool Pattern (ThreadPoolExecutor):
    
    📚 THE PROBLEM:
    Imagine you have 3 search queries. Searching them one-by-one:
        Query 1: 1.5 seconds
        Query 2: 1.3 seconds  
        Query 3: 1.2 seconds
        TOTAL:   4.0 seconds (sequential)
    
    📚 THE SOLUTION (ThreadPoolExecutor):
    Run all 3 queries at the same time in separate threads:
        Query 1: ─────1.5s─────┐
        Query 2: ────1.3s────┐ │
        Query 3: ───1.2s───┐ │ │
        TOTAL:   ──1.5s────┘ ┘ ┘  (parallel — only as slow as the LONGEST query)
    
    This gives us a ~3x speedup (from ~4.0s to ~1.5s).

    📚 WHY THREADS (not processes)?
    - Web searches are I/O-BOUND (waiting for network responses)
    - Python's GIL (Global Interpreter Lock) doesn't block I/O operations
    - Threads are lightweight and perfect for concurrent network requests
    - Processes would be overkill (they're for CPU-bound work)

🛡️ DUAL-LAYER FAULT TOLERANCE:
    1. Primary: Uses the `ddgs` library (DuckDuckGo Search) for clean API-based search
    2. Fallback: If `ddgs` fails, falls back to direct HTTP scraping of DuckDuckGo's
       HTML lite page. This ensures searches ALWAYS return results even if the
       ddgs library breaks due to API changes.

📁 ARCHITECTURE ROLE:
    LAYER 4 (Retrieval & Tool Concurrency) — Called by the Research node in graph.py.
"""

import re
import logging
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Setup module-level logger
logger = logging.getLogger(__name__)


def _single_query_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Executes a single search query using DuckDuckGo with HTTP scraper fallback.

    📚 DUAL-LAYER SEARCH STRATEGY:
        Layer 1 (Primary): Use the `ddgs` Python library for clean, structured results
        Layer 2 (Fallback): If ddgs fails, scrape DuckDuckGo's HTML lite page directly

    📚 WHY DUCKDUCKGO (not Google)?
        - DuckDuckGo doesn't require API keys or authentication
        - No rate limits for moderate usage
        - Privacy-friendly (no tracking)
        - The `ddgs` library provides a clean Python interface
        - Google Custom Search API costs money and has strict quotas

    📚 BEGINNER NOTE — What is a "private" function?
        The underscore prefix (_single_query_search) is a Python convention
        indicating this function is INTERNAL to this module. External code
        should call `search_web()` instead, which handles batching and dedup.

    Args:
        query: The search query string to execute.
        max_results: Maximum number of results to return per query (default: 3).

    Returns:
        List of dicts, each containing: query, title, url, snippet.
    """
    results = []

    # ──────────────────────────────────────────────────────────
    # STEP 1: Clean the query string
    #
    # Remove all special characters except word chars, spaces, and hyphens.
    # Search engines work best with plain keyword strings.
    # Example: "What's Shor's algorithm?" → "Whats Shors algorithm"
    # ──────────────────────────────────────────────────────────
    clean_q = re.sub(r'[^\w\s-]', '', query).strip()

    # ──────────────────────────────────────────────────────────
    # PRIMARY SEARCH: Using the ddgs library
    #
    # The ddgs library wraps DuckDuckGo's API and returns structured
    # results with title, href (URL), and body (snippet).
    #
    # 📚 WHY try/except?
    # The ddgs library can fail for many reasons:
    # - Network timeout
    # - DuckDuckGo changed their API format
    # - Rate limiting on rapid queries
    # - Library version incompatibility
    # We catch ALL exceptions to ensure the fallback runs.
    # ──────────────────────────────────────────────────────────
    try:
        from ddgs import DDGS
        ddgs_client = DDGS()
        raw_res = list(ddgs_client.text(clean_q, max_results=max_results))
        for r in raw_res:
            url = r.get("href", "")
            if url:
                results.append({
                    "query": query,
                    "title": r.get("title", "Web Source"),
                    "url": url,
                    "snippet": r.get("body", "")
                })
        logger.debug(f"DDGS returned {len(results)} results for query: '{clean_q}'")
    except Exception as e:
        logger.debug(f"DDGS primary search failed for '{clean_q}': {e}")

    # ──────────────────────────────────────────────────────────
    # FALLBACK SEARCH: Direct HTTP scraping of DuckDuckGo HTML
    #
    # 📚 HOW THE FALLBACK WORKS:
    # 1. Hit DuckDuckGo's "HTML lite" endpoint (designed for simple browsers)
    # 2. Parse the HTML response using regex to extract links and snippets
    # 3. Extract the actual URLs (DuckDuckGo wraps them in redirect links)
    #
    # 📚 WHY A CUSTOM USER-AGENT?
    # Some websites block requests without a valid User-Agent header.
    # We use a standard browser User-Agent to avoid being blocked.
    #
    # 📚 WHY "uddg=" PARSING?
    # DuckDuckGo's HTML lite page wraps result URLs in redirect links like:
    #   /l/?uddg=https%3A%2F%2Fexample.com&rut=...
    # We extract the actual URL by splitting on "uddg=" and URL-decoding it.
    # ──────────────────────────────────────────────────────────
    if not results:
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(clean_q)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            )
            html = urllib.request.urlopen(req, timeout=4).read().decode("utf-8")

            # Extract result URLs and titles using regex pattern matching
            links = re.findall(r'<a class="result__url" href="([^"]+)".*?>\s*(.*?)\s*</a>', html)
            # Extract result snippets
            snippets = re.findall(r'<a class="result__snippet".*?>(.*?)</a>', html, re.DOTALL)

            for i in range(min(max_results, len(links))):
                href, raw_title = links[i]
                # Strip HTML tags from title and snippet
                clean_title = re.sub(r'<[^>]+>', '', raw_title).strip()
                clean_snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else "Live web source."

                # Extract actual URL from DuckDuckGo redirect wrapper
                if "uddg=" in href:
                    href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])

                results.append({
                    "query": query,
                    "title": clean_title if clean_title else "Web Source",
                    "url": href,
                    "snippet": clean_snippet
                })

            logger.debug(f"HTTP fallback returned {len(results)} results for query: '{clean_q}'")
        except Exception as e:
            logger.warning(f"HTTP fallback search also failed for '{clean_q}': {e}")

    return results


def search_web(queries: List[str], max_results_per_query: int = 3) -> List[Dict[str, Any]]:
    """
    Executes concurrent multi-threaded web searches for a list of queries.

    📚 HOW CONCURRENT EXECUTION WORKS:
        1. We create a ThreadPoolExecutor with up to 5 worker threads
        2. Each query is submitted as a separate task to the thread pool
        3. Threads execute simultaneously (true I/O parallelism)
        4. as_completed() yields results as each thread finishes (not in order)
        5. We collect and deduplicate results by URL

    📚 THE DEDUPLICATION STRATEGY:
        Different queries may return the same URLs (e.g., "quantum computing RSA"
        and "post quantum cryptography" might both return a NIST page).
        We use a `seen_urls` set to ensure each URL appears only ONCE
        in the final results. This prevents the Writer from citing
        the same source multiple times with different IDs.

    📚 PERFORMANCE MATH:
        Sequential: 3 queries × ~1.5s each = ~4.5s total
        Parallel:   3 queries in ~1.5s (bottleneck = slowest query) = ~1.5s
        Speedup:    ~3x faster

    📚 BEGINNER NOTE — Why min(5, len(queries))?
        We don't want to create more threads than we have queries.
        If we have 2 queries, creating 5 threads wastes resources.
        min(5, 2) = 2 threads — exactly what we need.
        The cap of 5 prevents creating too many threads for large query lists.

    Args:
        queries: List of search query strings to execute concurrently.
        max_results_per_query: Maximum results per individual query (default: 3).

    Returns:
        List of deduplicated result dicts with: query, title, url, snippet.
    """
    all_results = []
    seen_urls = set()  # Set for O(1) duplicate URL detection

    if not queries:
        return all_results

    logger.info(f"Starting concurrent web search for {len(queries)} queries...")

    # ──────────────────────────────────────────────────────────
    # Create thread pool and submit all queries concurrently
    #
    # 📚 BEGINNER NOTE — Context Manager ("with" statement):
    # The "with" statement ensures the ThreadPoolExecutor is properly
    # cleaned up (all threads joined) when we're done, even if an
    # error occurs. This prevents orphan threads from leaking.
    #
    # 📚 INTERMEDIATE NOTE — executor.submit():
    # submit() doesn't WAIT for the function to finish. It returns
    # a Future object immediately. The actual work runs in a background
    # thread. We collect all Futures first, then iterate over them
    # with as_completed() to get results as they become available.
    # ──────────────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=min(5, len(queries))) as executor:
        # Submit all queries as concurrent tasks
        future_to_query = {
            executor.submit(_single_query_search, q, max_results_per_query): q
            for q in queries
        }

        # Collect results as threads complete (order is non-deterministic)
        for future in as_completed(future_to_query):
            try:
                res_list = future.result()  # Get the return value of _single_query_search
                for r in res_list:
                    url = r.get("url", "")
                    # Deduplicate: only add if we haven't seen this URL before
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)
            except Exception as e:
                query = future_to_query[future]
                logger.warning(f"Thread for query '{query}' raised an exception: {e}")
                continue

    logger.info(f"Web search complete: {len(all_results)} unique results across {len(queries)} queries.")
    return all_results
