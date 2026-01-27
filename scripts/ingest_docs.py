#!/usr/bin/env python3
"""
MindFu Documentation Ingestion Script

Crawls a documentation site and ingests all pages into the RAG system.

Usage:
    python scripts/ingest_docs.py https://docs.dev.sync.global/ --source-name "Splice Docs"
    python scripts/ingest_docs.py https://example.com/docs --dry-run
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


def get_page_content(url: str) -> tuple[str, list[str]]:
    """Fetch a page and extract its content and links."""
    response = httpx.get(url, follow_redirects=True, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # Try to find the main content area (common patterns)
    content_area = (
        soup.find("div", class_="document") or  # Sphinx
        soup.find("div", class_="md-content") or  # MkDocs Material
        soup.find("article") or  # Generic
        soup.find("main") or  # Generic
        soup.find("div", class_="content") or  # Generic
        soup.body
    )

    # Extract text
    if content_area:
        # Get the title
        title = ""
        h1 = content_area.find("h1")
        if h1:
            title = h1.get_text(strip=True)

        # Get the text content
        text = content_area.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)

        content = f"# {title}\n\n{text}" if title else text
    else:
        content = soup.get_text(separator="\n", strip=True)

    # Extract links
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Skip anchors, external links, and non-html
        if href.startswith("#") or href.startswith("mailto:"):
            continue
        full_url = urljoin(url, href)
        links.append(full_url)

    return content, links


def crawl_site(base_url: str, max_pages: int = 0, exclude_patterns: list[str] = None, cache_file: str = None, delay: float = 0.3, resume: bool = False) -> dict[str, str]:
    """Crawl a documentation site starting from base_url."""
    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

    visited = set()
    to_visit = [base_url]
    pages = {}

    # Resume from cache if requested
    if resume and cache_file and Path(cache_file).exists():
        pages = json.loads(Path(cache_file).read_text())
        visited = set(pages.keys())
        print(f"Resuming crawl: loaded {len(pages)} pages from cache")
        # Re-add links from cached pages to find new ones
        for url, content in pages.items():
            to_visit.append(url)

    print(f"Starting crawl from: {base_url}")
    print(f"Base domain: {base_domain}")

    save_every = 100  # Save cache every N pages

    while to_visit and (max_pages == 0 or len(visited) < max_pages):
        url = to_visit.pop(0)

        # Normalize URL (remove fragments)
        url = url.split("#")[0]

        # Skip if already visited
        if url in visited:
            continue

        # Only crawl pages from the same domain
        if not url.startswith(base_domain):
            continue

        # Skip non-documentation URLs
        skip_patterns = [
            "/_static/", "/_sources/", "/_images/",
            ".pdf", ".zip", ".tar", ".gz",
            "/search.html", "/genindex.html",
        ]
        if exclude_patterns:
            skip_patterns.extend(exclude_patterns)
        if any(pattern in url for pattern in skip_patterns):
            continue

        visited.add(url)

        try:
            print(f"[{len(visited)}/{max_pages}] Crawling: {url}")
            content, links = get_page_content(url)

            # Only store pages with meaningful content
            if len(content) > 100:
                pages[url] = content

            # Add new links to visit
            for link in links:
                if link not in visited and link.startswith(base_domain):
                    to_visit.append(link)

            # Be polite
            if delay > 0:
                time.sleep(delay)

            # Incremental cache save
            if cache_file and len(pages) % save_every == 0:
                save_cache(pages, cache_file)
                print(f"  [Cache saved: {len(pages)} pages]")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Final save
    if cache_file and pages:
        save_cache(pages, cache_file)

    print(f"\nCrawl complete: {len(pages)} pages collected")
    return pages


def save_cache(pages: dict[str, str], cache_file: str):
    """Save crawled pages to cache file."""
    Path(cache_file).write_text(json.dumps(pages, indent=2, ensure_ascii=False))
    print(f"Saved {len(pages)} pages to {cache_file}")


def load_cache(cache_file: str) -> dict[str, str]:
    """Load crawled pages from cache file."""
    pages = json.loads(Path(cache_file).read_text())
    print(f"Loaded {len(pages)} pages from {cache_file}")
    return pages


def ingest_to_rag(
    pages: dict[str, str],
    rag_url: str,
    collection: str = None,
    source_name: str = None,
):
    """Upload pages to the RAG service."""
    print(f"\nIngesting {len(pages)} pages to {rag_url}")

    success = 0
    updated = 0
    skipped = 0
    failed = 0

    with httpx.Client(timeout=60) as client:
        for url, content in pages.items():
            # Generate a stable ID from the URL
            doc_id = hashlib.md5(url.encode()).hexdigest()

            # Prepare metadata
            metadata = {
                "source": url,
                "source_name": source_name or urlparse(url).netloc,
                "type": "documentation",
            }

            # Upload to RAG
            payload = {
                "content": content,
                "metadata": metadata,
            }
            if collection:
                payload["collection"] = collection

            try:
                response = client.post(
                    f"{rag_url}/v1/documents",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                action = result.get("action", "created")
                if action == "skipped":
                    skipped += 1
                    print(f"  [SKIP] {url[:70]}...")
                elif action == "updated":
                    updated += 1
                    print(f"  [UPDATE] {url[:70]}...")
                else:
                    success += 1
                    print(f"  [NEW] {url[:70]}...")
            except Exception as e:
                failed += 1
                print(f"  [FAILED] {url}: {e}")

    print(f"\nIngestion complete: {success} new, {updated} updated, {skipped} skipped, {failed} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Crawl and ingest documentation into MindFu RAG"
    )
    parser.add_argument(
        "url",
        help="Base URL of the documentation site to crawl"
    )
    parser.add_argument(
        "--rag-url",
        default="http://localhost:8080",
        help="URL of the RAG service (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name (default: use service default)"
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help="Human-readable source name for metadata"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Maximum number of pages to crawl (0 = unlimited, default: 0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Crawl only, don't ingest"
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Cache file to save/load crawled pages (JSON)"
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Load pages from cache file instead of crawling"
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="URL patterns to exclude (can be used multiple times)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between requests in seconds (default: 0.3, use 0 for no delay)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume crawl from cache file (skips already crawled URLs)"
    )

    args = parser.parse_args()

    # Load from cache or crawl
    if args.from_cache:
        if not args.cache:
            print("Error: --from-cache requires --cache <file>")
            return
        pages = load_cache(args.cache)
    else:
        pages = crawl_site(args.url, max_pages=args.max_pages, exclude_patterns=args.exclude, cache_file=args.cache, delay=args.delay, resume=args.resume)

    if args.dry_run:
        print("\nDry run - pages that would be ingested:")
        for url in sorted(pages.keys()):
            print(f"  {url}")
        return

    # Ingest to RAG
    if pages:
        ingest_to_rag(
            pages=pages,
            rag_url=args.rag_url,
            collection=args.collection,
            source_name=args.source_name,
        )


if __name__ == "__main__":
    main()
