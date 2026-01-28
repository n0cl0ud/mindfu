#!/usr/bin/env python3
"""
MindFu Documentation Ingestion Script

Crawls a documentation site or ingests local files into the RAG system.

Usage:
    # Web crawl
    python scripts/ingest_docs.py https://docs.dev.sync.global/ --source-name "Splice Docs"

    # Local directory (markdown files)
    python scripts/ingest_docs.py --from-dir ./docs --source-name "My Docs"

    # Git repo
    python scripts/ingest_docs.py --from-git https://github.com/org/repo --git-path docs --source-name "Repo Docs"
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


def read_local_files(directory: str, extensions: list[str] = None) -> dict[str, str]:
    """Read all documentation files from a local directory."""
    if extensions is None:
        extensions = [".md", ".mdx", ".rst", ".txt"]

    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {directory}")

    pages = {}
    for ext in extensions:
        for file_path in dir_path.rglob(f"*{ext}"):
            # Skip hidden files and directories
            if any(part.startswith(".") for part in file_path.parts):
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                # Use relative path as the "URL"
                rel_path = file_path.relative_to(dir_path)
                source_key = f"file://{rel_path}"

                # Only store files with meaningful content
                if len(content.strip()) > 50:
                    pages[source_key] = content
                    print(f"  Read: {rel_path}")
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")

    print(f"\nLoaded {len(pages)} files from {directory}")
    return pages


def clone_git_repo(repo_url: str, git_path: str = None, branch: str = None) -> tuple[str, dict[str, str]]:
    """Clone a git repo (sparse if git_path specified) and read docs."""
    # Create temp directory
    tmp_dir = tempfile.mkdtemp(prefix="mindfu-git-")

    try:
        print(f"Cloning {repo_url}...")

        if git_path:
            # Sparse checkout - only get the specified path
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo_url, tmp_dir],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", tmp_dir, "sparse-checkout", "set", git_path],
                check=True,
                capture_output=True,
            )
            docs_dir = Path(tmp_dir) / git_path
        else:
            # Full clone (shallow)
            cmd = ["git", "clone", "--depth", "1", repo_url, tmp_dir]
            if branch:
                cmd.extend(["--branch", branch])
            subprocess.run(cmd, check=True, capture_output=True)
            docs_dir = Path(tmp_dir)

        print(f"Reading files from {docs_dir}...")
        pages = read_local_files(str(docs_dir))

        # Update source keys to include repo URL
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        updated_pages = {}
        for key, content in pages.items():
            # Replace file:// with github URL
            rel_path = key.replace("file://", "")
            if "github.com" in repo_url:
                # Convert to GitHub raw URL format
                github_base = repo_url.replace(".git", "").rstrip("/")
                branch_name = branch or "main"
                if git_path:
                    new_key = f"{github_base}/blob/{branch_name}/{git_path}/{rel_path}"
                else:
                    new_key = f"{github_base}/blob/{branch_name}/{rel_path}"
            else:
                new_key = f"{repo_url}:{rel_path}"
            updated_pages[new_key] = content

        return tmp_dir, updated_pages

    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Git clone failed: {e.stderr.decode() if e.stderr else str(e)}")


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
    cached_links = {}  # Store links separately for resume

    # Resume from cache if requested
    if resume and cache_file and Path(cache_file).exists():
        raw_cache = json.loads(Path(cache_file).read_text())

        # Handle both old format (str) and new format (dict with content/links)
        for url, data in raw_cache.items():
            if isinstance(data, dict):
                # New format: {"content": "...", "links": [...]}
                pages[url] = data.get("content", "")
                cached_links[url] = data.get("links", [])
            else:
                # Old format: just content string
                pages[url] = data
                cached_links[url] = []

        visited = set(pages.keys())
        print(f"Resuming crawl: loaded {len(pages)} pages from cache")

        # Extract unvisited links from cached pages
        unvisited_links = set()
        for url, links in cached_links.items():
            for link in links:
                link = link.split("#")[0]  # Remove fragments
                if link not in visited and link.startswith(base_domain):
                    unvisited_links.add(link)

        to_visit = list(unvisited_links)
        print(f"Found {len(to_visit)} unvisited links from cached pages")

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
                cached_links[url] = links  # Store links for resume

            # Add new links to visit
            for link in links:
                if link not in visited and link.startswith(base_domain):
                    to_visit.append(link)

            # Be polite
            if delay > 0:
                time.sleep(delay)

            # Incremental cache save
            if cache_file and len(pages) % save_every == 0:
                save_cache(pages, cached_links, cache_file)
                print(f"  [Cache saved: {len(pages)} pages]")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Final save
    if cache_file and pages:
        save_cache(pages, cached_links, cache_file)

    print(f"\nCrawl complete: {len(pages)} pages collected")
    return pages


def save_cache(pages: dict[str, str], cached_links: dict[str, list], cache_file: str):
    """Save crawled pages to cache file (new format with links)."""
    cache_data = {}
    for url, content in pages.items():
        cache_data[url] = {
            "content": content,
            "links": cached_links.get(url, [])
        }
    Path(cache_file).write_text(json.dumps(cache_data, indent=2, ensure_ascii=False))
    print(f"Saved {len(pages)} pages to {cache_file}")


def load_cache(cache_file: str) -> dict[str, str]:
    """Load crawled pages from cache file (handles both old and new format)."""
    raw_cache = json.loads(Path(cache_file).read_text())
    pages = {}
    for url, data in raw_cache.items():
        if isinstance(data, dict):
            # New format
            pages[url] = data.get("content", "")
        else:
            # Old format
            pages[url] = data
    print(f"Loaded {len(pages)} pages from {cache_file}")
    return pages


def ingest_to_rag(
    pages: dict[str, str],
    rag_url: str,
    collection: str = None,
    source_name: str = None,
    batch_size: int = 50,
):
    """Upload pages to the RAG service in batches."""
    print(f"\nIngesting {len(pages)} pages to {rag_url} (batch size: {batch_size})")

    success = 0
    updated = 0
    skipped = 0
    failed = 0

    # Convert to list for batching
    items = list(pages.items())

    with httpx.Client(timeout=300) as client:  # Longer timeout for batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(items) + batch_size - 1) // batch_size

            # Prepare batch payload
            documents = []
            for url, content in batch:
                metadata = {
                    "source": url,
                    "source_name": source_name or urlparse(url).netloc,
                    "type": "documentation",
                }
                doc = {
                    "content": content,
                    "metadata": metadata,
                    "chunk": True,
                }
                if collection:
                    doc["collection"] = collection
                documents.append(doc)

            try:
                print(f"  [Batch {batch_num}/{total_batches}] Sending {len(documents)} documents...")
                response = client.post(
                    f"{rag_url}/v1/documents/batch",
                    json=documents,
                )
                response.raise_for_status()
                result = response.json()

                success += result.get("created", 0)
                updated += result.get("updated", 0)
                skipped += result.get("skipped", 0)

                print(f"  [Batch {batch_num}/{total_batches}] Done: {result.get('created', 0)} new, {result.get('updated', 0)} updated, {result.get('skipped', 0)} skipped")
            except Exception as e:
                failed += len(batch)
                print(f"  [Batch {batch_num}/{total_batches}] FAILED: {e}")

    print(f"\nIngestion complete: {success} new, {updated} updated, {skipped} skipped, {failed} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Crawl and ingest documentation into MindFu RAG"
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="Base URL of the documentation site to crawl"
    )
    parser.add_argument(
        "--from-dir",
        default=None,
        help="Ingest from local directory instead of crawling"
    )
    parser.add_argument(
        "--from-git",
        default=None,
        help="Clone and ingest from git repository URL"
    )
    parser.add_argument(
        "--git-path",
        default=None,
        help="Subdirectory in git repo to ingest (sparse checkout)"
    )
    parser.add_argument(
        "--git-branch",
        default=None,
        help="Git branch to clone (default: main)"
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of documents per batch for ingestion (default: 50)"
    )

    args = parser.parse_args()

    tmp_dir = None  # For git cleanup

    # Determine ingestion mode
    if args.from_cache:
        if not args.cache:
            print("Error: --from-cache requires --cache <file>")
            return
        pages = load_cache(args.cache)

    elif args.from_dir:
        print(f"Reading local directory: {args.from_dir}")
        pages = read_local_files(args.from_dir)

    elif args.from_git:
        print(f"Cloning git repository: {args.from_git}")
        tmp_dir, pages = clone_git_repo(
            args.from_git,
            git_path=args.git_path,
            branch=args.git_branch,
        )

    elif args.url:
        pages = crawl_site(args.url, max_pages=args.max_pages, exclude_patterns=args.exclude, cache_file=args.cache, delay=args.delay, resume=args.resume)

    else:
        print("Error: Must provide URL, --from-dir, --from-git, or --from-cache")
        parser.print_help()
        return

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
            batch_size=args.batch_size,
        )

    # Cleanup temp directory if used
    if tmp_dir:
        print(f"\nCleaning up temp directory: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
