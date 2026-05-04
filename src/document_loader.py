"""
Document Loader -- Week 4 (RAG)

Loads documents from PDFs, plain text, web URLs, and the arXiv API into
a unified document representation: a list of dicts with `text`, `source`,
and `metadata` keys.

Functions:
  load_pdf(path)             - Extract text from a PDF (PyMuPDF preferred)
  load_text(path)            - Read a plain-text file
  load_directory(dir_path)   - Walk a directory and load supported files
  fetch_arxiv_papers(query)  - Query arXiv API and download N PDFs
  load_web_page(url)         - Fetch + clean a single web page
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path


def load_pdf(path: str, clean_pages: bool = True) -> Dict[str, Any]:
    """
    Extract text from a PDF using PyMuPDF (fitz). Falls back to pypdf.

    Args:
        path: Absolute or relative path to the PDF
        clean_pages: Strip headers/footers heuristically

    Returns:
        Dict with keys: text, source, metadata, num_pages, page_texts
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = []
        for page in doc:
            t = page.get_text()
            if clean_pages:
                # Strip very short header/footer lines (best-effort)
                lines = [ln for ln in t.split('\n') if len(ln.strip()) > 3]
                t = '\n'.join(lines)
            pages.append(t)
        meta = doc.metadata or {}
        doc.close()
        backend = "pymupdf"
    except ImportError:
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = [p.extract_text() or "" for p in reader.pages]
            meta = reader.metadata or {}
            backend = "pypdf"
        except ImportError:
            raise ImportError(
                "Install one of: pip install pymupdf  OR  pip install pypdf"
            )

    full_text = "\n\n".join(pages)
    print(f"  ✓ Loaded {os.path.basename(path)}: {len(pages)} pages, "
          f"{len(full_text):,} chars (via {backend})")

    return {
        "text": full_text,
        "source": path,
        "page_texts": pages,
        "num_pages": len(pages),
        "metadata": {
            "title":  str(meta.get("title", "") or ""),
            "author": str(meta.get("author", "") or ""),
            "backend": backend,
        },
    }


def load_text(path: str) -> Dict[str, Any]:
    """Load a plain-text file."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"  ✓ Loaded {os.path.basename(path)}: {len(text):,} chars")
    return {
        "text": text,
        "source": path,
        "metadata": {"type": "text"},
    }


def load_directory(
    dir_path: str,
    extensions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Walk a directory and load all supported files.

    Args:
        dir_path: Directory to scan
        extensions: File extensions to load (default: pdf, txt, md)

    Returns:
        List of document dicts
    """
    if extensions is None:
        extensions = [".pdf", ".txt", ".md"]
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    docs = []
    for root, _, files in os.walk(dir_path):
        for fn in sorted(files):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in extensions:
                continue
            full = os.path.join(root, fn)
            try:
                if ext == ".pdf":
                    docs.append(load_pdf(full))
                else:
                    docs.append(load_text(full))
            except Exception as e:
                print(f"  ⚠ Skipped {fn}: {e}")

    print(f"\n✓ Loaded {len(docs)} documents from {dir_path}")
    return docs


def fetch_arxiv_papers(
    query: str = "cat:cs.CL",
    max_results: int = 10,
    download_dir: str = "test_data/arxiv",
    download_pdfs: bool = True,
) -> List[Dict[str, Any]]:
    """
    Query the arXiv API and optionally download PDFs.

    Args:
        query: arXiv search query (e.g. "cat:cs.CL", "ti:RAG")
        max_results: Number of papers to fetch
        download_dir: Where to save PDFs
        download_pdfs: If False, only return metadata

    Returns:
        List of paper dicts with: id, title, summary, authors, pdf_url, pdf_path
    """
    try:
        import requests
        import xml.etree.ElementTree as ET
    except ImportError:
        raise ImportError("Install: pip install requests")

    os.makedirs(download_dir, exist_ok=True)

    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    print(f"Querying arXiv: '{query}' (max {max_results})...")
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    entries = root.findall("atom:entry", ns)

    papers = []
    for entry in entries:
        arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]

        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib["href"]
                break

        paper = {
            "id": arxiv_id,
            "title": title,
            "summary": summary,
            "authors": authors,
            "pdf_url": pdf_url,
            "pdf_path": None,
        }

        if download_pdfs and pdf_url:
            safe_name = arxiv_id.replace("/", "_") + ".pdf"
            pdf_path = os.path.join(download_dir, safe_name)
            if not os.path.exists(pdf_path):
                try:
                    pdf_resp = requests.get(pdf_url, timeout=60)
                    pdf_resp.raise_for_status()
                    with open(pdf_path, "wb") as f:
                        f.write(pdf_resp.content)
                    print(f"  ✓ {arxiv_id}: {title[:60]}...")
                    time.sleep(0.5)  # be polite to arXiv
                except Exception as e:
                    print(f"  ⚠ {arxiv_id}: download failed ({e})")
                    pdf_path = None
            else:
                print(f"  ↻ {arxiv_id}: cached")
            paper["pdf_path"] = pdf_path

        papers.append(paper)

    print(f"\n✓ Fetched {len(papers)} arXiv papers")
    return papers


def load_web_page(url: str) -> Dict[str, Any]:
    """
    Fetch a web page and extract clean text.

    Falls back gracefully if BeautifulSoup is unavailable.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install: pip install requests beautifulsoup4")

    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())

    print(f"  ✓ Fetched {url}: {len(text):,} chars")
    return {
        "text": text,
        "source": url,
        "metadata": {"type": "web", "title": (soup.title.string if soup.title else "")},
    }
