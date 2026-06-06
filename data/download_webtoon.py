#!/usr/bin/env python3
"""Webtoon chapter downloader.

Downloads one chapter (episode) of a webtoon from webtoons.com and saves it as a
single PDF (named <title>_ep<N>.pdf) under data/files/ — ready to feed into the
comic-audio pipeline (frontend/model_server.py / pdf_reader.py).

⚠️  Respect copyright and WEBTOON's Terms of Service. Use this ONLY for personal,
    offline reading/testing of content you are allowed to access, and do not
    redistribute. It is polite (rate-limited, normal headers) and downloads only
    publicly/freely accessible episodes — it does NOT bypass logins, paywalls, or
    age gates. Downloads land in data/files/, which is gitignored.

Usage (from anywhere):
  python data/download_webtoon.py --title-no 95 --episode 1        # most reliable
  python data/download_webtoon.py --name "Tower of God" --chapter 1

Requires:  pip install requests beautifulsoup4 pymupdf
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://www.webtoons.com"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Downloads always go to data/files/ (this file lives in data/). Created on demand.
FILES_DIR = Path(__file__).resolve().parent / "files"

session = requests.Session()
session.headers.update({"User-Agent": UA, "Referer": BASE + "/"})


def search_title(name: str) -> tuple[int, str]:
    """Best-effort: resolve a webtoon name -> (title_no, canonical list URL)."""
    r = session.get(f"{BASE}/en/search", params={"keyword": name}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.select("a[href*='title_no=']"):
        href = a.get("href", "")
        m = re.search(r"title_no=(\d+)", href)
        if m:
            full = href if href.startswith("http") else BASE + href
            return int(m.group(1)), full
    raise SystemExit(f"Could not find a webtoon named {name!r}. Try --title-no "
                     f"(open the series on webtoons.com and copy title_no=… from the URL).")


def canonical_list_url(title_no: int) -> str:
    """Follow the redirect from episodeList?titleNo=N to the canonical list URL."""
    r = session.get(f"{BASE}/episodeList", params={"titleNo": title_no},
                    timeout=20, allow_redirects=True)
    r.raise_for_status()
    return r.url  # e.g. https://www.webtoons.com/en/fantasy/tower-of-god/list?title_no=95


def title_slug(list_url: str, title_no: int) -> str:
    """Derive a readable title slug (e.g. 'tower_of_god') from the list URL."""
    parts = [p for p in urlparse(list_url).path.split("/") if p]
    if len(parts) >= 2 and parts[-1] == "list":
        return parts[-2].replace("-", "_")
    return f"title{title_no}"  # fallback if the URL shape is unexpected


def find_episode_viewer(title_no: int, episode_no: int, list_url: str,
                        max_pages: int = 400, delay: float = 0.3) -> str:
    """Find the viewer URL for episode_no (construct first, page the list as fallback)."""
    parts = [p for p in urlparse(list_url).path.split("/") if p]
    constructed = None
    if len(parts) >= 4 and parts[-1] == "list":
        genre, slug = parts[-3], parts[-2]
        constructed = (f"{BASE}/{parts[0]}/{genre}/{slug}/ep/viewer"
                       f"?title_no={title_no}&episode_no={episode_no}")
        if fetch_panel_urls(constructed, quiet=True):
            return constructed

    for page in range(1, max_pages + 1):
        r = session.get(list_url, params={"title_no": title_no, "page": page}, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        found_any = False
        for a in soup.select("a[href*='episode_no=']"):
            href = a.get("href", "")
            m = re.search(r"episode_no=(\d+)", href)
            if not m:
                continue
            found_any = True
            if int(m.group(1)) == episode_no:
                return href if href.startswith("http") else BASE + href
        if not found_any:
            break
        time.sleep(delay)

    if constructed:
        return constructed
    raise SystemExit(f"Episode {episode_no} not found for title {title_no}.")


def fetch_panel_urls(viewer_url: str, quiet: bool = False) -> list[str]:
    """Extract the ordered panel image URLs from an episode viewer page."""
    try:
        r = session.get(viewer_url, timeout=20)
        r.raise_for_status()
    except requests.RequestException:
        if quiet:
            return []
        raise
    soup = BeautifulSoup(r.text, "html.parser")
    urls = []
    for img in soup.select("#_imageList img, img._images, div.viewer_img img"):
        u = img.get("data-url") or img.get("src")
        if u and ("pstatic.net" in u or "webtoon" in u):
            urls.append(u)
    seen, ordered = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def download_panels(urls: list[str], out_dir: Path, delay: float) -> list[Path]:
    """Download panel images to a (temporary) directory. Returns ordered paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, u in enumerate(urls, 1):
        # Referer is required — the image CDN blocks hotlinking otherwise.
        resp = session.get(u, headers={"Referer": BASE + "/"}, timeout=30)
        resp.raise_for_status()
        ext = ".png" if ".png" in u.lower() else ".jpg"
        p = out_dir / f"panel_{i:03d}{ext}"
        p.write_bytes(resp.content)
        paths.append(p)
        print(f"  panel {i}/{len(urls)}  ({len(resp.content)//1024} KB)", flush=True)
        time.sleep(delay)  # be polite
    return paths


def make_pdf(image_paths: list[Path], pdf_path: Path) -> None:
    """Combine downloaded panels into a single PDF (one panel per page)."""
    import fitz  # PyMuPDF

    doc = fitz.open()
    for p in image_paths:
        img = fitz.open(p)
        pdf_bytes = img.convert_to_pdf()
        img.close()
        imgpdf = fitz.open("pdf", pdf_bytes)
        doc.insert_pdf(imgpdf)
        imgpdf.close()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(pdf_path)
    doc.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download one webtoon chapter as a PDF into data/files/")
    ap.add_argument("--name", help="Webtoon title to search for (best-effort)")
    ap.add_argument("--title-no", type=int, help="webtoons.com title_no (most reliable)")
    ap.add_argument("--chapter", "--episode", dest="episode", type=int, required=True,
                    help="Chapter / episode number")
    ap.add_argument("--delay", type=float, default=0.5, help="Seconds between requests (politeness)")
    args = ap.parse_args(argv)

    if not args.name and not args.title_no:
        ap.error("provide --name or --title-no")

    # 2.1: ensure data/files/ exists
    FILES_DIR.mkdir(parents=True, exist_ok=True)

    if args.title_no:
        title_no = args.title_no
        list_url = canonical_list_url(title_no)
    else:
        title_no, list_url = search_title(args.name)
        print(f"Resolved {args.name!r} -> title_no={title_no}", flush=True)

    slug = title_slug(list_url, title_no)
    print(f"Series: {slug}  ({list_url})", flush=True)

    viewer = find_episode_viewer(title_no, args.episode, list_url, delay=args.delay)
    print(f"Episode viewer: {viewer}", flush=True)

    urls = fetch_panel_urls(viewer)
    if not urls:
        sys.exit("No panels found. The chapter may be paywalled/age-gated/login-only, "
                 "or the site layout changed. (This tool does not bypass access controls.)")
    print(f"Found {len(urls)} panels. Downloading…", flush=True)

    # 2.3: PDF only — panels go to a temp dir and are discarded after the PDF is built.
    pdf_path = FILES_DIR / f"{slug}_ep{args.episode}.pdf"
    with tempfile.TemporaryDirectory(prefix="webtoon_panels_") as tmp:
        panel_paths = download_panels(urls, Path(tmp), args.delay)
        make_pdf(panel_paths, pdf_path)

    print(f"PDF saved: {pdf_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
