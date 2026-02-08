#!/usr/bin/env python3
"""Scrape SCOTUSblog OT2025 case pages for amicus curiae briefs."""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.scotusblog.com/case-files/terms/ot2025/"
CACHE_DIR = Path("cached_pages")
OUTPUT_FILE = "amici_data.json"
REQUEST_DELAY = 1  # seconds between requests

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "AmiciBriefCounter/1.0 (academic research tool)"
})


def fetch_page(url: str, cache_key: str | None = None) -> str:
    """Fetch a page, using disk cache if available."""
    if cache_key:
        cache_path = CACHE_DIR / f"{cache_key}.html"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

    time.sleep(REQUEST_DELAY)
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    html = resp.text

    if cache_key:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_path = CACHE_DIR / f"{cache_key}.html"
        cache_path.write_text(html, encoding="utf-8")

    return html


def get_case_list() -> list[dict]:
    """Parse the OT2025 index page to get all case URLs and metadata."""
    html = fetch_page(BASE_URL, cache_key="ot2025_index")
    soup = BeautifulSoup(html, "html.parser")

    cases = []
    seen_urls = set()

    # The page has an accordion div containing <h2> sitting headers followed
    # by <table class="caseindex"> elements.  Each table row has:
    #   - <a href="/cases/case-files/..."> with the case name
    #   - <span class="case-info-item"> containing "No. <a>XX-XXXX</a>"
    accordion = soup.find("div", id="accordion")
    if not accordion:
        # Fallback: search entire page
        accordion = soup

    current_category = "Unknown"

    for element in accordion.children:
        if not hasattr(element, "name") or element.name is None:
            continue

        # Track sitting category from <h2> headings
        if element.name == "h2":
            current_category = element.get_text(strip=True)
            continue

        # Process case rows inside tables
        for tr in element.find_all("tr"):
            # Find the case link (points to /cases/case-files/...)
            case_link = tr.find("a", href=re.compile(r"/cases/case-files/"))
            if not case_link:
                continue

            url = urljoin(BASE_URL, case_link["href"])
            if url in seen_urls:
                continue
            seen_urls.add(url)

            name = case_link.get_text(strip=True)
            if not name:
                continue

            # Extract docket number from "No. <a>XX-XXXX</a>" pattern
            docket = ""
            row_text = tr.get_text()
            docket_match = re.search(r"No\.\s*(\d{2,3}-\d+)", row_text)
            if docket_match:
                docket = docket_match.group(1)

            cases.append({
                "name": name,
                "docket_number": docket,
                "scotusblog_url": url,
                "category": current_category,
            })

    return cases


def classify_side_from_text(text: str) -> str | None:
    """Try to determine side from the brief description text.

    Returns a side string if a clear textual signal is found, or None
    if the text is ambiguous (caller should fall back to color).
    """
    t = text.lower()

    if re.search(r"in support of\s+(the\s+)?neither\s+party", t):
        return "neither"
    if re.search(r"in support of\s+(the\s+)?no\s+party", t):
        return "neither"
    if re.search(r"in support of\s+(the\s+)?(petitioner|appellant|plaintiff)", t):
        return "petitioner"
    if re.search(r"in support of\s+(the\s+)?(respondent|appellee|defendant)", t):
        return "respondent"
    if re.search(r"in support of\s+reversal", t):
        return "petitioner"
    if re.search(r"in support of\s+affirmance", t):
        return "respondent"
    if re.search(r"in support of\s+(the\s+)?judgment below", t):
        return "respondent"

    return None


# SCOTUSblog row background colors map to SCOTUS booklet cover colors.
# See: supremecourt.gov/casehand/USSC - Booklet-Format Specification Chart
#
#   #ffffd6  (cream)      → cert-stage amicus, no side distinction
#   #add6ad  (light green) → merits amicus supporting petitioner OR neither party
#   #32ad84  (dark green)  → merits amicus supporting respondent
#   #da6b6b  (light red)   → court-appointed amicus supporting judgment below
#   #DDD     (gray)        → U.S. Government (Solicitor General) amicus
#   #ffffff  (white)       → procedural / not-accepted / other
COLOR_TO_SIDE = {
    "#ffffd6": "petitioner",   # cert-stage; no side, but grouped with petitioner
    "#add6ad": "petitioner",   # light green = pet. side (or neither; text overrides)
    "#32ad84": "respondent",   # dark green = resp. side
    "#da6b6b": "respondent",   # court-appointed supporting judgment below
    "#ddd":    "unknown",      # government amicus — side unclear
    "#ffffff": "unknown",      # procedural / not accepted
}


def classify_side(text: str, row_bg_color: str) -> str:
    """Classify which side an amicus brief supports.

    Uses the row background color (from SCOTUS booklet cover colors) as
    the primary signal, with text-based overrides for "neither party"
    and other explicit statements.
    """
    # Text takes priority when it's explicit (especially for "neither party",
    # which shares light-green with petitioner-side briefs)
    text_side = classify_side_from_text(text)
    if text_side:
        return text_side

    # Fall back to color
    color = row_bg_color.strip().lower()
    return COLOR_TO_SIDE.get(color, "unknown")


def extract_filer_name(text: str) -> str:
    """Extract the filer name from a brief description."""
    # Common patterns:
    #   "Brief amicus curiae of [NAME] filed."
    #   "Brief amici curiae of [NAME] ... filed."
    #   "Brief of [NAME] as amicus curiae ..."
    #   "Brief of amicus curiae [NAME] ..."
    #   "Brief of amici curiae [NAME] ..."

    # Pattern: "Brief amicus curiae of [NAME]"
    m = re.search(
        r"[Bb]rief\s+(?:of\s+)?amici?\s+curiae?\s+of\s+(.+?)(?:\s+in\s+support|\s+filed|\s+urging|\.\s*$)",
        text,
    )
    if m:
        return m.group(1).strip().rstrip(".")

    # Pattern: "Brief of [NAME] as amicus curiae"
    m = re.search(
        r"[Bb]rief\s+of\s+(.+?)\s+as\s+amici?\s+curiae?",
        text,
    )
    if m:
        return m.group(1).strip().rstrip(".")

    # Pattern: "Brief of amicus curiae [NAME]"
    m = re.search(
        r"[Bb]rief\s+of\s+amici?\s+curiae?\s+(.+?)(?:\s+in\s+support|\s+filed|\s+urging|\.\s*$)",
        text,
    )
    if m:
        return m.group(1).strip().rstrip(".")

    # Fallback: just use the whole text, trimmed
    return text.strip()[:200]


def parse_briefs(html: str) -> list[dict]:
    """Parse amicus briefs from a SCOTUSblog case detail page."""
    soup = BeautifulSoup(html, "html.parser")
    briefs = []

    # SCOTUSblog case pages list filings in a <table> with columns:
    #   <td> Date | <td> Proceedings and Orders
    # Find the filings table by looking for a table whose header mentions
    # "Proceedings" or "Orders".
    filings_table = None
    for table in soup.find_all("table"):
        header_text = table.get_text(strip=True)[:200].lower()
        if "proceedings" in header_text or "orders" in header_text:
            filings_table = table
            break

    if not filings_table:
        return briefs

    for row in filings_table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        date_cell = cells[0]
        desc_cell = cells[1]
        text = desc_cell.get_text(strip=True)
        text_lower = text.lower()

        # Only process amicus/amici curiae briefs
        if "amicus curiae" not in text_lower and "amici curiae" not in text_lower:
            continue

        # Skip motion entries (e.g., "Motion for leave to file amicus curiae brief")
        if text_lower.startswith("motion"):
            continue

        # Skip "not accepted for filing" entries
        if "not accepted for filing" in text_lower:
            continue

        filing_date = date_cell.get_text(strip=True)

        # Extract row background color (SCOTUSblog uses SCOTUS booklet cover colors)
        row_bg = ""
        style = row.get("style", "")
        color_match = re.search(r"background-color:\s*([^;]+)", style)
        if color_match:
            row_bg = color_match.group(1).strip()

        # Extract PDF URL
        pdf_url = ""
        pdf_link = desc_cell.find("a", href=re.compile(r"\.pdf", re.I))
        if pdf_link:
            pdf_url = pdf_link.get("href", "")

        side = classify_side(text, row_bg)
        filer = extract_filer_name(text)

        briefs.append({
            "filer": filer,
            "side": side,
            "date": filing_date,
            "pdf_url": pdf_url,
            "raw_text": text,
        })

    return briefs


def scrape_all() -> dict:
    """Main scraping pipeline."""
    print("Fetching OT2025 case list...")
    cases = get_case_list()
    print(f"Found {len(cases)} cases.")

    if not cases:
        print("ERROR: No cases found. The page structure may have changed.")
        return {"cases": [], "scraped_at": datetime.now(timezone.utc).isoformat()}

    results = []

    for case in tqdm(cases, desc="Scraping case pages"):
        # Generate a cache key from the URL slug
        slug = case["scotusblog_url"].rstrip("/").split("/")[-1]
        cache_key = f"case_{slug}"

        try:
            html = fetch_page(case["scotusblog_url"], cache_key=cache_key)
        except requests.RequestException as e:
            print(f"\n  Warning: Failed to fetch {case['name']}: {e}")
            continue

        briefs = parse_briefs(html)

        # Organize briefs by side
        briefs_by_side = {
            "petitioner": [],
            "respondent": [],
            "neither": [],
            "unknown": [],
        }
        for b in briefs:
            entry = {
                "filer": b["filer"],
                "date": b["date"],
                "pdf_url": b["pdf_url"],
            }
            briefs_by_side[b["side"]].append(entry)

        results.append({
            "name": case["name"],
            "docket_number": case["docket_number"],
            "scotusblog_url": case["scotusblog_url"],
            "category": case["category"],
            "briefs": briefs_by_side,
            "total_briefs": len(briefs),
        })

    # Sort by total briefs descending
    results.sort(key=lambda c: c["total_briefs"], reverse=True)

    return {
        "cases": results,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    data = scrape_all()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    total = sum(c["total_briefs"] for c in data["cases"])
    print(f"\nDone! Wrote {OUTPUT_FILE}")
    print(f"  {len(data['cases'])} cases, {total} total amicus briefs")

    # Print top 10
    print("\nTop 10 cases by amicus brief count:")
    for i, case in enumerate(data["cases"][:10], 1):
        pet = len(case["briefs"]["petitioner"])
        resp = len(case["briefs"]["respondent"])
        neither = len(case["briefs"]["neither"])
        unk = len(case["briefs"]["unknown"])
        print(f"  {i:2}. {case['name'][:50]:50s} {case['total_briefs']:3d} "
              f"(pet:{pet} resp:{resp} neither:{neither} unk:{unk})")


if __name__ == "__main__":
    main()
