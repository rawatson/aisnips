# SCOTUS OT2025 Amici Brief Counter — Design Document

## Overview

This tool scrapes SCOTUSblog's OT2025 case listing, extracts all amicus curiae briefs filed in each case, classifies them by which side they support, and generates a static HTML report ranking cases by total amicus brief count.

## Data Source

**SCOTUSblog** (`scotusblog.com/case-files/terms/ot2025/`) was chosen because:

- It provides a comprehensive, well-structured listing of all Supreme Court cases for each term
- Case detail pages list every filing chronologically in a `<table>` with two columns: Date (`<td>` 0) and Proceedings/Orders (`<td>` 1), containing links to PDF documents on `supremecourt.gov`
- Filing descriptions include links to PDF documents hosted on `supremecourt.gov`
- The site is the de facto public resource for Supreme Court case tracking

## Side Classification

Amicus briefs are identified by the presence of "amicus curiae" or "amici curiae" (case-insensitive) in the filing description text. Entries marked "not accepted for filing" are excluded.

### Primary signal: row background color

SCOTUSblog color-codes each filing row's background to match the [SCOTUS booklet cover color scheme](https://www.supremecourt.gov/casehand/USSC%20-%20Booklet-Format%20Specification%20Chart%202019.pdf). This is the primary classification signal:

| Row Background | SCOTUS Cover | Classification |
|----------------|-------------|----------------|
| `#ffffd6` (cream) | Cream — cert-stage amicus | Petitioner (no side distinction at cert stage) |
| `#add6ad` (light green) | Light green — merits amicus for pet./neither | Petitioner (text override for "neither") |
| `#32ad84` (dark green) | Dark green — merits amicus for respondent | Respondent |
| `#da6b6b` (light red) | Light red — court-appointed amicus | Respondent (supports judgment below) |
| `#DDD` (gray) | — U.S. Government amicus | Unknown (side ambiguous) |
| `#ffffff` (white) | — procedural/not accepted | Unknown |

### Secondary signal: text pattern matching

When the filing description contains explicit side language, it overrides the color-based classification. This is particularly important for distinguishing "neither party" briefs (which share the light-green color with petitioner-side briefs):

- "in support of neither party" / "no party" → Neither
- "in support of petitioner/appellant/plaintiff" → Petitioner
- "in support of respondent/appellee/defendant" → Respondent
- "in support of reversal" → Petitioner
- "in support of affirmance" / "judgment below" → Respondent

This two-layer approach classifies ~92% of briefs by side.

## Caching Strategy

- All fetched HTML pages are saved to a `cached_pages/` directory
- The index page is cached as `ot2025_index.html`
- Each case page is cached as `case_{slug}.html` using the URL slug
- Subsequent runs skip the HTTP request for any page with a cached file
- To force a fresh scrape, delete the `cached_pages/` directory
- A 1-second delay is used between requests to be polite to the server

## Architecture

The tool is split into two scripts with a JSON intermediate format:

1. **`scrape_amici.py`** — Fetches pages, parses briefs, writes `amici_data.json`
2. **`generate_html.py`** — Reads `amici_data.json`, produces `amici_report.html`

This separation means:
- Re-generating the HTML report doesn't require re-scraping
- The JSON data can be used by other tools or analysis scripts
- Scraping and presentation concerns are decoupled

## HTML Report Design

The report is a single self-contained HTML file with no external dependencies:

- **Summary stats** at the top: total cases, total briefs, highest count
- **Ranked table** sorted by total amicus brief count (descending)
- **Color-coded "Total" column** using a blue → orange → red gradient to visually indicate interest level
- **Clickable rows** that expand to show three columns: Supporting Petitioner (green), Supporting Respondent (red), Neither/Unknown (gray)
- **PDF links** for each brief pointing to `supremecourt.gov`
- **Responsive layout** that stacks detail columns on mobile
- **Keyboard accessible**: Enter to toggle, Escape to close all

## Limitations

1. **Dependent on SCOTUSblog HTML structure**: If SCOTUSblog changes their page layout (CSS classes, element structure), the parser may break. The `<table>` filings structure and `<div id="accordion">` case index are key assumptions.

2. **Light-green ambiguity**: SCOTUS rules assign the same light-green cover to briefs supporting petitioner *and* briefs supporting neither party. We default light-green to "petitioner" unless the text explicitly says "neither party". A small number of "neither party" briefs may be miscounted as petitioner.

3. **Filer name extraction**: The regex-based name extraction handles common patterns but may truncate or misparse unusual brief titles.

4. **No cert/merits stage distinction**: We don't currently distinguish between cert-stage and merits-stage amicus briefs, as this would require knowing when certiorari was granted for each case.

5. **Point-in-time snapshot**: The data reflects what's on SCOTUSblog at scrape time. Newly filed briefs won't appear until the script is re-run (with cache cleared or updated).
