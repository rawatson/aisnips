#!/usr/bin/env python3
"""Generate a self-contained HTML report from amici_data.json."""

import json
import html
import sys
from pathlib import Path

INPUT_FILE = "amici_data.json"
OUTPUT_FILE = "amici_report.html"


def brief_count_color(count: int, max_count: int) -> str:
    """Return a background color on a gradient from cool (low) to warm (high)."""
    if max_count == 0:
        return "#f0f0f0"
    ratio = min(count / max_count, 1.0)
    # Gradient: light blue (#e3f2fd) → orange (#ff9800) → deep red (#c62828)
    if ratio < 0.5:
        t = ratio * 2
        r = int(227 + (255 - 227) * t)
        g = int(242 + (152 - 242) * t)
        b = int(253 + (0 - 253) * t)
    else:
        t = (ratio - 0.5) * 2
        r = int(255 + (198 - 255) * t)
        g = int(152 + (40 - 152) * t)
        b = int(0 + (40 - 0) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def text_color_for_bg(hex_color: str) -> str:
    """Return black or white text depending on background luminance."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000" if luminance > 0.5 else "#fff"


def render_brief_list(briefs: list[dict]) -> str:
    """Render a list of briefs as HTML."""
    if not briefs:
        return "<p class='empty'>None</p>"
    items = []
    for b in briefs:
        filer = html.escape(b.get("filer", "Unknown"))
        date = html.escape(b.get("date", ""))
        pdf_url = b.get("pdf_url", "")
        if pdf_url:
            filer_html = f'<a href="{html.escape(pdf_url)}" target="_blank" rel="noopener">{filer}</a>'
        else:
            filer_html = filer
        items.append(f'<li>{filer_html}<span class="date">{date}</span></li>')
    return "<ul>" + "\n".join(items) + "</ul>"


def generate_html(data: dict) -> str:
    """Generate the full HTML report."""
    cases = data["cases"]
    scraped_at = data.get("scraped_at", "unknown")
    max_briefs = max((c["total_briefs"] for c in cases), default=0)

    # Build table rows and detail sections
    rows = []
    details = []

    for i, case in enumerate(cases, 1):
        name = html.escape(case["name"])
        docket = html.escape(case["docket_number"])
        url = html.escape(case["scotusblog_url"])
        total = case["total_briefs"]
        pet = len(case["briefs"]["petitioner"])
        resp = len(case["briefs"]["respondent"])
        neither = len(case["briefs"]["neither"])
        unk = len(case["briefs"]["unknown"])
        bg = brief_count_color(total, max_briefs)
        fg = text_color_for_bg(bg)
        case_id = f"case-{i}"

        rows.append(f"""
        <tr class="case-row" onclick="toggleDetail('{case_id}')" role="button" tabindex="0"
            onkeydown="if(event.key==='Enter')toggleDetail('{case_id}')">
          <td class="rank">{i}</td>
          <td class="case-name"><a href="{url}" target="_blank" rel="noopener" onclick="event.stopPropagation()">{name}</a></td>
          <td class="docket">{docket}</td>
          <td class="count total" style="background:{bg};color:{fg}">{total}</td>
          <td class="count pet">{pet}</td>
          <td class="count resp">{resp}</td>
          <td class="count neither">{neither + unk}</td>
          <td class="expand-icon" id="icon-{case_id}">&#9654;</td>
        </tr>""")

        pet_html = render_brief_list(case["briefs"]["petitioner"])
        resp_html = render_brief_list(case["briefs"]["respondent"])
        neither_list = case["briefs"]["neither"] + case["briefs"]["unknown"]
        neither_html = render_brief_list(neither_list)

        details.append(f"""
        <tr class="detail-row" id="{case_id}" style="display:none">
          <td colspan="8">
            <div class="detail-content">
              <div class="detail-col">
                <h4>Supporting Petitioner ({pet})</h4>
                {pet_html}
              </div>
              <div class="detail-col">
                <h4>Supporting Respondent ({resp})</h4>
                {resp_html}
              </div>
              <div class="detail-col">
                <h4>Neither / Unknown ({neither + unk})</h4>
                {neither_html}
              </div>
            </div>
          </td>
        </tr>""")

    table_body = ""
    for row, detail in zip(rows, details):
        table_body += row + detail

    total_briefs = sum(c["total_briefs"] for c in cases)
    total_cases = len(cases)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SCOTUS OT2025 — Amici Brief Counter</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #fafafa;
  color: #222;
  line-height: 1.5;
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}}
h1 {{
  font-size: 1.6em;
  margin-bottom: 4px;
}}
.subtitle {{
  color: #666;
  font-size: 0.9em;
  margin-bottom: 20px;
}}
.stats {{
  display: flex;
  gap: 24px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}}
.stat-box {{
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 12px 20px;
  text-align: center;
}}
.stat-box .num {{
  font-size: 1.8em;
  font-weight: 700;
  color: #1565c0;
}}
.stat-box .label {{
  font-size: 0.85em;
  color: #666;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}}
thead th {{
  background: #263238;
  color: #fff;
  padding: 10px 12px;
  text-align: left;
  font-size: 0.85em;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  white-space: nowrap;
}}
thead th.count {{ text-align: center; }}
.case-row {{
  cursor: pointer;
  transition: background 0.15s;
}}
.case-row:hover {{
  background: #e8eaf6 !important;
}}
.case-row td {{
  padding: 8px 12px;
  border-bottom: 1px solid #eee;
  font-size: 0.9em;
}}
.rank {{ color: #999; width: 40px; text-align: center; }}
.case-name a {{
  color: #1565c0;
  text-decoration: none;
}}
.case-name a:hover {{ text-decoration: underline; }}
.docket {{ color: #666; white-space: nowrap; }}
.count {{ text-align: center; font-weight: 600; white-space: nowrap; }}
.total {{ border-radius: 4px; min-width: 50px; }}
.pet {{ color: #1b5e20; }}
.resp {{ color: #b71c1c; }}
.neither {{ color: #666; }}
.expand-icon {{
  width: 30px;
  text-align: center;
  color: #999;
  font-size: 0.8em;
  transition: transform 0.2s;
}}
.expand-icon.open {{
  transform: rotate(90deg);
}}
.detail-row td {{
  padding: 0;
  border-bottom: 2px solid #1565c0;
}}
.detail-content {{
  display: flex;
  gap: 20px;
  padding: 16px 20px;
  background: #f5f5f5;
  flex-wrap: wrap;
}}
.detail-col {{
  flex: 1;
  min-width: 250px;
}}
.detail-col h4 {{
  font-size: 0.85em;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  margin-bottom: 8px;
  padding-bottom: 4px;
  border-bottom: 2px solid #ccc;
}}
.detail-col:nth-child(1) h4 {{ color: #1b5e20; border-color: #1b5e20; }}
.detail-col:nth-child(2) h4 {{ color: #b71c1c; border-color: #b71c1c; }}
.detail-col:nth-child(3) h4 {{ color: #555; border-color: #555; }}
.detail-col ul {{
  list-style: none;
  padding: 0;
}}
.detail-col li {{
  padding: 4px 0;
  border-bottom: 1px solid #e0e0e0;
  font-size: 0.85em;
  display: flex;
  justify-content: space-between;
  gap: 8px;
}}
.detail-col li a {{
  color: #1565c0;
  text-decoration: none;
}}
.detail-col li a:hover {{ text-decoration: underline; }}
.date {{
  color: #999;
  white-space: nowrap;
  font-size: 0.85em;
  flex-shrink: 0;
}}
.empty {{
  color: #999;
  font-style: italic;
  font-size: 0.85em;
}}
@media (max-width: 768px) {{
  body {{ padding: 12px; }}
  .detail-content {{ flex-direction: column; }}
  table {{ font-size: 0.85em; }}
}}
</style>
</head>
<body>

<h1>SCOTUS OT2025 &mdash; Amici Brief Counter</h1>
<p class="subtitle">Data scraped from SCOTUSblog &middot; {html.escape(scraped_at)} &middot; Click a row to expand brief details</p>

<div class="stats">
  <div class="stat-box"><div class="num">{total_cases}</div><div class="label">Cases</div></div>
  <div class="stat-box"><div class="num">{total_briefs}</div><div class="label">Amicus Briefs</div></div>
  <div class="stat-box"><div class="num">{max_briefs}</div><div class="label">Most in One Case</div></div>
</div>

<table>
<thead>
  <tr>
    <th>#</th>
    <th>Case</th>
    <th>Docket</th>
    <th class="count">Total</th>
    <th class="count">Pet.</th>
    <th class="count">Resp.</th>
    <th class="count">Other</th>
    <th></th>
  </tr>
</thead>
<tbody>
{table_body}
</tbody>
</table>

<script>
function toggleDetail(id) {{
  var row = document.getElementById(id);
  var icon = document.getElementById('icon-' + id);
  if (row.style.display === 'none') {{
    row.style.display = 'table-row';
    icon.classList.add('open');
  }} else {{
    row.style.display = 'none';
    icon.classList.remove('open');
  }}
}}

// Allow keyboard navigation
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') {{
    document.querySelectorAll('.detail-row').forEach(function(r) {{
      r.style.display = 'none';
    }});
    document.querySelectorAll('.expand-icon').forEach(function(i) {{
      i.classList.remove('open');
    }});
  }}
}});
</script>

</body>
</html>"""


def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found. Run scrape_amici.py first.")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    html_content = generate_html(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated {OUTPUT_FILE} with {len(data['cases'])} cases")


if __name__ == "__main__":
    main()
