"""
report_html.py

Script to do the following:

- build HTML tables and sections
- provide collapsible sections
- initialise DataTables with sorting, searching, pagination, and CSV export
- draw Lorenz curves in Plotly
- provide formatting helpers for outputs

"""

from __future__ import annotations

from html import escape
from itertools import count
import json


# ---------------------------------------------------------------------
# Table IDs
# ---------------------------------------------------------------------

_TABLE_COUNTER = count(1)


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def f_int(value: int | float) -> str:
    """
    Format an integer-like value with thousands separators.
    """
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}"


def f_float(value: float, digits: int = 4) -> str:
    """
    Format a float with a fixed number of decimal places.
    """
    return f"{value:.{digits}f}"


def f_pct(value: float, digits: int = 2) -> str:
    """
    Format a proportion in [0, 1] as a percentage.
    E.g. 0.1234 would be formatted as 12.34%
    """
    return f"{value * 100:.{digits}f}%"


def f_pp(value: float, digits: int = 2) -> str:
    """
    Format a percentage-point difference.
    Works on inputs already in percentage-point units.
    E.g. 1.25 would be formatted as +1.25 pp
    """
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f} pp"


# ---------------------------------------------------------------------
# Builders for HTML report
# ---------------------------------------------------------------------

def html_note(text: str) -> str:
    """
    Render an informational block.
    """
    return f'<div class="note">{escape(text)}</div>'


def html_table(
    headers: list[str],
    rows: list[list[object]],
    caption: str | None = None,
    datatable: bool = True,
    csv_name: str | None = None,
) -> str:
    """
    Render an HTML table.

    Parameters
    ----------
    headers:
        Column headers.
    rows:
        Body rows, already formatted for display
    caption:
        Optional caption above table
    datatable:
        If True, the table is initialised with DataTables
    csv_name:
        Optional filename stem used by the in-browser CSV export button

    Returns
    -------
    str
        HTML string for the table.
    """
    table_id = f"tbl_{next(_TABLE_COUNTER)}"
    class_attr = "report-table datatable" if datatable else "report-table"

    caption_html = f"<caption>{escape(caption)}</caption>" if caption else ""
    export_attr = f'data-export-name="{escape(csv_name)}"' if csv_name else ""

    thead = "".join(f"<th>{escape(str(h))}</th>" for h in headers)

    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")

    tbody = "".join(body_rows)

    return f"""
    <table id="{table_id}" class="{class_attr}" {export_attr}>
      {caption_html}
      <thead>
        <tr>{thead}</tr>
      </thead>
      <tbody>
        {tbody}
      </tbody>
    </table>
    """


def subsection(title: str, content: str) -> str:
    """
    Render a subsection inside subsection in the report
    """
    return f"""
    <div class="subsection">
      <h3>{escape(title)}</h3>
      {content}
    </div>
    """


def section(title: str, content: str, open_by_default: bool = False) -> str:
    """
    Render a collapsible report section

    Parameters
    ----------
    title:
        Summary text shown in the collapsible header
    content:
        HTML body for the section
    open_by_default:
        If True, the section is expanded on page load
    """
    open_attr = " open" if open_by_default else ""
    return f"""
    <details class="report-section"{open_attr}>
      <summary>{escape(title)}</summary>
      <div class="section-content">
        {content}
      </div>
    </details>
    """


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def lorenz_plot_block(lorenz_data: dict[str, list[tuple[float, float]]]) -> str:
    """
    Render one Lorenz curve in Plotly per style

    Parameters
    ----------
    lorenz_data:
        Mapping:
            style -> list of (cumulative_doc_share, cumulative_edit_share)

    Returns
    -------
    str
        HTML block containing one plot per style.
    """
    blocks = ['<div class="lorenz-grid">']

    for style in sorted(lorenz_data):
        plot_id = f"lorenz_{style.replace(' ', '_').replace('/', '_')}"
        points = lorenz_data[style]

        xs = [x for x, _ in points]
        ys = [y for _, y in points]

        blocks.append(f"""
        <div class="plot-box">
          <div><strong>{escape(style)}</strong></div>
          <div id="{plot_id}" style="height:360px;"></div>
          <script>
            Plotly.newPlot(
              "{plot_id}",
              [
                {{
                  x: {json.dumps(xs)},
                  y: {json.dumps(ys)},
                  mode: "lines",
                  name: "Lorenz curve"
                }},
                {{
                  x: [0, 1],
                  y: [0, 1],
                  mode: "lines",
                  name: "Equality line"
                }}
              ],
              {{
                margin: {{l: 50, r: 20, t: 20, b: 50}},
                xaxis: {{title: "Cumulative share of documents"}},
                yaxis: {{title: "Cumulative share of edits", range: [0, 1]}},
                showlegend: true
              }},
              {{responsive: true}}
            );
          </script>
        </div>
        """)

    blocks.append("</div>")
    return "".join(blocks)


# ---------------------------------------------------------------------
# Page wrapper
# ---------------------------------------------------------------------

def html_page(title: str, body: str) -> str:
    """
    Builds the full HTML page
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>

<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>

<style>
  :root {{
    --bg: #f7f7f8;
    --panel: #ffffff;
    --border: #d9d9de;
    --text: #1f2328;
    --muted: #606770;
    --accent: #214f8b;
    --note: #eef5ff;
  }}

  body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 32px;
    background: var(--bg);
    color: var(--text);
  }}

  h1 {{
    margin-top: 0;
    margin-bottom: 24px;
  }}

  h2, h3 {{
    margin-bottom: 8px;
  }}

  .report-section {{
    border: 1px solid var(--border);
    background: var(--panel);
    border-radius: 10px;
    margin-bottom: 18px;
    overflow: hidden;
  }}

  .report-section > summary {{
    cursor: pointer;
    font-size: 1.1rem;
    font-weight: 700;
    padding: 14px 18px;
    background: #eef1f5;
    list-style: none;
  }}

  .report-section > summary::-webkit-details-marker {{
    display: none;
  }}

  .section-content {{
    padding: 16px 18px 20px 18px;
  }}

  .subsection {{
    margin-bottom: 28px;
  }}

  .note {{
    background: var(--note);
    border-left: 4px solid var(--accent);
    padding: 10px 12px;
    margin: 8px 0 14px 0;
    line-height: 1.4;
  }}

  .report-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 18px 0;
    background: white;
  }}

  .report-table caption {{
    caption-side: top;
    text-align: left;
    font-weight: 700;
    margin-bottom: 8px;
  }}

  .report-table th,
  .report-table td {{
    border: 1px solid #e4e6eb;
    padding: 6px 8px;
    vertical-align: top;
  }}

  .report-table th {{
    background: #f0f2f5;
  }}

  .lorenz-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 16px;
  }}

  .plot-box {{
    border: 1px solid var(--border);
    background: white;
    border-radius: 8px;
    padding: 10px;
  }}

  .small {{
    color: var(--muted);
    font-size: 0.92rem;
  }}

  .topnav {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 20px;
  }}

  .topnav ul {{
    margin: 8px 0 0 18px;
  }}

  .topnav li {{
    margin-bottom: 4px;
  }}
</style>

<script>
document.addEventListener("DOMContentLoaded", function() {{
  $('table.datatable').each(function() {{
    const exportName = $(this).data('export-name') || 'table';
    $(this).DataTable({{
      pageLength: 25,
      autoWidth: false,
      order: [],
      dom: 'Bfrtip',
      buttons: [
        {{
          extend: 'csvHtml5',
          title: null,
          filename: exportName
        }}
      ]
    }});
  }});

  document.querySelectorAll("details.report-section").forEach(function(el) {{
    el.addEventListener("toggle", function() {{
      if (el.open) {{
        setTimeout(function() {{
          $.fn.dataTable.tables({{visible: true, api: true}}).columns.adjust();
        }}, 25);
      }}
    }});
  }});
}});
</script>
</head>
<body>
<h1>{escape(title)}</h1>

<div class="topnav">
  <strong>Report structure</strong>
  <ul>
    <li>Metadata and analytical scope</li>
    <li>Corpus description and issue overview</li>
    <li>Primary style comparison</li>
    <li>Error concentration and problematic documents</li>
    <li>Error drivers by style: character, bigram, and word</li>
    <li>Per-style document diagnostics</li>
  </ul>
</div>

{body}
</body>
</html>
"""


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------

def csv_ready_rows(rows: list[list[object]]) -> list[list[object]]:
    """
    Convert rendered-table rows into plain string values to write to CSV
    """
    return [[str(cell) for cell in row] for row in rows]