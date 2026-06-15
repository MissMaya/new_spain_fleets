"""
report_html.py

HTML rendering helpers for the revised HTR corpus diagnostics report.

This version is designed for the v7 report builder.  It keeps the helpers
used by the earlier report, but updates the page wrapper, navigation, table
styling, and Plotly layout support for the newer report structure:

- corpus shape and style comparison;
- line-structure integrity;
- error concentration;
- error ecology and recurrent confusion contexts;
- structural topology plots and peak-position summaries;
- risk-weighted stratification, document-status governance, and downstream status-index outputs;
- appendices for metric definitions, index-derived risk stratification, and human review.
"""

from __future__ import annotations

from html import escape
from itertools import count
import json
from typing import Any


# ---------------------------------------------------------------------
# Table IDs
# ---------------------------------------------------------------------

_TABLE_COUNTER = count(1)


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def f_int(value: int | float | None) -> str:
    """
    Format an integer-like value using thousands separators.
    """
    if value is None:
        return ""
    try:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return f"{value:,}"
    except (TypeError, ValueError):
        return escape(str(value))


def f_float(value: float | int | None, digits: int = 4) -> str:
    """
    Format a float with a fixed number of decimal places.
    """
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return escape(str(value))


def f_pct(value: float | int | None, digits: int = 2) -> str:
    """
    Format a proportion in [0, 1] as a percentage.
    """
    if value is None:
        return ""
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return escape(str(value))


def f_pp(value: float | int | None, digits: int = 2) -> str:
    """
    Format a percentage-point difference.
    Input is already in percentage-point units.
    """
    if value is None:
        return ""
    try:
        value = float(value)
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.{digits}f} pp"
    except (TypeError, ValueError):
        return escape(str(value))


# ---------------------------------------------------------------------
# Basic HTML builders
# ---------------------------------------------------------------------

def html_note(text: str) -> str:
    """
    Render an informational note block. Text is escaped deliberately.
    """
    return f'<div class="note">{escape(text)}</div>'


def html_warning(text: str) -> str:
    """
    Render a caution / limitations note block. Text is escaped deliberately.
    """
    return f'<div class="warning-note">{escape(text)}</div>'


def html_small(text: str) -> str:
    """
    Render small muted helper text.
    """
    return f'<p class="small">{escape(text)}</p>'


def html_badge(text: str, kind: str = "neutral") -> str:
    """
    Render a small badge. Useful for labels such as Stable, Review, High risk.
    """
    safe_kind = "".join(ch for ch in kind.lower() if ch.isalnum() or ch in {"-", "_"}) or "neutral"
    return f'<span class="badge badge-{safe_kind}">{escape(text)}</span>'


def html_table(
    headers: list[str],
    rows: list[list[object]],
    caption: str | None = None,
    datatable: bool = True,
    csv_name: str | None = None,
    wide: bool = True,
    sticky_first_col: bool = True,
    scroll_y: str = "520px",
    page_length: int = 25,
) -> str:
    """
    Render an HTML table.

    The helper defaults to wide, scrollable DataTables because the report
    intentionally includes several operational handoff tables with many fields.
    """
    table_id = f"tbl_{next(_TABLE_COUNTER)}"

    classes = ["report-table"]
    if datatable:
        classes.append("datatable")
    if wide:
        classes.append("wide-table")
    if sticky_first_col:
        classes.append("sticky-first-col")

    class_attr = " ".join(classes)
    caption_html = f"<caption>{escape(caption)}</caption>" if caption else ""
    export_attr = f'data-export-name="{escape(csv_name)}"' if csv_name else ""
    scroll_attr = f'data-scroll-y="{escape(scroll_y)}"'
    page_len_attr = f'data-page-length="{int(page_length)}"'

    thead = "".join(f"<th>{escape(str(h))}</th>" for h in headers)

    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")

    tbody = "".join(body_rows)

    table_html = f"""
    <table id="{table_id}" class="{class_attr}" {export_attr} {scroll_attr} {page_len_attr}>
      {caption_html}
      <thead>
        <tr>{thead}</tr>
      </thead>
      <tbody>
        {tbody}
      </tbody>
    </table>
    """

    if wide:
        return f'<div class="table-scroll-wrap">{table_html}</div>'

    return table_html


def subsection(title: str, content: str) -> str:
    """
    Render a subsection inside a report section.
    """
    return f"""
    <div class="subsection">
      <h3>{escape(title)}</h3>
      {content}
    </div>
    """


def section(title: str, content: str, open_by_default: bool = False) -> str:
    """
    Render a collapsible report section.
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
# Optional plot helpers
# ---------------------------------------------------------------------

def plotly_block(
    div_id: str,
    traces: list[dict[str, Any]],
    layout: dict[str, Any] | None = None,
    height: int = 420,
    title: str | None = None,
) -> str:
    """
    Generic Plotly block used by report builders that prefer to create traces
    in the builder rather than in this helper module.
    """
    layout = layout or {}
    safe_div_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in div_id)
    title_html = f"<div><strong>{escape(title)}</strong></div>" if title else ""

    return f"""
    <div class="plot-box">
      {title_html}
      <div id="{safe_div_id}" style="height:{int(height)}px;"></div>
      <script>
        Plotly.newPlot(
          "{safe_div_id}",
          {json.dumps(traces)},
          {json.dumps(layout)},
          {{responsive: true}}
        );
      </script>
    </div>
    """


def boxplot_block(
    distribution_data: dict[str, dict[str, list[float]]],
    metric_labels: dict[str, str] | None = None,
) -> str:
    """
    Render one Plotly boxplot per metric.
    """
    metric_labels = metric_labels or {}
    blocks = ['<div class="plot-grid">']

    for metric in sorted(distribution_data):
        plot_id = f"box_{metric.replace(' ', '_').replace('/', '_')}"
        style_map = distribution_data[metric]

        traces = []
        for style in sorted(style_map):
            traces.append({
                "y": style_map[style],
                "type": "box",
                "name": style,
                "boxpoints": "outliers",
            })

        title = metric_labels.get(metric, metric)

        blocks.append(plotly_block(
            div_id=plot_id,
            traces=traces,
            layout={
                "margin": {"l": 60, "r": 20, "t": 20, "b": 90},
                "yaxis": {"title": title},
                "xaxis": {"tickangle": -30},
                "showlegend": False,
            },
            height=420,
            title=title,
        ))

    blocks.append("</div>")
    return "".join(blocks)


# For backwards compatibility just in case older report builders call it.
def lorenz_plot_block(lorenz_data: dict[str, list[tuple[float, float]]]) -> str:
    """
    Deprecated Lorenz helper retained for backwards compatibility.
    """
    return html_note(
        "Lorenz curves are omitted from the refactored report. "
        "Use concentration bars and exported concentration tables instead."
    )


# ---------------------------------------------------------------------
# Page wrapper
# ---------------------------------------------------------------------

def html_page(title: str, body: str) -> str:
    """
    Build the full HTML page.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>

<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/fixedheader/3.4.0/css/fixedHeader.dataTables.min.css">

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
<script src="https://cdn.datatables.net/fixedheader/3.4.0/js/dataTables.fixedHeader.min.js"></script>
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
    --warning: #fff8e6;
    --warning-accent: #d99a00;
    --header: #eef1f5;
    --table-header: #f0f2f5;
    --sticky-bg: #ffffff;
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

  a {{
    color: var(--accent);
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
    background: var(--header);
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
    line-height: 1.45;
  }}

  .warning-note {{
    background: var(--warning);
    border-left: 4px solid var(--warning-accent);
    padding: 10px 12px;
    margin: 8px 0 14px 0;
    line-height: 1.45;
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
    columns: 2;
  }}

  .topnav li {{
    margin-bottom: 4px;
    break-inside: avoid;
  }}

  .badge {{
    display: inline-block;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 1px 8px;
    font-size: 0.82rem;
    background: #f4f6f8;
    white-space: nowrap;
  }}

  .badge-high, .badge-severe {{ background: #ffeceb; border-color: #ffb3ad; }}
  .badge-medium, .badge-moderate {{ background: #fff8e6; border-color: #ffd36a; }}
  .badge-low, .badge-stable {{ background: #ecf8ef; border-color: #9ad0a5; }}
  .badge-review {{ background: #fff8e6; border-color: #ffd36a; }}
  .badge-exclude {{ background: #ffeceb; border-color: #ffb3ad; }}

  .table-scroll-wrap {{
    width: 100%;
    overflow-x: auto;
    margin: 10px 0 18px 0;
  }}

  .report-table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
    font-size: 0.93rem;
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
    white-space: nowrap;
  }}

  .report-table th {{
    background: var(--table-header);
    position: sticky;
    top: 0;
    z-index: 2;
  }}

  .report-table.sticky-first-col th:first-child,
  .report-table.sticky-first-col td:first-child {{
    position: sticky;
    left: 0;
    z-index: 3;
    background: var(--sticky-bg);
    box-shadow: 2px 0 3px rgba(0, 0, 0, 0.06);
  }}

  .report-table.sticky-first-col th:first-child {{
    z-index: 4;
    background: var(--table-header);
  }}

  div.dataTables_wrapper {{
    width: 100%;
  }}

  div.dataTables_scrollBody {{
    border-bottom: 1px solid var(--border);
  }}

  .dt-buttons {{
    margin-bottom: 8px;
  }}

  .plot-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
    gap: 16px;
  }}

  .plot-box {{
    border: 1px solid var(--border);
    background: white;
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0 16px 0;
  }}

  code {{
    background: #f4f4f5;
    border: 1px solid #e4e6eb;
    border-radius: 4px;
    padding: 1px 4px;
  }}

  @media (max-width: 900px) {{
    body {{ margin: 18px; }}
    .topnav ul {{ columns: 1; }}
    .plot-grid {{ grid-template-columns: 1fr; }}
  }}
</style>

<script>
document.addEventListener("DOMContentLoaded", function() {{
  $('table.datatable').each(function() {{
    const $table = $(this);
    const exportName = $table.data('export-name') || 'table';
    const scrollY = $table.data('scroll-y') || '520px';
    const pageLength = parseInt($table.data('page-length') || '25', 10);
    const isWide = $table.hasClass('wide-table');

    $table.DataTable({{
      pageLength: pageLength,
      lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, 'All']],
      autoWidth: false,
      order: [],
      scrollX: isWide,
      scrollY: isWide ? scrollY : '',
      scrollCollapse: true,
      fixedHeader: true,
      dom: 'Blfrtip',
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
          if (window.Plotly) {{
            el.querySelectorAll('.js-plotly-plot').forEach(function(plot) {{
              Plotly.Plots.resize(plot);
            }});
          }}
        }}, 80);
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
    <li>Corpus shape and issue context</li>
    <li>Style behaviour and clean-subset potential</li>
    <li>Line-structure integrity and segmentation risk</li>
    <li>Error concentration and high-burden subsets</li>
    <li>Error ecology and recurrent confusion contexts</li>
    <li>Structural topology of transcription instability</li>
    <li>Risk Weighted Stratification by Style</li>
    <li>Document Status Governance</li>
    <li>Downstream use of the Document Status Index</li>
    <li>Appendix A. Metric definitions</li>
    <li>Appendix B. Index-Derived Risk Stratification</li>
    <li>Appendix C. Human Review Protocol</li>
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
    Convert rendered-table rows into plain string values to write to CSV.
    """
    return [[str(cell) for cell in row] for row in rows]
