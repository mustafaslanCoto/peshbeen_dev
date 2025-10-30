from great_tables import GT, html
from IPython.display import HTML, display
def make_main_gt(df,
                    n_regimes: int,
                    inner_labels=None,
                    param_names=None,
                    regime_labels=None,
                    decimals=None,
                    p_threshold=0.001,
                    title="Model summary",
                    subtitle="Multi-regime coefficients"):
    """
    Build a great_tables GT that groups columns into n_regimes with identical visible
    inner labels under each regime tab_spanner.
    
    Parameters
    - arr_or_df: numpy array (n_rows x (n_regimes * n_inner)) or pandas DataFrame
                columns must be ordered regime1..regimeN.
    - n_regimes: int, number of regimes
    - inner_labels: list of strings, inner column headings (default ['Coef','Std. Err','t-stat','p-value'])
    - param_names: optional list of row index labels (len == n_rows)
    - regime_labels: optional list of regime labels for the spanners (default 'Regime 1', 'Regime 2', ...)
    - decimals: dict mapping inner label -> decimal places (e.g. {'Coef':4, 't-stat':3})
    - p_threshold: float, p-value threshold to show as "<threshold"
    - title/subtitle: strings for the table header
    
    Returns: GT object (ready to render in Jupyter)
    """
    if inner_labels is None:
        inner_labels = ['Coef', 'Std. Err', 't-stat', 'p-value']
    n_inner = len(inner_labels)
    if decimals is None:
        decimals = {'Coef':4, 'Std. Err':3, 't-stat':3, 'p-value':3}


    # expected_cols = n_regimes * n_inner
    # if df.shape[1] != expected_cols:
    #     raise ValueError(f"Data has {df.shape[1]} columns but expected {expected_cols} "
    #                      f"({n_regimes} regimes * {n_inner} inner labels).")

    # Prepare regime labels
    if regime_labels is None:
        regime_labels = [f"Regime {i+1}" for i in range(n_regimes)]
    if len(regime_labels) != n_regimes:
        raise ValueError("regime_labels length must equal n_regimes")

    # Create unique internal names while keeping visible labels identical:
    zw = '\u200b'   # zero-width space
    cols = ["variable"]
    for r in range(n_regimes):
        suffix = zw * r  # r=0 -> '', r=1 -> one zw, r=2 -> two zw, etc.
        for lbl in inner_labels:
            cols.append(lbl + suffix)

    df.columns = cols

    # Optional: nicer row labels
    if param_names is not None:
        if len(param_names) != df.shape[0]:
            raise ValueError("param_names length must equal number of rows")
        df.index = param_names
    else:
        df.index = [f"param_{i}" for i in range(df.shape[0])]

    # Create a formatted copy (strings) for display:
    df_fmt = df.copy()

    # Format each column according to its inner label
    for r in range(n_regimes):
        for i, lbl in enumerate(inner_labels):
            col = lbl + (zw * r)

            d = decimals.get(lbl, 4)
            def fmt_numeric(x, d=d):
                try:
                    xv = float(x)
                except Exception:
                    return "" if pd.isna(x) else str(x)
                fmt = f"{{:.{d}f}}"
                return fmt.format(xv)
            df_fmt[col] = df_fmt[col].apply(fmt_numeric)

    # Build GT with tab_spanner groups using internal (unique) column names
    gt = GT(df_fmt)
    gt = gt.tab_header(title=title, subtitle=subtitle)

    for r in range(n_regimes):
        cols_for_regime = [lbl + (zw * r) for lbl in inner_labels]
        gt = gt.tab_spanner(label=regime_labels[r], columns=cols_for_regime)

    gt = gt.tab_stub(rowname_col="variable").tab_stubhead(label="Variables")
    return gt


def gt_mini(df_small, title=None, show_col_labels=True):
    g = (
        GT(df_small.reset_index(drop=True))
        .tab_header(title=title or "")
        .opt_vertical_padding(scale=0.8)
        .opt_horizontal_padding(scale=0.8)
    )
    if not show_col_labels:
        g = g.tab_options(
            column_labels_hidden=True,                # remove the labels band
            heading_border_bottom_style="solid",      # keep one rule under the title
            heading_border_bottom_width="1px",
        )
    else:
        g = g.tab_options(
            heading_border_bottom_style="solid",
            heading_border_bottom_width="1px",
        )
    return g

def inject_header_table_groups(gt_main, columns, subtitle_text,
                            col_min_px=260, gap_px=16, font_px=12):
    # columns: list of columns; each column is a list of (caption, df, show_col_labels)
    col_html = []
    for stack in columns:
        minis = [gt_mini(df, cap, show_labels).as_raw_html() for cap, df, show_labels in stack]
        col_html.append("<div class='gt-mini-col'>" + "".join(minis) + "</div>")

    block = f"""
    <style>
    .gt-mini-grid {{
        display: grid;
        grid-template-columns: repeat({len(columns)}, minmax({int(col_min_px)}px, 1fr));
        gap: {int(gap_px)}px; align-items: start; justify-items: center;
    }}
    .gt-mini-col {{ display: flex; flex-direction: column; gap: {int(gap_px)}px; }}
    .gt-mini-grid .gt_table {{ font-size: {int(font_px)}px; width: auto; }}
    </style>
    <div class="gt-mini-grid">{''.join(col_html)}</div>
    """
    return gt_main.tab_header(title="Model summary", subtitle=html(f"<div>{subtitle_text}</div>{block}"))

def cov_table(cov_matrixes, font_size_px=350):
    gt_tables = [
        GT(df).tab_header(title=title)
          .tab_options(
              heading_border_bottom_style="solid",
              heading_border_bottom_width="2px",
              heading_border_bottom_color="#444",    # darker for visibility
          )
        for title, df in cov_matrixes
    ]

    # Export each GT to inline HTML
    html_tables = [gt.as_raw_html(inline_css=True) for gt in gt_tables]

    # Combine them horizontally using flexbox
    combined_html = (
        '<div style="display:flex; flex-wrap:wrap; gap:16px; align-items:flex-start;">'
        + "".join([f'<div style="flex:0 1 {font_size_px}px;">{html}</div>' for html in html_tables])
        + "</div>"
    )

    # Display in notebook or Quarto
    return display(HTML(combined_html))