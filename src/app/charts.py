"""Plotly chart builders for the Streamlit ACWR application."""

from __future__ import annotations

from typing import Any

from app.constants import ZONE_BANDS


def _fmt_load(val: float | None, unit: str) -> str:
    if val is None:
        return "—"
    return f"{val:,.0f} {unit}" if unit == "m" else f"{val:.0f}"


def _acts_str(acts: list[str]) -> str:
    return " · ".join(acts) if acts else "Rest"


def build_acwr_chart(mdata: dict, meta: dict, show_load: bool = False) -> Any:
    import plotly.graph_objects as go

    hist_x = mdata["hist_dates"]
    fore_x = mdata["fore_dates"]
    hist_y = mdata["hist_acwr"]
    fore_y = mdata["fore_acwr"]

    hist_load = mdata.get("hist_load", [None] * len(hist_x))
    hist_acts = mdata.get("hist_activities", [[] for _ in hist_x])
    fore_load = mdata.get("fore_load", [None] * len(fore_x))
    fore_acts = mdata.get("fore_activities", [[] for _ in fore_x])
    unit = meta.get("unit", "")

    # customdata rows: [load_str, activities_str]
    hist_custom = [[_fmt_load(v, unit), _acts_str(a)] for v, a in zip(hist_load, hist_acts, strict=False)]
    fore_custom = [[_fmt_load(v, unit), _acts_str(a)] for v, a in zip(fore_load, fore_acts, strict=False)]

    fig = go.Figure()

    for band in ZONE_BANDS:
        fig.add_hrect(y0=band["y0"], y1=band["y1"],
                      fillcolor=band["color"], layer="below", line_width=0)

    for y_val in [0.8, 1.3, 1.5]:
        fig.add_hline(y=y_val, line_dash="dot",
                      line_color="rgba(0,82,159,0.20)", line_width=1)

    if fore_x:
        fig.add_vrect(x0=fore_x[0], x1=fore_x[-1],
                      fillcolor="rgba(254,190,16,0.05)", layer="below", line_width=0)
        fig.add_annotation(
            x=fore_x[0], y=2.1, text="Forecast Window",
            showarrow=False, font=dict(size=10, color="#B8920A"),
            xanchor="left", yanchor="top",
        )

    # ── Load bars (below ACWR lines so they don't obscure) ────────────────────
    if show_load:
        fig.add_trace(go.Bar(
            x=hist_x,
            y=hist_load,
            name=f"Load ({unit})",
            marker=dict(
                color="#334D6E",
                opacity=0.55,
                line=dict(color="#000000", width=0.6),
            ),
            yaxis="y2",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Bar(
            x=fore_x,
            y=fore_load,
            name=f"Forecast Load ({unit})",
            marker=dict(
                color="#334D6E",
                opacity=0.35,
                line=dict(color="#000000", width=0.6),
            ),
            yaxis="y2",
            hoverinfo="skip",
        ))

    # ── Historical ACWR — spline for smooth curve, extend one point for join ──
    join_x      = hist_x + ([fore_x[0]] if fore_x else [])
    join_y      = hist_y + ([fore_y[0]] if fore_y else [])
    join_custom = hist_custom + ([fore_custom[0]] if fore_custom else [])

    fig.add_trace(go.Scatter(
        x=join_x, y=join_y,
        customdata=join_custom,
        mode="lines", name="Historical",
        line=dict(color="rgba(0,82,159,0.65)", width=2, shape="spline", smoothing=0.6),
        connectgaps=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Load: %{customdata[0]}<br>"
            "%{customdata[1]}"
            "<extra>Historical</extra>"
        ),
    ))

    # ── Forecast ACWR — spline ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=fore_x, y=fore_y,
        customdata=fore_custom,
        mode="lines+markers", name="Forecast",
        line=dict(color=meta["color"], width=2.5, shape="spline", smoothing=0.6),
        marker=dict(size=5, color=meta["color"],
                    line=dict(color="#FFFFFF", width=1.5)),
        fill="tozeroy", fillcolor=meta["fill"],
        connectgaps=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "ACWR: %{y:.3f}<br>"
            "Load: %{customdata[0]}<br>"
            "%{customdata[1]}"
            "<extra>Forecast</extra>"
        ),
    ))

    fig.update_layout(
        height=400,
        plot_bgcolor="#FAFCFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, system-ui", color="#334D6E", size=12),
        margin=dict(l=8, r=8, t=32, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#DCE8F5",
                        font=dict(color="#0F172A", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#DCE8F5", borderwidth=1,
                    font=dict(size=11, color="#334D6E")),
        barmode="overlay",
        xaxis=dict(tickfont=dict(size=10, color="#64748B"),
                   gridcolor="rgba(0,82,159,0.06)",
                   showline=False, tickangle=40, nticks=22),
        yaxis=dict(range=[0, 2.3],
                   tickfont=dict(size=10, color="#64748B"),
                   gridcolor="rgba(0,82,159,0.06)",
                   title=dict(text="ACWR", font=dict(size=11, color="#64748B")),
                   zeroline=False),
        **({"yaxis2": dict(
            title=dict(text=f"Load ({unit})", font=dict(size=11, color="#64748B")),
            side="right",
            overlaying="y",
            tickfont=dict(size=10, color="#64748B"),
            showgrid=False,
            zeroline=False,
        )} if show_load else {}),
    )
    return fig
