"""Plotly chart builders for the Streamlit ACWR application."""

from __future__ import annotations

from typing import Any

from app.constants import ZONE_BANDS


def build_acwr_chart(mdata: dict, meta: dict) -> Any:
    import plotly.graph_objects as go

    hist_x = mdata["hist_dates"]
    fore_x = mdata["fore_dates"]
    hist_y = mdata["hist_acwr"]
    fore_y = mdata["fore_acwr"]

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

    # Extend historical trace by one forecast point so the two lines visually join at the boundary
    join_x = hist_x + ([fore_x[0]] if fore_x else [])
    join_y = hist_y + ([fore_y[0]] if fore_y else [])
    fig.add_trace(go.Scatter(
        x=join_x, y=join_y,
        mode="lines", name="Historical",
        line=dict(color="rgba(0,82,159,0.40)", width=2),
        connectgaps=False,
        hovertemplate="<b>%{x}</b><br>ACWR: %{y:.3f}<extra>Historical</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=fore_x, y=fore_y,
        mode="lines+markers", name="Forecast",
        line=dict(color=meta["color"], width=2.5),
        marker=dict(size=5, color=meta["color"],
                    line=dict(color="#FFFFFF", width=1.5)),
        fill="tozeroy", fillcolor=meta["fill"],
        connectgaps=False,
        hovertemplate="<b>%{x}</b><br>ACWR: %{y:.3f}<extra>Forecast</extra>",
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
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        xaxis=dict(tickfont=dict(size=10, color="#64748B"),
                   gridcolor="rgba(0,82,159,0.06)",
                   showline=False, tickangle=40, nticks=22),
        yaxis=dict(range=[0, 2.3],
                   tickfont=dict(size=10, color="#64748B"),
                   gridcolor="rgba(0,82,159,0.06)",
                   title=dict(text="ACWR", font=dict(size=11, color="#64748B")),
                   zeroline=False),
    )
    return fig
