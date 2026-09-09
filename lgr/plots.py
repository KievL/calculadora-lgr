"""Figuras Plotly para o LGR."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

_POLO = "#1f4e8c"
_ZERO = "#c0392b"
_LGR = "#2e7d32"
_ASSINT = "#8e8e8e"
_PONTO = "#8e44ad"


def _base_layout(fig: go.Figure, titulo: str) -> None:
    fig.update_layout(
        title=titulo,
        template="plotly_white",
        xaxis_title="Eixo Real",
        yaxis_title="Eixo Imaginário",
        height=460,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, zeroline=True, zerolinecolor="#bbb")
    fig.update_xaxes(zeroline=True, zerolinecolor="#bbb")


def _limits(system, pad: float = 2.0):
    pts = list(system.poles) + list(system.zeros)
    xs = [p.real for p in pts] + [0.0]
    ys = [p.imag for p in pts] + [0.0]
    xmin, xmax = min(xs), max(xs)
    ymax = max(abs(min(ys)), abs(max(ys)), 1.0)
    return (xmin - pad, xmax + pad), (-ymax - pad, ymax + pad)


def _add_poles_zeros(fig: go.Figure, system) -> None:
    if system.poles:
        fig.add_trace(go.Scatter(
            x=[p.real for p in system.poles], y=[p.imag for p in system.poles],
            mode="markers", name="Polos de MA",
            marker=dict(symbol="x-thin", size=12, line=dict(width=3, color=_POLO)),
        ))
    if system.zeros:
        fig.add_trace(go.Scatter(
            x=[z.real for z in system.zeros], y=[z.imag for z in system.zeros],
            mode="markers", name="Zeros de MA",
            marker=dict(symbol="circle-open", size=12, line=dict(width=3, color=_ZERO)),
        ))


def pole_zero_figure(system, *, segments=None, asymptotes=None, test_point=None,
                     titulo="Polos e zeros de malha aberta") -> go.Figure:
    fig = go.Figure()
    (x0, x1), (y0, y1) = _limits(system)

    if segments:
        first = True
        for a, bnd in segments:
            a_plot = a if np.isfinite(a) else x0
            fig.add_trace(go.Scatter(
                x=[a_plot, bnd], y=[0, 0], mode="lines",
                line=dict(color=_LGR, width=6), opacity=0.5,
                name="Segmento do eixo real (LGR)", legendgroup="seg",
                showlegend=first,
            ))
            first = False

    if asymptotes:
        sigma = asymptotes["sigma"]
        first = True
        for ang in asymptotes["angulos_deg"]:
            r = max(abs(x1 - x0), abs(y1 - y0))
            dx, dy = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            fig.add_trace(go.Scatter(
                x=[sigma, sigma + r * dx], y=[0, r * dy], mode="lines",
                line=dict(color=_ASSINT, width=1, dash="dash"),
                name="Assíntotas", legendgroup="asy", showlegend=first,
            ))
            first = False
        fig.add_trace(go.Scatter(
            x=[sigma], y=[0], mode="markers", name="σ_A (centroide)",
            marker=dict(symbol="triangle-up", size=10, color=_ASSINT),
        ))

    _add_poles_zeros(fig, system)

    if test_point is not None:
        fig.add_trace(go.Scatter(
            x=[test_point.real], y=[test_point.imag], mode="markers",
            name="Ponto de teste sᵢ",
            marker=dict(symbol="square-open", size=13, line=dict(width=3, color=_PONTO)),
        ))

    _base_layout(fig, titulo)
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(range=[y0, y1])
    return fig


def root_locus_figure(system, *, test_point=None, k_mark=None,
                      titulo="Lugar Geométrico das Raízes (LGR)") -> go.Figure:
    import control as ct

    fig = go.Figure()
    plant = ct.tf(system.num_coeffs.real,
                  system.den_coeffs.real)
    data = ct.root_locus_map(plant)
    loci = np.asarray(data.loci)
    gains = np.asarray(data.gains)

    for j in range(loci.shape[1]):
        fig.add_trace(go.Scatter(
            x=loci[:, j].real, y=loci[:, j].imag, mode="lines",
            line=dict(color=_LGR, width=2), name="Ramo do LGR",
            legendgroup="lgr", showlegend=(j == 0),
            customdata=gains,
            hovertemplate="s = %{x:.3f} %{y:+.3f}i<br>K = %{customdata:.4g}<extra></extra>",
        ))

    _add_poles_zeros(fig, system)

    if k_mark is not None:
        clp = system.closed_loop_poles(k_mark)
        fig.add_trace(go.Scatter(
            x=clp.real, y=clp.imag, mode="markers",
            name=f"Polos de MF (K={k_mark:.4g})",
            marker=dict(symbol="diamond", size=10, color="#e67e22"),
        ))

    if test_point is not None:
        fig.add_trace(go.Scatter(
            x=[test_point.real], y=[test_point.imag], mode="markers",
            name="Ponto de teste sᵢ",
            marker=dict(symbol="square-open", size=13, line=dict(width=3, color=_PONTO)),
        ))

    (x0, x1), (y0, y1) = _limits(system)
    _base_layout(fig, titulo)
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(range=[y0, y1])
    return fig


def angle_vectors_figure(system, test_point, contribs,
                         titulo="Critério de ângulo em sᵢ") -> go.Figure:
    """contribs: lista de dicts {tipo:'polo'|'zero', ponto:complex, ang:float}."""
    fig = go.Figure()
    si = test_point
    for c in contribs:
        p = c["ponto"]
        cor = _ZERO if c["tipo"] == "zero" else _POLO
        fig.add_trace(go.Scatter(
            x=[p.real, si.real], y=[p.imag, si.imag], mode="lines",
            line=dict(color=cor, width=1.5),
            name=f"{'ϕ' if c['tipo'] == 'zero' else 'θ'} = {c['ang']:.2f}°",
        ))
    _add_poles_zeros(fig, system)
    fig.add_trace(go.Scatter(
        x=[si.real], y=[si.imag], mode="markers", name="sᵢ",
        marker=dict(symbol="square-open", size=13, line=dict(width=3, color=_PONTO)),
    ))
    (x0, x1), (y0, y1) = _limits(system)
    _base_layout(fig, titulo)
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(range=[y0, y1])
    return fig
