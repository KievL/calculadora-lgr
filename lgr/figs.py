"""Figuras estáticas (Matplotlib) usadas nas soluções da lista e no PDF."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_POLO = "#1f4e8c"
_ZERO = "#c0392b"
_LGR = "#2e7d32"
_ALVO = "#8e44ad"
_MF = "#e67e22"
_AUX = "#8e8e8e"


def _novo(largura: float = 7.0, altura: float = 4.2):
    fig, ax = plt.subplots(figsize=(largura, altura), dpi=110)
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    return fig, ax


def lgr(num, den, *, alvo: complex | None = None, k_mark: float | None = None,
        zeta: float | None = None, sigma: float | None = None,
        titulo: str = "Lugar Geométrico das Raízes") -> plt.Figure:
    """LGR de 1 + K·num/den = 0, com polos desejados e polos de MF em `k_mark`."""
    import control as ct

    num = np.real(np.asarray(num, dtype=complex))
    den = np.real(np.asarray(den, dtype=complex))
    dados = ct.root_locus_map(ct.tf(num, den))
    loci = np.asarray(dados.loci)
    polos = np.roots(den)
    zeros = np.roots(num) if len(num) > 1 else np.array([])

    pts = list(polos) + list(zeros) + [0j]
    if alvo is not None:
        pts += [alvo, alvo.conjugate()]
    xmin = min(p.real for p in pts) - 2.0
    xmax = max(max(p.real for p in pts) + 1.5, 1.0)
    ymax = max(max(abs(p.imag) for p in pts) + 1.5, 2.5)

    fig, ax = _novo()
    for j in range(loci.shape[1]):
        ax.plot(loci[:, j].real, loci[:, j].imag, color=_LGR, linewidth=1.8,
                label="LGR" if j == 0 else None)
    if zeta is not None:
        r = np.hypot(xmin, ymax) * 2
        th = np.arccos(zeta)
        for sgn in (1, -1):
            ax.plot([0, -r * np.cos(th)], [0, sgn * r * np.sin(th)], color=_AUX,
                    linestyle="--", linewidth=0.9,
                    label=r"$\zeta_{min}$" if sgn == 1 else None)
    if sigma is not None:
        ax.axvline(-sigma, color=_AUX, linestyle=":", linewidth=1.1,
                   label=r"$-(\zeta\omega_n)_{min}$")
    ax.scatter(polos.real, polos.imag, marker="x", s=70, linewidths=2.2, color=_POLO,
               label="Polos de MA", zorder=5)
    if len(zeros):
        ax.scatter(zeros.real, zeros.imag, marker="o", s=70, facecolors="none",
                   linewidths=2.0, edgecolors=_ZERO, label="Zeros de MA", zorder=5)
    if k_mark is not None:
        mf = np.roots(np.polyadd(den, k_mark * num))
        ax.scatter(mf.real, mf.imag, marker="D", s=38, color=_MF, zorder=6,
                   label="Polos de MF (projeto)")
    if alvo is not None:
        ax.scatter([alvo.real] * 2, [alvo.imag, -alvo.imag], marker="s", s=110,
                   facecolors="none", edgecolors=_ALVO, linewidths=2.0, zorder=7,
                   label=r"$s_d$ desejado")
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.axvline(0, color="#999", linewidth=0.8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("Eixo Real")
    ax.set_ylabel("Eixo Imaginário")
    ax.set_title(titulo, fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def degrau(curvas: list[dict], *, faixa: float = 0.05, mp_max: float | None = None,
           ts_max: float | None = None, titulo: str = "Resposta ao degrau unitário") -> plt.Figure:
    """curvas: [{t, y, rotulo, y_inf}] — a primeira é a curva principal."""
    fig, ax = _novo(7.0, 3.8)
    cores = [_LGR, _AUX, _POLO]
    for i, c in enumerate(curvas):
        ax.plot(c["t"], c["y"], color=cores[i % len(cores)],
                linewidth=2.0 if i == 0 else 1.3,
                linestyle="-" if i == 0 else "--", label=c["rotulo"])
    y_inf = curvas[0]["y_inf"]
    ax.axhspan(y_inf * (1 - faixa), y_inf * (1 + faixa), color=_LGR, alpha=0.08,
               label=f"Faixa de ±{faixa * 100:.0f}%")
    if mp_max is not None:
        ax.axhline(y_inf * (1 + mp_max / 100.0), color=_ZERO, linestyle=":",
                   linewidth=1.1, label=f"$M_P$ = {mp_max:g}%")
    if ts_max is not None:
        ax.axvline(ts_max, color=_ALVO, linestyle=":", linewidth=1.1,
                   label=f"$t_s$ = {ts_max:g} s")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Saída c(t)")
    ax.set_title(titulo, fontsize=11)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.set_xlim(0, float(curvas[0]["t"][-1]))
    fig.tight_layout()
    plt.close(fig)
    return fig


def plano_z(polos, zeros, titulo: str = "Polos e zeros no plano z") -> plt.Figure:
    polos = np.asarray(polos, dtype=complex)
    zeros = np.asarray(zeros, dtype=complex)
    fig, ax = _novo(5.6, 5.0)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=_AUX, linestyle="--", linewidth=1.0,
            label="Círculo unitário")
    if len(polos):
        ax.scatter(polos.real, polos.imag, marker="x", s=70, linewidths=2.2,
                   color=_POLO, label="Polos", zorder=5)
    if len(zeros):
        ax.scatter(zeros.real, zeros.imag, marker="o", s=70, facecolors="none",
                   linewidths=2.0, edgecolors=_ZERO, label="Zeros", zorder=5)
    lim = max([1.3] + [abs(p) + 0.4 for p in list(polos) + list(zeros)])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.axvline(0, color="#999", linewidth=0.8)
    ax.set_xlabel("Re{z}")
    ax.set_ylabel("Im{z}")
    ax.set_title(titulo, fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    plt.close(fig)
    return fig
