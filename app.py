"""Calculadora do Lugar Geométrico das Raízes (LGR) — DCA-3701 / UFRN.

Resolve, passo a passo, os 12 passos da apostila "Sistemas de Controle"
(seções 2.2 e 2.4) para um sistema realimentado 1 + G(s)H(s) = 1 + K·P(s) = 0.
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import sympy as sp

from lgr import fmt, plots, steps
from lgr.routh import imaginary_axis_crossings
from lgr.system import OpenLoopSystem, ParseError

st.set_page_config(page_title="Calculadora de LGR", page_icon="📈", layout="wide")

st.title("📈 Calculadora do Lugar Geométrico das Raízes (LGR)")
st.caption(
    "DCA-3701 · Projeto de Sistemas de Controle · UFRN — "
    "resolve explicitamente os 12 passos da apostila para a construção do LGR "
    "e o teste de pertinência de um ponto."
)

with st.sidebar:
    st.header("Sistema em malha aberta")
    with st.form("entrada"):
        g_str = st.text_input(
            "G(s)", value="K*(s+1)/(s^2+13*s)",
            help="Use 's' e o ganho 'K'. Aceita '^' e polinômios/fatores. "
                 "Ex.: K*(s+2)/(s^2+4*s)",
        )
        h_str = st.text_input(
            "H(s)", value="1/s",
            help="Função de transferência do ramo de realimentação. "
                 "Ex.: 1, 1/s, (s+1)/(s+5)",
        )
        st.divider()
        testar = st.checkbox("Testar um ponto sᵢ (item b)", value=True)
        c1, c2 = st.columns(2)
        re_si = c1.number_input("Re{sᵢ}", value=-2.0, step=0.1, format="%.4f")
        im_si = c2.number_input("Im{sᵢ}", value=2.0, step=0.1, format="%.4f")
        enviado = st.form_submit_button("Resolver", type="primary",
                                        width='stretch')

    st.caption(
        "O ganho **K** deve aparecer no numerador (ex.: `K*(s+1)/...`). "
        "Se omitido, assume-se `K` multiplicando todo o numerador."
    )


@st.cache_data(show_spinner="Resolvendo os 12 passos…")
def resolver(g_str: str, h_str: str, re_si: float, im_si: float, testar: bool):
    si = complex(re_si, im_si) if testar else None
    system = OpenLoopSystem(g_str, h_str)
    resultados = steps.solve_all(system, si)
    cruz = imaginary_axis_crossings(system)
    if cruz:
        k_max = 2.0 * max(c["K"] for c in cruz)
    elif si is not None and np.isfinite(system.gain_at(si)):
        k_max = max(4.0 * system.gain_at(si), 10.0)
    else:
        k_max = 50.0
    return system, resultados, float(np.clip(k_max, 1.0, 1e6))


si = complex(re_si, im_si) if testar else None

try:
    system, resultados, k_max = resolver(g_str, h_str, re_si, im_si, testar)
except ParseError as exc:
    st.error(f"⚠️ {exc}")
    st.stop()
except Exception as exc:  # pragma: no cover
    st.exception(exc)
    st.stop()

# ---- resumo ----------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Polos de malha aberta (nₚ)", system.n_p)
col2.metric("Zeros de malha aberta (n_z)", system.n_z)
col3.metric("Ramos ao infinito (nₚ − n_z)", system.relative_degree())
st.latex(r"1 + G(s)H(s) = 1 + K\left[" + sp.latex(system.P) + r"\right] = 0")
st.divider()


# ---- os 12 passos --------------------------------------------------------
def render(r: steps.StepResult) -> None:
    st.subheader(f"Passo {r.n} — {r.titulo}")
    if r.aplicavel:
        st.markdown(f"📖 **Regra da apostila:** {r.regra}")
    else:
        st.info(f"📖 {r.regra}")
    for linha in r.desenvolvimento:
        st.latex(linha)
    if r.figura is not None:
        st.plotly_chart(r.figura, width='stretch', key=f"fig_step_{r.n}")
    if r.resultado:
        (st.success if r.aplicavel else st.caption)(r.resultado)


for r in resultados:
    render(r)
    st.divider()

# ---- LGR interativo completo -------------------------------------------
st.header("🎯 LGR completo (interativo)")
st.caption(
    "Passe o mouse sobre as curvas para ler o par (s, K). "
    "Use o controle abaixo para marcar os polos de malha fechada."
)
step = max(k_max / 200, 1e-6)
k_slider = st.slider("Ganho K", min_value=0.0, max_value=float(round(k_max, 2)),
                     value=float(round(k_max / 2, 2)), step=float(step))
fig = plots.root_locus_figure(system, test_point=si, k_mark=k_slider)
st.plotly_chart(fig, width='stretch', key="fig_lgr_final")

clp = system.closed_loop_poles(k_slider)
st.markdown(f"**Polos de malha fechada** para K = {fmt.num(k_slider)}:")
st.latex(",\\quad ".join(fmt.cplx(p) for p in sorted(clp, key=lambda z: (z.real, z.imag))))

with st.expander("📋 Referência — os 12 passos da apostila"):
    st.markdown(
        r"""
| # | Passo | Ideia |
|---|---|---|
| 1 | Polinômio característico | escrever $1+G(s)H(s)=1+K\,P(s)=0$ |
| 2 | Fatorar $P(s)$ | identificar $n_P$ polos e $n_Z$ zeros |
| 3 | Marcar no plano $s$ | **X** = polos, **O** = zeros |
| 4 | Segmentos do eixo real | à esquerda de nº ímpar de polos+zeros |
| 5 | Lugares separados | $LS = n_P$ (quando $n_P \ge n_Z$) |
| 6 | Simetria | LGR simétrico ao eixo real |
| 7 | Assíntotas | $\sigma_A$ e $\phi_A = \frac{(2q+1)180^\circ}{n_P-n_Z}$ |
| 8 | Pontos de saída/entrada | raízes de $dp(s)/ds = 0$ com $K=p(s)$ |
| 9 | Cruzamento do eixo imaginário | critério de Routh-Hurwitz |
| 10 | Ângulos de partida/chegada | $180^\circ - \sum\theta_i + \sum\phi_i$ |
| 11 | Critério do ângulo de fase | testar se $s_i$ pertence ao LGR |
| 12 | Valor de $K$ em $s_i$ | critério de módulo $K=\frac{\prod|s_i-p_j|}{\prod|s_i-z_k|}$ |
        """
    )
