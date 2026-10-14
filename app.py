"""Calculadora de Sistemas de Controle — DCA-3701 / UFRN.

- 2ª Unidade, 2º Exercício: Questão 1 (PID com z1 = z2 + Tustin) e Questão 2
  (G_c = a/(s+b) + Euler), com geração da solução em PDF;
- 1ª Unidade: os 12 passos da construção do LGR (seções 2.2 e 2.4 da apostila).
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import sympy as sp

from lgr import exercicios, fmt, plots, report, steps
from lgr.discrete import METODOS
from lgr.routh import imaginary_axis_crossings
from lgr.system import OpenLoopSystem, ParseError

st.set_page_config(page_title="Sistemas de Controle — DCA-3701", page_icon="📈", layout="wide")

_METODO_NOMES = {k: v["nome"] for k, v in METODOS.items()}
_QUESTOES = {
    "q1": "Questão 1 — PID (z₁ = z₂) + discretização",
    "q2": "Questão 2 — a/(s+b) + discretização",
}
_TIPOS_MANUAIS = {
    "q1": "PID (z₁ = z₂) + discretização — como na Questão 1",
    "q2": "a/(s+b) + discretização — como na Questão 2",
}
_PRESETS = {"q1": exercicios.DEFAULTS_Q1, "q2": exercicios.DEFAULTS_Q2}

with st.sidebar:
    unidade = st.segmented_control("Unidade", ["1ª Unidade", "2ª Unidade"],
                                   default="2ª Unidade", required=True)
    st.divider()


# --------------------------------------------------------------------------- #
# 2ª Unidade — soluções da lista
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Resolvendo a Questão 1…")
def solucao_q1(g, h, mp, ts, criterio, T, metodo):
    return exercicios.resolver_q1(g, h, mp, ts, criterio, T, metodo)


@st.cache_resource(show_spinner="Resolvendo a Questão 2…")
def solucao_q2(g, h, re, im, T, metodo):
    return exercicios.resolver_q2(g, h, re, im, T, metodo)


def _resolver(chave: str, p: dict):
    if chave == "q1":
        return solucao_q1(p["g"], p["h"], p["mp"], p["ts"], p["criterio"], p["T"], p["metodo"])
    return solucao_q2(p["g"], p["h"], p["re"], p["im"], p["T"], p["metodo"])


def _resumo_preset(chave: str) -> None:
    p = _PRESETS[chave]
    linhas = [f"**G(s)** = `{p['g']}`", f"**H(s)** = `{p['h']}`"]
    if chave == "q1":
        linhas += [f"**Mₚ** ≤ {fmt.num(p['mp'], 0)}%  ·  **tₛ({p['criterio']:g}%)** < "
                   f"{fmt.num(p['ts'], 1)} s"]
    else:
        linhas += ["**G_C(s)** = a/(s+b)",
                   f"**Polos de MF:** {fmt.num(p['re'])} ± {fmt.num(p['im'])}i"]
    linhas += [f"**Discretização:** {_METODO_NOMES[p['metodo']]}, T = {fmt.num(p['T'], 1)} s"]
    with st.sidebar.container(border=True):
        st.caption("Parâmetros da lista (preset)")
        st.markdown("  \n".join(linhas))


def _form_manual(chave: str) -> dict | None:
    """Formulário vazio; devolve os parâmetros após o envio ou None."""
    salvo = st.session_state.get(f"manual_{chave}")
    with st.sidebar.form(f"form_manual_{chave}"):
        st.subheader("Parâmetros do problema")
        g = st.text_input("G(s)", placeholder="ex.: 5*(s+3)/(s*(s+4))")
        h = st.text_input("H(s)", placeholder="ex.: 1/(s+1)  ou  1")
        if chave == "q1":
            c1, c2 = st.columns(2)
            mp = c1.number_input("Mₚ máx. (%)", 0.1, 99.0, None, 1.0, placeholder="ex.: 10")
            ts = c2.number_input("tₛ máx. (s)", 0.01, 1e3, None, 0.5, placeholder="ex.: 3")
            criterio = st.selectbox(
                "Critério de tₛ", [5.0, 2.0],
                format_func=lambda c: f"{c:g}%  →  tₛ = {3 if c == 5 else 4}/(ζωₙ)")
        else:
            st.markdown("G_C(s) = a/(s+b) · polos desejados de MF: **s_d = Re ± j·Im**")
            c1, c2 = st.columns(2)
            re = c1.number_input("Re{s_d}", -1e3, -1e-6, None, 0.1, format="%.4f",
                                 placeholder="ex.: -2,5")
            im = c2.number_input("Im{s_d}", 1e-6, 1e3, None, 0.1, format="%.4f",
                                 placeholder="ex.: 2")
        st.divider()
        T = st.number_input("Período de amostragem T (s)", 0.001, 100.0, None, 0.1,
                            placeholder="ex.: 1")
        metodo = st.selectbox("Método de discretização", list(METODOS),
                              format_func=_METODO_NOMES.get)
        enviado = st.form_submit_button("Resolver", type="primary", width="stretch")
    if not enviado:
        return salvo
    campos = {"G(s)": g.strip(), "H(s)": h.strip(), "T": T}
    if chave == "q1":
        campos |= {"Mₚ": mp, "tₛ": ts}
    else:
        campos |= {"Re{s_d}": re, "Im{s_d}": im}
    faltando = [k for k, v in campos.items() if v in (None, "")]
    if faltando:
        st.sidebar.error("Preencha: " + ", ".join(faltando))
        return salvo
    p = {"g": g.strip(), "h": h.strip(), "T": T, "metodo": metodo}
    p |= {"mp": mp, "ts": ts, "criterio": criterio} if chave == "q1" else {"re": re, "im": im}
    st.session_state[f"manual_{chave}"] = p
    return p


_CAIXA = ("<div style='background:{bg};border-left:4px solid {borda};padding:0.6rem 0.9rem;"
          "border-radius:4px;margin:0.4rem 0 0.8rem 0;color:#1d1d1d'>{html}</div>")


def _render_itens(itens) -> None:
    for tipo, valor in itens:
        if tipo == "txt":
            st.markdown(valor, unsafe_allow_html=True)
        elif tipo == "eq":
            st.latex(fmt.tex(valor))
        elif tipo == "fig":
            st.pyplot(valor, clear_figure=False, width="content")
        elif tipo == "ok":
            st.markdown(_CAIXA.format(bg="#e8f5e9", borda="#2e7d32", html="✅ " + valor),
                        unsafe_allow_html=True)
        elif tipo == "nota":
            st.markdown(_CAIXA.format(bg="#fff8e1", borda="#e6a100",
                                      html="<b>Observação:</b> " + valor),
                        unsafe_allow_html=True)


def _botao_pdf(coluna, rotulo: str, chave_pdf: str, arquivo: str, gerar) -> None:
    if coluna.button(f"Gerar {rotulo}", width="stretch", key=f"gerar_{chave_pdf}"):
        with st.spinner("Gerando PDF…"):
            st.session_state[chave_pdf] = gerar()
    if chave_pdf in st.session_state:
        coluna.download_button(f"⬇️ Baixar {rotulo}", st.session_state[chave_pdf],
                               file_name=arquivo, mime="application/pdf", type="primary",
                               width="stretch", key=f"baixar_{chave_pdf}")


def _secao_pdf(chave: str, sol, manual: bool, p: dict) -> None:
    st.header("📄 Gerar a solução em PDF")
    st.caption("O PDF identifica a disciplina DCA-3701, a 2ª Unidade e o discente "
               f"{report.DISCENTE}.")
    nome = "Kiev_Luiz_Freitas_Guedes_DCA3701_U2"
    if st.session_state.get("pdf_params") != (chave, manual, p):
        for k in ("pdf_questao", "pdf_lista"):
            st.session_state.pop(k, None)
        st.session_state["pdf_params"] = (chave, manual, p)
    if manual:
        _botao_pdf(st, "PDF deste problema", "pdf_questao", f"{nome}_problema_manual.pdf",
                   lambda: report.gerar_pdf([sol], manual=True))
        return
    c1, c2 = st.columns(2)
    _botao_pdf(c1, f"PDF da Questão {sol.numero}", "pdf_questao",
               f"{nome}_Exercicio2_Q{sol.numero}.pdf", lambda: report.gerar_pdf([sol]))
    _botao_pdf(c2, "PDF completo (Questões 1 e 2)", "pdf_lista", f"{nome}_Exercicio2.pdf",
               lambda: report.gerar_pdf([_resolver(k, _PRESETS[k]) for k in ("q1", "q2")]))


def pagina_unidade2() -> None:
    with st.sidebar:
        modo = st.radio("Parâmetros", ["preset", "manual"],
                        format_func={"preset": "Preset da lista (2º Exercício)",
                                     "manual": "Inserir manualmente"}.get)
        chave = st.radio("Tipo de problema" if modo == "manual" else "Questão", list(_QUESTOES),
                         format_func=(_TIPOS_MANUAIS if modo == "manual" else _QUESTOES).get,
                         key="questao")
    manual = modo == "manual"
    if manual:
        p = _form_manual(chave)
        if p is None:
            st.title(f"Entrada manual — {_TIPOS_MANUAIS[chave]}")
            st.info("Preencha os parâmetros na barra lateral e clique em **Resolver**.")
            return
    else:
        p = dict(_PRESETS[chave])
        _resumo_preset(chave)

    try:
        sol = _resolver(chave, p)
    except (ParseError, ValueError) as exc:
        st.error(f"⚠️ {exc}")
        return

    rotulo = "Problema" if manual else f"Questão {sol.numero}"
    st.title(f"{rotulo} — {sol.titulo}")
    origem = ("parâmetros informados manualmente" if manual
              else "parâmetros do 2º Exercício da 2ª Unidade")
    st.caption(f"DCA-3701 · Projeto de Sistemas de Controle · UFRN — 2ª Unidade, {origem}. "
               "Solução passo a passo com base na apostila *Sistemas de Controle* "
               "(cap. 4 — projeto pelo LGR; cap. 5 — aproximação discreta).")
    with st.container(border=True):
        st.markdown("**Enunciado**")
        _render_itens(sol.enunciado.itens)
    for parte in sol.partes:
        st.header(parte.titulo)
        for passo in parte.passos:
            st.subheader(passo.titulo)
            _render_itens(passo.itens)
        st.divider()
    with st.container(border=True):
        st.markdown("### Respostas" + ("" if manual else f" da Questão {sol.numero}"))
        for r in sol.respostas:
            st.markdown(r, unsafe_allow_html=True)
    st.divider()
    _secao_pdf(chave, sol, manual, p)


# --------------------------------------------------------------------------- #
# 1ª Unidade — LGR em 12 passos
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Resolvendo os 12 passos…")
def resolver_lgr(g_str: str, h_str: str, re_si: float, im_si: float, testar: bool):
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


def _render_passo(r: steps.StepResult) -> None:
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


def pagina_lgr() -> None:
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
            st.form_submit_button("Resolver", type="primary", width='stretch')

        st.caption(
            "O ganho **K** deve aparecer no numerador (ex.: `K*(s+1)/...`). "
            "Se omitido, assume-se `K` multiplicando todo o numerador."
        )

    st.title("📈 Calculadora do Lugar Geométrico das Raízes (LGR)")
    st.caption(
        "DCA-3701 · Projeto de Sistemas de Controle · UFRN — "
        "resolve explicitamente os 12 passos da apostila para a construção do LGR "
        "e o teste de pertinência de um ponto."
    )
    si = complex(re_si, im_si) if testar else None

    try:
        system, resultados, k_max = resolver_lgr(g_str, h_str, re_si, im_si, testar)
    except ParseError as exc:
        st.error(f"⚠️ {exc}")
        st.stop()
    except Exception as exc:  # pragma: no cover
        st.exception(exc)
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Polos de malha aberta (nₚ)", system.n_p)
    col2.metric("Zeros de malha aberta (n_z)", system.n_z)
    col3.metric("Ramos ao infinito (nₚ − n_z)", system.relative_degree())
    st.latex(r"1 + G(s)H(s) = 1 + K\left[" + sp.latex(system.P) + r"\right] = 0")
    st.divider()

    for r in resultados:
        _render_passo(r)
        st.divider()

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


if unidade == "1ª Unidade":
    pagina_lgr()
else:
    pagina_unidade2()
