"""Soluções do 2º Exercício da 2ª Unidade (DCA-3701), passo a passo.

Questão 1 — PID com zeros iguais (apostila 4.1 e 4.4) + discretização (cap. 5).
Questão 2 — G_c(s) = a/(s+b) pelos critérios de ângulo e módulo + discretização.

Cada solução é uma lista de partes/passos com itens ("txt", "eq", "fig", "ok",
"nota") que são renderizados tanto pelo Streamlit quanto pelo gerador de PDF.
As equações usam o subconjunto de LaTeX comum ao KaTeX e ao mathtext.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from . import design, discrete, figs, fmt
from .system import OpenLoopSystem, s

DEFAULTS_Q1 = {"g": "5*(s+3)/(s*(s+4))", "h": "1/(s+1)", "mp": 10.0, "ts": 3.0,
               "criterio": 5.0, "T": 2.0, "metodo": "tustin"}
DEFAULTS_Q2 = {"g": "2*(s+1)/(s^2+2*s+2)", "h": "(s+3)/(s+5)", "re": -2.5, "im": 2.0,
               "T": 1.0, "metodo": "euler"}


@dataclass
class Passo:
    titulo: str
    itens: list[tuple[str, Any]] = field(default_factory=list)

    def txt(self, html: str) -> Passo:
        self.itens.append(("txt", html))
        return self

    def eq(self, *latex: str) -> Passo:
        self.itens.extend(("eq", e) for e in latex)
        return self

    def fig(self, figura) -> Passo:
        self.itens.append(("fig", figura))
        return self

    def ok(self, html: str) -> Passo:
        self.itens.append(("ok", html))
        return self

    def nota(self, html: str) -> Passo:
        self.itens.append(("nota", html))
        return self


@dataclass
class Parte:
    titulo: str
    passos: list[Passo] = field(default_factory=list)


@dataclass
class Solucao:
    numero: int
    titulo: str
    enunciado: Passo
    partes: list[Parte] = field(default_factory=list)
    respostas: list[str] = field(default_factory=list)
    dados: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# helpers de formatação
# --------------------------------------------------------------------------- #

def _tf_coeffs(expr: sp.Expr) -> tuple[np.ndarray, np.ndarray]:
    num, den = sp.fraction(sp.cancel(sp.together(expr)))
    n = np.array([float(c) for c in sp.Poly(num, s).all_coeffs()])
    d = np.array([float(c) for c in sp.Poly(den, s).all_coeffs()])
    return n, d


def _tf_latex(expr: sp.Expr) -> str:
    return sp.latex(sp.factor(expr)).replace(r"\frac", r"\dfrac")


def _frac(num, den, var: str = "s") -> str:
    return f"\\dfrac{{{fmt.poly_str(num, var)}}}{{{fmt.poly_str(den, var)}}}"


def _deg(x: float) -> str:
    return fmt.deg_raw(x)


def _menos(p: complex) -> str:
    """'s_d - p' com sinal adequado: p=-3 → '+ 3', p=0 → ''."""
    p = complex(p)
    if abs(p) < 1e-12:
        return ""
    if abs(p.imag) < 1e-9:
        return f" + {fmt.num(-p.real)}" if p.real < 0 else f" - {fmt.num(p.real)}"
    return f" - ({fmt.cplx(p)})"


def _fatores(raizes, S: str) -> str:
    """Produto de fatores (S - r) agrupando pares complexos e multiplicidades."""
    raizes = sorted((complex(r) for r in raizes), key=lambda r: (-r.real, -r.imag))
    usados = [False] * len(raizes)
    partes: list[tuple[str, int]] = []

    def potencia(base: str, k: int) -> str:
        return base if k == 1 else f"{base}^{{{k}}}"

    sq = "s" if S == "s" else f"\\left({S}\\right)"
    for i, r in enumerate(raizes):
        if usados[i]:
            continue
        usados[i] = True
        if abs(r.imag) > 1e-9:
            for j in range(i + 1, len(raizes)):
                if not usados[j] and abs(raizes[j] - r.conjugate()) < 1e-6:
                    usados[j] = True
                    break
            a1 = -2 * r.real
            a0 = abs(r) ** 2
            lin = "" if abs(a1) < 1e-12 else (
                f" + {fmt.num(a1)}{S if S == 's' else sq}" if a1 > 0
                else f" - {fmt.num(-a1)}{S if S == 's' else sq}")
            partes.append((f"\\left({potencia(sq, 2)}{lin} + {fmt.num(a0)}\\right)", 1))
            continue
        if abs(r.real) < 1e-12:
            base = S if S == "s" else sq
        else:
            sinal = "+" if r.real < 0 else "-"
            base = f"\\left({S} {sinal} {fmt.num(abs(r.real))}\\right)"
        for k, (b, m) in enumerate(partes):
            if b == base:
                partes[k] = (b, m + 1)
                break
        else:
            partes.append((base, 1))
    if not partes:
        return "1"
    return "".join(potencia(b, m) for b, m in partes)


def _zpk_latex(ganho: float, zeros, polos, S: str = "s") -> str:
    g = fmt.num(ganho)
    return f"{g}\\,\\dfrac{{{_fatores(zeros, S)}}}{{{_fatores(polos, S)}}}"


def _subst_latex(metodo: str, T: float) -> str:
    """Expressão que substitui s, já com o valor de T."""
    if metodo == "euler":
        return "z-1" if abs(T - 1.0) < 1e-12 else f"\\dfrac{{z-1}}{{{fmt.num(T)}}}"
    if metodo == "backward":
        return "\\dfrac{z-1}{z}" if abs(T - 1.0) < 1e-12 else f"\\dfrac{{z-1}}{{{fmt.num(T)}\\,z}}"
    alfa = 2.0 / T
    return "\\dfrac{z-1}{z+1}" if abs(alfa - 1.0) < 1e-12 else \
        f"{fmt.num(alfa)}\\,\\dfrac{{z-1}}{{z+1}}"


def _contrib_eqs(thetas, phis, sd: complex) -> list[str]:
    linhas = []
    for i, t in enumerate(thetas, start=1):
        linhas.append(
            f"\\theta_{{{i}}} = \\angle\\left(s_d{_menos(t['ponto'])}\\right) = "
            f"\\angle\\left({fmt.cplx(sd - t['ponto'])}\\right) = {_deg(t['ang'])}")
    for i, f in enumerate(phis, start=1):
        linhas.append(
            f"\\phi_{{{i}}} = \\angle\\left(s_d{_menos(f['ponto'])}\\right) = "
            f"\\angle\\left({fmt.cplx(sd - f['ponto'])}\\right) = {_deg(f['ang'])}")
    return linhas


def _passo_discretizacao(titulo: str, ganho: float, zeros, polos, num_s, den_s,
                         T: float, metodo: str, nome: str, y: str, u: str):
    d = discrete.discretizar(num_s, den_s, T, metodo)
    info = discrete.METODOS[metodo]
    S = _subst_latex(metodo, T)
    p = Passo(titulo)
    p.txt(f"Pelo método <b>{info['nome']}</b> (apostila, seção 5.2, eq. {info['eq']}), "
          f"substitui-se s na função de transferência contínua, com período de "
          f"amostragem T = {fmt.num(T, 1)} s:")
    p.eq(f"{info['subst']} = {S}")
    p.eq(f"{nome}(s) = {_zpk_latex(ganho, zeros, polos)}")
    p.eq(f"{nome}(z) = {_zpk_latex(ganho, zeros, polos, S)}")
    p.txt("Multiplicando numerador e denominador pela maior potência do denominador "
          "da substituição e normalizando o denominador (mônico):")
    p.eq(f"{nome}(z) = {_frac(d['num_z'], d['den_z'], 'z')}")
    p.txt("Em potências de z<sup>-1</sup>:")
    p.eq(f"{nome}(z) = \\dfrac{{{y.upper()}(z)}}{{{u.upper()}(z)}} = {discrete.forma_zinv(d)}")
    p.txt("Equação de diferenças correspondente (implementação digital):")
    p.eq(discrete.eq_diferencas(d, y, u))
    if len(d["polos_z"]):
        p.txt("Polos do sistema discreto e sua posição em relação ao círculo unitário:")
        for zp in sorted(d["polos_z"], key=lambda w: (w.real, w.imag)):
            pos = "dentro do" if abs(zp) < 1 - 1e-9 else (
                "sobre o" if abs(abs(zp) - 1) < 1e-9 else "fora do")
            p.eq(f"z = {fmt.cplx(zp)}\\quad |z| = {fmt.num(abs(zp))}\\quad "
                 f"\\text{{({pos} círculo unitário)}}")
    return p, d


# --------------------------------------------------------------------------- #
# Questão 1 — PID com z1 = z2
# --------------------------------------------------------------------------- #

def resolver_q1(g: str, h: str, mp: float, ts: float, criterio: float,
                T: float, metodo: str) -> Solucao:
    sistema = OpenLoopSystem(g, h)
    num_g, den_g = _tf_coeffs(sistema.G)
    num_h, den_h = _tf_coeffs(sistema.H)
    num_l = np.real(sistema.num_coeffs)
    den_l = np.real(sistema.den_coeffs)
    crit_txt = f"{criterio:g}%"
    crit_tex = crit_txt.replace("%", "\\%")

    en = Passo("Enunciado")
    en.eq(f"G(s) = {_tf_latex(sistema.G)};\\qquad H(s) = {_tf_latex(sistema.H)}")
    en.txt(f"<b>a)</b> Projetar um controlador PID, com z<sub>1</sub> = z<sub>2</sub>, para "
           f"que em MF o sistema apresente M<sub>P</sub>(%) ≤ {fmt.num(mp, 0)}% e tempo de "
           f"acomodação ({crit_txt}) inferior a {fmt.num(ts, 1)} s.")
    en.txt(f"<b>b)</b> Encontre uma aproximação discretizada para o controlador encontrado "
           f"utilizando um período de amostragem de {fmt.num(T, 1)} s e o método "
           f"{discrete.METODOS[metodo]['nome']}.")
    sol = Solucao(1, "Projeto de controlador PID (z₁ = z₂) e discretização", en)

    # ---- (a) ----------------------------------------------------------------
    pa = Parte("a) Projeto do controlador PID com z₁ = z₂")
    esp = design.specs_to_poles(mp, ts, criterio)
    sd = esp["sd"]

    p1 = Passo("Passo 1 — Especificações → polos dominantes desejados")
    p1.txt("Apostila, seção 4.1: as especificações são traduzidas em termos de um sistema "
           "de 2ª ordem; M<sub>P</sub> ≤ M impõe ζ ≥ ζ<sub>min</sub> e t<sub>s</sub> ≤ T impõe "
           "ζω<sub>n</sub> ≥ (ζω<sub>n</sub>)<sub>min</sub>.")
    p1.eq(
        f"\\zeta_{{min}} = \\dfrac{{\\left|\\ln\\left(\\dfrac{{M_P}}{{100}}\\right)\\right|}}"
        f"{{\\sqrt{{\\pi^2 + \\ln^2\\left(\\dfrac{{M_P}}{{100}}\\right)}}}} = "
        f"\\dfrac{{\\left|\\ln({fmt.num(mp / 100, 2)})\\right|}}"
        f"{{\\sqrt{{\\pi^2 + ({fmt.num(esp['ln'])})^2}}}} = {fmt.num(esp['zeta'])}",
        f"t_s({crit_tex}) = \\dfrac{{{esp['num_ts']:g}}}{{\\zeta\\omega_n}} \\leq {fmt.num(ts, 1)}"
        f"\\;\\Rightarrow\\; (\\zeta\\omega_n)_{{min}} = \\dfrac{{{esp['num_ts']:g}}}{{{fmt.num(ts, 1)}}}"
        f" = {fmt.num(esp['sigma'])}",
        f"\\omega_n = \\dfrac{{(\\zeta\\omega_n)_{{min}}}}{{\\zeta_{{min}}}} = "
        f"\\dfrac{{{fmt.num(esp['sigma'])}}}{{{fmt.num(esp['zeta'])}}} = {fmt.num(esp['wn'])}"
        f"\\ \\mathrm{{rad/s}}",
        f"\\omega_d = \\omega_n\\sqrt{{1-\\zeta^2}} = {fmt.num(esp['wn'])}"
        f"\\sqrt{{1-{fmt.num(esp['zeta'])}^2}} = {fmt.num(esp['wd'])}\\ \\mathrm{{rad/s}}",
        f"s_d = -\\zeta\\omega_n \\pm j\\,\\omega_d = {fmt.cplx_pm(sd)}",
    )
    p1.ok(f"Polos dominantes desejados de MF: s<sub>d</sub> = {fmt.num(sd.real)} ± "
          f"{fmt.num(sd.imag)}i (ζ = {fmt.num(esp['zeta'])}, ω<sub>n</sub> = "
          f"{fmt.num(esp['wn'])} rad/s).")
    pa.passos.append(p1)

    ang_planta = sistema.angle_deg_at(sd)
    p2 = Passo("Passo 2 — Verificar se um controlador mais simples (P) é suficiente")
    p2.txt("Com a malha aberta G(s)H(s), o LGR só passa por s<sub>d</sub> se a condição de "
           "ângulo for satisfeita (apostila, passo 11 do LGR):")
    p2.eq(f"G(s)H(s) = {_zpk_latex(sistema.b, sistema.zeros, sistema.poles)}",
          f"\\angle G(s_d)H(s_d) = {_deg(ang_planta)} \\neq \\pm 180^\\circ")
    if abs(abs(ang_planta) - 180.0) < 1.0:  # pragma: no cover
        p2.nota("O LGR original já passa por s<sub>d</sub>; um controlador P seria suficiente, "
                "mas o enunciado pede um PID.")
    else:
        p2.ok("O LGR do sistema não compensado não passa por s<sub>d</sub>: um controlador "
              "proporcional não atende às especificações. Usa-se o PID pedido no enunciado.")
    pa.passos.append(p2)

    pid = design.pid_zeros_iguais(sistema, sd)
    z = pid["z"]
    p3 = Passo("Passo 3 — Condição de ângulo: polo na origem e zeros iguais")
    p3.txt("Apostila, seção 4.4 (eq. 4.4). Com dois zeros idênticos, o número de incógnitas "
           "(K<sub>c</sub> e z) fica igual ao número de equações (ângulo e módulo):")
    p3.eq("G_c(s) = K_p + \\dfrac{K_i}{s} + K_d s = K_c\\,\\dfrac{(s+z_1)(s+z_2)}{s}"
          "= K_c\\,\\dfrac{(s+z)^2}{s}")
    p3.txt("Contribuições angulares dos polos (θ) e zeros (φ) de G(s)H(s) em s<sub>d</sub>:")
    p3.eq(*_contrib_eqs(pid["thetas"], pid["phis"], sd))
    p3.txt("Polo do PID na origem:")
    p3.eq(f"\\theta_c = \\angle\\left(s_d\\right) = \\angle\\left({fmt.cplx(sd)}\\right) = "
          f"{_deg(pid['theta_c'])}")
    soma_t_txt = _deg(pid["soma_t"])
    soma_f_txt = _deg(pid["soma_f"])
    q_txt = ""
    if abs(pid["excesso"] / 2.0 - pid["phi_c"]) > 1e-6:
        q_txt = f" + {fmt.num((2 * pid['phi_c'] - pid['excesso']) / 360.0, 0)}\\cdot 360^\\circ"
    p3.eq(
        "\\left(\\sum\\theta_i + \\theta_c\\right) - \\sum\\phi_j - 2\\phi_c = 180^\\circ \\pm q\\,360^\\circ",
        f"\\phi_c = \\dfrac{{\\left(\\sum\\theta_i + \\theta_c\\right) - \\sum\\phi_j - 180^\\circ{q_txt}}}{{2}} = "
        f"\\dfrac{{{soma_t_txt} - {soma_f_txt} - 180^\\circ{q_txt}}}{{2}} = {_deg(pid['phi_c'])}",
        f"\\tan\\phi_c = \\dfrac{{\\omega_d}}{{z - \\zeta\\omega_n}}\\;\\Rightarrow\\;"
        f"z = \\dfrac{{\\omega_d}}{{\\tan\\phi_c}} + \\zeta\\omega_n = "
        f"\\dfrac{{{fmt.num(sd.imag)}}}{{\\tan({_deg(pid['phi_c'])})}} + {fmt.num(-sd.real)}"
        f" = {fmt.num(z)}",
    )
    p3.ok(f"Zeros do PID: z<sub>1</sub> = z<sub>2</sub> = {fmt.num(z)} "
          f"(zeros em s = {fmt.num(-z)}).")
    pa.passos.append(p3)

    p4 = Passo("Passo 4 — Condição de módulo: ganho Kc")
    p4.txt("Aplicando |G<sub>c</sub>(s)G(s)H(s)| = 1 em s = s<sub>d</sub> "
           "(apostila, passo 12 do LGR):")
    mods_p = " \\cdot ".join([fmt.num(abs(sd))] + [fmt.num(t["mod"]) for t in pid["thetas"]])
    mods_z = " \\cdot ".join([fmt.num(f["mod"]) for f in pid["phis"]]) or "1"
    p4.eq(
        f"K_c = \\dfrac{{|s_d|\\,\\prod|s_d - p_j|}}{{{fmt.num(abs(pid['b']))}\\,|s_d + z|^2"
        f"\\,\\prod|s_d - z_i|}} = \\dfrac{{{mods_p}}}{{{fmt.num(abs(pid['b']))} \\cdot "
        f"{fmt.num(pid['mod_zc'])}^2 \\cdot {mods_z}}} = {fmt.num(pid['Kc'])}",
    )
    p4.txt("Parâmetros do PID na forma paralela:")
    p4.eq(
        f"K_d = K_c = {fmt.num(pid['Kd'])};\\quad K_p = 2zK_c = {fmt.num(pid['Kp'])};\\quad "
        f"K_i = z^2K_c = {fmt.num(pid['Ki'])}",
        f"G_c(s) = {fmt.num(pid['Kc'])}\\,\\dfrac{{\\left(s + {fmt.num(z)}\\right)^2}}{{s}} = "
        f"{fmt.num(pid['Kp'])} + \\dfrac{{{fmt.num(pid['Ki'])}}}{{s}} + {fmt.num(pid['Kd'])}\\,s",
    )
    p4.ok(f"G<sub>c</sub>(s) = {fmt.num(pid['Kc'])}·(s + {fmt.num(z)})²/s  →  "
          f"K<sub>p</sub> = {fmt.num(pid['Kp'])}, K<sub>i</sub> = {fmt.num(pid['Ki'])}, "
          f"K<sub>d</sub> = {fmt.num(pid['Kd'])}.")
    pa.passos.append(p4)

    num_t, char = design.malha_fechada(num_g, den_g, num_h, den_h, pid["num_c"], pid["den_c"])
    polos_mf = sorted(np.roots(char), key=lambda w: (w.real, w.imag))
    p5 = Passo("Passo 5 — Verificação: polos de MF, LGR e simulação")
    p5.txt("Equação característica do sistema compensado, 1 + G<sub>c</sub>(s)G(s)H(s) = 0:")
    p5.eq(f"{fmt.poly_str(char)} = 0",
          "\\text{Polos de MF: }" + ";\\ ".join(fmt.cplx(w) for w in polos_mf))
    num_lc = np.polymul([1.0, 2 * z, z ** 2], num_l)
    den_lc = np.polymul([1.0, 0.0], den_l)
    p5.fig(figs.lgr(num_lc, den_lc, alvo=sd, k_mark=pid["Kc"], zeta=esp["zeta"],
                    sigma=esp["sigma"],
                    titulo="LGR do sistema compensado: 1 + Kc·(s+z)²/s · G(s)H(s) = 0"))
    resp = design.resposta_degrau(num_t, char, tol=criterio / 100.0)
    num_t0, char0 = design.malha_fechada(num_g, den_g, num_h, den_h, [1.0], [1.0])
    resp0 = design.resposta_degrau(num_t0, char0, t_final=float(resp["t"][-1]),
                                   tol=criterio / 100.0)
    p5.fig(figs.degrau(
        [{"t": resp["t"], "y": resp["y"], "y_inf": resp["y_inf"], "rotulo": "Com PID"},
         {"t": resp0["t"], "y": resp0["y"], "y_inf": resp0["y_inf"], "rotulo": "Sem controlador"}],
        faixa=criterio / 100.0, mp_max=mp, ts_max=ts,
        titulo="Resposta ao degrau em malha fechada  C(s)/R(s)"))
    p5.eq(f"\\text{{Simulação com PID: }} M_P = {fmt.num(resp['mp'], 2)}\\%;\\quad "
          f"t_s({crit_tex}) = {fmt.num(resp['ts'], 2)}\\ \\mathrm{{s}}")
    atende = resp["mp"] <= mp + 1e-6 and resp["ts"] <= ts + 1e-6
    if atende:
        p5.ok("A simulação confirma que as especificações são atendidas.")
    else:
        zeros_mf = np.roots(num_t)
        zeros_txt = ", ".join(fmt.num(w.real) if abs(w.imag) < 1e-9 else fmt.cplx(w).replace("\\,", "")
                              for w in sorted(zeros_mf, key=lambda w: w.real))
        p5.nota(
            "Os polos dominantes estão exatamente em s<sub>d</sub>, porém a resposta simulada "
            "não reproduz a de um sistema de 2ª ordem puro: a função de transferência de MF "
            f"tem zeros em s = {zeros_txt} (zeros do PID, da planta e o polo de H(s), que vira "
            "zero de MF) e, como o PID é ideal (termo derivativo puro), C(s)/R(s) é própria "
            f"com transmissão direta: c(0⁺) = {fmt.num(resp['y0'])}. Conforme o passo 5 do "
            "projeto de PID da apostila, nesse caso faz-se um ajuste fino de K<sub>c</sub>, "
            "z<sub>1</sub> e z<sub>2</sub> (ou usa-se uma variante como PI-D/I-PD, seção 3.4) "
            "para atender às especificações na prática.")
    pa.passos.append(p5)
    sol.partes.append(pa)

    # ---- (b) ----------------------------------------------------------------
    pb = Parte(f"b) Discretização do controlador — {discrete.METODOS[metodo]['nome']}, "
               f"T = {fmt.num(T, 1)} s")
    passo_d, dz = _passo_discretizacao(
        "Passo 6 — Aproximação discreta de Gc(s)", pid["Kc"], [-z, -z], [0.0],
        pid["num_c"], pid["den_c"], T, metodo, "G_c", "u", "e")
    if metodo == "tustin":
        passo_d.nota("O polo em z = 1 corresponde à ação integral; o polo em z = −1 resulta da "
                      "aproximação de Tustin do termo derivativo ideal (K<sub>d</sub>s).")
    passo_d.ok(f"G<sub>c</sub>(z) obtida; u[k] é o sinal de controle e e[k] o erro "
               f"amostrado a cada {fmt.num(T, 1)} s.")
    pb.passos.append(passo_d)
    sol.partes.append(pb)

    sol.respostas = [
        f"a) s<sub>d</sub> = {fmt.num(sd.real)} ± {fmt.num(sd.imag)}i;  "
        f"G<sub>c</sub>(s) = {fmt.num(pid['Kc'])}·(s + {fmt.num(z)})²/s  "
        f"(K<sub>p</sub> = {fmt.num(pid['Kp'])}, K<sub>i</sub> = {fmt.num(pid['Ki'])}, "
        f"K<sub>d</sub> = {fmt.num(pid['Kd'])})",
        f"b) G<sub>c</sub>(z) = ({_poly_txt(dz['num_z'], 'z')}) / ({_poly_txt(dz['den_z'], 'z')})",
    ]
    sol.dados = {"esp": esp, "pid": pid, "resp": resp, "dz": dz, "polos_mf": polos_mf}
    return sol


# --------------------------------------------------------------------------- #
# Questão 2 — G_c(s) = a/(s+b)
# --------------------------------------------------------------------------- #

def resolver_q2(g: str, h: str, re_sd: float, im_sd: float, T: float, metodo: str) -> Solucao:
    sistema = OpenLoopSystem(g, h)
    num_g, den_g = _tf_coeffs(sistema.G)
    num_h, den_h = _tf_coeffs(sistema.H)
    num_l = np.real(sistema.num_coeffs)
    den_l = np.real(sistema.den_coeffs)
    sd = complex(re_sd, abs(im_sd))

    en = Passo("Enunciado")
    en.eq(f"G(s) = {_tf_latex(sistema.G)};\\qquad H(s) = {_tf_latex(sistema.H)};\\qquad "
          f"G_C(s) = \\dfrac{{a}}{{s+b}}")
    en.txt(f"<b>a)</b> Quais devem ser os valores de a e b para que o sistema controlado, em "
           f"malha fechada, apresente um par de polos em {fmt.num(sd.real)} ± {fmt.num(sd.imag)}i?")
    en.txt(f"<b>b)</b> Discretize o sistema encontrado {{G(s)H(s)G<sub>C</sub>(s)}}, utilizando "
           f"um período de amostragem de {fmt.num(T, 1)} s e o método "
           f"{discrete.METODOS[metodo]['nome']}.")
    sol = Solucao(2, "Compensador a/(s+b) por alocação de polos e discretização", en)

    pa = Parte("a) Cálculo de a e b")
    c = design.compensador_a_sb(sistema, sd)
    a, b = c["a"], c["b"]

    p1 = Passo("Passo 1 — Equação característica e condições do LGR")
    p1.txt("Os polos de MF são as raízes de 1 + G<sub>C</sub>(s)G(s)H(s) = 0. Para que "
           "s<sub>d</sub> seja raiz, ele deve satisfazer as condições de ângulo e de módulo "
           "(apostila, passos 11 e 12 do LGR), com a como ganho e o polo −b do controlador "
           "como singularidade adicional:")
    p1.eq(f"G(s)H(s) = {_zpk_latex(sistema.b, sistema.zeros, sistema.poles)}",
          f"1 + \\dfrac{{a}}{{s+b}}\\,G(s)H(s) = 0,\\qquad s_d = {fmt.cplx_pm(sd)}")
    pa.passos.append(p1)

    p2 = Passo("Passo 2 — Condição de ângulo: polo do controlador (b)")
    p2.txt("Contribuições angulares dos polos (θ) e zeros (φ) de G(s)H(s) em s<sub>d</sub>:")
    p2.eq(*_contrib_eqs(c["thetas"], c["phis"], sd))
    alvo = "180^\\circ" if c["sinal"] > 0 else "0^\\circ"
    p2.eq(
        f"\\sum\\theta_i + \\theta_b - \\sum\\phi_j = {alvo} \\pm q\\,360^\\circ",
        f"\\theta_b = {alvo} + \\sum\\phi_j - \\sum\\theta_i = {alvo} + {_deg(c['soma_f'])} - "
        f"{_deg(c['soma_t'])} = {_deg(c['theta_b'])}",
        f"\\tan\\theta_b = \\dfrac{{{fmt.num(sd.imag)}}}{{b - {fmt.num(-sd.real)}}}\\;\\Rightarrow\\;"
        f"b = \\dfrac{{{fmt.num(sd.imag)}}}{{\\tan({_deg(c['theta_b'])})}} + {fmt.num(-sd.real)}"
        f" = {fmt.num(b)}",
    )
    if c["sinal"] < 0:
        p2.nota("Não há solução com a > 0 (critério de 180°); usa-se o critério de 0°, "
                "o que leva a a < 0.")
    p2.ok(f"Polo do controlador em s = {fmt.num(-b)}  →  b = {fmt.num(b)}.")
    pa.passos.append(p2)

    p3 = Passo("Passo 3 — Condição de módulo: ganho a")
    mods_p = " \\cdot ".join([fmt.num(t["mod"]) for t in c["thetas"]] + [fmt.num(c["mod_b"])])
    mods_z = " \\cdot ".join(fmt.num(f["mod"]) for f in c["phis"]) or "1"
    p3.eq(
        f"\\left|\\dfrac{{a}}{{s_d+b}}\\,G(s_d)H(s_d)\\right| = 1 \\;\\Rightarrow\\; "
        f"a = \\dfrac{{|s_d+b|\\,\\prod|s_d-p_j|}}{{{fmt.num(abs(sistema.b))}\\,\\prod|s_d-z_i|}}",
        f"a = \\dfrac{{{mods_p}}}{{{fmt.num(abs(sistema.b))} \\cdot {mods_z}}} = {fmt.num(a)}",
    )
    p3.ok(f"a = {fmt.num(a)}  e  b = {fmt.num(b)}  →  G<sub>C</sub>(s) = "
          f"{fmt.num(a)}/(s + {fmt.num(b)}).")
    pa.passos.append(p3)

    num_t, char = design.malha_fechada(num_g, den_g, num_h, den_h, c["num_c"], c["den_c"])
    polos_mf = sorted(np.roots(char), key=lambda w: (w.real, w.imag))
    p4 = Passo("Passo 4 — Verificação")
    p4.txt("Alternativamente, impondo diretamente Δ(s<sub>d</sub>) = 0 na equação "
           "característica (s+b)·den(s) + a·num(s) = 0, as partes real e imaginária formam "
           "um sistema linear em a e b:")
    d_sd, n_sd = c["d_sd"], c["n_sd"]
    p4.eq(
        f"\\mathrm{{den}}(s_d) = {fmt.cplx(d_sd)};\\quad \\mathrm{{num}}(s_d) = {fmt.cplx(n_sd)}",
        f"(s_d + b)\\,({fmt.cplx(d_sd)}) + a\\,({fmt.cplx(n_sd)}) = 0"
        f"\\;\\Rightarrow\\; a = {fmt.num(c['a_lin'])},\\ b = {fmt.num(c['b_lin'])}",
        f"\\Delta(s) = {fmt.poly_str(char)} = 0",
        "\\text{Polos de MF: }" + ";\\ ".join(fmt.cplx(w) for w in polos_mf),
    )
    num_lc = num_l
    den_lc = np.polymul([1.0, b], den_l)
    p4.fig(figs.lgr(num_lc, den_lc, alvo=sd, k_mark=a,
                    titulo="LGR de 1 + a/(s+b) · G(s)H(s) = 0  (variando a)"))
    outros = [w for w in polos_mf if abs(w - sd) > 1e-6 and abs(w - sd.conjugate()) > 1e-6]
    if any(w.real > sd.real for w in outros):
        p4.nota("O par em s<sub>d</sub> é atendido exatamente, mas há polo(s) de MF mais "
                "próximo(s) do eixo imaginário ("
                + ", ".join(fmt.num(w.real) for w in outros if w.real > sd.real)
                + "), de modo que s<sub>d</sub> não é o par dominante.")
    p4.ok(f"O sistema em MF possui o par de polos {fmt.num(sd.real)} ± {fmt.num(sd.imag)}i.")
    pa.passos.append(p4)
    sol.partes.append(pa)

    pb = Parte(f"b) Discretização de G(s)H(s)Gc(s) — {discrete.METODOS[metodo]['nome']}, "
               f"T = {fmt.num(T, 1)} s")
    p5 = Passo("Passo 5 — Função de transferência contínua a ser discretizada")
    num_tot = a * num_l
    den_tot = np.polymul([1.0, b], den_l)
    p5.eq(f"G(s)H(s)G_C(s) = {_zpk_latex(a * sistema.b, sistema.zeros, list(sistema.poles) + [-b])}",
          f"G(s)H(s)G_C(s) = {_frac(num_tot, den_tot)}")
    pb.passos.append(p5)
    passo_d, dz = _passo_discretizacao(
        "Passo 6 — Aproximação discreta", a * sistema.b, sistema.zeros,
        list(sistema.poles) + [-b], num_tot, den_tot, T, metodo, "L", "y", "u")
    passo_d.fig(figs.plano_z(dz["polos_z"], dz["zeros_z"],
                             titulo="Polos e zeros de L(z) = G(z)H(z)G_C(z)"))
    polos_s = list(sistema.poles) + [-b]
    estavel_s = all(p.real < 0 for p in polos_s)
    if estavel_s and not dz["estavel"]:
        passo_d.nota(
            "Observe que todos os polos contínuos estão no semiplano esquerdo, mas o sistema "
            "discreto possui polos sobre ou fora do círculo unitário. Como alerta a apostila "
            "(seção 5.2), o método de Euler pode transformar um sistema contínuo estável em um "
            "discreto instável; um período de amostragem menor (ou o método de Tustin) evita "
            "esse efeito." if metodo == "euler" else
            "Há polos discretos sobre ou fora do círculo unitário.")
    passo_d.ok(f"L(z) = G(z)H(z)G<sub>C</sub>(z) obtida com T = {fmt.num(T, 1)} s.")
    pb.passos.append(passo_d)
    sol.partes.append(pb)

    sol.respostas = [
        f"a) a = {fmt.num(a)};  b = {fmt.num(b)}  →  G<sub>C</sub>(s) = {fmt.num(a)}/(s + {fmt.num(b)})",
        f"b) L(z) = ({_poly_txt(dz['num_z'], 'z')}) / ({_poly_txt(dz['den_z'], 'z')})",
    ]
    sol.dados = {"comp": c, "dz": dz, "polos_mf": polos_mf}
    return sol


def _poly_txt(coeffs, var: str) -> str:
    """Polinômio em texto simples (HTML) para o quadro de respostas."""
    tex = fmt.poly_str(coeffs, var)
    return re.sub(r"\^\{(-?\d+)\}", r"<sup>\1</sup>", tex)
