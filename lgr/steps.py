"""Os 12 passos da construção do LGR (tabela da apostila SisContr, seções 2.2 e 2.4).

Cada função `step_XX` devolve um :class:`StepResult` com a regra citada da
apostila, o desenvolvimento algébrico (linhas em LaTeX) e o resultado numérico.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from . import fmt, plots
from .routh import char_poly, imaginary_axis_crossings, routh_table
from .system import K, OpenLoopSystem, s

_ANG_TOL = 2.0  # graus de tolerância para o critério de ângulo


@dataclass
class StepResult:
    n: int
    titulo: str
    regra: str
    desenvolvimento: list[str] = field(default_factory=list)
    resultado: str = ""
    figura: object | None = None
    aplicavel: bool = True


# --------------------------------------------------------------------------- #
# helpers de geometria
# --------------------------------------------------------------------------- #

def _ang(origem: complex, destino: complex) -> float:
    """Ângulo (graus) do vetor que vai de `origem` até `destino`."""
    return float(np.degrees(np.angle(complex(destino) - complex(origem))))


def _norm180(x: float) -> float:
    y = ((x + 180.0) % 360.0) - 180.0
    return 180.0 if abs(y + 180.0) < 1e-9 else y


def _is_real(z: complex, tol: float = 1e-7) -> bool:
    return abs(complex(z).imag) < tol


def real_axis_segments(system: OpenLoopSystem):
    """Intervalos do eixo real que pertencem ao LGR (à esquerda de nº ímpar)."""
    reais: list[float] = []
    for z, m in system.zeros_grp + system.poles_grp:
        if _is_real(z):
            reais.extend([z.real] * m)
    if not reais:
        return [], []
    marcos = sorted(set(round(r, 9) for r in reais))
    # intervalos: (-inf, marcos[0]), (marcos[0], marcos[1]), ...
    bordas = [-np.inf] + marcos
    segs = []
    detalhes = []
    for i in range(len(bordas) - 1):
        esq, dir_ = bordas[i], bordas[i + 1]
        teste = (dir_ - 1.0) if not np.isfinite(esq) else 0.5 * (esq + dir_)
        cont = sum(1 for r in reais if r > teste + 1e-9)
        impar = cont % 2 == 1
        detalhes.append({"esq": esq, "dir": dir_, "contagem": cont, "lgr": impar})
        if impar:
            segs.append((esq, dir_))
    return segs, detalhes


def asymptotes(system: OpenLoopSystem):
    rd = system.relative_degree()
    if rd <= 0:
        return None
    soma_p = sum(system.poles)
    soma_z = sum(system.zeros)
    sigma = ((soma_p - soma_z) / rd).real
    angs = [(2 * q + 1) * 180.0 / rd for q in range(rd)]
    return {"sigma": sigma, "angulos_deg": angs, "rd": rd,
            "soma_p": soma_p, "soma_z": soma_z}


def breakaway_points(system: OpenLoopSystem):
    num, den = system.num_expr, system.den_expr
    Ks = sp.simplify(-den / num)
    dKds = sp.together(sp.diff(Ks, s))
    numer = sp.expand(sp.numer(dKds))
    poly = sp.Poly(numer, s)
    try:
        raizes = [complex(r) for r in sp.nroots(poly, n=15)]
    except sp.PolynomialError:  # pragma: no cover
        raizes = []
    candidatos = []
    for r in raizes:
        re = 0.0 if abs(r.real) < 1e-8 else r.real
        im = 0.0 if abs(r.imag) < 1e-8 else r.imag
        rc = complex(re, im)
        kval = system.gain_at(rc)
        ang = system.angle_deg_at(rc)
        no_lgr = np.isfinite(kval) and kval > 1e-9 and abs(_norm180(ang - 180.0)) < _ANG_TOL
        candidatos.append({"s": rc, "K": kval, "no_lgr": no_lgr})
    return {"Ks": Ks, "dKds": dKds, "poly": poly, "candidatos": candidatos}


def departure_arrival_angles(system: OpenLoopSystem):
    """Ângulos de partida (polos complexos) e de chegada (zeros complexos).

    Considera polos/zeros simples (escopo do curso).
    """
    resultados = []
    for polo, _ in system.poles_grp:
        if _is_real(polo):
            continue
        thetas = [(_ang(q, polo), q) for q, mq in system.poles_grp
                  for _ in range(mq) if abs(q - polo) > 1e-9]
        phis = [(_ang(z, polo), z) for z, mz in system.zeros_grp for _ in range(mz)]
        ang = _norm180(180.0 - sum(t for t, _ in thetas) + sum(f for f, _ in phis))
        resultados.append({"tipo": "partida", "ponto": polo,
                           "thetas": thetas, "phis": phis, "angulo": ang})
    for zero, _ in system.zeros_grp:
        if _is_real(zero):
            continue
        phis = [(_ang(z, zero), z) for z, mz in system.zeros_grp
                for _ in range(mz) if abs(z - zero) > 1e-9]
        thetas = [(_ang(p, zero), p) for p, mp in system.poles_grp for _ in range(mp)]
        ang = _norm180(180.0 - sum(f for f, _ in phis) + sum(t for t, _ in thetas))
        resultados.append({"tipo": "chegada", "ponto": zero,
                           "thetas": thetas, "phis": phis, "angulo": ang})
    return resultados


def angle_test(system: OpenLoopSystem, si: complex):
    thetas = []  # contribuições dos polos  (θ)
    for polo, m in system.poles_grp:
        for _ in range(m):
            thetas.append({"ponto": polo, "ang": _ang(polo, si), "tipo": "polo"})
    phis = []  # contribuições dos zeros  (ϕ)
    for zero, m in system.zeros_grp:
        for _ in range(m):
            phis.append({"ponto": zero, "ang": _ang(zero, si), "tipo": "zero"})
    soma_t = sum(c["ang"] for c in thetas)
    soma_f = sum(c["ang"] for c in phis)
    total = soma_t - soma_f            # deve ser 180° ± q·360°
    erro = _norm180(total - 180.0)
    pertence = abs(erro) < _ANG_TOL
    return {"thetas": thetas, "phis": phis, "soma_t": soma_t, "soma_f": soma_f,
            "total": total, "erro": erro, "pertence": pertence}


def gain_at_point(system: OpenLoopSystem, si: complex):
    mods_p = []
    for polo, m in system.poles_grp:
        for _ in range(m):
            mods_p.append({"ponto": polo, "mod": abs(si - polo)})
    mods_z = []
    for zero, m in system.zeros_grp:
        for _ in range(m):
            mods_z.append({"ponto": zero, "mod": abs(si - zero)})
    prod_p = float(np.prod([d["mod"] for d in mods_p])) if mods_p else 1.0
    prod_z = float(np.prod([d["mod"] for d in mods_z])) if mods_z else 1.0
    b = abs(system.b)
    kval = prod_p / (prod_z * b) if prod_z > 0 else float("inf")
    return {"mods_p": mods_p, "mods_z": mods_z, "prod_p": prod_p,
            "prod_z": prod_z, "b": b, "K": kval}


# --------------------------------------------------------------------------- #
# helpers de LaTeX
# --------------------------------------------------------------------------- #

def _roots_latex(grupos, simbolo: str) -> list[str]:
    linhas = []
    i = 1
    for val, m in grupos:
        suf = f"\\ (\\text{{multiplicidade }} {m})" if m > 1 else ""
        linhas.append(f"{simbolo}_{{{i}}} = {fmt.cplx(val)}{suf}")
        i += 1
    return linhas


def _routh_latex(M: sp.Matrix, labels: list[str]) -> str:
    linhas = []
    for i in range(M.rows):
        celulas = [sp.latex(sp.nsimplify(M[i, j], rational=False)) for j in range(M.cols)]
        linhas.append(f"{labels[i].replace('^', '^{') + '}'} & " + " & ".join(celulas))
    corpo = " \\\\[2pt] ".join(linhas)
    cols = "c|" + "c" * M.cols
    return f"\\begin{{array}}{{{cols}}} {corpo} \\end{{array}}"


# --------------------------------------------------------------------------- #
# os 12 passos
# --------------------------------------------------------------------------- #

def step_01(system: OpenLoopSystem, si=None) -> StepResult:
    dev = [
        f"G(s)\\,H(s) = {sp.latex(system.L)}",
        f"1 + G(s)H(s) = 0 \\;\\Rightarrow\\; 1 + K\\,P(s) = 0",
        f"P(s) = {sp.latex(system.P)}",
        f"\\underbrace{{{fmt.poly_str(system.den_coeffs)}}}_{{\\text{{den}}(s)}}"
        f" + K\\underbrace{{\\left({fmt.poly_str(system.num_coeffs)}\\right)}}_{{\\text{{num}}(s)}} = 0",
    ]
    return StepResult(
        1, "Polinômio característico com K em evidência",
        "Escrever o polinômio característico de modo que o parâmetro de interesse "
        "(K) apareça claramente: $1 + G(s)H(s) = 1 + K\\,P(s)$.",
        dev,
        "Equação característica: $\\;\\text{den}(s) + K\\,\\text{num}(s) = 0$.",
    )


def step_02(system: OpenLoopSystem, si=None) -> StepResult:
    dev = [
        f"P(s) = \\dfrac{{{sp.latex(system.num_expr)}}}{{{sp.latex(system.den_expr)}}}"
        f" = {sp.latex(system.factored_P)}",
    ]
    dev += [f"\\text{{Polos: }} " + ";\\quad ".join(_roots_latex(system.poles_grp, "p"))]
    if system.zeros_grp:
        dev += [f"\\text{{Zeros: }} " + ";\\quad ".join(_roots_latex(system.zeros_grp, "z"))]
    else:
        dev += ["\\text{Zeros finitos: nenhum}"]
    return StepResult(
        2, "Fatoração de P(s): polos e zeros",
        "Fatorar o polinômio $P(s)$ em termos dos $n_P$ polos e $n_Z$ zeros. "
        "As curvas do LGR **começam nos polos** (K = 0) e **terminam nos zeros** (K → ∞).",
        dev,
        f"$n_P = {system.n_p}$ polos finitos e $n_Z = {system.n_z}$ zeros finitos.",
    )


def step_03(system: OpenLoopSystem, si=None) -> StepResult:
    dev = ["\\text{X = polo} \\qquad \\text{O = zero}"]
    fig = plots.pole_zero_figure(system, test_point=si)
    return StepResult(
        3, "Marcar polos e zeros no plano s",
        "Assinalar os polos e zeros de malha aberta no plano $s$ com os símbolos "
        "correspondentes (**X** = polos, **O** = zeros).",
        dev,
        "Polos e zeros marcados no plano complexo (figura).",
        figura=fig,
    )


def step_04(system: OpenLoopSystem, si=None) -> StepResult:
    segs, detalhes = real_axis_segments(system)
    dev = []
    for d in detalhes:
        esq = "-\\infty" if not np.isfinite(d["esq"]) else fmt.num(d["esq"])
        parid = "\\textbf{ímpar} \\Rightarrow \\text{pertence ao LGR}" if d["lgr"] \
            else "\\text{par} \\Rightarrow \\text{não pertence}"
        dev.append(f"({esq},\\ {fmt.num(d['dir'])}):\\ \\text{{polos+zeros à direita}} = "
                   f"{d['contagem']} \\;({parid})")
    if not detalhes:
        dev.append("\\text{Não há polos nem zeros reais.}")
    fig = plots.pole_zero_figure(system, segments=segs, test_point=si,
                                 titulo="Passo 4 — segmentos do eixo real")
    if segs:
        txt = "; ".join(
            f"({'−∞' if not np.isfinite(a) else fmt.num(a)} , {fmt.num(b)})"
            for a, b in segs)
        res = f"Segmentos do eixo real no LGR: {txt}."
    else:
        res = "Nenhum segmento do eixo real pertence ao LGR."
    return StepResult(
        4, "Segmentos do eixo real",
        "Assinalar os segmentos do eixo real que são LGR: **o LGR se situa à "
        "esquerda de um número ímpar de polos e zeros** (reais).",
        dev, res, figura=fig,
    )


def step_05(system: OpenLoopSystem, si=None) -> StepResult:
    rd = system.relative_degree()
    dev = [
        f"LS = n_P = {system.n_p} \\quad (n_P \\ge n_Z)",
        f"\\text{{Ramos que terminam em zeros finitos}} = n_Z = {system.n_z}",
        f"\\text{{Ramos que seguem para zeros no infinito}} = n_P - n_Z = {rd}",
    ]
    return StepResult(
        5, "Número de lugares separados (LS)",
        "Determinar o número de lugares separados, $LS$ (segmentos de curva que "
        "compõem o LGR): $LS = n_P$ quando $n_P \\ge n_Z$.",
        dev,
        f"$LS = {system.n_p}$ ramos, sendo {rd} deles para o infinito.",
    )


def step_06(system: OpenLoopSystem, si=None) -> StepResult:
    complexos = [f"{fmt.cplx(v)}" for v, _ in system.poles_grp if not _is_real(v)]
    dev = ["\\text{Polos/zeros complexos ocorrem em pares conjugados.}"]
    if complexos:
        dev.append("\\text{Polos complexos: }" + ",\\ ".join(complexos))
    return StepResult(
        6, "Simetria em relação ao eixo real",
        "O LGR é **simétrico em relação ao eixo real** (eixo horizontal). "
        "Basta desenhar a parte acima do eixo real e depois espelhar.",
        dev,
        "O traçado acima do eixo real é espelhado abaixo dele.",
    )


def step_07(system: OpenLoopSystem, si=None) -> StepResult:
    asy = asymptotes(system)
    if asy is None:
        return StepResult(
            7, "Assíntotas",
            "$(n_P - n_Z)$ ramos seguem para os zeros no infinito ao longo de "
            "assíntotas centralizadas em $\\sigma_A$ com ângulos $\\phi_A$.",
            ["n_P - n_Z \\le 0 \\Rightarrow \\text{todos os ramos terminam em zeros finitos.}"],
            "Não há assíntotas (todos os ramos terminam em zeros finitos).",
            aplicavel=False,
        )
    rd = asy["rd"]
    dev = [
        f"\\sigma_A = \\dfrac{{\\sum(-p_j) - \\sum(-z_i)}}{{n_P - n_Z}}"
        f" = \\dfrac{{({fmt.num(asy['soma_p'].real)}) - ({fmt.num(asy['soma_z'].real)})}}{{{rd}}}"
        f" = {fmt.num(asy['sigma'])}",
    ]
    for q, ang in enumerate(asy["angulos_deg"]):
        dev.append(f"\\phi_A = \\dfrac{{(2\\cdot{q}+1)\\,180^\\circ}}{{{rd}}} = {fmt.deg_raw(ang)}")
    fig = plots.pole_zero_figure(
        system, segments=real_axis_segments(system)[0], asymptotes=asy, test_point=si,
        titulo="Passo 7 — assíntotas")
    angs = ", ".join(fmt.deg_raw(a) for a in asy["angulos_deg"])
    return StepResult(
        7, "Assíntotas",
        "$(n_P - n_Z)$ ramos seguem para os zeros no infinito ao longo de "
        "assíntotas centralizadas em $\\sigma_A$ com ângulos "
        "$\\phi_A = \\dfrac{(2q+1)180^\\circ}{n_P-n_Z}$, $q = 0,1,\\dots,(n_P-n_Z-1)$.",
        dev,
        f"$\\sigma_A = {fmt.num(asy['sigma'])}$; ângulos das assíntotas: {angs}.",
        figura=fig,
    )


def step_08(system: OpenLoopSystem, si=None) -> StepResult:
    bk = breakaway_points(system)
    dev = [
        f"K = p(s) = -\\dfrac{{\\text{{den}}(s)}}{{\\text{{num}}(s)}} = {sp.latex(bk['Ks'])}",
        f"\\dfrac{{dp(s)}}{{ds}} = {sp.latex(sp.simplify(bk['dKds']))} = 0",
    ]
    validos = [c for c in bk["candidatos"] if c["no_lgr"]]
    todas = ",\\ ".join(fmt.cplx(c["s"]) for c in bk["candidatos"]) or "\\varnothing"
    dev.append(f"\\text{{Raízes de }} dp/ds = 0:\\ {todas}")
    for c in bk["candidatos"]:
        marca = "\\ \\checkmark\\ \\text{(sobre o LGR)}" if c["no_lgr"] \
            else "\\ \\text{(fora do LGR — descartada)}"
        kk = fmt.num(c["K"]) if np.isfinite(c["K"]) else "\\infty"
        dev.append(f"s = {fmt.cplx(c['s'])}:\\quad K = {kk}{marca}")
    if validos:
        res = "; ".join(f"$s = {fmt.cplx(c['s'])}$ com $K = {fmt.num(c['K'])}$" for c in validos)
        res = "Ponto(s) de saída/entrada: " + res + "."
    else:
        res = "Não há ponto de saída/entrada sobre o eixo real dentro do LGR."
    return StepResult(
        8, "Pontos de saída / entrada no eixo real",
        "Determinar os pontos sobre o eixo real onde curvas do LGR se encontram: "
        "fazer $K = p(s)$ e determinar as raízes de $\\dfrac{dp(s)}{ds} = 0$.",
        dev, res,
    )


def step_09(system: OpenLoopSystem, si=None) -> StepResult:
    cp = char_poly(system)
    coeffs = cp.all_coeffs()
    M, labels = routh_table(coeffs)
    cruz = imaginary_axis_crossings(system)
    dev = [
        f"D(s) = {fmt.poly_str(system.den_coeffs)} + K\\left({fmt.poly_str(system.num_coeffs)}\\right) = 0",
        _routh_latex(M, labels),
    ]
    if cruz:
        for c in cruz:
            dev.append(
                f"\\text{{Linha nula em }} K = {fmt.num(c['K'])} \\Rightarrow "
                f"\\text{{polinômio auxiliar}} \\Rightarrow s = \\pm\\,{fmt.num(c['omega'])}\\,i")
        res = "; ".join(
            f"cruza o eixo imaginário em $s = \\pm {fmt.num(c['omega'])}\\,i$ para "
            f"$K = {fmt.num(c['K'])}$" for c in cruz)
        res = res[0].upper() + res[1:] + "."
    else:
        res = "O LGR não cruza o eixo imaginário para $K > 0$ (sistema permanece estável)."
    return StepResult(
        9, "Cruzamento do eixo imaginário (Routh-Hurwitz)",
        "Utilizando o critério de Routh-Hurwitz, determinar o(s) ponto(s) no(s) "
        "qual(is) o eixo imaginário é cruzado por curvas do LGR (se isso ocorrer).",
        dev, res,
    )


def step_10(system: OpenLoopSystem, si=None) -> StepResult:
    dados = departure_arrival_angles(system)
    if not dados:
        return StepResult(
            10, "Ângulos de partida / chegada",
            "Usando a condição de ângulo, determinar o ângulo de partida para os "
            "polos complexos e o ângulo de chegada para os zeros complexos.",
            ["\\text{Não há polos nem zeros complexos.}"],
            "Não se aplica (sem polos/zeros complexos).",
            aplicavel=False,
        )
    def _soma(vals):
        if not vals:
            return "0^\\circ"
        out = fmt.deg_raw(vals[0])
        for v in vals[1:]:
            out += (f" + {fmt.deg_raw(v)}" if v >= 0 else f" - {fmt.deg_raw(abs(v))}")
        return out

    dev = []
    for d in dados:
        nome = "partida" if d["tipo"] == "partida" else "chegada"
        st = _soma([t for t, _ in d["thetas"]])
        sf = _soma([f for f, _ in d["phis"]])
        if d["tipo"] == "partida":
            dev.append(
                f"\\text{{Polo }} {fmt.cplx(d['ponto'])}:\\ \\theta_{{{nome}}}"
                f" = 180^\\circ - (\\sum\\theta_i) + (\\sum\\phi_i)"
                f" = 180^\\circ - ({st}) + ({sf}) = {fmt.deg_raw(d['angulo'])}")
        else:
            dev.append(
                f"\\text{{Zero }} {fmt.cplx(d['ponto'])}:\\ \\phi_{{{nome}}}"
                f" = 180^\\circ - (\\sum\\phi_i) + (\\sum\\theta_i)"
                f" = 180^\\circ - ({sf}) + ({st}) = {fmt.deg_raw(d['angulo'])}")
    res = "; ".join(
        f"{'partida' if d['tipo']=='partida' else 'chegada'} em {fmt.cplx(d['ponto'])}: "
        f"${fmt.deg_raw(d['angulo'])}$" for d in dados)
    return StepResult(
        10, "Ângulos de partida / chegada",
        "Usando a condição de ângulo, determinar o ângulo de partida para os polos "
        "complexos e o de chegada para os zeros complexos: "
        "$\\text{partida} = 180^\\circ - \\sum\\theta_i + \\sum\\phi_i$.",
        dev, "Ângulo de " + res + ".",
    )


def step_11(system: OpenLoopSystem, si=None) -> StepResult:
    if si is None:
        return StepResult(
            11, "Critério do ângulo de fase no ponto de teste",
            "Determinar se o ponto $s_i$ satisfaz o critério do ângulo de fase: "
            "$\\left(\\sum_{n_P}\\theta_i - \\sum_{n_Z}\\phi_j\\right)_{s=s_i} "
            "= 180^\\circ \\pm q\\,360^\\circ$.",
            ["\\text{Informe um ponto de teste } s_i \\text{ para executar este passo.}"],
            "Informe um ponto de teste $s_i$.",
            aplicavel=False,
        )
    at = angle_test(system, si)
    dev = [f"s_i = {fmt.cplx(si)}"]
    for c in at["thetas"]:
        dev.append(f"\\theta\\ \\text{{(do polo }} {fmt.cplx(c['ponto'])}) "
                   f"= \\angle\\,(s_i - ({fmt.cplx(c['ponto'])})) = {fmt.deg_raw(c['ang'])}")
    for c in at["phis"]:
        dev.append(f"\\phi\\ \\text{{(do zero }} {fmt.cplx(c['ponto'])}) "
                   f"= \\angle\\,(s_i - ({fmt.cplx(c['ponto'])})) = {fmt.deg_raw(c['ang'])}")
    dev.append(
        f"\\sum\\theta_i - \\sum\\phi_j = {fmt.deg_raw(at['soma_t'])} - "
        f"{fmt.deg_raw(at['soma_f'])} = {fmt.deg_raw(at['total'])}")
    dev.append(f"\\text{{Erro em relação a }} 180^\\circ:\\ {fmt.deg_raw(at['erro'])}")
    contribs = [{"tipo": c["tipo"], "ponto": c["ponto"], "ang": c["ang"]}
                for c in at["thetas"] + at["phis"]]
    fig = plots.angle_vectors_figure(system, si, contribs)
    if at["pertence"]:
        res = (f"✅ **O ponto $s_i = {fmt.cplx(si)}$ PERTENCE ao LGR** "
               f"(erro de ângulo = ${fmt.deg_raw(at['erro'])}$).")
    else:
        res = (f"❌ **O ponto $s_i = {fmt.cplx(si)}$ NÃO pertence ao LGR** "
               f"(erro de ângulo = ${fmt.deg_raw(at['erro'])}$, fora da tolerância de "
               f"±{fmt.num(_ANG_TOL, 1)}°).")
    return StepResult(
        11, "Critério do ângulo de fase no ponto de teste",
        "Determinar se o ponto $s_i$ satisfaz o critério do ângulo de fase: "
        "$\\left(\\sum_{n_P}\\theta_i - \\sum_{n_Z}\\phi_j\\right)_{s=s_i} "
        "= 180^\\circ \\pm q\\,360^\\circ$.",
        dev, res, figura=fig,
    )


def step_12(system: OpenLoopSystem, si=None) -> StepResult:
    if si is None:
        return StepResult(
            12, "Valor do ganho K no ponto s_i",
            "Determinar o valor do parâmetro $K$ na raiz $s_i$: "
            "$|K\\,P(s)|_{s=s_i} = 1 \\Rightarrow "
            "K = \\dfrac{\\prod |s_i - p_j|}{\\prod |s_i - z_k|}$.",
            ["\\text{Informe um ponto de teste } s_i."],
            "Informe um ponto de teste $s_i$.",
            aplicavel=False,
        )
    at = angle_test(system, si)
    if not at["pertence"]:
        return StepResult(
            12, "Valor do ganho K no ponto s_i",
            "Determinar o valor do parâmetro $K$ na raiz $s_i$ pelo critério de módulo.",
            ["\\text{O ponto não pertence ao LGR (passo 11) — não há } K > 0 "
             "\\text{ que coloque um polo de MF exatamente em } s_i."],
            "Não se aplica: o ponto não pertence ao LGR.",
            aplicavel=False,
        )
    g = gain_at_point(system, si)
    dev = []
    for d in g["mods_p"]:
        dev.append(f"|s_i - ({fmt.cplx(d['ponto'])})| = {fmt.num(d['mod'])}")
    for d in g["mods_z"]:
        dev.append(f"|s_i - ({fmt.cplx(d['ponto'])})| = {fmt.num(d['mod'])}")
    num_txt = " \\cdot ".join(fmt.num(d["mod"]) for d in g["mods_p"]) or "1"
    den_txt = " \\cdot ".join(fmt.num(d["mod"]) for d in g["mods_z"]) or "1"
    b_txt = "" if abs(g["b"] - 1.0) < 1e-9 else f" \\cdot {fmt.num(g['b'])}"
    dev.append(
        f"K = \\dfrac{{\\prod|s_i - p_j|}}{{\\prod|s_i - z_k|}}"
        f" = \\dfrac{{{num_txt}}}{{{den_txt}{b_txt}}} = {fmt.num(g['K'])}")
    clp = system.closed_loop_poles(g["K"])
    polos_txt = ",\\ ".join(fmt.cplx(p) for p in sorted(clp, key=lambda z: (z.real, z.imag)))
    dev.append(f"\\text{{Polos de MF em }} K = {fmt.num(g['K'])}:\\ {polos_txt}")
    return StepResult(
        12, "Valor do ganho K no ponto s_i",
        "Determinar o valor do parâmetro $K$ na raiz $s_i$: "
        "$|K\\,P(s)|_{s=s_i} = 1 \\Rightarrow "
        "K = \\dfrac{\\prod |s_i - p_j|}{\\prod |s_i - z_k|}$.",
        dev,
        f"**$K = {fmt.num(g['K'])}$** coloca um polo de malha fechada em "
        f"$s_i = {fmt.cplx(si)}$.",
    )


ALL_STEPS = [step_01, step_02, step_03, step_04, step_05, step_06,
             step_07, step_08, step_09, step_10, step_11, step_12]


def solve_all(system: OpenLoopSystem, si: complex | None = None) -> list[StepResult]:
    return [fn(system, si) for fn in ALL_STEPS]
