"""Critério de Routh-Hurwitz aplicado a D(s) = den(s) + K·num(s).

Fornece a tabela de Routh simbólica (em função de K) e os pontos de cruzamento
do eixo imaginário pelo LGR.
"""
from __future__ import annotations

import sympy as sp

from .system import K, s

_eps = sp.Symbol("epsilon", positive=True)


def char_poly(system) -> sp.Poly:
    return sp.Poly(sp.expand(system.den_expr + K * system.num_expr), s)


def routh_table(coeffs: list[sp.Expr]) -> tuple[sp.Matrix, list[str]]:
    """Monta a tabela de Routh. `coeffs` em grau decrescente (a_n ... a_0)."""
    n = len(coeffs)
    width = (n + 1) // 2
    M = sp.zeros(n, width)

    for j in range(width):
        if 2 * j < n:
            M[0, j] = sp.nsimplify(coeffs[2 * j], rational=False)
        if 2 * j + 1 < n:
            M[1, j] = sp.nsimplify(coeffs[2 * j + 1], rational=False)

    for i in range(2, n):
        if all(M[i - 1, j] == 0 for j in range(width)):
            # linha de zeros -> derivada do polinômio auxiliar (linha i-1)
            ordem = n - i  # potência de s da linha i-1
            for j in range(width):
                p = ordem - 2 * j
                if p > 0:
                    M[i - 1, j] = M[i - 2, j] * p
            # se ainda tudo zero, mantém
        b = M[i - 1, 0]
        if b == 0:
            b = _eps
            M[i - 1, 0] = _eps
        for j in range(width - 1):
            a1 = M[i - 2, j + 1]
            a2 = M[i - 1, j + 1]
            M[i, j] = sp.cancel(sp.expand((b * a1 - M[i - 2, 0] * a2) / b))

    labels = [f"s^{n - 1 - i}" for i in range(n)]
    return M, labels


def imaginary_axis_crossings(system) -> list[dict]:
    """Resolve D(jw) = 0 para w>0 e K>0 (cruzamentos do eixo imaginário)."""
    w = sp.Symbol("w", positive=True)
    D = char_poly(system).as_expr()
    expr = sp.expand(D.subs(s, sp.I * w))
    re = sp.re(expr)
    im = sp.im(expr)
    try:
        sols = sp.solve([sp.Eq(re, 0), sp.Eq(im, 0)], [w, K], dict=True)
    except Exception:  # pragma: no cover
        sols = []
    out = []
    seen = set()
    for sol in sols:
        if w not in sol or K not in sol:
            continue
        try:
            wv = complex(sol[w])
            kv = complex(sol[K])
        except (TypeError, ValueError):
            continue
        if abs(wv.imag) > 1e-6 or abs(kv.imag) > 1e-6:
            continue
        wv, kv = wv.real, kv.real
        if wv <= 1e-9 or kv <= 1e-9:
            continue
        key = (round(wv, 6), round(kv, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append({"omega": wv, "K": kv})
    out.sort(key=lambda d: d["K"])
    return out


def routh_stability_limit(system) -> list[dict]:
    """Valores de K>0 que anulam algum elemento da 1ª coluna (limite de estabilidade)."""
    coeffs = char_poly(system).all_coeffs()
    M, labels = routh_table(coeffs)
    n = M.rows
    limites = []
    for i in range(2, n):
        entry = sp.simplify(M[i, 0])
        if entry.free_symbols <= {K} and K in entry.free_symbols:
            try:
                raizes = sp.solve(sp.Eq(entry, 0), K)
            except Exception:  # pragma: no cover
                raizes = []
            for r in raizes:
                try:
                    rv = complex(r)
                except (TypeError, ValueError):
                    continue
                if abs(rv.imag) < 1e-9 and rv.real > 1e-9:
                    limites.append({"linha": labels[i], "K": rv.real, "expr": entry})
    return limites
