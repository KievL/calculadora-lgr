"""Modelo do sistema de malha aberta L(s) = G(s)H(s) = K * P(s).

Faz o parsing das expressões de G(s) e H(s), coloca o ganho K em evidência e
extrai numerador/denominador de P(s), polos e zeros de malha aberta.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

s = sp.Symbol("s")
K = sp.Symbol("K", positive=True)

_CLUSTER_TOL = 1e-6


class ParseError(ValueError):
    """Erro amigável de entrada do usuário."""


def _normalize(txt: str) -> str:
    t = (txt or "").strip()
    if not t:
        raise ParseError("Expressão vazia.")
    t = t.replace("^", "**")
    return t


def _sympify(txt: str) -> sp.Expr:
    try:
        expr = sp.sympify(_normalize(txt), locals={"s": s, "K": K}, rational=False)
    except (sp.SympifyError, SyntaxError, TypeError) as exc:  # pragma: no cover
        raise ParseError(f"Não consegui interpretar a expressão: {txt!r}") from exc
    extra = expr.free_symbols - {s, K}
    if extra:
        nomes = ", ".join(sorted(str(x) for x in extra))
        raise ParseError(f"Símbolos desconhecidos na expressão: {nomes}. Use apenas 's' e 'K'.")
    return expr


def _group(vals: list[complex]) -> list[tuple[complex, int]]:
    """Agrupa raízes iguais (dentro da tolerância) em (valor, multiplicidade)."""
    grupos: list[list[complex]] = []
    for v in vals:
        for g in grupos:
            if abs(g[0] - v) < _CLUSTER_TOL:
                g.append(v)
                break
        else:
            grupos.append([v])
    out = [(complex(np.mean(g)), len(g)) for g in grupos]
    out.sort(key=lambda t: (round(t[0].real, 6), round(abs(t[0].imag), 6), t[0].imag))
    return out


def _roots(poly: sp.Poly) -> list[complex]:
    if poly.degree() <= 0:
        return []
    coeffs = [complex(c) for c in poly.all_coeffs()]
    r = np.roots(coeffs)
    # limpa ruído numérico
    cleaned = []
    for z in r:
        re = 0.0 if abs(z.real) < 1e-9 else z.real
        im = 0.0 if abs(z.imag) < 1e-9 else z.imag
        cleaned.append(complex(round(re, 10), round(im, 10)))
    return cleaned


@dataclass
class OpenLoopSystem:
    g_str: str
    h_str: str

    def __post_init__(self) -> None:
        G = _sympify(self.g_str)
        H = _sympify(self.h_str)
        self.G, self.H = G, H
        GH = sp.cancel(sp.together(G * H))
        self.L = GH

        if K in GH.free_symbols:
            P = sp.cancel(GH / K)
            if K in P.free_symbols:
                raise ParseError(
                    "O ganho K deve aparecer linearmente, multiplicando todo o "
                    "numerador de G(s)·H(s)."
                )
            self.k_explicito = True
        else:
            P = GH
            self.k_explicito = False

        P = sp.cancel(sp.together(P))
        num, den = sp.fraction(P)
        num = sp.expand(num)
        den = sp.expand(den)
        den_poly = sp.Poly(den, s)
        num_poly = sp.Poly(num, s)
        if den_poly.degree() < 1:
            raise ParseError("O denominador de G(s)·H(s) deve ter grau ≥ 1.")

        lc = den_poly.LC()
        den_poly = sp.Poly(sp.expand(den / lc), s)
        num_poly = sp.Poly(sp.expand(num / lc), s)

        self.P = sp.cancel(num_poly.as_expr() / den_poly.as_expr())
        self.num_poly = num_poly
        self.den_poly = den_poly
        self.num_expr = num_poly.as_expr()
        self.den_expr = den_poly.as_expr()

        self.num_coeffs = np.array([complex(c) for c in num_poly.all_coeffs()])
        self.den_coeffs = np.array([complex(c) for c in den_poly.all_coeffs()])

        self.zeros = _roots(num_poly)
        self.poles = _roots(den_poly)
        self.zeros_grp = _group(self.zeros)
        self.poles_grp = _group(self.poles)
        self.n_z = len(self.zeros)
        self.n_p = len(self.poles)

        # P(s) = b * prod(s - z) / prod(s - p)
        self.b = complex(num_poly.LC()) if num_poly.degree() >= 0 else 1.0
        if abs(self.b.imag) < 1e-9:
            self.b = self.b.real

        self.factored_P = sp.factor(self.P)

        if self.n_p < 1:
            raise ParseError("Sistema sem polos finitos.")

    # ---- numérico -------------------------------------------------------

    def _polyval(self, coeffs: np.ndarray, val: complex) -> complex:
        return complex(np.polyval(coeffs, complex(val)))

    def num_at(self, val: complex) -> complex:
        return self._polyval(self.num_coeffs, val)

    def den_at(self, val: complex) -> complex:
        return self._polyval(self.den_coeffs, val)

    def char_coeffs(self, k: float) -> np.ndarray:
        """Coeficientes de D(s) = den(s) + k·num(s)."""
        n = max(len(self.den_coeffs), len(self.num_coeffs))
        d = np.zeros(n, dtype=complex)
        nu = np.zeros(n, dtype=complex)
        d[n - len(self.den_coeffs):] = self.den_coeffs
        nu[n - len(self.num_coeffs):] = self.num_coeffs
        return (d + k * nu).real

    def closed_loop_poles(self, k: float) -> np.ndarray:
        c = self.char_coeffs(k)
        c = np.trim_zeros(c, "f")
        if len(c) <= 1:
            return np.array([])
        return np.roots(c)

    def gain_at(self, val: complex) -> float:
        """K = |den(s_i)| / |num(s_i)| (critério de módulo)."""
        nu = abs(self.num_at(val))
        if nu < 1e-12:
            return float("inf")
        return abs(self.den_at(val)) / nu

    def angle_deg_at(self, val: complex) -> float:
        """∠L(s_i) = ∠[num(s_i)/den(s_i)] em graus, em (-180, 180]."""
        nu = self.num_at(val)
        de = self.den_at(val)
        if abs(nu) < 1e-12 or abs(de) < 1e-12:
            return float("nan")
        ang = np.degrees(np.angle(nu / de))
        ang = ((ang + 180.0) % 360.0) - 180.0
        return float(ang)

    def relative_degree(self) -> int:
        return self.n_p - self.n_z
