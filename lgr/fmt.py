"""Formatação numérica no padrão da apostila (vírgula decimal, 4 casas)."""
from __future__ import annotations

import numpy as np

TOL = 1e-6


def _clean(x: float, casas: int) -> float:
    x = float(x)
    if abs(x) < 0.5 * 10 ** (-casas):
        return 0.0
    return x


def num(x: float, casas: int = 4) -> str:
    """Formata um número real com vírgula decimal."""
    x = _clean(x, casas)
    s = f"{x:.{casas}f}"
    return s.replace(".", ",")


def cplx(z: complex, casas: int = 4) -> str:
    """Formata um número complexo como 'a', 'b i' ou 'a ± b i'."""
    z = complex(z)
    re = _clean(z.real, casas)
    im = _clean(z.imag, casas)
    if abs(im) < TOL:
        return num(re, casas)
    if abs(re) < TOL:
        return f"{num(im, casas)}\\,i"
    sinal = "+" if im >= 0 else "-"
    return f"{num(re, casas)} {sinal} {num(abs(im), casas)}\\,i"


def cplx_pm(z: complex, casas: int = 4) -> str:
    """Formata como 'a ± b i' (par conjugado)."""
    z = complex(z)
    re = _clean(z.real, casas)
    im = abs(_clean(z.imag, casas))
    if im < TOL:
        return num(re, casas)
    if abs(re) < TOL:
        return f"\\pm\\,{num(im, casas)}\\,i"
    return f"{num(re, casas)} \\pm {num(im, casas)}\\,i"


def deg(theta: float, casas: int = 2) -> str:
    """Ângulo em graus, normalizado para (-180, 180]."""
    t = float(np.degrees(theta)) if abs(theta) <= 2 * np.pi else float(theta)
    t = ((t + 180.0) % 360.0) - 180.0
    if abs(t + 180.0) < 1e-9:
        t = 180.0
    return f"{num(t, casas)}^\\circ"


def deg_raw(theta_deg: float, casas: int = 2) -> str:
    return f"{num(theta_deg, casas)}^\\circ"


def poly_str(coeffs, var: str = "s") -> str:
    """Lista de coeficientes (grau decrescente) -> string LaTeX do polinômio."""
    coeffs = [complex(c) for c in coeffs]
    n = len(coeffs) - 1
    termos: list[str] = []
    for i, c in enumerate(coeffs):
        p = n - i
        if abs(c) < TOL:
            continue
        cr = c.real
        mag = abs(cr)
        # coeficiente
        if abs(mag - 1.0) < TOL and p != 0:
            coef = ""
        else:
            coef = num(mag)
        # variável
        if p == 0:
            v = ""
        elif p == 1:
            v = var
        else:
            v = f"{var}^{{{p}}}"
        termo = (coef + v) if (coef or v) else num(mag)
        sinal = "-" if cr < 0 else "+"
        termos.append((sinal, termo))
    if not termos:
        return "0"
    out = ""
    for k, (sinal, termo) in enumerate(termos):
        if k == 0:
            out = ("-" + termo) if sinal == "-" else termo
        else:
            out += f" {sinal} {termo}"
    return out
