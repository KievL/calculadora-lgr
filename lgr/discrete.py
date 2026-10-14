"""Aproximação discreta de funções de transferência contínuas (apostila, cap. 5)."""
from __future__ import annotations

import numpy as np

from . import fmt

METODOS = {
    "euler": {"nome": "Euler (Forward)", "subst": r"s = \dfrac{z-1}{T}", "eq": "5.3"},
    "backward": {"nome": "Backward", "subst": r"s = \dfrac{z-1}{Tz}", "eq": "5.5"},
    "tustin": {"nome": "Tustin (trapezoidal / bilinear)",
               "subst": r"s = \dfrac{2}{T}\,\dfrac{z-1}{z+1}", "eq": "5.7"},
}


def _subst(metodo: str, T: float):
    """s = p(z)/q(z) para cada método."""
    if metodo == "euler":
        return np.array([1.0, -1.0]), np.array([T])
    if metodo == "backward":
        return np.array([1.0, -1.0]), np.array([T, 0.0])
    if metodo == "tustin":
        return np.array([2.0, -2.0]), np.array([T, T])
    raise ValueError(f"Método desconhecido: {metodo}")


def _trim(c: np.ndarray) -> np.ndarray:
    c = np.real_if_close(np.asarray(c, dtype=complex)).real
    c = np.where(np.abs(c) < 1e-12 * max(1.0, np.max(np.abs(c))), 0.0, c)
    nz = np.flatnonzero(c)
    return c[nz[0]:] if len(nz) else np.array([0.0])


def _apply(coeffs: np.ndarray, p: np.ndarray, q: np.ndarray, n: int) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float)
    grau = len(coeffs) - 1
    out = np.zeros(1)
    for i, c in enumerate(coeffs):
        k = grau - i
        out = np.polyadd(out, c * np.polymul(_polypow(p, k), _polypow(q, n - k)))
    return out


def _polypow(p: np.ndarray, k: int) -> np.ndarray:
    out = np.array([1.0])
    for _ in range(k):
        out = np.polymul(out, p)
    return out


def discretizar(num_s, den_s, T: float, metodo: str) -> dict:
    """H(z) = H(s)|_{s = p(z)/q(z)}, normalizada com denominador mônico."""
    num_s = _trim(num_s)
    den_s = _trim(den_s)
    p, q = _subst(metodo, T)
    n = max(len(num_s), len(den_s)) - 1
    num_z = _trim(_apply(num_s, p, q, n))
    den_z = _trim(_apply(den_s, p, q, n))
    lc = den_z[0]
    num_z, den_z = num_z / lc, den_z / lc
    polos_z = np.roots(den_z) if len(den_z) > 1 else np.array([])
    zeros_z = np.roots(num_z) if len(num_z) > 1 else np.array([])
    return {"metodo": metodo, "T": T, "num_s": num_s, "den_s": den_s,
            "num_z": num_z, "den_z": den_z, "polos_z": polos_z, "zeros_z": zeros_z,
            "estavel": bool(np.all(np.abs(polos_z) < 1.0 - 1e-9)),
            "n": len(den_z) - 1, "m": len(num_z) - 1}


def poly_zinv(coeffs, atraso: int = 0) -> str:
    """Coeficientes [c0, c1, ...] → c0 z^{-atraso} + c1 z^{-(atraso+1)} + ..."""
    termos = []
    for i, c in enumerate(coeffs):
        if abs(c) < 1e-12:
            continue
        k = atraso + i
        mag = abs(c)
        coef = "" if (abs(mag - 1.0) < 1e-12 and k) else fmt.num(mag)
        var = "" if k == 0 else f"z^{{-{k}}}"
        termos.append(("-" if c < 0 else "+", coef + var))
    if not termos:
        return "0"
    out = ("-" if termos[0][0] == "-" else "") + termos[0][1]
    for sinal, t in termos[1:]:
        out += f" {sinal} {t}"
    return out


def forma_zinv(d: dict) -> str:
    atraso = d["n"] - d["m"]
    return (f"\\dfrac{{{poly_zinv(d['num_z'], atraso)}}}"
            f"{{{poly_zinv(d['den_z'])}}}")


def eq_diferencas(d: dict, y: str = "y", u: str = "u") -> str:
    """y[k] = -a1 y[k-1] - ... + b0 u[k-d] + ..."""
    atraso = d["n"] - d["m"]
    termos = []
    for i, a in enumerate(d["den_z"][1:], start=1):
        if abs(a) > 1e-12:
            termos.append((-a, f"{y}[k-{i}]"))
    for j, b in enumerate(d["num_z"]):
        if abs(b) > 1e-12:
            k = atraso + j
            termos.append((b, f"{u}[k]" if k == 0 else f"{u}[k-{k}]"))
    if not termos:
        return f"{y}[k] = 0"
    out = f"{y}[k] = "
    for i, (c, v) in enumerate(termos):
        mag = fmt.num(abs(c))
        if i == 0:
            out += ("-" if c < 0 else "") + f"{mag}\\,{v}"
        else:
            out += f" {'-' if c < 0 else '+'} {mag}\\,{v}"
    return out
