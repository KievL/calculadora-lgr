"""Projeto de controladores pelo método do LGR (apostila SisContr, cap. 4).

Especificações de desempenho → polos dominantes desejados (seção 4.1) e
cálculo dos parâmetros do controlador pelos critérios de ângulo e de módulo.
"""
from __future__ import annotations

import numpy as np

from .system import OpenLoopSystem


def _ang(origem: complex, destino: complex) -> float:
    return float(np.degrees(np.angle(complex(destino) - complex(origem))))


def _norm360(x: float) -> float:
    return float(x % 360.0)


def specs_to_poles(mp_pct: float, ts: float, criterio: float = 5.0) -> dict:
    """M_P(%) ≤ mp_pct e t_s ≤ ts → ζ_min, (ζω_n)_min e s_d = -ζω_n ± jω_d."""
    if not 0 < mp_pct < 100:
        raise ValueError("M_P deve estar entre 0 e 100%.")
    if ts <= 0:
        raise ValueError("t_s deve ser positivo.")
    num_ts = {5.0: 3.0, 2.0: 4.0}[float(criterio)]
    ln = float(np.log(mp_pct / 100.0))
    zeta = abs(ln) / float(np.sqrt(np.pi ** 2 + ln ** 2))
    sigma = num_ts / ts
    wn = sigma / zeta
    wd = wn * float(np.sqrt(1.0 - zeta ** 2))
    return {"mp": mp_pct, "ts": ts, "criterio": float(criterio), "num_ts": num_ts,
            "ln": ln, "zeta": zeta, "sigma": sigma, "wn": wn, "wd": wd,
            "theta": float(np.degrees(np.arccos(zeta))),
            "sd": complex(-sigma, wd)}


def _contribuicoes(system: OpenLoopSystem, sd: complex):
    thetas = [{"ponto": p, "ang": _ang(p, sd), "mod": abs(sd - p)} for p in system.poles]
    phis = [{"ponto": z, "ang": _ang(z, sd), "mod": abs(sd - z)} for z in system.zeros]
    return thetas, phis


def pid_zeros_iguais(system: OpenLoopSystem, sd: complex) -> dict:
    """PID G_c(s) = K_c (s+z)²/s com z1 = z2 = z (seção 4.4).

    Condição de ângulo: Σθ_planta + θ_c − Σφ_planta − 2φ_c = 180° + q·360°.
    """
    thetas, phis = _contribuicoes(system, sd)
    theta_c = _ang(0.0, sd)
    soma_t = sum(t["ang"] for t in thetas) + theta_c
    soma_f = sum(f["ang"] for f in phis)
    excesso = soma_t - soma_f - 180.0
    phi_c = None
    for q in range(-4, 5):
        cand = (excesso + 360.0 * q) / 2.0
        if 0.0 < cand < 180.0:
            phi_c = cand
            break
    if phi_c is None:  # pragma: no cover
        raise ValueError("Não há zeros reais que satisfaçam a condição de ângulo.")
    z = sd.imag / float(np.tan(np.radians(phi_c))) - sd.real

    prod_p = float(np.prod([t["mod"] for t in thetas])) * abs(sd)
    prod_z = float(np.prod([f["mod"] for f in phis])) if phis else 1.0
    mod_zc = abs(sd + z)
    k_total = prod_p / (prod_z * mod_zc ** 2)
    kc = k_total / abs(system.b)
    return {"sd": sd, "thetas": thetas, "phis": phis, "theta_c": theta_c,
            "soma_t": soma_t, "soma_f": soma_f, "excesso": excesso,
            "phi_c": phi_c, "z": z, "mod_zc": mod_zc, "prod_p": prod_p,
            "prod_z": prod_z, "k_total": k_total, "b": system.b, "Kc": kc,
            "Kp": 2.0 * z * kc, "Ki": kc * z ** 2, "Kd": kc,
            "num_c": kc * np.array([1.0, 2.0 * z, z ** 2]),
            "den_c": np.array([1.0, 0.0])}


def compensador_a_sb(system: OpenLoopSystem, sd: complex) -> dict:
    """G_c(s) = a/(s+b): polo do controlador pela condição de ângulo e a pelo módulo."""
    thetas, phis = _contribuicoes(system, sd)
    soma_t = sum(t["ang"] for t in thetas)
    soma_f = sum(f["ang"] for f in phis)
    theta_b = _norm360(180.0 + soma_f - soma_t)
    sinal = 1.0
    if not 0.0 < theta_b < 180.0:
        theta_b = _norm360(theta_b - 180.0)
        sinal = -1.0
    b = sd.imag / float(np.tan(np.radians(theta_b))) - sd.real
    mod_b = abs(sd + b)
    prod_p = float(np.prod([t["mod"] for t in thetas])) * mod_b
    prod_z = float(np.prod([f["mod"] for f in phis])) if phis else 1.0
    a = sinal * prod_p / (prod_z * abs(system.b))

    num = np.real(system.num_coeffs)
    den = np.real(system.den_coeffs)
    d_sd = np.polyval(den, sd)
    n_sd = np.polyval(num, sd)
    A = np.array([[d_sd.real, n_sd.real], [d_sd.imag, n_sd.imag]])
    rhs = -np.array([(sd * d_sd).real, (sd * d_sd).imag])
    b_lin, a_lin = np.linalg.solve(A, rhs)
    return {"sd": sd, "thetas": thetas, "phis": phis, "soma_t": soma_t,
            "soma_f": soma_f, "theta_b": theta_b, "sinal": sinal, "b": b,
            "mod_b": mod_b, "prod_p": prod_p, "prod_z": prod_z, "kL": system.b,
            "a": a, "a_lin": float(a_lin), "b_lin": float(b_lin), "d_sd": d_sd, "n_sd": n_sd,
            "num_c": np.array([a]), "den_c": np.array([1.0, b])}


def malha_fechada(num_g, den_g, num_h, den_h, num_c, den_c):
    """C(s)/R(s) = Gc·G / (1 + Gc·G·H) e polinômio característico."""
    ng = np.polymul(num_c, num_g)
    dg = np.polymul(den_c, den_g)
    num_t = np.polymul(ng, den_h)
    char = np.polyadd(np.polymul(dg, den_h), np.polymul(ng, num_h))
    return np.real(num_t), np.real(char)


def resposta_degrau(num_t, den_t, t_final: float | None = None, tol: float = 0.05):
    """Resposta ao degrau e medidas M_P(%) e t_s (faixa `tol`)."""
    from scipy import signal

    polos = np.roots(den_t)
    if t_final is None:
        lento = min(abs(p.real) for p in polos if abs(p.real) > 1e-9)
        t_final = float(np.clip(8.0 / lento, 5.0, 60.0))
    t = np.linspace(0.0, t_final, 4000)
    _, y = signal.step(signal.TransferFunction(num_t, den_t), T=t)
    y_inf = float(np.polyval(num_t, 0.0) / np.polyval(den_t, 0.0))
    pico = float(np.max(y))
    mp = max(0.0, (pico - y_inf) / abs(y_inf) * 100.0)
    fora = np.where(np.abs(y - y_inf) > tol * abs(y_inf))[0]
    ts = float(t[fora[-1] + 1]) if len(fora) and fora[-1] + 1 < len(t) else 0.0
    return {"t": t, "y": y, "y_inf": y_inf, "pico": pico, "t_pico": float(t[np.argmax(y)]),
            "mp": mp, "ts": ts, "y0": float(y[0]), "polos": polos}
