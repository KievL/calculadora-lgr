"""Testes de regressão contra os exemplos da apostila e da 1ª lista."""
from __future__ import annotations

import numpy as np
import pytest

from lgr import steps
from lgr.routh import imaginary_axis_crossings
from lgr.system import OpenLoopSystem, ParseError


def test_parsing_basico():
    s = OpenLoopSystem("K/(s*(s+4))", "1")
    assert s.n_p == 2 and s.n_z == 0
    polos = sorted(p.real for p in s.poles)
    assert polos == pytest.approx([-4.0, 0.0])


def test_parsing_erros():
    with pytest.raises(ParseError):
        OpenLoopSystem("K/(x+1)", "1")          # símbolo desconhecido
    with pytest.raises(ParseError):
        OpenLoopSystem("1/(K*s+1)", "1")        # K não-linear


# --- apostila p.5 / p.17: G = K/(s(s+4)) ---------------------------------
def test_apostila_segunda_ordem():
    s = OpenLoopSystem("K/(s*(s+4))", "1")
    si = complex(-2, 2)

    at = steps.angle_test(s, si)
    assert at["pertence"]
    assert at["erro"] == pytest.approx(0.0, abs=1e-6)

    g = steps.gain_at_point(s, si)
    assert g["K"] == pytest.approx(8.0, rel=1e-6)

    asy = steps.asymptotes(s)
    assert asy["sigma"] == pytest.approx(-2.0)
    assert sorted(asy["angulos_deg"]) == pytest.approx([90.0, 270.0])

    bk = steps.breakaway_points(s)
    validos = [c for c in bk["candidatos"] if c["no_lgr"]]
    assert len(validos) == 1
    assert validos[0]["s"].real == pytest.approx(-2.0)
    assert validos[0]["K"] == pytest.approx(4.0)


# --- apostila p.9: 4 polos, 1 zero --------------------------------------
def test_apostila_quatro_polos_um_zero():
    s = OpenLoopSystem("K*(s+1)/(s^4+10*s^3+32*s^2+32*s)", "1")

    asy = steps.asymptotes(s)
    assert asy["sigma"] == pytest.approx(-3.0)
    assert sorted(asy["angulos_deg"]) == pytest.approx([60.0, 180.0, 300.0])

    bk = steps.breakaway_points(s)
    validos = [c for c in bk["candidatos"] if c["no_lgr"]]
    assert any(c["s"].real == pytest.approx(-2.5994, abs=1e-3) for c in validos)


# --- apostila p.11: 2 polos reais + 2 complexos ------------------------
def test_apostila_polos_complexos():
    s = OpenLoopSystem("K/(s*(s^2+8*s+32))", "1/(s+4)")

    assert s.den_coeffs.real == pytest.approx([1, 12, 64, 128, 0])

    cruz = imaginary_axis_crossings(s)
    assert len(cruz) == 1
    assert cruz[0]["omega"] == pytest.approx(3.2660, abs=1e-3)
    assert cruz[0]["K"] == pytest.approx(568.8889, abs=1e-2)

    ang = {round(d["ponto"].imag, 3): d["angulo"]
           for d in steps.departure_arrival_angles(s)}
    # polo -4+4j: ângulo de partida 225° ≡ -135°
    assert ang[4.0] == pytest.approx(-135.0, abs=1e-6)


# --- 1ª lista, item 1: ponto no SPD não pertence ao LGR ----------------
def test_lista_item1_rejeita_ponto():
    s = OpenLoopSystem("K*(s+1)/(s^2+13*s)", "1/s")
    at = steps.angle_test(s, complex(4, 2))
    assert not at["pertence"]

    r12 = steps.step_12(s, complex(4, 2))
    assert not r12.aplicavel


# --- conferência cruzada com python-control ---------------------------
@pytest.mark.parametrize("g,h", [
    ("K/(s*(s+4))", "1"),
    ("K*(s+1)/(s^2+13*s)", "1/s"),
    ("K*(s^2+4*s+5)/(s*(s+5))", "(s+1)/s"),
])
def test_polos_zeros_vs_control(g, h):
    import control as ct

    key = lambda z: (round(z.real, 6), round(z.imag, 6))
    s = OpenLoopSystem(g, h)
    plant = ct.tf(s.num_coeffs.real, s.den_coeffs.real)
    p_ct = sorted((complex(x) for x in plant.poles()), key=key)
    z_ct = sorted((complex(x) for x in plant.zeros()), key=key)
    p_ours = sorted(s.poles, key=key)
    z_ours = sorted(s.zeros, key=key)
    assert p_ours == pytest.approx(p_ct, abs=1e-6)
    assert z_ours == pytest.approx(z_ct, abs=1e-6)


def test_solve_all_roda_sem_ponto():
    s = OpenLoopSystem("K*(s^2+8*s+17)/(s^3+7*s^2+12*s+10)", "(s+1)/(s+5)")
    rs = steps.solve_all(s, None)
    assert len(rs) == 12
    assert not rs[10].aplicavel  # passo 11 exige ponto
