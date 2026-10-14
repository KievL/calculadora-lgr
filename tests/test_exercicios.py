"""Testes do projeto de controladores, da discretização e do 2º exercício da 2ª unidade."""
from __future__ import annotations

import numpy as np
import pytest

from lgr import design, discrete, exercicios, report
from lgr.system import OpenLoopSystem


def test_especificacoes_para_polos():
    esp = design.specs_to_poles(10.0, 3.0, 5.0)
    assert esp["zeta"] == pytest.approx(0.5912, abs=1e-4)
    assert esp["sigma"] == pytest.approx(1.0)
    assert esp["sd"] == pytest.approx(complex(-1.0, 1.3644), abs=1e-4)


def test_apostila_pd_especificacoes():
    esp = design.specs_to_poles(20.0, 4.0, 2.0)
    assert esp["zeta"] == pytest.approx(0.46, abs=5e-3)
    assert esp["wd"] == pytest.approx(1.95, abs=1e-2)


# --- apostila p.32: G = 1/(s²+1), polos de MF em -1 ± √3 i -----------------
def test_apostila_pid_zeros_iguais():
    pid = design.pid_zeros_iguais(OpenLoopSystem("1/(s^2+1)", "1"), complex(-1, np.sqrt(3)))
    assert pid["phi_c"] == pytest.approx(96.95, abs=1e-2)
    assert pid["z"] == pytest.approx(0.79, abs=1e-2)
    assert pid["Kc"] == pytest.approx(2.37, abs=1e-2)


def test_questao1():
    d = exercicios.DEFAULTS_Q1
    sol = exercicios.resolver_q1(d["g"], d["h"], d["mp"], d["ts"], d["criterio"], d["T"], d["metodo"])
    pid = sol.dados["pid"]
    assert pid["z"] == pytest.approx(1.3322, abs=1e-4)
    assert pid["Kc"] == pytest.approx(0.5390, abs=1e-4)
    sd = sol.dados["esp"]["sd"]
    assert min(abs(p - sd) for p in sol.dados["polos_mf"]) < 1e-6
    dz = sol.dados["dz"]
    assert dz["num_z"] == pytest.approx([2.9319, 0.8352, 0.0595], abs=1e-4)
    assert dz["den_z"] == pytest.approx([1.0, 0.0, -1.0], abs=1e-9)


def test_questao2():
    d = exercicios.DEFAULTS_Q2
    sol = exercicios.resolver_q2(d["g"], d["h"], d["re"], d["im"], d["T"], d["metodo"])
    c = sol.dados["comp"]
    assert c["a"] == pytest.approx(3.7999, abs=1e-4)
    assert c["b"] == pytest.approx(2.8061, abs=1e-4)
    assert c["a_lin"] == pytest.approx(c["a"], rel=1e-9)
    assert c["b_lin"] == pytest.approx(c["b"], rel=1e-9)
    assert min(abs(p - complex(-2.5, 2)) for p in sol.dados["polos_mf"]) < 1e-6
    polos_z = sorted(sol.dados["dz"]["polos_z"], key=lambda w: (w.real, w.imag))
    assert polos_z == pytest.approx([-4, 1 - c["b"], -1j, 1j], abs=1e-6)


@pytest.mark.parametrize("metodo, num, den", [
    ("euler", [0.5], [1.0, -1.0]),
    ("backward", [0.5, 0.0], [1.0, -1.0]),
    ("tustin", [0.25, 0.25], [1.0, -1.0]),
])
def test_discretizacao_integrador(metodo, num, den):
    d = discrete.discretizar([1.0], [1.0, 0.0], 0.5, metodo)
    assert d["num_z"] == pytest.approx(num)
    assert d["den_z"] == pytest.approx(den)


def test_equacao_diferencas():
    d = discrete.discretizar([1.0], [1.0, 0.0], 0.5, "euler")
    assert discrete.eq_diferencas(d) == r"y[k] = 1,0000\,y[k-1] + 0,5000\,u[k-1]"


def test_pdf():
    d1, d2 = exercicios.DEFAULTS_Q1, exercicios.DEFAULTS_Q2
    sols = [exercicios.resolver_q1(d1["g"], d1["h"], d1["mp"], d1["ts"], d1["criterio"],
                                   d1["T"], d1["metodo"]),
            exercicios.resolver_q2(d2["g"], d2["h"], d2["re"], d2["im"], d2["T"], d2["metodo"])]
    pdf = report.gerar_pdf(sols)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 50_000
    manual = report.gerar_pdf(sols[1:], manual=True)
    assert manual.startswith(b"%PDF")
