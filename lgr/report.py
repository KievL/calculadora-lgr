"""Geração do PDF com a solução do 2º Exercício da 2ª Unidade."""
from __future__ import annotations

import io
import os
from datetime import date

import matplotlib
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from . import fmt
from .exercicios import Solucao

DISCENTE = "Kiev Luiz Freitas Guedes"
DISCIPLINA = "DCA-3701.0 – Projeto de Sistemas de Controle"
PERIODO = "2026.2"
ATIVIDADE = "2º Exercício da 2ª Unidade"
ATIVIDADE_MANUAL = "Problema com parâmetros manuais"

_LARGURA = A4[0] - 4.0 * cm
_EQ_DPI = 220
_EQ_PT = 11


def _registrar_fontes() -> str:
    pasta = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(pasta, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(pasta, "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Oblique", os.path.join(pasta, "DejaVuSans-Oblique.ttf")))
        pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                                      italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")
    return "DejaVu"


def _estilos(fonte: str) -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("base", fontName=fonte, fontSize=9.5, leading=13.5, spaceAfter=4)
    return {
        "base": base,
        "cab": ParagraphStyle("cab", parent=base, alignment=TA_CENTER, fontSize=10, leading=13),
        "cab_b": ParagraphStyle("cab_b", parent=base, alignment=TA_CENTER, fontSize=11,
                                leading=14, fontName=fonte + "-Bold"),
        "titulo": ParagraphStyle("titulo", parent=base, alignment=TA_CENTER, fontSize=14,
                                 leading=18, fontName=fonte + "-Bold", spaceBefore=6, spaceAfter=8),
        "questao": ParagraphStyle("questao", parent=base, fontSize=13, leading=17,
                                  fontName=fonte + "-Bold", spaceBefore=6, spaceAfter=6,
                                  textColor=colors.HexColor("#1f4e8c")),
        "parte": ParagraphStyle("parte", parent=base, fontSize=11.5, leading=15,
                                fontName=fonte + "-Bold", spaceBefore=8, spaceAfter=4,
                                textColor=colors.HexColor("#2e7d32")),
        "passo": ParagraphStyle("passo", parent=base, fontSize=10.5, leading=14,
                                fontName=fonte + "-Bold", spaceBefore=6, spaceAfter=3),
        "caixa": ParagraphStyle("caixa", parent=base, spaceAfter=0),
    }


def _eq_image(latex: str) -> Image:
    buf = io.BytesIO()
    mathtext.math_to_image(f"${fmt.tex(latex)}$", buf,
                           prop=FontProperties(size=_EQ_PT, math_fontfamily="cm"),
                           dpi=_EQ_DPI, format="png")
    buf.seek(0)
    w_px, h_px = ImageReader(buf).getSize()
    buf.seek(0)
    w, h = w_px / _EQ_DPI * 72.0, h_px / _EQ_DPI * 72.0
    max_w = _LARGURA - 0.6 * cm
    if w > max_w:
        h *= max_w / w
        w = max_w
    img = Image(buf, width=w, height=h)
    img.hAlign = "CENTER"
    return img


def _fig_image(figura, largura: float = 15.0 * cm) -> Image:
    buf = io.BytesIO()
    figura.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    buf.seek(0)
    w_px, h_px = ImageReader(buf).getSize()
    buf.seek(0)
    img = Image(buf, width=largura, height=largura * h_px / w_px)
    img.hAlign = "CENTER"
    return img


def _caixa(conteudo, fundo: str, borda: str) -> Table:
    t = Table([[conteudo]], colWidths=[_LARGURA])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(fundo)),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(borda)),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _cabecalho(st: dict, data_txt: str, atividade: str) -> list:
    linhas = [
        [Paragraph("Universidade Federal do Rio Grande do Norte", st["cab_b"])],
        [Paragraph("Centro de Tecnologia", st["cab"])],
        [Paragraph("Departamento de Engenharia de Computação e Automação", st["cab"])],
        [Paragraph(f"{DISCIPLINA} – Teoria / Período: {PERIODO}", st["cab"])],
    ]
    t = Table(linhas, colWidths=[_LARGURA])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.2, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    ident = Table(
        [[Paragraph(f"<b>Discente:</b> {DISCENTE}", st["base"]),
          Paragraph(f"<b>Data:</b> {data_txt}", st["base"])],
         [Paragraph("<b>Unidade:</b> 2ª Unidade", st["base"]),
          Paragraph(f"<b>Atividade:</b> {atividade}", st["base"])]],
        colWidths=[_LARGURA * 0.5, _LARGURA * 0.5])
    ident.setStyle(TableStyle([
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#888888")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [t, Spacer(1, 8), ident, Spacer(1, 4),
            Paragraph(f"{atividade if atividade == ATIVIDADE else '2ª Unidade — ' + atividade}"
                      " — Solução", st["titulo"])]


def _rodape(atividade: str):
    rotulo = atividade if atividade == ATIVIDADE else f"2ª Unidade · {atividade}"

    def desenhar(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(2.0 * cm, 1.2 * cm,
                          f"DCA-3701 · {rotulo} · Discente: {DISCENTE}")
        canvas.drawRightString(A4[0] - 2.0 * cm, 1.2 * cm, f"Página {doc.page}")
        canvas.restoreState()
    return desenhar


def _itens(itens, st: dict) -> list:
    out = []
    for tipo, valor in itens:
        if tipo == "txt":
            out.append(Paragraph(valor, st["base"]))
        elif tipo == "eq":
            out += [_eq_image(valor), Spacer(1, 4)]
        elif tipo == "fig":
            out += [Spacer(1, 2), _fig_image(valor), Spacer(1, 4)]
        elif tipo == "ok":
            out += [_caixa(Paragraph(valor, st["caixa"]), "#e8f5e9", "#2e7d32"), Spacer(1, 4)]
        elif tipo == "nota":
            out += [_caixa(Paragraph("<b>Observação:</b> " + valor, st["caixa"]),
                           "#fff8e1", "#e6a100"), Spacer(1, 4)]
    return out


def _solucao(sol: Solucao, st: dict, manual: bool) -> list:
    rotulo = "Problema" if manual else f"Questão {sol.numero}"
    out = [Paragraph(f"{rotulo} — {sol.titulo}", st["questao"])]
    out += _itens(sol.enunciado.itens, st)
    for parte in sol.partes:
        out.append(Paragraph(parte.titulo, st["parte"]))
        for passo in parte.passos:
            bloco = [Paragraph(passo.titulo, st["passo"])] + _itens(passo.itens, st)
            if len(bloco) <= 6:
                out.append(KeepTogether(bloco))
            else:
                out.append(KeepTogether(bloco[:3]))
                out += bloco[3:]
    resp = [Paragraph(f"<b>Respostas{'' if manual else f' da Questão {sol.numero}'}</b>", st["caixa"])]
    resp += [Paragraph(r, st["caixa"]) for r in sol.respostas]
    out += [Spacer(1, 6), _caixa(resp, "#e3ecf7", "#1f4e8c")]
    return out


def gerar_pdf(solucoes: list[Solucao], data_ref: date | None = None,
              manual: bool = False) -> bytes:
    """`manual=True` identifica um problema com parâmetros fora da lista."""
    atividade = ATIVIDADE_MANUAL if manual else ATIVIDADE
    fonte = _registrar_fontes()
    st = _estilos(fonte)
    data_txt = (data_ref or date.today()).strftime("%d/%m/%Y")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=1.6 * cm, bottomMargin=1.8 * cm,
        title=f"{atividade} — DCA-3701 — {DISCENTE}", author=DISCENTE,
        subject=f"{DISCIPLINA} · 2ª Unidade")
    historia = _cabecalho(st, data_txt, atividade)
    for i, sol in enumerate(solucoes):
        if i:
            historia.append(PageBreak())
        historia += _solucao(sol, st, manual)
    rodape = _rodape(atividade)
    doc.build(historia, onFirstPage=rodape, onLaterPages=rodape)
    return buf.getvalue()
