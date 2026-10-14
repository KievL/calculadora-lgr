# Calculadora de Sistemas de Controle — DCA-3701

App Streamlit para a disciplina **DCA-3701 – Projeto de Sistemas de Controle (UFRN)**.

## 2ª Unidade — 2º Exercício

Na barra lateral escolhe-se a **unidade** (1ª ou 2ª). Na 2ª Unidade há dois modos:

- **Preset da lista** — parâmetros das questões já carregados. Ao final, é possível
  baixar o PDF da questão ou o **PDF completo** (Questões 1 e 2);
- **Inserir manualmente** — formulário vazio para um problema do mesmo tipo
  (PID com z₁ = z₂ ou compensador a/(s+b)); ao final, apenas o PDF desse problema.

Questões do preset:

- **Questão 1** — `G(s) = 5(s+3)/(s(s+4))`, `H(s) = 1/(s+1)`:
  - (a) projeto de PID `Kc(s+z)²/s` (z₁ = z₂) para Mₚ ≤ 10% e tₛ(5%) < 3 s,
    pelos critérios de ângulo e módulo (apostila, seções 4.1 e 4.4), com LGR do
    sistema compensado e simulação da resposta ao degrau;
  - (b) discretização do controlador por Tustin, T = 2 s (cap. 5), com forma em
    z⁻¹ e equação de diferenças.
- **Questão 2** — `G(s) = 2(s+1)/(s²+2s+2)`, `H(s) = (s+3)/(s+5)`, `Gc(s) = a/(s+b)`:
  - (a) a e b para polos de MF em −2,5 ± 2i (ângulo → b, módulo → a, verificação
    por Δ(s_d) = 0);
  - (b) discretização de G·H·Gc por Euler, T = 1 s, com análise dos polos em z.

Os PDFs são identificados com a disciplina, a 2ª Unidade e o discente.

## 1ª Unidade — LGR em 12 passos

Resolve, passo a passo, os **12 passos** da apostila *Sistemas de Controle*
(seções 2.2 e 2.4) para a construção do Lugar Geométrico das Raízes de um sistema
realimentado `1 + G(s)H(s) = 1 + K·P(s) = 0`:

- **(a)** esboço do LGR — polos/zeros, segmentos do eixo real, assíntotas,
  pontos de saída/entrada, cruzamento do eixo imaginário (Routh-Hurwitz),
  ângulos de partida/chegada;
- **(b)** teste de um ponto `sᵢ` pelo critério do ângulo de fase e cálculo do
  ganho `K` pelo critério de módulo.

Cada passo mostra a **regra citada da apostila**, o **desenvolvimento algébrico**
(com os números do problema) e o **resultado**, além de gráficos interativos
(Plotly).

## Como executar

```bash
uv sync
uv run streamlit run app.py
```

## Entrada

- **G(s)** e **H(s)**: expressões em `s` e no ganho `K`. Aceita `^` para potência
  e polinômios ou fatores. Exemplos:
  - `K*(s+1)/(s^2+13*s)` e `1/s`
  - `K*(s^2+0.2*s+4)/(s^3+2*s^2+2*s+1)` e `1/(s+2)`
  - `K/(s*(s+4))` e `1`
- **sᵢ**: parte real e imaginária do ponto a testar (item b).

O ganho `K` deve aparecer no numerador. Se for omitido, assume-se `K`
multiplicando todo o numerador de `G(s)·H(s)`.

## Estrutura

| Arquivo | Papel |
|---|---|
| `app.py` | interface Streamlit |
| `lgr/system.py` | parsing de G(s), H(s) → `K·P(s)`; polos, zeros, coeficientes |
| `lgr/steps.py` | os 12 passos (uma função por passo) |
| `lgr/routh.py` | tabela de Routh-Hurwitz simbólica e cruzamento do eixo imaginário |
| `lgr/plots.py` | figuras Plotly |
| `lgr/fmt.py` | formatação numérica (vírgula decimal) |
| `lgr/design.py` | especificações → polos desejados; PID com z₁ = z₂; compensador a/(s+b); simulação |
| `lgr/discrete.py` | discretização (Euler, Backward, Tustin) e equação de diferenças |
| `lgr/exercicios.py` | soluções passo a passo do 2º exercício da 2ª unidade |
| `lgr/figs.py` | figuras Matplotlib (LGR, degrau, plano z) usadas no app e no PDF |
| `lgr/report.py` | geração do PDF (ReportLab + mathtext) |
| `tests/test_lgr.py` | regressão contra os exemplos da apostila e da 1ª lista |
| `tests/test_exercicios.py` | regressão do projeto/discretização (apostila e 2ª unidade) |

## Testes

```bash
uv run pytest
```
