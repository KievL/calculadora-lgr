# Calculadora de LGR — 12 passos

App Streamlit para a disciplina **DCA-3701 – Projeto de Sistemas de Controle (UFRN)**.
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
| `tests/test_lgr.py` | regressão contra os exemplos da apostila e da 1ª lista |

## Testes

```bash
uv run pytest
```
