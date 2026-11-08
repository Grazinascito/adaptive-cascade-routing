# Adaptive Cascade Routing (benchmark PT-BR)

Contexto de um benchmark empírico de engenharia de inferência que compara três cenários de
execução para NLI em Português do Brasil. Este glossário padroniza a linguagem usada nos
docs (`docs/`) e no código (`src/`).

## Language

### Cenários de execução

**System 1 (fast-path)**:
Caminho rápido de decisão tipada: 1 token via endpoint `/v1/systemone`, inspecionando logprobs.
_Avoid_: modelo leve, classificador rápido, typed decision engine

**System 2 (slow-path)**:
Caminho lento de raciocínio: geração autorregressiva com Chain-of-Thought e saída JSON estrita
via `/v1/chat/completions`.
_Avoid_: modelo pesado, LLM de raciocínio

**Cascata Adaptativa (Adaptive Cascade)**:
Estratégia que aceita a resposta do System 1 quando a confiança é alta e escala para o
System 2 quando a incerteza estatística é alta.
_Avoid_: fallback geral, híbrido, roteador dinâmico genérico

**Limiar α (alpha)**:
Valor de corte da confiança que separa aceitar o System 1 de escalar para o System 2
(default 0.85).
_Avoid_: threshold genérico, cutoff

### Sinais estatísticos

**Logprobs**:
Log-probabilidades que o servidor retorna para os rótulos candidatos numa decisão tipada.
_Avoid_: logits, probabilidades brutas

**Confiança (P_top)**:
A maior probabilidade da distribuição normalizada por softmax sobre os logprobs.
_Avoid_: score, probabilidade do modelo

**Entropia de Shannon H(P)**:
Medida de incerteza da distribuição normalizada; alta entropia significa decisão incerta.
_Avoid_: impureza, variância

### Dados

**ASSIN 2**:
Dataset público de NLI em Português do Brasil (NILC/USP), origem das amostras do benchmark.
_Avoid_: ASSIN (sem o 2), dataset de NLI genérico

**NLI (Natural Language Inference)**:
Tarefa de decidir se uma hipótese segue (entailment) ou contradiz a premissa.
_Avoid_: classificação de texto genérica

**Entailment**:
Rótulo binário positivo do benchmark (mapeado do `ENTAILMENT` original do ASSIN 2).
_Avoid_: positivo, sim

**Contradiction**:
Rótulo binário negativo do benchmark (mapeado do `NONE` original do ASSIN 2).
_Avoid_: negativo, não, neutral

**Frozen test set**:
Conjunto congelado de 200 amostras balanceadas (seed=42) sobre o qual todos os cenários são
comparados; não muda entre execuções.
_Avoid_: dataset de teste genérico, amostra aleatória

### Métricas

**APGR (Average Performance Gap Recovered)**:
Fração da diferença de acurácia entre System 2 e System 1 que a Cascata Adaptativa consegue
recuperar.
_Avoid_: ganho de acurácia, acurácia da cascata

**p50/p95/p99**:
Percentis de latência das requisições do benchmark.
_Avoid_: latência média, tempo médio de resposta

**Prefix caching**:
Reutilização do KV cache do servidor para o prefixo estático do prompt, compartilhado entre
requisições.
_Avoid_: cache de prompt genérico, cache de resultado
