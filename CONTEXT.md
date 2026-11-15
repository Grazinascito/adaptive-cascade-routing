# Adaptive Cascade Routing (benchmark PT-BR)

Contexto de um benchmark empírico de engenharia de inferência que compara três cenários
de execução para classificação de itens de catálogo brasileiro em códigos NCM. Este
glossário padroniza a linguagem usada nos docs (`docs/`) e no código (`src/`).

## Language

### Cenários de execução

**System 1 (fast-path)**:
Caminho rápido: geração curta (max_tokens=12, sem raciocínio) do código NCM via
`/v1/chat/completions`, com confiança pela média geométrica dos logprobs dos tokens gerados.
_Avoid_: modelo leve, typed decision engine, decisão de 1 token (era ASSIN, ver histórico)

**System 2 (slow-path)**:
Caminho lento de raciocínio: geração autorregressiva com Chain-of-Thought e saída JSON estrita
via `/v1/chat/completions`, com a lista de capítulos NCM como contexto da taxonomia.
_Avoid_: modelo pesado, LLM de raciocínio

**Cascata Adaptativa (Adaptive Cascade)**:
Estratégia que aceita a resposta do System 1 quando a confiança é alta e escala para o
System 2 quando a incerteza estatística é alta.
_Avoid_: fallback geral, roteador dinâmico genérico

**Limiar α (alpha)**:
Valor de corte da confiança que separa aceitar o System 1 de escalar para o System 2
(default 0.85).
_Avoid_: threshold genérico, cutoff

### Sinais estatísticos

**Logprobs**:
Log-probabilidades por token devolvidas pelo servidor na geração; base da confiança do
System 1.
_Avoid_: logits, probabilidades brutas

**Confiança (média geométrica)**:
exp(Σ ln p / n) sobre os tokens gerados do código; menos punitiva que o produto puro em
códigos de 8 dígitos. Código inexistente na taxonomia → confiança 0.0 (escala imediato).
_Avoid_: score, P_top (era ASSIN, ver histórico)

**Entropia de Shannon H(P)**:
Medida de incerteza sobre a distribuição top-logprobs do PRIMEIRO token gerado — onde o
modelo decide o início do código.
_Avoid_: impureza, variância

### Dados

**NCM (Nomenclatura Comum do Mercosul)**:
Taxonomia fiscal brasileira oficial em árvore: capítulo (2 dígitos) → posição (4) →
subposição (6) → item (8). A tarefa do benchmark: nome de item → código de 8 dígitos.
_Avoid_: HS Code genérico, categoria fiscal

**NCM-RAFT-samall**:
Dataset público no Hugging Face (yurifacanha/NCM-RAFT-samall; 105k linhas, PT-BR, formato
conversacional) de onde o pool é parseado. Taxonomia derivada: config/ncm_taxonomy.csv
(114 códigos, 19 capítulos).
_Avoid_: o dataset do governo, RAFT (o método de fine-tuning, sem "NCM-")

**Cabeça e cauda (head/tail)**:
Cabeça = códigos com >= 50 nomes únicos no pool (itens fáceis); cauda = códigos raros
(itens ambíguos — o long tail do blog do iFood). O conjunto congelado usa ~65%/35%.
_Avoid_: classe majoritária/minoritária, treino/teste

**Granularidade do rótulo (label_digits)**:
Número de dígitos do código usados na comparação (2, 4, 6 ou 8). Congelada pela sonda do
notebook: o nível mais fino em que a acurácia do System 1 fica em 0.30–0.70.
_Avoid_: nível da taxonomia, profundidade

**Frozen eval set**:
Conjunto congelado de 800 amostras head/tail (seed=42) sobre o qual todos os cenários são
comparados; não muda entre execuções.
_Avoid_: dataset de teste genérico, amostra aleatória

**ASSIN 2 (histórico)**:
Dataset de NLI em PT-BR usado na primeira iteração medida do benchmark (commit 79a8654);
saiu da linha principal no pivot de 2026-11-14 por não ter cauda ambígua.
_Avoid_: citar como dataset atual do benchmark

### Métricas

**SC-A (gap da cauda)**:
Diferença de acurácia S2 puro − S1 puro no subconjunto cauda, em pontos percentuais
(gate >= 10 pp).
_Avoid_: ganho de acurácia geral

**SC-B (recuperação)**:
Fração do gap da cauda que a cascata recupera (gate >= 50%, com a cascata >= S1 puro no
conjunto completo).
_Avoid_: APGR como gate (APGR é informativo pós-pivot)

**p50/p95/p99**:
Percentis de latência das requisições do benchmark.
_Avoid_: latência média, tempo médio de resposta

**Prefix caching**:
Reutilização do KV cache do servidor para o prefixo estático do prompt (instrução + lista
de capítulos), compartilhado entre requisições.
_Avoid_: cache de prompt genérico, cache de resultado

**Taxa de fallback**:
Fração de itens escalados ao System 2; comparada à banda 30–50% reportada pelo blog do
iFood.
_Avoid_: taxa de erro, taxa de escalonamento (usar uma só forma)
