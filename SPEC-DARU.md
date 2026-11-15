# Spec — Benchmark DARU (v1, 15/11/2026)

## 1. Objetivo

Tese: a economia de custo da arquitetura híbrida do iFood (Sugar) também existe quando trocamos "modelo pequeno vs modelo grande" por "System 1 vs System 2 no mesmo modelo".

- **System 1** = decisão de 1 token (probabilidades dos rótulos). Rápido e barato.
- **System 2** = geração com raciocínio passo a passo (CoT). Lento e caro.
- **Cascata** = System 1 primeiro; se a confiança dele for baixa, a requisição sobe para o System 2.

Prova: 3 cenários no mesmo conjunto fixo de itens, mesmo modelo, mesmo servidor.

**Gates** (critérios de aprovação, medidos uma vez no conjunto de teste):

| Cenário | Precisa mostrar |
|---|---|
| SC-A: System 1 sozinho | custo e latência baixos; acurácia na cauda pelo menos 10 pontos percentuais abaixo do System 2 |
| SC-B: System 2 sozinho | maior acurácia dos três; maior custo de tokens e latência |
| SC-C: cascata | recupera ≥ 50% da diferença de acurácia entre S1 e S2 (APGR ≥ 0,5) gastando ≤ 40% dos tokens do System 2 |

### 1.1 O que este benchmark afirma — e o que não afirma

**Afirmamos:** no mesmo modelo, a cascata (System 1 primeiro, System 2 só na incerteza) retém a maior parte da acurácia do System 2 com uma fração do custo. Esse é o mesmo *padrão* de ganho da arquitetura híbrida do iFood: caminho barato na frente, caminho caro só quando necessário.

**Não afirmamos:** (1) números iguais aos do iFood — os 60–80% de economia deles foram medidos no catálogo deles, com modelos deles, em escala de produção; o nosso número será o que medirmos, contra o nosso baseline; (2) que reproduzimos o ciclo completo do iFood. Correspondência de papéis:

| Papel no iFood | Papel neste benchmark |
|---|---|
| Sugar: roteamento híbrido (modelo leve primeiro, frontier no fallback) | Cascata S1→S2 (decisão de 1 token primeiro, CoT no fallback) — núcleo replicado |
| Pepper: geração de ground truth para medir | Rótulos reais do DARU (18 maiores varejistas do Brasil) — medido contra ouro real |
| Revisor: auditoria + transformar erro em aprendizado (melhoria contínua em produção) | Fora do escopo da medição. O System 2 já registra o insight (regra) a cada caso resolvido, sem custo extra; a reinjeção é trabalho futuro com experimento de 2 runs já desenhado |

A acurácia da cascata **não** depende do Revisor: quem sustenta a qualidade é o fallback (todo caso incerto sobe para o System 2), exatamente como no iFood — o blog deles atribui a economia ao roteamento (50–70% das inferências no modelo leve), e não ao Revisor. O Revisor existe para melhorar o sistema ao longo do tempo em produção; uma medição congelada de um dia não consegue (nem deve) demonstrar isso.

## 2. Dataset (DARU)

Fonte: Kaggle `gilsileydaru/classificao-produtos-varejo-cpg-prbr`. Descrições reais dos 18 maiores varejistas do Brasil (ranking ABRAS). Paper: Darú et al., Semina 2022 (USP/ICMC). Download anônimo confirmado hoje.

Fatos verificados hoje no arquivo real (`DATASETPRODUTOSVAREJOBRASIL.xlsx`):

- 158.113 células de dados no Excel (+1 de cabeçalho). 119.225 (75%) se separam de forma limpa em 5 campos: descrição, segmento, categoria, subcategoria, produto canônico. As outras linhas (25%) ficam fora. O parser é determinístico e fica protegido por teste.
- O Excel veio com as 5 colunas embolidas em 1 célula. Regra de leitura: os 4 últimos campos são a taxonomia; o que sobra antes é a descrição.
- Exemplos reais: `APRESUNTADO REZENDE PEC KG`, `CARNE SUIN ESPINHACO KG`, `lte cond moca desn tp 395G` — estilo de nota fiscal, igual ao exemplo `HMBR ART C/ CHED` do blog do iFood.

**Nível de rótulo escolhido: categoria** (65 valores no arquivo; 35 com ≥ 100 exemplos). Comida e bebida dominam: MERCEARIA 23.283, FRIOS E LATICINIOS 12.133, AÇOUGUE E PEIXARIA 6.308, HORTIFRUTI (FLV) 6.292, BEBIDAS ALCOÓLICAS 4.513, SUCOS E REFRESCOS 4.186, REFRIGERANTES 2.514.

**Classes do benchmark: 12 categorias** de comida/bebida (lista final, medidas no arquivo real): MERCEARIA, FRIOS E LATICINIOS, AÇOUGUE E PEIXARIA, PREPARADOS E CONGELADOS, HORTIFRUTI (FLV), PADARIA E CONFEITARIA, BEBIDAS ALCOÓLICAS, SUCOS E REFRESCOS, REFRIGERANTES, LACTEOS, VINHOS E FERMENTADOS, AGUA MINERAL — todas com ≥ 1.400 exemplos. Lista fechada no `config/benchmark_config.yaml`, protegida por teste. Limpeza, higiene e baby care ficam de fora (escopo = comida e bebida, como no contexto do iFood). Motivo de 12: o System 1 usa 12 opções com apelido de 1 letra (A a L), o que mantém a pergunta de decisão confiável.

## 3. Conjuntos (100% amostrados de linhas reais; nada é gerado)

- **Cauda** (`is_tail = true`): descrição com ≤ 20 caracteres. 8.359 itens no arquivo. São os nomes truncados e abreviados — a parte difícil.
- **Cabeça**: descrição com > 20 caracteres. A parte fácil.
- **Validação: 200 itens** (140 cabeça + 60 cauda). Único lugar onde o α é escolhido.
- **Teste congelado: 600 itens** (420 cabeça + 180 cauda). Número final medido uma única vez.
- seed 42, sem sobreposição entre validação e teste, sem descrição duplicada.

## 4. Motor

- **Modelo: `Qwen/Qwen3-4B-AWQ`**, servidor vLLM único com `--enable-prefix-caching`. Motivo: o histórico do repo (commits `86daec3`, `aa6c5d1`) mostra que o Qwen2.5-7B-AWQ não funciona no endpoint `/v1/systemone`.
- **System 1**: endpoint `/v1/systemone`, pergunta do tipo escolha ("choice") com as 12 categorias como opções (apelidos A–L). O servidor devolve a probabilidade de cada opção. O cliente calcula P_top (maior probabilidade) e entropia.
- **Roteador**: se P_top ≥ α, aceita a resposta do System 1; senão, marca `routed_to_system_2 = true` e chama o System 2. α inicial 0,85; α final escolhido na validação (varredura de 0,70 a 0,90).
- **System 2**: `/v1/chat/completions`, raciocínio passo a passo, resposta em JSON com os campos `reasoning` e `category`.
- **Revisor do iFood: fora do escopo de hoje.** O System 2 registra um `revisor_insight` (frase com a regra do acerto) no log, porque já sai de graça. Reinjetar regras no prompt fica para uma fase futura.

## 5. Métricas

- Acurácia por cenário: total e só na cauda.
- Latência p50, p95 e p99 por item.
- Tokens de saída (decode) somados por cenário.
- APGR = (acurácia_cascata − acurácia_S1) / (acurácia_S2 − acurácia_S1).
- Tudo salvo por item em `results/raw_benchmark_data.json` (item, rótulo real, é-cauda, predição do S1, P_top, entropia, latências, roteado?, predição do S2, tokens, acerto).
- Saída inválida do System 2 (JSON quebrado, ou categoria fora da lista mesmo após casamento aproximado determinístico com corte 0,85 — ex.: `SUCCOS E REFRESCOS` → `SUCOS E REFRESCOS`) vira **erro contado do item** (`s2_parse_ok`, `s2_category_fuzzy` no registro e no sumário), nunca uma exceção que derruba o run.
- Checkpoints parciais por fase em `results/partial_*.json`: se o run cair no meio, o trabalho de GPU já feito fica salvo em disco.

## 6. Pilot antes do run completo

100 itens (70 cabeça + 30 cauda). Se a acurácia do System 2 na cauda não superar a do System 1 por pelo menos 10 pontos percentuais, paramos e reportamos. Motivo: é a mesma falha do ASSIN 2, e desta vez queremos descobrir em 1 hora, não depois do run completo de GPU.

## 7. O que NÃO fazemos hoje

Artigo longo, README em inglês, reinjeção do Revisor, mais de 4 valores no sweep de α.

## 8. Ordem de execução (hoje, começo 14h10)

| Horário | O quê | Quem |
|---|---|---|
| 14:10–14:25 | Você lê esta spec e aprova (ou aponta mudanças) | você |
| 14:25–17:00 | Implementação em PRs pequenos: loader DARU, restaurar System 1 do histórico (commit `79a8654`), prompts das 12 categorias, config, testes, avaliador, notebook | eu (subagentes) |
| 17:00–17:30 | Você abre o notebook no Colab (GPU T4) e roda tudo | você (conta Google) |
| 17:30–19:30 | Pilot de 100 itens → se passar, run completo (3 cenários + varredura de α na validação + teste final) | notebook |
| 19:30–20:30 | Gráficos + rascunho do post em PT-BR com os números medidos | eu |

Riscos declarados: (1) o Colab gratuito pode estar sem GPU T4 disponível — aí o run desloca para o primeiro horário com GPU; (2) o pilot pode reprovar o gate — aí o dia termina com um diagnóstico honesto em vez de um número inválido.

## 9. Referências (fontes fixadas — nada aqui depende da memória de nenhuma IA)

1. **iFood — Sugar, Revisor e Pepper** (arquitetura híbrida de referência): https://tech.ifood.com.br/data-ai/sugar-revisor-and-pepper-the-agentic-cycle-that-maintains-ifoods-catalog-scale-and-consistency/
2. **iFood — Tokens are money** (princípio de custo): https://tech.ifood.com.br/en/data-ai/tokens-are-money-a-strategic-guide-to-cost-optimization-in-llms
3. **RouteLLM** (regra R_α e métrica APGR): Ong et al., arXiv:2406.18665 — https://arxiv.org/abs/2406.18665
4. **FrugalGPT** (mecânica da cascata sequencial): Chen et al., arXiv:2305.05176 — https://arxiv.org/abs/2305.05176
5. **DARU** (dataset): https://www.kaggle.com/datasets/gilsileydaru/classificao-produtos-varejo-cpg-prbr · paper: Darú et al., DOI 10.5433/1679-0375.2022v43n2p189
6. **vLLM** (servidor de inferência e endpoint `/v1/systemone`): https://docs.vllm.ai

Propósito (por que este projeto existe): provar que a economia da arquitetura híbrida do iFood também existe trocando "modelo pequeno vs grande" por "System 1 vs System 2 no mesmo modelo" — material de abordagem para a vaga de estágio no iFood.
