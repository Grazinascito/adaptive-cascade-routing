# Dependências: uv + pyproject com extra `serving`; vLLM roda apenas no Colab

O desenvolvimento acontece em macOS (sem CUDA), enquanto o servidor vLLM do benchmark só roda
no Google Colab (GPU T4) ou em máquina com CUDA 12.x. Decidimos: `pyproject.toml` gerenciado
com uv é a fonte da verdade (lockfile `uv.lock` commitado), vLLM vive no extra `serving`
(nunca instalado no macOS), e `requirements.txt` (core) + `requirements-colab.txt` (com vLLM)
são exportados do lock e commitados porque o pip do Colab não lê `uv.lock`.

## Opções Consideradas

- **requirements.txt único incluindo vllm**: rejeitado — `pip install -r requirements.txt`
  falha no macOS (vLLM exige CUDA), quebrando o ambiente de desenvolvimento.
- **Dois requirements.txt mantidos à mão**: rejeitado — sem lockfile, perde-se a
  reprodutibilidade que um benchmark congelado exige.
