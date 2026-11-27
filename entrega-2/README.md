# Entrega 2 | Censo Escolar 2025 | Grupo 9

O `relatorio.pdf` é o documento para leitura. O `relatorio.qmd` e os módulos Python reproduzem as tabelas, figuras e modelos usando os arquivos da pasta `data`. O pacote inclui o recorte de microdados da Entrega 1 e a malha estadual de 2025 do IBGE.

## Reprodução com o Positron e o Quarto

1. Extraia o ZIP inteiro em uma pasta local e abra essa pasta no Positron.
2. Use Python 3.12 e uma instalação do Quarto com XeLaTeX disponível. A renderização foi verificada com Quarto 1.11.9 e TinyTeX 2026.11. `quarto check` verifica a instalação. Se precisar instalar LaTeX, execute `quarto install tinytex`.
3. Ative o ambiente Python escolhido e instale as dependências uma vez com `python -m pip install -r requirements.txt`. Se o ambiente for Conda, use `conda activate <nome>` antes de executar o Quarto, para disponibilizar as bibliotecas nativas.
4. Na raiz da pasta extraída, execute `quarto render relatorio.qmd --to pdf`. Se o Positron selecionar outro Python, ajuste `QUARTO_PYTHON` para o interpretador correto.

O treinamento pode levar alguns minutos. Com as dependências instaladas, a renderização lê os dados e a malha localmente. O relatório define sementes para a amostragem e os modelos.

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `relatorio.qmd` | Relatório principal e células de execução |
| `base.py`, `exploratoria.py` e `evidencias.py` | Preparação, análise exploratória e saídas complementares |
| `espacial.py` | Integração com a malha e mapas estaduais |
| `agrupamento.py` | Comparação de agrupamentos e perfis em PCA |
| `supervisionado.py` | Engenharia de atributos, comparação de modelos e teste por município |
| `regressao.py` | Estimativa da quantidade de computadores e validação por município |
| `caderno.py` | Dossiês de dados, território e diagnósticos por modelo |
| `data/` | Recorte do Censo e malha estadual do IBGE |
| `defs_simple.tex`, `cover.png`, `logo.png` e `fonts/` | Estilo e capa do template do professor e arquivos locais da fonte Lato |

No clone do repositório, a pasta `entrega-2/` pode reutilizar o CSV já versionado em `data/` na raiz. Execute o comando na pasta `entrega-2/`.
