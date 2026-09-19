# Análise de Dados — Censo Escolar 2025

Projeto colaborativo de análise de dados para a disciplina de Análise de Dados (UFSC, 2026.2).

## Escopo

- Limpeza e tratamento dos dados
- Análise exploratória univariada
- Análise exploratória multivariada
- Visualizações com Pandas, Matplotlib e Seaborn

## Estrutura do repositório

```
├── data/
│   └── dataset_escolas_censo2025.csv
├── scripts/
│   ├── limpeza.py
│   └── analise.py
├── requirements.txt
└── README.md
```

## Como rodar

1. Clone o repositório:
   ```
   git clone <url-do-repositorio>
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

4. Rode os scripts a partir da raiz do projeto (ex: `python scripts/limpeza.py`), para que os caminhos relativos ao `data/` funcionem corretamente.

## Dataset

`dataset_escolas_censo2025.csv` — Censo Escolar 2025. Está versionado em `data/` para que todo o time trabalhe sobre a mesma base.

## Time

- Pedro Bertoncini Oliveira (25150202)
- Leonardo Ghizoni (24250183)
- Henrique Silva Antonelli (24100868)
