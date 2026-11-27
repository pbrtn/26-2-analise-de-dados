from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display


ROOT = Path(__file__).resolve().parent
SEED = 42
MINHA_PALETTE = ['#0267C1', '#D65108', '#591F0A', '#EFA00B', '#0075C4', '#647789']
minha_palette = MINHA_PALETTE
REGIOES = ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul']
LOCAL = {1: 'Urbana', 2: 'Rural'}
REDE = {1: 'Federal', 2: 'Estadual', 3: 'Municipal', 4: 'Privada'}
COMPUTADORES = ['qt_desktop_aluno', 'qt_comp_portatil_aluno']
RECURSOS = {
    'in_agua_potavel': 'Água potável', 'in_energia_rede_publica': 'Energia da rede',
    'in_banheiro': 'Banheiro', 'in_biblioteca_sala_leitura': 'Biblioteca ou sala de leitura',
    'in_laboratorio_informatica': 'Laboratório de informática',
    'in_quadra_esportes': 'Quadra', 'in_acessibilidade_rampas': 'Rampas',
}


def configurar_graficos():
    sns.set_theme(style='whitegrid', palette=minha_palette, font='DejaVu Sans', font_scale=0.88)
    plt.rcParams.update({'axes.titleweight': 'bold', 'grid.alpha': 0.3,
                         'savefig.bbox': 'tight', 'figure.dpi': 145})


def carregar():
    caminho = ROOT / 'data' / 'dataset_escolas_censo2025.csv'
    if not caminho.is_file():
        caminho = ROOT.parent / 'data' / caminho.name
    bruto = pd.read_csv(caminho, low_memory=False)
    bruto.columns = bruto.columns.str.strip().str.lower()
    campos = [col for col in bruto if col.startswith(('in_', 'qt_', 'tp_', 'co_'))]
    dados = bruto.dropna(subset=['co_entidade']).drop_duplicates('co_entidade').copy()
    dados[campos] = dados[campos].apply(pd.to_numeric, errors='coerce')
    contagens = dados.filter(regex=r'^qt_').columns
    codificados = dados[contagens].isin([88888])
    dados[contagens] = dados[contagens].mask(codificados)
    for coluna in dados.filter(regex=r'^in_'):
        dados[coluna] = dados[coluna].where(dados[coluna].isin([0, 1]))
    ativas = dados.loc[dados['tp_situacao_funcionamento'].eq(1)].copy()
    ativas['local'] = ativas['tp_localizacao'].map(LOCAL)
    ativas['rede'] = ativas['tp_dependencia'].map(REDE)
    ativas['sem_internet'] = 1 - ativas['in_internet']
    ativas['porte'] = pd.cut(ativas['qt_mat_bas'], [-0.1, 50, 200, 500, np.inf],
                           labels=['Até 50', '51–200', '201–500', 'Mais de 500'])
    ativas['computadores'] = ativas[COMPUTADORES].sum(axis=1, min_count=len(COMPUTADORES))
    ativas['alunos_computador'] = ativas['qt_mat_bas'].div(
        ativas['computadores'].where(ativas['computadores'].gt(0)))
    auditoria = {
        'Linhas na base inicial': len(bruto), 'Após validar o identificador': len(dados),
        'Escolas em atividade': len(ativas),
        'Ativas com resposta sobre internet': int(ativas['sem_internet'].notna().sum()),
        'Valores 88888 nas colunas de quantidade (toda a base)': int(codificados.to_numpy().sum()),
    }
    return dados, ativas, auditoria


def numero(valor, casas=0):
    return f'{valor:,.{casas}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def pct(valor, casas=1):
    return f'{numero(valor * 100, casas)}%'


def tabela(quadro):
    def celula(valor):
        return str(valor).replace('|', r'\|').replace('\n', ' ')

    linhas = [list(quadro.columns), ['---'] * len(quadro.columns),
              *quadro.to_numpy().tolist()]
    display(Markdown('\n' + '\n'.join(
        '| ' + ' | '.join(map(celula, linha)) + ' |' for linha in linhas) + '\n'))


def cobertura(dados, campo):
    return dados.groupby(['no_regiao', 'local'], observed=True)[campo].mean().unstack()
