import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

from base import MINHA_PALETTE, REGIOES, RECURSOS, cobertura, numero, pct


def panorama(escolas):
    fig, eixos = plt.subplots(2, 2, figsize=(9, 5.1), layout='constrained')
    for eixo, campo, titulo in zip(eixos.flat, ['no_regiao', 'rede', 'local', 'porte'],
                                 ['Região', 'Rede administrativa', 'Localização', 'Porte por matrículas']):
        valores = escolas[campo].value_counts(sort=campo != 'porte')
        barras = eixo.barh(valores.index, valores / 1000, color=MINHA_PALETTE[0])
        eixo.bar_label(barras, labels=[numero(x / 1000, 1) for x in valores], padding=3)
        eixo.invert_yaxis()
        eixo.margins(x=0.19)
        eixo.set(title=titulo, xlabel='Mil escolas')
    plt.show()


def diferencas(escolas):
    fig, eixos = plt.subplots(1, 3, figsize=(10, 3.8), layout='constrained')
    for eixo, (campo, titulo) in zip(eixos, {'in_internet': 'Internet',
            'in_agua_potavel': 'Água potável', 'in_energia_rede_publica': 'Energia da rede'}.items()):
        valores = cobertura(escolas, campo).reindex(REGIOES)[['Urbana', 'Rural']] * 100
        sns.heatmap(valores, annot=True, fmt='.1f', cmap='crest', vmin=40, vmax=100,
                    cbar=False, ax=eixo, linewidths=0.8)
        eixo.set(title=titulo + ' (%)', xlabel='', ylabel='')
    plt.show()


def recursos(escolas):
    fig, (esq, dir_) = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    campos = ['in_biblioteca_sala_leitura', 'in_laboratorio_informatica', 'in_quadra_esportes']
    rec = escolas.groupby('porte', observed=True)[campos].mean().rename(columns=RECURSOS)
    rec.plot(ax=esq, marker='o', color=MINHA_PALETTE[:3], rot=18)
    esq.yaxis.set_major_formatter(PercentFormatter(1))
    esq.set(title='Oferta de recursos por porte', xlabel='Matrículas', ylabel='Escolas com recurso')
    esq.legend(title=None, fontsize=7, loc='upper left')
    mediana = escolas.groupby('no_regiao')['alunos_computador'].median().sort_values()
    barras = dir_.barh(mediana.index, mediana, color=MINHA_PALETTE[1])
    dir_.bar_label(barras, labels=[numero(x, 1) for x in mediana], padding=3)
    dir_.margins(x=.2)
    dir_.set(title='Alunos por computador', xlabel='Mediana entre escolas elegíveis')
    plt.show()


def perfis_exploratorios(escolas):
    fig, eixos = plt.subplots(2, 2, figsize=(10, 6.3), layout='constrained')
    positivas = escolas.loc[escolas.qt_mat_bas.gt(0), 'qt_mat_bas']
    sns.histplot(x=positivas, log_scale=True, bins=26, color=MINHA_PALETTE[0], ax=eixos[0, 0])
    eixos[0, 0].set(title='Distribuição do porte', xlabel='Matrículas (escala log)', ylabel='Escolas')
    matriculas = escolas.groupby(['no_regiao', 'rede'], observed=True)['qt_mat_bas'].sum().unstack()
    (matriculas.reindex(REGIOES) / 1e6).plot.barh(stacked=True, ax=eixos[0, 1],
                                                color=MINHA_PALETTE[:4])
    eixos[0, 1].set(title='Matrículas por região e rede', xlabel='Milhões de matrículas', ylabel='')
    eixos[0, 1].legend(fontsize=7, loc='lower right')
    eja = escolas['qt_mat_eja'].gt(0).where(escolas['qt_mat_eja'].notna()).groupby(escolas.rede).mean()
    eixos[1, 0].bar(eja.index, eja, color=MINHA_PALETTE[1])
    eixos[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    eixos[1, 0].set(title='Educação de jovens e adultos', ylabel='Escolas com matrículas em EJA')
    escolhidos = {'in_biblioteca_sala_leitura': 'Biblioteca', 'in_quadra_esportes': 'Quadra',
                  'in_laboratorio_informatica': 'Lab. informática', 'in_acessibilidade_rampas': 'Rampas'}
    corr = escolas[list(escolhidos)].rename(columns=escolhidos).corr()
    sns.heatmap(corr, vmin=0, vmax=1, cmap='crest', annot=True, cbar=False, ax=eixos[1, 1])
    eixos[1, 1].set(title='Correlação entre indicadores de recurso')
    eixos[1, 1].tick_params(axis='x', labelrotation=22)
    plt.show()


def texto_exploratorio(escolas):
    urbano = escolas.loc[escolas.local.eq('Urbana'), 'in_internet'].mean()
    rural = escolas.loc[escolas.local.eq('Rural'), 'in_internet'].mean()
    norte = escolas.loc[escolas.local.eq('Rural') & escolas.no_regiao.eq('Norte')]
    porte = escolas.groupby('porte', observed=True)['in_laboratorio_informatica'].mean()
    mediana = escolas.groupby('no_regiao')['alunos_computador'].median()
    display(Markdown(f'''
A cobertura de internet alcança **{pct(urbano)} nas escolas urbanas** e **{pct(rural)} nas rurais**,
uma diferença de {numero((urbano-rural)*100, 1)} pontos percentuais. No Norte rural, {pct(norte.in_agua_potavel.mean())}
declaram água potável e {pct(norte.in_energia_rede_publica.mean())} estão ligadas à rede pública de energia.
A medida de energia corresponde à rede pública e não abrange fontes alternativas.

Laboratório de informática aparece em {pct(porte.iloc[0])} das escolas com até 50 matrículas
e em {pct(porte.iloc[-1])} das com mais de 500. Entre escolas elegíveis para a razão de computadores,
a mediana chega a {numero(mediana['Norte'],1)} alunos por equipamento no Norte e {numero(mediana['Sul'],1)} no Sul.
Essa razão descreve disponibilidade nas escolas onde foi possível calculá-la. A combinação de diferenças
territoriais e de porte orienta a escolha de atributos das duas etapas de modelagem.
'''))
