import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from base import MINHA_PALETTE, RECURSOS, SEED, numero, pct, tabela


CAMPOS = ['in_agua_potavel', 'in_energia_rede_publica', 'in_banheiro',
          'in_biblioteca_sala_leitura', 'in_laboratorio_informatica',
          'in_quadra_esportes', 'in_acessibilidade_rampas']


def agrupar(escolas):
    amostra = escolas.sample(n=min(4000, len(escolas)), random_state=SEED).copy()
    atributos = amostra[CAMPOS].fillna(np.nan).copy()
    atributos['log_matriculas'] = np.log1p(amostra['qt_mat_bas'])
    matriz = StandardScaler().fit_transform(SimpleImputer(strategy='most_frequent').fit_transform(atributos))
    resultados, rotulos = [], {}
    with threadpool_limits(limits=2):
        for k in range(2, 7):
            modelos = {'K-Means': KMeans(k, n_init=15, random_state=SEED),
                       'Ward': AgglomerativeClustering(n_clusters=k, linkage='ward'),
                       'Mistura gaussiana': GaussianMixture(k, n_init=3, covariance_type='diag',
                                                           reg_covar=1e-4, random_state=SEED)}
            for nome, modelo in modelos.items():
                grupos = modelo.fit_predict(matriz)
                tamanhos = pd.Series(grupos).value_counts()
                escore = silhouette_score(matriz, grupos) if 1 < len(tamanhos) < len(grupos) else np.nan
                resultados.append((nome, k, escore, tamanhos.min() / len(grupos), len(tamanhos)))
                rotulos[nome, k] = grupos
    grade = pd.DataFrame(resultados, columns=['Método', 'Grupos', 'Silhueta', 'Menor grupo', 'Efetivos'])
    validos = grade[(grade['Menor grupo'] >= .05) & (grade.Grupos == grade.Efetivos)]
    melhor = validos.sort_values(['Silhueta', 'Grupos'], ascending=[False, True]).iloc[0]
    amostra['Perfil'] = rotulos[melhor['Método'], melhor['Grupos']] + 1
    pca = PCA(n_components=2)
    xy = pca.fit_transform(matriz)
    return grade, melhor, amostra, xy, pca.explained_variance_ratio_


def tabela_agrupamento(grade):
    resumo = grade.pivot(index='Grupos', columns='Método', values='Silhueta').reset_index()
    for coluna in resumo.columns[1:]:
        resumo[coluna] = resumo[coluna].map(lambda x: numero(x, 3))
    tabela(resumo)


def graficos_agrupamento(amostra, xy, variancia):
    fig, (esq, dir_) = plt.subplots(1, 2, figsize=(10, 4), layout='constrained',
                                   gridspec_kw={'width_ratios': [1, 1.07]})
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=amostra['Perfil'].astype(str),
                    hue_order=sorted(amostra['Perfil'].astype(str).unique()), palette=MINHA_PALETTE[:amostra.Perfil.nunique()],
                    s=10, alpha=.32, linewidth=0, ax=esq)
    esq.set(xlabel=f'PC1 ({pct(variancia[0])})', ylabel=f'PC2 ({pct(variancia[1])})',
            title='Projeção dos perfis')
    indicadores = amostra.groupby('Perfil')[CAMPOS].mean().rename(columns=RECURSOS).T * 100
    sns.heatmap(indicadores, vmin=0, vmax=100, annot=True, fmt='.0f', cmap='crest',
                cbar=False, ax=dir_, linewidths=.5)
    dir_.set(title='Escolas com o recurso (%)', ylabel='')
    plt.show()


def diagnostico_perfis(amostra, melhor, variancia):
    grupos = amostra.groupby('Perfil')
    resumo = grupos.agg(Escolas=('co_entidade', 'size'), Matrículas=('qt_mat_bas', 'median'),
                         Rural=('tp_localizacao', lambda s: s.eq(2).mean()),
                         **{'Sem internet': ('sem_internet', 'mean')}).reset_index()
    resumo = resumo.rename(columns={'Matrículas': 'Matrículas (mediana)'})
    for campo in ['Rural', 'Sem internet']:
        resumo[campo] = resumo[campo].map(pct)
    tabela(resumo)
    pior = amostra.groupby('Perfil')['sem_internet'].mean().idxmax()
    melhor_conexao = amostra.groupby('Perfil')['sem_internet'].mean().idxmin()
    grupos_conexao = amostra.groupby('Perfil')['sem_internet'].mean()
    cobertura = amostra.groupby('Perfil')[CAMPOS].mean()
    contrastes = (cobertura.loc[melhor_conexao] - cobertura.loc[pior]).nlargest(2)
    contrastes_texto = ' e '.join(f'{RECURSOS[c].lower()} ({pct(cobertura.loc[pior,c])} contra '
                                 f'{pct(cobertura.loc[melhor_conexao,c])})' for c in contrastes.index)
    display(Markdown(f'''
A regra de seleção escolheu **{melhor['Método']} com {melhor['Grupos']} grupos**, silhueta de
**{numero(melhor['Silhueta'],3)}** e {pct(melhor['Menor grupo'])} de observações no menor perfil.
As duas primeiras componentes explicam {pct(sum(variancia))} da variância. A projeção mostra a estrutura
em duas dimensões, enquanto a silhueta mede a separação no espaço completo dos atributos padronizados.
O perfil {pior} tem {pct(grupos_conexao[pior])} de escolas sem conexão na amostra, frente a
{pct(grupos_conexao[melhor_conexao])} no perfil {melhor_conexao}. Os maiores contrastes em infraestrutura
entre esses perfis estão em {contrastes_texto}. A conectividade foi usada para descrever
os grupos depois do ajuste. A comparação representa os perfis da amostra e não mede efeitos de uma intervenção.
'''))
