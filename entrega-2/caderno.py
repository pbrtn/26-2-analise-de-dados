"""Tabelas e figuras do caderno analítico ampliado."""
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import adjusted_rand_score, brier_score_loss, confusion_matrix, roc_curve
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from agrupamento import CAMPOS
from base import MINHA_PALETTE, REGIOES, RECURSOS, SEED, numero, pct
from regressao import metricas


def barras(valores, eixo, porcentagem=False):
    _, ax = plt.subplots(figsize=(8.5, 2.85), layout='constrained')
    quadro = valores if isinstance(valores, pd.DataFrame) else valores.to_frame()
    if len(quadro.columns) == 1:
        ax.barh(quadro.index.astype(str), quadro.iloc[:, 0], color=MINHA_PALETTE[0])
    else:
        quadro.plot.barh(ax=ax, color=MINHA_PALETTE[:len(quadro.columns)], width=.72)
        ax.legend(loc='best', fontsize=8, title=None)
    ax.invert_yaxis()
    ax.set(xlabel=eixo, ylabel='')
    if porcentagem:
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    plt.show()


def formatar(dados, formatos, indice):
    quadro = dados.copy()
    for coluna, fmt in formatos.items():
        quadro[coluna] = quadro[coluna].map(lambda x: fmt(x) if pd.notna(x) else '—')
    return quadro.rename_axis(indice).reset_index()


def faltantes(dados, escolas, prefixo):
    campos = [c for c in dados if c.startswith(prefixo)]
    q = pd.DataFrame({'Todos': dados[campos].isna().mean(), 'Ativas': escolas[campos].isna().mean()})
    q = q.sort_values(['Ativas', 'Todos'], ascending=False).head(7)
    return q, formatar(q, {'Todos': pct, 'Ativas': pct}, 'Campo')


def quantidades(escolas, campos):
    q = escolas[list(campos)].rename(columns=campos).agg(['count', 'min', 'median', 'max']).T
    q['p90'] = escolas[list(campos)].rename(columns=campos).quantile(.9)
    q.columns = ['Válidas', 'Mínimo', 'Mediana', 'Máximo', 'P90']
    return q[['Mediana', 'P90']], formatar(q, {c: numero for c in q}, 'Indicador')


def recurso(escolas, campo, agrupador='no_regiao'):
    q = escolas.groupby([agrupador, 'local'], observed=True)[campo].agg(['mean', 'count']).unstack()
    q = q.reindex(REGIOES) if agrupador == 'no_regiao' else q
    taxas = q['mean'][['Urbana', 'Rural']].where(q['count'] >= 30)
    out = taxas.copy()
    out['Válidas urb.'] = q['count']['Urbana']
    out['Válidas rur.'] = q['count']['Rural']
    out['Hiato U–R (p.p.)'] = (taxas.Urbana - taxas.Rural) * 100
    return taxas, formatar(out, {'Urbana': pct, 'Rural': pct, 'Válidas urb.': numero, 'Válidas rur.': numero,
                                  'Hiato U–R (p.p.)': lambda x: numero(x, 1)}, 'Grupo')


def regional(escolas, regiao):
    d = escolas.loc[escolas.no_regiao.eq(regiao)]
    q = d.groupby('sg_uf').sem_internet.agg(['size', 'count', 'mean'])
    q.columns = ['Ativas', 'Respostas', 'Taxa']
    q['Sem internet'] = d.groupby('sg_uf').sem_internet.sum()
    q['Matrículas'] = d.groupby('sg_uf').qt_mat_bas.sum(min_count=1)
    q = q.sort_values('Taxa', ascending=False)
    return q[['Taxa']], formatar(q, {'Ativas': numero, 'Respostas': numero, 'Taxa': pct,
                                     'Sem internet': numero, 'Matrículas': numero}, 'UF')


def municipio(escolas, local):
    d = escolas.loc[escolas.local.eq(local)]
    q = d.groupby(['co_municipio', 'no_municipio', 'sg_uf']).sem_internet.agg(['count', 'mean', 'sum'])
    q = q.loc[q['count'].ge(30)].nlargest(8, ['mean', 'count']).reset_index()
    q.index = q.no_municipio + ' (' + q.sg_uf + ')'
    q = q[['count', 'sum', 'mean']].rename(columns={'count': 'Válidas', 'sum': 'Sem internet', 'mean': 'Taxa'})
    return q[['Taxa']], formatar(q, {'Válidas': numero, 'Sem internet': numero, 'Taxa': pct}, 'Município')


def pior_distancia(mapa):
    q = mapa.set_index('sg_uf').nlargest(8, 'hiato')[['urbana', 'rural', 'hiato', 'n_urbana', 'n_rural']]
    q.columns = ['Urbana', 'Rural', 'Diferença R–U', 'Válidas urb.', 'Válidas rur.']
    return q[['Urbana', 'Rural']], formatar(q, {'Urbana': pct, 'Rural': pct, 'Diferença R–U': pct,
                                              'Válidas urb.': numero, 'Válidas rur.': numero}, 'UF')


def mapa_servico(mapa, escolas, campo):
    q = escolas.groupby('sg_uf')[campo].mean().rename('Taxa')
    geo = mapa.merge(q, left_on='sg_uf', right_index=True, validate='one_to_one')
    _, ax = plt.subplots(figsize=(7.5, 3.35), layout='constrained')
    geo.plot(column='Taxa', cmap='cividis', vmin=0, vmax=1, legend=True, ax=ax, rasterized=True,
             edgecolor='white', linewidth=.35, legend_kwds={'format': PercentFormatter(1), 'shrink': .7})
    ax.set_axis_off()
    plt.show()


def tabela_servico(escolas, campo):
    q = escolas.groupby('sg_uf')[campo].agg(['count', 'mean']).nsmallest(8, 'mean')
    q.columns = ['Válidas', 'Cobertura']
    return formatar(q, {'Válidas': numero, 'Cobertura': pct}, 'UF')


def curvas_clusters(grade):
    q = grade.pivot(index='Grupos', columns='Método', values='Silhueta')
    _, ax = plt.subplots(figsize=(8.5, 2.85), layout='constrained')
    q.plot(ax=ax, marker='o', color=MINHA_PALETTE[:3])
    ax.set(xlabel='Número solicitado de grupos', ylabel='Silhueta', xticks=q.index)
    ax.legend(title=None, fontsize=8)
    plt.show()


def comparacao_clusters(amostra, k):
    x = amostra[CAMPOS].assign(log_matriculas=np.log1p(amostra.qt_mat_bas))
    x = StandardScaler().fit_transform(SimpleImputer(strategy='most_frequent').fit_transform(x))
    k = int(k)
    with threadpool_limits(limits=2):
        rotulos = {'K-Means': KMeans(k, n_init=15, random_state=SEED).fit_predict(x),
                   'Ward': AgglomerativeClustering(n_clusters=k).fit_predict(x),
                   'Mistura gaussiana': GaussianMixture(k, n_init=3, covariance_type='diag',
                                                       reg_covar=1e-4, random_state=SEED).fit_predict(x)}
        pca = PCA(n_components=x.shape[1]).fit(x)
    pares = [(a, b, adjusted_rand_score(rotulos[a], rotulos[b])) for i, a in enumerate(rotulos) for b in list(rotulos)[i+1:]]
    q = pd.DataFrame(pares, columns=['Método A', 'Método B', 'ARI'])
    q['ARI'] = q.ARI.map(lambda x: numero(x, 3))
    cargas = pd.DataFrame(pca.components_[:2].T, index=[RECURSOS[c] for c in CAMPOS] + ['Log de matrículas'],
                          columns=['PC1', 'PC2'])
    return q, cargas, np.cumsum(pca.explained_variance_ratio_)


def rodar_classificador(res, nome):
    y = res['yv']
    p = res['previsoes'][nome]
    fpr, tpr, _ = roc_curve(y, p)
    efetiva, estimada = calibration_curve(y, p, n_bins=8, strategy='quantile')
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.5, 2.85), layout='constrained')
    for ax, eixo_x, eixo_y, titulo in [(a, fpr, tpr, 'Curva ROC'), (b, estimada, efetiva, 'Calibração')]:
        ax.plot([0, 1], [0, 1], '--', color='#647789', lw=1)
        ax.plot(eixo_x, eixo_y, color=MINHA_PALETTE[0], marker='o' if ax is b else None, ms=3)
        ax.set(title=titulo, xlim=(0, 1), ylim=(0, 1))
    a.set(xlabel='Taxa de falsos positivos', ylabel='Recall')
    b.set(xlabel='Probabilidade estimada', ylabel='Frequência observada')
    plt.show()


def quadro_classificador(res, nome):
    y, p = res['yv'], res['previsoes'][nome]
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    linha = res['teste'].loc[res['teste'].Modelo.eq(nome)].iloc[0]
    medidas = {'Precisão média (AP)': numero(linha.AP, 3), 'ROC-AUC': numero(linha['ROC-AUC'], 3),
               'Brier': numero(brier_score_loss(y, p), 3), 'Falsos positivos': numero(fp),
               'Falsos negativos': numero(fn), 'Verdadeiros positivos': numero(tp),
               'Verdadeiros negativos': numero(tn)}
    return pd.DataFrame(medidas.items(), columns=['Métrica no teste', 'Valor'])


def acumulado(res):
    y = np.asarray(res['yv'])
    p = res['previsoes'][res['melhor']]
    corte = pd.DataFrame({'p': p, 'y': y}).groupby('p', sort=True).y.agg(['sum', 'size']).iloc[::-1]
    x = corte['size'].cumsum() / len(y)
    z = corte['sum'].cumsum() / y.sum()
    _, ax = plt.subplots(figsize=(8.5, 2.85), layout='constrained')
    ax.plot(np.r_[0, x], np.r_[0, z], color=MINHA_PALETTE[0]); ax.plot([0, 1], [0, 1], '--', color='#647789')
    ax.set(xlabel='Fração de estabelecimentos na fila', ylabel='Fração de casos detectados')
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(PercentFormatter(1))
    plt.show()


def quadro_regressao(reg, nome):
    linha = reg['resultados'].set_index('Modelo').loc[nome]
    return pd.DataFrame([(k, numero(v, 3)) for k, v in linha.items()], columns=['Métrica', 'Valor'])


def figura_regressao(reg, nome):
    y, p = reg['yv'], reg['previsoes'][nome]
    idx = np.random.default_rng(SEED).choice(len(y), min(3500, len(y)), replace=False)
    erro = p - y
    grupos = pd.cut(y, [-1, 0, 5, 15, 30, 60, np.inf], labels=['0', '1–5', '6–15', '16–30', '31–60', '61+'])
    q = pd.DataFrame({'grupo': grupos, 'erro': erro}).groupby('grupo', observed=True).erro.mean()
    _, (a, b) = plt.subplots(1, 2, figsize=(8.5, 2.85), layout='constrained')
    limite = max(np.log1p(y[idx]).max(), np.log1p(p[idx]).max())
    a.scatter(np.log1p(y[idx]), np.log1p(p[idx]), s=4, alpha=.14, color=MINHA_PALETTE[0], rasterized=True)
    a.plot([0, limite], [0, limite], '--', color='#647789')
    a.set(title='Observado × previsto', xlabel='log(1 + observado)', ylabel='log(1 + previsto)')
    b.bar(q.index.astype(str), q, color=MINHA_PALETTE[1]); b.axhline(0, color='#647789')
    b.set(title='Viés por faixa observada', xlabel='Computadores por escola', ylabel='Previsto menos observado')
    plt.show()


def diagnostico_regressao(reg, campo):
    q = reg['segmentos'].assign(verdade=reg['yv'], previsao=reg['previsoes'][reg['melhor']])
    linhas = []
    for nome, bloco in q.groupby(campo, observed=True):
        medidas = metricas(bloco.verdade, bloco.previsao)
        linhas.append({'Grupo': str(nome), 'N': len(bloco), 'Observado médio': bloco.verdade.mean(),
                       'MAE': medidas['MAE'], 'Viés': medidas['Viés']})
    estat = pd.DataFrame(linhas).set_index('Grupo')
    return estat[['MAE']], formatar(estat, {'N': numero, 'Observado médio': lambda x: numero(x, 1),
                                          'MAE': lambda x: numero(x, 2), 'Viés': lambda x: numero(x, 2)}, 'Grupo')
