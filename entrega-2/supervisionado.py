import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from base import MINHA_PALETTE, REGIOES, SEED, numero, pct, tabela


CATEGORIAS = ['no_regiao', 'rede', 'local']
BINARIOS = ['in_agua_potavel', 'in_energia_rede_publica', 'in_banheiro', 'in_esgoto_rede_publica',
            'in_biblioteca_sala_leitura', 'in_quadra_esportes', 'in_acessibilidade_rampas']
QUANTIDADES = ['qt_mat_bas', 'qt_mat_inf', 'qt_mat_fund', 'qt_mat_med', 'qt_salas_utilizadas']
DERIVADOS = ['indice_servicos', 'indice_recursos', 'frac_infantil', 'frac_medio',
             'matriculas_sala', 'porte_rural']


def atributos(dados):
    quadro = dados[CATEGORIAS + BINARIOS + QUANTIDADES].copy()
    quadro['indice_servicos'] = quadro[BINARIOS[:4]].mean(axis=1, skipna=False)
    quadro['indice_recursos'] = quadro[BINARIOS[4:]].mean(axis=1, skipna=False)
    total = quadro['qt_mat_bas'].where(quadro['qt_mat_bas'].gt(0))
    quadro['frac_infantil'] = quadro['qt_mat_inf'].div(total)
    quadro['frac_medio'] = quadro['qt_mat_med'].div(total)
    quadro['matriculas_sala'] = total.div(quadro['qt_salas_utilizadas'].where(quadro['qt_salas_utilizadas'].gt(0)))
    quadro['porte_rural'] = np.log1p(quadro['qt_mat_bas']) * quadro['local'].eq('Rural')
    quadro[QUANTIDADES] = np.log1p(quadro[QUANTIDADES])
    return quadro


def pipeline(modelo, derivados=True):
    numericas = BINARIOS + QUANTIDADES + (DERIVADOS if derivados else [])
    tratamento_numerico = Pipeline([
        ('imputar', SimpleImputer(strategy='median', add_indicator=True)),
        ('escala', StandardScaler()),
    ])
    tratamento_categorico = Pipeline([
        ('imputar', SimpleImputer(strategy='most_frequent')),
        ('codificar', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    prep = ColumnTransformer([
        ('num', tratamento_numerico, numericas),
        ('cat', tratamento_categorico, CATEGORIAS),
    ])
    return Pipeline([('preparar', prep), ('modelo', modelo)])


def resultado(nome, y, previsao, limiar):
    estimativa = previsao >= limiar
    return {'Modelo': nome, 'AP': average_precision_score(y, previsao),
            'ROC-AUC': roc_auc_score(y, previsao), 'Precisão': precision_score(y, estimativa, zero_division=0),
            'Recall': recall_score(y, estimativa, zero_division=0),
            'F2': fbeta_score(y, estimativa, beta=2, zero_division=0)}


def treinar(escolas):
    dados = escolas.dropna(subset=['sem_internet', 'co_municipio']).reset_index(drop=True)
    x, y, municipios = atributos(dados), dados.sem_internet.astype(int), dados.co_municipio
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    itrain, itest = next(folds.split(x, y, municipios))
    assert set(municipios.iloc[itrain]).isdisjoint(municipios.iloc[itest])
    xt, yt, gt = x.iloc[itrain], y.iloc[itrain], municipios.iloc[itrain]
    xv, yv = x.iloc[itest], y.iloc[itest]
    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1).split(xt, yt, gt))
    assert all(set(gt.iloc[a]).isdisjoint(gt.iloc[b]) for a, b in cv)
    candidatos = {
        'Logística': (LogisticRegression(max_iter=350, random_state=SEED),
                      {'modelo__C': [.05, .5, 3]}),
        'Random forest': (RandomForestClassifier(n_estimators=65, random_state=SEED, n_jobs=2),
                          {'modelo__max_depth': [8, 15], 'modelo__min_samples_leaf': [12, 45]}),
        'Gradient boosting': (HistGradientBoostingClassifier(max_iter=90, random_state=SEED,
                                                            early_stopping=False),
                              {'modelo__max_leaf_nodes': [15, 31], 'modelo__learning_rate': [.06, .12]}),
    }
    res_cv, res_teste, previsoes, combinacoes = [], [], {}, []
    buscas = {}
    with threadpool_limits(limits=2):
        for nome, (modelo, grade) in candidatos.items():
            busca = GridSearchCV(pipeline(modelo), grade, cv=cv, scoring='average_precision', error_score='raise')
            busca.fit(xt, yt)
            buscas[nome] = busca
            i = busca.best_index_
            parametros = ', '.join(f'{k.split("__")[-1]}={v}' for k, v in busca.best_params_.items())
            res_cv.append((nome, busca.cv_results_['mean_test_score'][0], busca.best_score_,
                           busca.cv_results_['std_test_score'][i], parametros))
            for j, cfg in enumerate(busca.cv_results_['params']):
                valores = ' / '.join(f'{v:g}'.replace('.', ',') for v in cfg.values())
                ap_dobras = [busca.cv_results_[f'split{d}_test_score'][j] for d in range(len(cv))]
                combinacoes.append((nome, valores, *ap_dobras, busca.cv_results_['mean_test_score'][j]))
            p = busca.predict_proba(xv)[:, 1]
            previsoes[nome] = p
            res_teste.append(resultado(nome, yv, p, .5))

        melhor = max(buscas, key=lambda nome: buscas[nome].best_score_)
        escolhido = buscas[melhor].best_estimator_
        oof = cross_val_predict(clone(escolhido), xt, yt, cv=cv, method='predict_proba')[:, 1]
        precisao, recall, limiares = precision_recall_curve(yt, oof)
        f2 = 5 * precisao[:-1] * recall[:-1] / np.maximum(4 * precisao[:-1] + recall[:-1], 1e-10)
        limiar = limiares[np.argmax(f2)]
        reduzido = GridSearchCV(pipeline(clone(candidatos[melhor][0]), derivados=False),
                               candidatos[melhor][1], cv=cv, scoring='average_precision', error_score='raise')
        reduzido.fit(xt, yt)
        ab = reduzido.predict_proba(xv)[:, 1]

    dummy = DummyClassifier(strategy='prior').fit(xt, yt)
    prob_dummy = dummy.predict_proba(xv)[:, 1]
    previsoes['Referência'] = prob_dummy
    res_teste.insert(0, resultado('Referência', yv, prob_dummy, .5))
    p = previsoes[melhor]
    res_teste.append(resultado('Escolhido · limiar F2', yv, p, limiar))
    testes = pd.DataFrame(res_teste)
    pred = p >= limiar
    rural_urbano = dados.loc[itest, 'local'].to_numpy()
    detalhes = []
    for grupo in ['Urbana', 'Rural']:
        mascara = rural_urbano == grupo
        rea, est = yv.to_numpy()[mascara], pred[mascara]
        detalhes.append((grupo, len(rea), rea.sum(), pct(recall_score(rea, est, zero_division=0)),
                         pct(precision_score(rea, est, zero_division=0)), int(((rea == 1) & ~est).sum())))
    regioes = dados.loc[itest, 'no_regiao'].to_numpy()
    segmentos = []
    for regiao in REGIOES:
        mascara = regioes == regiao
        rea, est = yv.to_numpy()[mascara], pred[mascara]
        segmentos.append((regiao, len(rea), int(rea.sum()), precision_score(rea, est, zero_division=0),
                          recall_score(rea, est, zero_division=0), int(((rea == 1) & ~est).sum())))
    return {'cv': pd.DataFrame(res_cv, columns=['Modelo', 'AP inicial', 'AP média', 'Desvio-padrão',
                                               'Melhor configuração']),
            'combinacoes': pd.DataFrame(combinacoes, columns=['Modelo', 'Configuração', 'D1', 'D2', 'D3', 'D4', 'D5', 'Média']),
            'segmentos': pd.DataFrame(segmentos, columns=['regiao', 'total', 'positivos', 'precisao', 'recall', 'perdidos']),
            'limiares': pd.DataFrame({'limiar': limiares, 'precisao': precisao[:-1], 'recall': recall[:-1]}),
            'teste': testes, 'melhor': melhor, 'limiar': limiar, 'previsoes': previsoes, 'yv': yv,
            'matriz': confusion_matrix(yv, pred), 'ab_cv': reduzido.best_score_,
            'ab_teste': average_precision_score(yv, ab), 'n_treino': len(yt), 'n_teste': len(yv),
            'municipios_treino': gt.nunique(), 'municipios_teste': municipios.iloc[itest].nunique(),
            'grupos': pd.DataFrame(detalhes, columns=['Localização', 'Escolas', 'Sem internet',
                                                      'Recall', 'Precisão', 'Casos perdidos'])}


def tabela_modelos(quadro):
    quadro = quadro.copy()
    for coluna in quadro.select_dtypes(include='number'):
        quadro[coluna] = quadro[coluna].map(lambda x: numero(x, 3))
    tabela(quadro)


def graficos_modelos(res):
    fig, (esq, dir_) = plt.subplots(1, 2, figsize=(9.5, 3.8), layout='constrained')
    linhas = zip(((n, p) for n, p in res['previsoes'].items() if n != 'Referência'), MINHA_PALETTE)
    for (nome, p), cor in linhas:
        prec, rec, _ = precision_recall_curve(res['yv'], p)
        esq.step(rec, prec, label=nome, color=cor, alpha=.9)
    esq.axhline(res['yv'].mean(), label='Prevalência', color=MINHA_PALETTE[-1], linestyle='--')
    esq.set(xlabel='Recall', ylabel='Precisão', title='Curva precisão–recall', xlim=(0, 1), ylim=(0, 1))
    esq.legend(fontsize=7)
    matriz = res['matriz']
    taxas = matriz / matriz.sum(axis=1, keepdims=True)
    legendas = np.array([f'{numero(n)}\n{pct(t)}' for n, t in zip(matriz.flat, taxas.flat)]).reshape(2, 2)
    sns.heatmap(taxas, annot=legendas, fmt='', cmap='crest', vmin=0, vmax=1, cbar=False,
                xticklabels=['Com internet', 'Sem internet'], yticklabels=['Com internet', 'Sem internet'], ax=dir_)
    dir_.set(xlabel='Classificação', ylabel='Observado', title='Matriz no limiar de triagem')
    dir_.tick_params(axis='y', labelrotation=0)
    plt.show()


def texto_modelos(res):
    r = res['teste'].iloc[-1]
    prevalencia = res['yv'].mean()
    escolhido = res['cv'].loc[res['cv']['Modelo'].eq(res['melhor'])].iloc[0]
    rural = res['grupos'].loc[res['grupos']['Localização'].eq('Rural')].iloc[0]
    urbana = res['grupos'].loc[res['grupos']['Localização'].eq('Urbana')].iloc[0]
    display(Markdown(f'''
O conjunto de treino tem **{numero(res['n_treino'])} escolas em {numero(res['municipios_treino'])} municípios**.
O teste independente tem **{numero(res['n_teste'])} escolas em {numero(res['municipios_teste'])} municípios**,
com prevalência de {pct(prevalencia)} de escolas sem internet. O modelo **{res['melhor']}** foi escolhido pela
maior AP média em validação cruzada. No teste, alcançou AP de **{numero(r['AP'],3)}** e ROC-AUC de
{numero(r['ROC-AUC'],3)}. A comparação dos atributos derivados mostra AP de validação de
{numero(escolhido['AP média'],3)} com eles e {numero(res['ab_cv'],3)} sem eles. No teste, as APs foram
{numero(r['AP'],3)} e {numero(res['ab_teste'],3)}, respectivamente. A diferença pequena e a inversão
entre validação e teste sugerem contribuição limitada dos derivados neste recorte.

Para triagem, o limiar de **{pct(res['limiar'])}** foi escolhido nas previsões fora da dobra do treino
maximizando F2, que dá mais peso ao recall. No teste, atingiu **{pct(r['Recall'])} de recall** com
**{pct(r['Precisão'])} de precisão**. O recall rural ({rural['Recall']}) supera o urbano ({urbana['Recall']}).
Essa diferença exige cautela para identificar falta de internet em escolas urbanas, onde há menos casos positivos.
As probabilidades e o limiar são operacionais para este recorte. Uma aplicação externa exigiria
calibração e validação em outro ano ou contexto.
'''))
