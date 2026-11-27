"""Regressão da disponibilidade de computadores para estudantes."""
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from base import SEED
from supervisionado import atributos, pipeline


def metricas(observado, previsto):
    return {
        'MAE': mean_absolute_error(observado, previsto),
        'RMSE': root_mean_squared_error(observado, previsto),
        'R²': r2_score(observado, previsto),
        'Viés': np.mean(previsto - observado),
    }


def treinar_regressao(escolas):
    dados = escolas.loc[escolas.computadores.ge(0)].dropna(subset=['co_municipio']).reset_index(drop=True)
    x, y, grupos = atributos(dados), dados.computadores.astype(float), dados.co_municipio
    treino, teste = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SEED).split(x, y, grupos))
    xt, yt, gt = x.iloc[treino], y.iloc[treino], grupos.iloc[treino]
    xv, yv = x.iloc[teste], y.iloc[teste]
    assert set(gt).isdisjoint(grupos.iloc[teste])
    dobras = list(GroupKFold(5).split(xt, yt, gt))
    assert all(set(gt.iloc[a]).isdisjoint(gt.iloc[b]) for a, b in dobras)
    opcoes = {
        'Ridge': (Ridge(), {'regressor__modelo__alpha': [.5, 10, 70]}),
        'Random forest': (RandomForestRegressor(n_estimators=40, n_jobs=2, random_state=SEED),
                          {'regressor__modelo__max_depth': [7, 13],
                           'regressor__modelo__min_samples_leaf': [20]}),
        'Gradient boosting': (HistGradientBoostingRegressor(max_iter=85, early_stopping=False, random_state=SEED),
                              {'regressor__modelo__max_leaf_nodes': [15, 31],
                               'regressor__modelo__learning_rate': [.07]}),
    }
    linhas, grades, previsoes = [], [], {}
    escolha = None
    with threadpool_limits(limits=2):
        for nome, (modelo, parametros) in opcoes.items():
            # O escore vê as predições na escala original, após inversão de log1p.
            estimador = TransformedTargetRegressor(regressor=pipeline(modelo), func=np.log1p, inverse_func=np.expm1)
            busca = GridSearchCV(estimador, parametros, cv=dobras, scoring='neg_mean_absolute_error', error_score='raise')
            busca.fit(xt, yt)
            cv_mae = -busca.best_score_
            if escolha is None or cv_mae < escolha[1]:
                escolha = (nome, cv_mae)
            pred = np.maximum(busca.predict(xv), 0)
            previsoes[nome] = pred
            linhas.append({'Modelo': nome, 'MAE CV': cv_mae, **metricas(yv, pred)})
            for i, cfg in enumerate(busca.cv_results_['params']):
                ajustes = ' / '.join(f'{v:g}'.replace('.', ',') for v in cfg.values())
                esc = [-busca.cv_results_[f'split{k}_test_score'][i] for k in range(len(dobras))]
                grades.append((nome, ajustes, *esc, np.mean(esc)))

    ref = DummyRegressor(strategy='median').fit(xt, yt)
    pred_ref = ref.predict(xv)
    previsoes['Mediana'] = pred_ref
    ref_cv = np.mean([mean_absolute_error(yt.iloc[b], np.repeat(yt.iloc[a].median(), len(b))) for a, b in dobras])
    linhas.insert(0, {'Modelo': 'Mediana', 'MAE CV': ref_cv, **metricas(yv, pred_ref)})
    metadados = dados.loc[teste, ['no_regiao', 'local', 'rede', 'porte']].reset_index(drop=True)
    return {
        'treino': len(treino), 'teste': len(teste), 'mun_treino': gt.nunique(),
        'mun_teste': grupos.iloc[teste].nunique(), 'yv': yv.to_numpy(), 'previsoes': previsoes,
        'melhor': escolha[0], 'segmentos': metadados, 'elegiveis': len(dados),
        'zeros': int(y.eq(0).sum()), 'resultados': pd.DataFrame(linhas),
        'busca': pd.DataFrame(grades, columns=['Modelo', 'Config.', 'D1', 'D2', 'D3', 'D4', 'D5', 'MAE']),
    }
