import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from IPython.display import Markdown, display

from base import MINHA_PALETTE, REGIOES, numero, pct, tabela


CAMPOS_AUDITORIA = {
    'co_entidade': 'Identificador do estabelecimento',
    'co_municipio': 'Código do município',
    'tp_dependencia': 'Dependência administrativa',
    'tp_situacao_funcionamento': 'Situação de funcionamento',
    'tp_localizacao': 'Localização urbana ou rural',
    'in_internet': 'Escola declara acesso à internet',
    'in_agua_potavel': 'Escola declara água potável',
    'in_energia_rede_publica': 'Ligação à rede pública de energia',
    'qt_mat_bas': 'Matrículas de educação básica',
    'qt_salas_utilizadas': 'Salas utilizadas',
    'qt_desktop_aluno': 'Desktops para alunos',
    'qt_comp_portatil_aluno': 'Computadores portáteis para alunos',
    'qt_mat_eja': 'Matrículas em EJA',
}


def tabela_auditoria(base, escolas):
    linhas = []
    for campo, descricao in CAMPOS_AUDITORIA.items():
        codigo = r'\texttt{' + campo.replace('_', r'\_\allowbreak{}') + '}'
        linhas.append((codigo, descricao, pct(base[campo].isna().mean()),
                       pct(escolas[campo].isna().mean())))
    tabela(pd.DataFrame(linhas, columns=['Campo', 'Significado', 'Ausentes total', 'Ausentes nas ativas']))


def cobertura_por_regiao(escolas):
    resumo = escolas.groupby('no_regiao').agg(
        escolas=('co_entidade', 'size'), respostas=('sem_internet', 'count'),
        sem_internet=('sem_internet', 'sum'), razao_valida=('alunos_computador', 'count'),
        mediana=('alunos_computador', 'median')).reindex(REGIOES)
    tabela(pd.DataFrame({
        'Região': resumo.index, 'Ativas': resumo.escolas.map(numero),
        'Sem internet': resumo.sem_internet.map(numero),
        '% sem internet': resumo.sem_internet.div(resumo.respostas).map(pct),
        'Razão elegível': resumo.razao_valida.div(resumo.escolas).map(pct),
        'Alunos por computador': resumo.mediana.map(lambda x: numero(x, 1)),
    }))


def recortes_conectividade(escolas):
    recorte = escolas.dropna(subset=['sem_internet', 'porte', 'rede', 'local'])
    fig, (esq, dir_) = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for eixo, campo, titulo in [(esq, 'porte', 'Por porte e localização'),
                              (dir_, 'rede', 'Por rede e localização')]:
        dados = recorte.groupby([campo, 'local'], observed=True).sem_internet.agg(['mean', 'count'])
        proporcoes = dados['mean'].where(dados['count'] >= 30).unstack()[['Urbana', 'Rural']]
        proporcoes.plot.bar(ax=eixo, color=MINHA_PALETTE[:2], width=.78)
        eixo.set(title=titulo, xlabel='', ylabel='Escolas sem internet')
        eixo.yaxis.set_major_formatter(PercentFormatter(1))
        eixo.tick_params(axis='x', labelrotation=22)
        eixo.legend(title=None, fontsize=8)
    plt.show()


def texto_conectividade(escolas):
    recorte = escolas.dropna(subset=['sem_internet', 'porte', 'rede', 'local'])
    perfis = recorte.groupby(['porte', 'local'], observed=True).sem_internet.agg(['mean', 'count'])
    taxas = perfis['mean'].unstack()
    hiato = taxas.Rural - taxas.Urbana
    display(Markdown(f'''
Entre escolas com **até 50 matrículas**, a proporção sem internet é de {pct(taxas.Urbana.iloc[0])}
nas urbanas e {pct(taxas.Rural.iloc[0])} nas rurais. O hiato é de **{numero(hiato.iloc[0]*100, 1)} p.p.**
Entre as escolas com **mais de 500 matrículas**, o hiato é de {numero(hiato.iloc[-1]*100, 1)} p.p.
O recorte compara escolas de porte semelhante, ainda com possíveis diferenças de rede, região e composição.
'''))


def tabelas_validacao(resultado):
    teste = resultado['segmentos'].copy()
    tabela(pd.DataFrame({
        'Região': teste.regiao, 'Escolas': teste.total.map(numero), 'Sem internet': teste.positivos.map(numero),
        'Precisão': teste.precisao.map(pct), 'Recall': teste.recall.map(pct),
        'Casos perdidos': teste.perdidos.map(numero),
    }))


def grafico_limiares(resultado):
    dados = resultado['limiares']
    fig, eixo = plt.subplots(figsize=(8.3, 3.6), layout='constrained')
    eixo.plot(dados.limiar, dados.precisao, color=MINHA_PALETTE[0], label='Precisão')
    eixo.plot(dados.limiar, dados.recall, color=MINHA_PALETTE[1], label='Recall')
    eixo.axvline(resultado['limiar'], color=MINHA_PALETTE[2], linestyle='--',
                 label=f"Ponto de triagem ({pct(resultado['limiar'])})")
    eixo.set(xlabel='Limiar de classificação', ylabel='Proporção', ylim=(0, 1))
    eixo.xaxis.set_major_formatter(PercentFormatter(1))
    eixo.yaxis.set_major_formatter(PercentFormatter(1))
    eixo.legend()
    plt.show()
