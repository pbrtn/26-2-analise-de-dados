import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

from base import MINHA_PALETTE, ROOT, numero, pct, tabela


def territorial(escolas):
    validas = escolas.dropna(subset=['sem_internet', 'local', 'sg_uf', 'co_municipio']).copy()
    validas['cod_uf'] = validas.co_municipio.astype(int).astype(str).str[:2]
    por_local = validas.groupby(['cod_uf', 'sg_uf', 'local'])['sem_internet'].agg(['mean', 'count'])
    taxas = por_local['mean'].unstack().rename(columns={'Rural': 'rural', 'Urbana': 'urbana'})
    taxas['n_rural'] = por_local['count'].unstack()['Rural']
    taxas['n_urbana'] = por_local['count'].unstack()['Urbana']
    taxas['hiato'] = taxas.rural - taxas.urbana
    taxas = taxas.reset_index()

    malha = gpd.read_file(f'zip://{ROOT / "data" / "BR_UF_2025.zip"}', engine='pyogrio')
    if set(taxas.cod_uf) != set(malha['CD_UF'].astype(str)):
        raise ValueError('Os códigos de UF da malha e dos microdados diferem.')
    malha['CD_UF'] = malha['CD_UF'].astype(str)
    mapa = malha.merge(taxas, left_on='CD_UF', right_on='cod_uf', validate='one_to_one').to_crs(5880)
    municipios = validas.loc[validas.local.eq('Rural')].groupby(['sg_uf', 'co_municipio'])['sem_internet'].agg(['mean', 'count'])
    municipios = municipios.loc[municipios['count'].ge(10)]
    return mapa, municipios


def mapas(mapa):
    fig, eixos = plt.subplots(1, 3, figsize=(11, 4), layout='constrained',
                              gridspec_kw={'width_ratios': [1, 1, .065]})
    cmap = sns.light_palette(MINHA_PALETTE[0], as_cmap=True)
    teto = max(.1, float(mapa[['urbana', 'rural']].max().max()))
    norma = Normalize(0, teto)
    for eixo, coluna, titulo in zip(eixos, ['urbana', 'rural'], ['Escolas urbanas', 'Escolas rurais']):
        mapa.plot(column=coluna, ax=eixo, norm=norma, cmap=cmap, linewidth=.45, edgecolor='white', rasterized=True,
                  missing_kwds={'color': '#d9dce0', 'hatch': '///'})
        eixo.set_title(titulo)
        eixo.set_axis_off()
    barra = fig.colorbar(plt.cm.ScalarMappable(norm=norma, cmap=cmap), cax=eixos[2],
                        format=PercentFormatter(1), label='Escolas sem internet')
    barra.ax.tick_params(labelsize=8)
    plt.show()


def resumo_espacial(mapa, municipios):
    maior_hiato = mapa.nlargest(5, 'hiato')
    quadro = pd.DataFrame({'UF': maior_hiato.sg_uf, 'Rurais válidas': maior_hiato.n_rural.map(numero),
                           'Sem internet (rural)': maior_hiato.rural.map(pct),
                           'Sem internet (urbana)': maior_hiato.urbana.map(pct),
                           'Hiato (p.p.)': maior_hiato.hiato.mul(100).map(lambda x: numero(x, 1))})
    tabela(quadro)
    lider = maior_hiato.iloc[0]
    exemplos = municipios.loc[lider.sg_uf, 'mean']
    p25, p75 = exemplos.quantile([.25, .75])
    display(Markdown(f'''
A junção cobriu **{len(mapa)} unidades federativas**. O maior hiato rural–urbano foi observado em
**{lider.sg_uf}**, com {pct(lider.rural)} de escolas rurais sem acesso e {pct(lider.urbana)} das urbanas.
Dentro dessa UF, entre os {numero(len(exemplos))} municípios com pelo menos dez respostas em escolas rurais,
os percentis 25 e 75 das proporções municipais foram {pct(p25)} e {pct(p75)}. Essa dispersão mostra
quanto a média estadual pode ocultar. O corte de dez respostas se aplica a essa comparação municipal.
No mapa, cada proporção estadual usa todas as respostas válidas do recorte.
'''))
