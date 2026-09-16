"""
Limpeza e tratamento de dados - Censo Escolar 2025
"""

import pandas as pd

# 1. Carregar o CSV como DataFrame
df = pd.read_csv("data/dataset_escolas_censo2025.csv")

# 2. Inspeção inicial
print("=== INFO GERAL ===")
df.info()

print("\n=== PRIMEIRAS LINHAS ===")
print(df.head())

print("\n=== ESTATÍSTICAS DAS COLUNAS NUMÉRICAS ===")
print(df.describe())

# 3. Remover duplicatas (código único de escola)
duplicadas = df.duplicated(subset="CO_ENTIDADE").sum()
print(f"\nLinhas duplicadas: {duplicadas}")
df = df.drop_duplicates(subset="CO_ENTIDADE")

# 4. Padronizar nomes de colunas
df.columns = df.columns.str.lower()

# 5. Tratar valores ausentes
print("\n=== COLUNAS COM NaN ===")
nan_por_coluna = df.isna().sum()
print(nan_por_coluna[nan_por_coluna > 0])

# 5.1 NaN em região/UF/município: erro pontual de preenchimento -> remove linha
colunas_identificacao = ["no_regiao", "no_uf", "sg_uf", "no_municipio"]
df = df.dropna(subset=colunas_identificacao)

# 5.2 NaN em infraestrutura: ocorre só em escolas paralisadas/extintas
# (situação de funcionamento 2 ou 3) -> preenche com 0
colunas_infraestrutura = [
    "in_agua_potavel", "in_energia_rede_publica", "in_esgoto_rede_publica",
    "in_banheiro", "in_banheiro_pne", "in_biblioteca", "in_biblioteca_sala_leitura",
    "in_cozinha", "in_laboratorio_ciencias", "in_laboratorio_informatica",
    "in_patio_coberto", "in_patio_descoberto", "in_parque_infantil",
    "in_quadra_esportes", "in_refeitorio", "in_acessibilidade_corrimao",
    "in_acessibilidade_rampas", "in_computador", "in_internet",
    "in_internet_alunos", "in_banda_larga", "in_alimentacao",
    "qt_salas_utilizadas_dentro", "qt_salas_utilizadas_fora", "qt_salas_utilizadas",
    "qt_salas_utiliza_climatizadas", "qt_salas_utilizadas_acessiveis", "qt_salas_leitura",
    "qt_equip_dvd", "qt_equip_som", "qt_equip_tv", "qt_equip_lousa_digital",
    "qt_equip_multimidia", "qt_desktop_aluno", "qt_comp_portatil_aluno", "qt_tablet_aluno",
    "qt_prof_administrativos", "qt_prof_servicos_gerais", "qt_prof_bibliotecario",
    "qt_prof_saude", "qt_prof_coordenador", "qt_prof_psicologo", "qt_prof_alimentacao",
    "qt_prof_pedagogia", "qt_prof_secretario", "qt_prof_seguranca", "qt_prof_monitores",
    "qt_prof_gestao", "qt_prof_assist_social",
]
df[colunas_infraestrutura] = df[colunas_infraestrutura].fillna(0)

# 5.3 NaN em tp_categoria_escola_privada: só existe para escola privada -> -1 (não aplicável)
df["tp_categoria_escola_privada"] = df["tp_categoria_escola_privada"].fillna(-1)

print(f"\nTotal de NaN restantes: {df.isna().sum().sum()}")

# 6. Corrigir tipos (float -> int, já que são contagens/categorias)
colunas_para_int = colunas_infraestrutura + ["tp_categoria_escola_privada"]
df[colunas_para_int] = df[colunas_para_int].astype(int)

# 7. Remover coluna redundante (tem_matricula duplica qt_mat_bas)
df = df.drop(columns=["tem_matricula"])

# 8. Exportar dataset limpo
print("\n=== INFO FINAL ===")
df.info()

df.to_csv("data/dataset_escolas_censo2025_limpo.csv", index=False)
print("\nArquivo salvo em data/dataset_escolas_censo2025_limpo.csv")
