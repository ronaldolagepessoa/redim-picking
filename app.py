import streamlit as st
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pyutilib.subprocess.GlobalData
import base64
from io import BytesIO

pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

st.set_page_config(page_title='Endereçador', layout='wide')

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="acomodação_de_itens.xlsx">Baixar tabela</a>' # decode b'abc' => abc   



def solve(df_agrupado, df_restricoesr1, volume_pallet):
    model = pyo.ConcreteModel()

    model.Produtos = pyo.Set(initialize=df_agrupado.NOME_PROD.values)
    model.ProdutosFracionados = pyo.Set(initialize=df_agrupado.loc[df_agrupado.VOLUME_FECHADO == 'não'].NOME_PROD.values)
    model.ProdutosFechados = pyo.Set(initialize=df_agrupado.loc[df_agrupado.VOLUME_FECHADO == 'sim'].NOME_PROD.values)
    model.PosicoesFracionados = pyo.Set(initialize=['RUA1_NIVEL2_D', 'RUA1_NIVEL2_E', 'RUA2_NIVEL2_D', 'RUA2_NIVEL2_E', 'RUA3_NIVEL2'])
    model.PosicoesFechados = pyo.Set(initialize=['RUA1_NIVEL1_D', 'RUA1_NIVEL1_E', 'RUA2_NIVEL1_D', 'RUA2_NIVEL1_E', 'RUA3_NIVEL1'])
    model.ProdutosRua1Proibidos = pyo.Set(initialize=df_restricoesr1.DESCRICAO.unique())

    model.volumePallets = volume_pallet

    model.numeroPalletsFracionados = pyo.Param(
        model.PosicoesFracionados,
        initialize = {'RUA1_NIVEL2_D': 27 * 2, 'RUA1_NIVEL2_E': 26 * 2, 'RUA2_NIVEL2_D': 26 * 2, 'RUA2_NIVEL2_E': 26 * 2, 'RUA3_NIVEL2': 26 * 2}
    )

    model.numeroPalletsFechados = pyo.Param(
        model.PosicoesFechados,
        initialize = {'RUA1_NIVEL1_D': 27 * 2, 'RUA1_NIVEL1_E': 26 * 2, 'RUA2_NIVEL1_D': 26 * 2, 'RUA2_NIVEL1_E': 26 * 2, 'RUA3_NIVEL1': 26 * 2}
    )

    # model.produtoVolume = pyo.Param(
    #     model.Produtos,
    #     initialize = df_agrupado.set_index('NOME_PROD').VOLUME_UNIT.to_dict()
    # )

    model.produtoQuantidadeMax = pyo.Param(
        model.Produtos,
        initialize = df_agrupado.set_index('NOME_PROD').N_MAXIMO_PALLETS.to_dict()
    )

    model.produtoQuantidadeMin = pyo.Param(
        model.Produtos,
        initialize = df_agrupado.set_index('NOME_PROD').N_MINIMO_PALLETS.to_dict()
    #     initialize = df_agrupado.set_index('NOME_PROD')[f'QUANTIDADE ({quantil * 100}%)'].to_dict()
    )

    model.produtoImportanciaFracionado = pyo.Param(
        model.ProdutosFracionados,
        initialize=df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'não')].set_index('NOME_PROD')['LEAD TIME'].to_dict()
    )
    model.produtoImportanciaFechado = pyo.Param(
        model.ProdutosFechados,
        initialize=df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'sim')].set_index('NOME_PROD')['LEAD TIME'].to_dict()
    )


    grauFrac = {(prod, posicao): 1000 if posicao == 'RUA3_NIVEL2' else 1
                for prod in df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'não') & ((df_agrupado.CLASSIFICACAO == 'A')), 'NOME_PROD'].values 
                for posicao in model.PosicoesFracionados
            }
    grauFrac.update({(prod, posicao): 1000 if posicao in ['RUA2_NIVEL2_D', 'RUA2_NIVEL2_E'] else 1
                for prod in df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'não') & ((df_agrupado.CLASSIFICACAO == 'B')), 'NOME_PROD'].values 
                for posicao in model.PosicoesFracionados
            })
    grauFrac.update({(prod, posicao): 500 
                for prod in df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'não') & ((df_agrupado.CLASSIFICACAO == 'B')), 'NOME_PROD'].values 
                for posicao in model.PosicoesFracionados if posicao == 'RUA3_NIVEL2'
            })
    grauFrac.update({(prod, posicao): 1000 if posicao in ['RUA1_NIVEL2_D', 'RUA1_NIVEL2_E'] else 0
                for prod in df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'não') & ((df_agrupado.CLASSIFICACAO == 'C')), 'NOME_PROD'].values 
                for posicao in model.PosicoesFracionados
            })
    grauFrac.update({(prod, posicao): 100 
                for prod in df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'não') & ((df_agrupado.CLASSIFICACAO == 'C')), 'NOME_PROD'].values 
                for posicao in model.PosicoesFracionados if posicao in ['RUA2_NIVEL2_D', 'RUA2_NIVEL2_E'] 
            })

    model.grauFrac = pyo.Param(
        model.ProdutosFracionados,
        model.PosicoesFracionados,
        initialize=grauFrac
    )

    grauFech = {(prod, posicao): 1000 if posicao == 'RUA3_NIVEL1' else 1
                for prod in df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'sim') & ((df_agrupado.CLASSIFICACAO == 'A')), 'NOME_PROD'].values 
                for posicao in model.PosicoesFechados
            }
    grauFech.update({(prod, posicao): 1000 if posicao in ['RUA2_NIVEL1_D', 'RUA2_NIVEL1_E'] else 1
                for prod in df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'sim') & ((df_agrupado.CLASSIFICACAO == 'B')), 'NOME_PROD'].values 
                for posicao in model.PosicoesFechados
            })
    grauFech.update({(prod, posicao): 500 
                for prod in df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'sim') & ((df_agrupado.CLASSIFICACAO == 'B')), 'NOME_PROD'].values 
                for posicao in model.PosicoesFechados if posicao == 'RUA3_NIVEL1' 
            })
    grauFech.update({(prod, posicao): 1000 if posicao in ['RUA1_NIVEL1_D', 'RUA1_NIVEL1_E'] else 1
                for prod in df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'sim') & ((df_agrupado.CLASSIFICACAO == 'C')), 'NOME_PROD'].values 
                for posicao in model.PosicoesFechados
            })
    model.grauFech = pyo.Param(
        model.ProdutosFechados,
        model.PosicoesFechados,
        initialize=grauFech
    )

    model.xFrac = pyo.Var(model.ProdutosFracionados, model.PosicoesFracionados, within=pyo.NonNegativeReals)
    model.xFech = pyo.Var(model.ProdutosFechados, model.PosicoesFechados, within=pyo.NonNegativeReals)
    model.yFrac = pyo.Var(model.ProdutosFracionados, model.PosicoesFracionados, within=pyo.Binary)
    model.yFech = pyo.Var(model.ProdutosFechados, model.PosicoesFechados, within=pyo.Binary)

    model.obj = pyo.Objective(
        expr=sum(model.produtoImportanciaFracionado[i] * model.grauFrac[i, j] * model.xFrac[i, j] 
                for i in model.ProdutosFracionados 
                for j in model.PosicoesFracionados) + 
        sum(model.produtoImportanciaFechado[i] * model.grauFech[i, j] * model.xFech[i, j] 
                for i in model.ProdutosFechados 
                for j in model.PosicoesFechados),
        sense=pyo.maximize
    )

    model.c1 = pyo.ConstraintList()
    for j in model.PosicoesFracionados:
        model.c1.add(
            sum(model.xFrac[i, j] for i in model.ProdutosFracionados) <= 
            model.numeroPalletsFracionados[j]
        )
    for j in model.PosicoesFechados:
        model.c1.add(
            sum(model.xFech[i, j] for i in model.ProdutosFechados) <= 
            model.numeroPalletsFechados[j]
        )
    model.c2 = pyo.ConstraintList()
    for i in model.ProdutosFracionados:
        model.c2.add(
            sum(model.xFrac[i, j] for j in model.PosicoesFracionados) <= model.produtoQuantidadeMax[i]
        )
        model.c2.add(
            sum(model.xFrac[i, j] for j in model.PosicoesFracionados) >= model.produtoQuantidadeMin[i]
        )
    for i in model.ProdutosFechados:
        model.c2.add(
            sum(model.xFech[i, j] for j in model.PosicoesFechados) <= model.produtoQuantidadeMax[i]
        )
        model.c2.add(
            sum(model.xFech[i, j] for j in model.PosicoesFechados) >= model.produtoQuantidadeMin[i]
        )
    model.c3 = pyo.ConstraintList()
    for i in model.ProdutosFracionados:
        for j in model.PosicoesFracionados:
            model.c3.add(
                model.xFrac[i, j] <= model.produtoQuantidadeMax[i] * model.yFrac[i, j]
            )
    for i in model.ProdutosFechados:
        for j in model.PosicoesFechados:
            model.c3.add(
                model.xFech[i, j] <= model.produtoQuantidadeMax[i] * model.yFech[i, j]
            )
    model.c4 = pyo.ConstraintList()
    for i in model.ProdutosFracionados:
        model.c4.add(
            sum(model.yFrac[i, j] for j in model.PosicoesFracionados) == 1
        )
    for i in model.ProdutosFechados:
        model.c4.add(
            sum(model.yFech[i, j] for j in model.PosicoesFechados) == 1
        )
    model.c5 = pyo.ConstraintList()
    for i in model.ProdutosFracionados:
        if i in model.ProdutosRua1Proibidos:
            model.c5.add(
                model.yFrac[i, 'RUA1_NIVEL2_D']  == 0
            )
            model.c5.add(
                model.yFrac[i, 'RUA1_NIVEL2_E']  == 0
            )
    for i in model.ProdutosFechados:
        if i in model.ProdutosRua1Proibidos:
            model.c5.add(
                model.yFech[i, 'RUA1_NIVEL1_D']  == 0
            )
            model.c5.add(
                model.yFech[i, 'RUA1_NIVEL1_E']  == 0
            )
    model.c6 = pyo.ConstraintList()

    for i in model.ProdutosFracionados:
        if i in df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'não') & (df_agrupado.CLASSIFICACAO == 'C'), 'NOME_PROD'].values:                        
            model.c6.add(model.yFrac[i, 'RUA3_NIVEL2']  == 0)
    for i in model.ProdutosFechados:
        if i in df_agrupado.loc[(df_agrupado.VOLUME_FECHADO == 'sim') & (df_agrupado.CLASSIFICACAO == 'C'), 'NOME_PROD'].values:
            model.c6.add(model.yFech[i, 'RUA3_NIVEL1']  == 0)

    solver = SolverFactory('cbc')

    solver.solve(model, timelimit=10)

    df_results1 = pd.DataFrame([(i, j, model.xFrac[i, j].value) for i in model.ProdutosFracionados for j in model.PosicoesFracionados],
                columns=['NOME_PROD', 'POSICAO', 'QUANTIDADE_PALLETS_ALOCADA'])
    df_results1 = df_results1.loc[df_results1.QUANTIDADE_PALLETS_ALOCADA > 0]

    df_results2 = pd.DataFrame([(i, j, model.xFech[i, j].value) for i in model.ProdutosFechados for j in model.PosicoesFechados],
                columns=['NOME_PROD', 'POSICAO', 'QUANTIDADE_PALLETS_ALOCADA'])
    df_results2 = df_results2.loc[df_results2.QUANTIDADE_PALLETS_ALOCADA > 0]
    df_results = pd.concat([df_results1, df_results2])
    return df_results
    
def main():
    st.header('Otimizador de posicionamento de itens')
    st.subheader('Configuração de volume dos pallets')
    cols = st.beta_columns(2)
    with cols[0]:
        volume_pallet_bruto = st.number_input('Volume do pallet (m³)', value=2.3 * 1.3 * 1.13 * 0.7)
    with cols[1]:
        fator_de_correcao = st.number_input('Fator de correção de volume', value=0.5)
    volume_pallet = volume_pallet_bruto * fator_de_correcao
    st.subheader('Quantidade de pallets por rua')
    cols = st.beta_columns(5)
    with cols[0]:
        rua1D = st.number_input('Rua 1D', value=(28 * 2 - 2) * 2)
    with cols[1]:
        rua1E = st.number_input('Rua 1E', value=26 * 2 * 2)
    with cols[2]:
        rua2D = st.number_input('Rua 2D', value=26 * 2 * 2)
    with cols[3]:
        rua2E = st.number_input('Rua 2E', value=26 * 2 * 2)
    with cols[4]:
        rua3D = st.number_input('Rua 3D', value=26 * 2 * 2)
    n_pallets_picking = rua1D + rua1E + rua2D + rua2E + rua3D
    st.subheader('Configuração das estatísticas')
    cols = st.beta_columns(2)
    with cols[0]:
        cobertura_leadtime =st.number_input('Cobertura do leadtime (decimal)', value=0.1)
    with cols[1]:
        quantil = st.number_input('Quantil (decimal)', value=0.75)
    st.subheader('Upload de arquivos')
    cols = st.beta_columns(3)
    with cols[0]:
        produtos = st.file_uploader('Produtos', type=['xlsx', 'xls', 'csv'])
    with cols[1]:
        grandes_contas = st.file_uploader('Grandes contas', type=['xlsx', 'xls', 'csv'])
    with cols[2]:
        producao = st.file_uploader('Vendas', type=['xlsx', 'xls', 'csv'])
    cols = st.beta_columns(4)
    with cols[0]:
        frac =  st.file_uploader('Produtos fracionados x fechado', type=['xlsx', 'xls', 'csv'])
    with cols[1]:
        lead = st.file_uploader('Lead time dos fornecedores', type=['xlsx', 'xls', 'csv'])
    with cols[2]:
        restricoesr1 = st.file_uploader('Restrições da Rua 1', type=['xlsx', 'xls', 'csv'])
    with cols[3]:
        inativos = st.file_uploader('Produtos inativos', type=['xlsx', 'xls', 'csv'])
    if st.button('Calcular'):
        try:
            df_produtos = pd.read_csv(produtos)
        except Exception:
            df_produtos = pd.read_excel(produtos)
        try:
            df_grandes_contas = pd.read_csv(grandes_contas)
        except Exception:
            df_grandes_contas = pd.read_excel(grandes_contas)
        try:
            df_producao = pd.read_csv(producao)
        except Exception:
            df_producao = pd.read_excel(producao)
        try:
            df_frac = pd.read_csv(frac)
        except Exception:
            df_frac = pd.read_excel(frac)
        try:
            df_lead = pd.read_csv(lead)
        except Exception:
            df_lead = pd.read_excel(lead)
        try:
            df_restricoesr1 = pd.read_csv(restricoesr1)
        except Exception:
            df_restricoesr1 = pd.read_excel(restricoesr1)
        else:
            df_restricoesr1 = None
        if inativos:
            try:
                df_inativos = pd.read_csv(inativos)
            except Exception:
                df_inativos = pd.read_excel(inativos)
        else:
            df_inativos = None
        df_produtos['VOLUME_UNIT'] = df_produtos.ALTURA * df_produtos.LARGURA * df_produtos.COMPRIMENTO
        clientes_eliminar = df_grandes_contas.codclie.values
        df_producao = df_producao.loc[~df_producao['COD CLIENTE'].isin(clientes_eliminar)]
        df_producao = pd.merge(df_producao, df_frac[['NOME_PROD', 'FRACIONADO', 'FECHADO']], left_on='NOME_PROD', right_on='NOME_PROD')
        df_producao = pd.merge(df_producao, df_produtos, left_on='PRODUTOID', right_on='PRODUTOID')
        df_producao['data'] = pd.to_datetime(df_producao['DATAFATURAMENTO'])
        df_producao['Soma de QUANT'] = pd.to_numeric(df_producao['Soma de QUANT'], errors='coerce')
        df_producao['VOLUME'] = df_producao['ALTURA'] * df_producao['LARGURA'] * df_producao['COMPRIMENTO'] * df_producao['Soma de QUANT']
        df_leadtime = df_lead.groupby(['NOME_PROD'])['LEAD TIME'].max().reset_index()
        
        if inativos:
            df_producao_temp = df_producao.loc[~df_producao.NOME_PROD.isin(df_inativos.DESCRICAO.values)].copy()
        else:
            df_producao_temp = df_producao.copy()
        df_producao1 = df_producao_temp.groupby([df_producao_temp['data'].dt.date, 'NOME_PROD'])[['Soma de QUANT', 'VOLUME']].sum().reset_index()
        df_agrupado = df_producao1.groupby(['NOME_PROD'])[['Soma de QUANT', 'VOLUME']].quantile(quantil).sort_values('VOLUME', ascending=False)
        df_agrupado = df_agrupado.reset_index()
        df_agrupado = pd.merge(df_agrupado, df_leadtime, left_on='NOME_PROD', right_on='NOME_PROD')
        df_agrupado.drop_duplicates(subset=['NOME_PROD'], inplace=True)

        df_agrupado['N_PALLETS_1_DIA'] = df_agrupado.VOLUME / volume_pallet
        df_agrupado['COBERTURA'] = np.ceil(df_agrupado['LEAD TIME'] * (1 + cobertura_leadtime))
        df_agrupado['N_PALLETS_TOTAL'] = df_agrupado.N_PALLETS_1_DIA * df_agrupado.COBERTURA
        df_agrupado['NUMERO_UNIDADES'] = df_agrupado.COBERTURA * df_agrupado['Soma de QUANT']
        df_agrupado['PONTO_REPOSICAO'] = df_agrupado['LEAD TIME'] * df_agrupado['Soma de QUANT']
        df_agrupado = pd.merge(df_agrupado, df_frac[['NOME_PROD', 'FECHADO', 'FRACIONADO']], left_on='NOME_PROD', right_on='NOME_PROD')
        df_agrupado.loc[df_agrupado.FECHADO == 'x', 'VOLUME_FECHADO'] = 'sim'
        df_agrupado.loc[df_agrupado.FRACIONADO == 'x', 'VOLUME_FECHADO'] = 'não'
        df_agrupado.rename(columns={'Soma de QUANT': f'QUANTIDADE ({quantil * 100}%)'}, inplace=True)
        df_agrupado.sort_values(['VOLUME_FECHADO', f'QUANTIDADE ({quantil * 100}%)'], inplace=True, ascending=False)
        df_agrupado['perc_acum'] = df_agrupado.groupby('VOLUME_FECHADO')[f'QUANTIDADE ({quantil * 100}%)'].apply(lambda value: value.cumsum() / value.sum())
        df_agrupado.loc[df_agrupado.perc_acum < 0.21, 'CLASSIFICACAO'] = 'A'
        df_agrupado.loc[(df_agrupado.perc_acum >= 0.21) & (df_agrupado.perc_acum <= 0.5), 'CLASSIFICACAO'] = 'B'
        df_agrupado.loc[df_agrupado.perc_acum > 0.5 , 'CLASSIFICACAO'] = 'C'
        df_agrupado = df_agrupado[['NOME_PROD', 'VOLUME_FECHADO', 'CLASSIFICACAO', f'QUANTIDADE ({quantil * 100}%)', 'LEAD TIME',
       'N_PALLETS_1_DIA', 'COBERTURA', 'N_PALLETS_TOTAL',
       'NUMERO_UNIDADES', 'PONTO_REPOSICAO']]
        df_agrupado = df_agrupado.loc[df_agrupado.N_PALLETS_TOTAL > 0]
        df_agrupado['N_MAXIMO_PALLETS'] = df_agrupado.N_PALLETS_TOTAL
        df_agrupado['N_MINIMO_PALLETS'] = df_agrupado.N_PALLETS_1_DIA * df_agrupado['LEAD TIME']
        
        df_results = solve(df_agrupado, df_restricoesr1, volume_pallet)
        st.markdown(get_table_download_link(df_results), unsafe_allow_html=True)
        st.dataframe(df_results)
        
        
        
    
    
if __name__ == '__main__':
    main()