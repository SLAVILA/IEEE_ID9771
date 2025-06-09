import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
from hmmlearn import hmm
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import subprocess

# script_path = "hmm_cod_LK6_planilhas_com_datas.py"  

# # Verifica se o arquivo existe
# if os.path.isfile(script_path):
#     result = subprocess.run(["python", script_path], capture_output=True, text=True)
    
#     # Exibe a saída e erros, se houver
#     print("Saída:", result.stdout)
#     print("Erros:", result.stderr)
# else:
#     print("O arquivo não existe:", script_path)
    
    

df_iteracao_atual1 = []
df_iteracoes_list = [] 
df_iteracoes_list1 = [] 
# diretorio = r"C:/Users/Dell/Desktop/P&D Urca/dados_preco_atualizado/raw/"

grupoEstag = pd.DataFrame()
grupoLower = pd.DataFrame()
grupoUpper= pd.DataFrame()
grupoEstagMedian = pd.DataFrame()
grupoLowerMedian = pd.DataFrame()
grupoUpperMedian = pd.DataFrame()
grupoUpperMedian = pd.DataFrame()
grupoUpperMedian = pd.DataFrame()
grupoUpperMedian = pd.DataFrame()
grupoUpperMedian = pd.DataFrame()
grupoUpperMedian = pd.DataFrame()
grupoUpperMedian = pd.DataFrame()
df_dados = pd.DataFrame()
dataset0 = pd.DataFrame()
dataset = pd.DataFrame()
df_iteracoes1 = pd.DataFrame()
df_iteracoes1T = pd.DataFrame()

# Caminho para o diretório com as planilhas
diretorio = r"C:/Users/Dell/Desktop/P&D Urca/dados_preco_atualizado/raw"

# Pasta destino onde os gráficos serão salvos
pasta_destino = r'C:/Users/Dell/Desktop/dados_LK6 (1)/dados_LK6/'

def analisar_planilha(nomeDaPlanilha):

    
    global grupoEstag
    global grupoLower
    global grupoUpper
    global grupoEstagMedian
    global grupoLowerMedian
    global grupoUpperMedian
    global df_dados
    global df_iteracoes1
    global df_iteracoes1T
    global dataset0 
    global dataset 

    # numero de iterações definido pelo usuario
    for i in range(500):
        print(i)
    
    
    
        # parametros do modelo
        NUM_TEST = 15
        K = 50
        NUM_ITERS = 10000
    
        #plnanilha a ser acessada "como banco de dados" 
       
        dados_planilha = nomeDaPlanilha
        
        nome_planilha = dados_planilha[0:-5]
        
     
        
        dataset0 = pd.read_excel(dados_planilha)
        dataset = pd.read_excel(dados_planilha)
        dataset = dataset[:-15] # tirando os ultimos 15 dados das planilhas 
        dataset = dataset['VWAP'].values
        dataset = dataset.reshape(-1, 1)
    
    
    
        
        df_iteracoes = pd.DataFrame()
    
        likelihood_vect = np.empty([0, 1])
        aic_vect = np.empty([0, 1])
        bic_vect = np.empty([0, 1])
    
        # Possíveis números de estados no Modelo de Markov
        STATE_SPACE = range(2, 15)
    
        dataset_normal = dataset.reshape(-1, 1)  # Ajusta a forma para torná-lo bidimensional
        dataset = np.flip(dataset_normal)
    
        predicted_stock_data = np.empty([0, 1])
        likelihood_vect = np.empty([0, 1])
        aic_vect = np.empty([0, 1])
        bic_vect = np.empty([0, 1])
        new_train = np.empty([0, 1])
        
        for states in STATE_SPACE:
            try:
                num_params = states**2 + states
                dirichlet_params_states = np.random.randint(1, 50, states)
                model = hmm.GaussianHMM(n_components=states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS)
                model.fit(dataset[NUM_TEST:])
            except ValueError as e:
                print(f"Erro: {e}. Continuando para o próximo estado.")
                continue
            
            try:
                
                likelihood_vect = np.vstack((likelihood_vect, model.score(dataset)))
            except ValueError as e:
                print(f"Erro: {e}. Numero de estado deve ser escolhido manualmente")
                continue  
            try:
                aic_vect = np.vstack((aic_vect, -2 * model.score(dataset) + 2 * num_params))
            except ValueError as e:
                print(f"Erro: {e}. Numero de estado deve ser escolhido manualmente")
                continue  
            try:
                bic_vect = np.vstack((bic_vect, -2 * model.score(dataset) +  num_params * np.log(dataset.shape[0])))
            except ValueError as e:
                print(f"Erro: {e}. Numero de estado deve ser escolhido manualmente")
                continue  
            # Verifique se todas as três partes do código têm erro
            if "likelihood_vect" in locals() and "aic_vect" in locals() and "bic_vect" in locals():
                opt_states = np.random.randint(1, 16) # se houver erro de não convergencia. O numero de estados é alterado aleatório 
            else:
                # Se não houver erro, defina opt_states com base no índice do mínimo valor em bic_vect mais 2
                opt_states = np.argmin(bic_vect) + 2
        print('O número ótimo de estados é {}'.format(opt_states))
        
        for idx in reversed(range(NUM_TEST)):
            train_dataset = dataset[idx + 1:].reshape(-1, 1)
            test_data = dataset[idx]
            num_examples = train_dataset.shape[0]
            if idx == NUM_TEST - 1:
                model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS,
                                        init_params='stmc')
            else:
                model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS,
                                        init_params='')
                model.transmat_ = transmat_retune_prior
                model.startprob_ = startprob_retune_prior
                model.means_ = means_retune_prior
                model.covars_ = covars_retune_prior
        
            if idx == NUM_TEST - 1:
                current_data = np.flipud(train_dataset) # se 1º iteração 
            else:
                current_data = np.flipud(new_train)  # a partir da 2° iteração 
        
            try:
                model.fit(current_data)
            except ValueError as e:
                print(f"Erro: {e}. Numero de estado deve ser escolhido manualmente")
                continue 
        
          
            try:
                model.fit(current_data)
            except ValueError as e:
                print(f"Erro: {e}. Numero de estado deve ser escolhido manualmente")
                continue
           
               
            try:
                transmat_retune_prior = model.transmat_ 
            except ValueError as e:
                print(f"Erro: {e}. Numero de estado deve ser escolhido manualmente")
                continue
            
            try:
                startprob_retune_prior = model.startprob_
            except ValueError as e:
                print(f"Erro: {e}. Numero de estado deve ser escolhido manualmente")
                continue
            
            try:
                means_retune_prior = model.means_
            except ValueError as e:
                print(f"Erro: {e}. Numero de estado deve ser escolhido manualmente")
                continue
                
            try:
                covars_retune_prior = model.covars_
            except ValueError as e:
                print(f"Erro: {e}. Numero de estado deve ser escolhido manualmente")
                continue
            if model.monitor_.iter == NUM_ITERS:
                print('Aumente o número de iterações')
                        
            iters = 1
            past_likelihood = []
            curr_likelihood = model.score(np.flipud(train_dataset[0:K - 1].reshape(-1, 1)))
            
            while iters < num_examples / K - 1:
                past_likelihood = np.append(past_likelihood, model.score(np.flipud(train_dataset[iters:iters + K - 1].reshape(-1, 1))))
                iters = iters + 1
            likelihood_diff_idx = np.argmin(np.absolute(past_likelihood - curr_likelihood))
            predicted_change = train_dataset[likelihood_diff_idx] - train_dataset[likelihood_diff_idx + 1]
            # predicted_stock_data = np.vstack((predicted_stock_data, dataset[idx + 1] - predicted_change))
            
            predicted_stock_data = np.vstack((predicted_stock_data, current_data[-1, :] + predicted_change))
            
            if idx == NUM_TEST - 1:    # se primeira iteraçao 
                new_train = train_dataset.copy()
                new_train = np.vstack((predicted_stock_data[NUM_TEST-idx-1, 0], new_train)) # acumula a 1° previsão 
            else:
                if idx != NUM_TEST - 1 :    
                    new_train = np.vstack((predicted_stock_data[NUM_TEST-idx-1, 0], new_train))  # acumula as demais previsões 
        
        
        
         # figura para guiar a analise 
        
        try:
            
            plt.figure()
            plt.plot(range(15), predicted_stock_data[:, 0], color='blue', label='Preço previsto ' )
            plt.plot(range(15), np.flipud(dataset[range(15)]), 'r--', label='Preço real ')
            plt.xlabel('Passos temporais')
            plt.ylabel('Preço')
            plt.title('previsão '+str(i)+' '+ dados_planilha[:-4])
            plt.grid(True)
            plt.legend(loc='upper left')
            
            
          
        except ValueError as e:
            print(f"Erro: {e}.continue sem plotagem")
            continue
        
              
        # Adicionar resultados desta iteração ao DataFrame
        
        df_iteracao_atual1 = pd.DataFrame({f'HMM Iteracao_{i}': predicted_stock_data[:, 0]})
        df_iteracao_atual1.index.name = 'Passos temporais'
    
        # Adicionar à lista de DataFrames para posterior concatenação
        df_iteracoes_list.append(df_iteracao_atual1)
    
        # Concatenar todos os DataFrames da lista em um único DataFrame
        df_iteracoes1 = pd.concat(df_iteracoes_list, axis=1)
        df_iteracoes1T = df_iteracoes1.T
        df_iteracoes1T_mean = df_iteracoes1T.mean()
    
    
    #################################  montagem do dataframe que servira as informações para os graficos em plotly
    
    
    
    
    df_iteracoes1 = df_iteracoes1.abs() 
    
    #limpei colunas que não seram mais usadas 
    columns = ['date', 'high_mark', 'low_mark', 'estag_mark']
    df_dados = pd.DataFrame(columns=columns)
    
    
    # ultimo dia negociado do banco de dados
    d0 = dataset0['VWAP'].iloc[-1]
    
    dmais = d0*1.005
    dmenos = d0*0.995
    
    for i in range(len(df_iteracoes1)):
        
        # Contagem de valores na linha i
        count_values = len(df_iteracoes1.columns) 
        # Valor d0
        # d0 = count_values
        
        # Calcular d0 * 1.005
        threshold = d0 * 1.005
        
        # Contar quantos valores na linha i são maiores que o threshold
        values_greater_than_threshold = (df_iteracoes1.iloc[i, 0:] > dmais).sum()
        
        # Calcular a porcentagem
        high_mark_value = (values_greater_than_threshold * 100) / count_values
        
        # Preencher a coluna 'high_mark'
        df_dados.loc[i, 'high_mark'] = high_mark_value
    
    
        # Contar quantos valores na linha i são maiores que o threshold
        values_smaller_than_threshold = (df_iteracoes1.iloc[i, 0:] < dmenos).sum()
        
        # Calcular a porcentagem
        low_mark_value = (values_smaller_than_threshold * 100) / count_values
        
        # Preencher a coluna 'high_mark'
        df_dados.loc[i, 'low_mark'] = low_mark_value
        
        df_dados.loc[i, 'estag_mark'] = -1 * high_mark_value - 1 * df_dados.loc[i, 'low_mark'] + 100
        
        #Preencher coluna Media_preço
        df_iteracoes1T = df_iteracoes1.T
        df_iteracoes1T_mean = df_iteracoes1T.mean()
        df_dados['media_precos'] = df_iteracoes1T_mean
    
    
    data = dataset0['data'].iloc[-1]
    data = pd.to_datetime(data)
    
    
    # preenchendo o dataframe df_dados
    
    df_dados.loc[0, 'date'] = data + pd.DateOffset(days=1)
    df_dados.loc[1, 'date'] = data + pd.DateOffset(days=2)
    df_dados.loc[2, 'date'] = data + pd.DateOffset(days=3)
    df_dados.loc[3, 'date'] = data + pd.DateOffset(days=4)
    df_dados.loc[4, 'date'] = data + pd.DateOffset(days=5)
    df_dados.loc[5, 'date'] = data + pd.DateOffset(days=6)
    df_dados.loc[6, 'date'] = data + pd.DateOffset(days=7)
    df_dados.loc[7, 'date'] = data + pd.DateOffset(days=8)
    df_dados.loc[8, 'date'] = data + pd.DateOffset(days=9)
    df_dados.loc[9, 'date'] = data + pd.DateOffset(days=10)
    df_dados.loc[10, 'date'] = data + pd.DateOffset(days=11)
    df_dados.loc[11, 'date'] = data + pd.DateOffset(days=12)
    df_dados.loc[12, 'date'] = data + pd.DateOffset(days=13)
    df_dados.loc[13, 'date'] = data + pd.DateOffset(days=14)
    df_dados.loc[14, 'date'] = data + pd.DateOffset(days=15)
        
    
    ##############################################################  Figura para a Dash
    ### a partir daqui é acrescentado informações novas 
    
    
    d0_ = dataset0['VWAP'].iloc[-1]
    
    porcentagem_padrao = 0.5
    
    dviMaior =  ((porcentagem_padrao/d0_)+1)*d0_
    
    dviMenor =  (1-(porcentagem_padrao/d0_))*d0_
    
    
    # Inicializar grupoUpper,grupoLower, grupoEstag com as mesmas colunas e índice de df_iteracoes1T, mas vazio
    grupoUpper = pd.DataFrame(np.nan, index=df_iteracoes1T.index, columns=df_iteracoes1T.columns)
    grupoLower = pd.DataFrame(np.nan, index=df_iteracoes1T.index, columns=df_iteracoes1T.columns)
    grupoEstag = pd.DataFrame(np.nan, index=df_iteracoes1T.index, columns=df_iteracoes1T.columns)
    
    
    
    # Preencher grupoUpper conforme a condição
    for i in range(15):  # Loop de 0 a 14
        grupoUpper.loc[df_iteracoes1T[i] >= dviMaior, i] = df_iteracoes1T.loc[df_iteracoes1T[i] >= dviMaior, i]
    
        
    grupoUpperMedian = grupoUpper.median()
    grupoUpperMedian = grupoUpperMedian.to_frame()
        
    
    
    for i in range(15):
        grupoUpperMedian.loc[i, 'date'] = data + pd.DateOffset(days=i+1)
        grupoUpperMedian['date'] = pd.to_datetime(grupoUpperMedian['date'])
        
        
        
        
    # Preencher grupoLower conforme a condição
    for i in range(15):  # Loop de 0 a 14
        grupoLower.loc[df_iteracoes1T[i] <= dviMenor, i] = df_iteracoes1T.loc[df_iteracoes1T[i] <= dviMenor, i]
        grupoLowerMedian = grupoLower.median()    
        
    grupoLowerMedian = grupoLowerMedian.to_frame()
      
    
    
    
    # grupoLowerMedian = grupoLowerMedian.T
        
    for i in range(15):
        grupoLowerMedian.loc[i, 'date'] = data + pd.DateOffset(days=i+1)
        grupoLowerMedian['date'] = pd.to_datetime(grupoLowerMedian['date'])
      
         
    for i in range(15):  # Loop de 0 a 14
        grupoEstag.loc[(df_iteracoes1T[i] > dviMenor) & (df_iteracoes1T[i] < dviMaior), i] = df_iteracoes1T.loc[(df_iteracoes1T[i] > dviMenor) & (df_iteracoes1T[i] < dviMaior), i]
        grupoEstagMedian = grupoEstag.median()  
    
    grupoEstagMedian = grupoEstagMedian.to_frame()
        
    
    for i in range(15):
        grupoEstagMedian.loc[i, 'date'] = data + pd.DateOffset(days=i+1)
        grupoEstagMedian['date'] = pd.to_datetime(grupoEstagMedian['date'])
        
        
        
    
    
    # valores percentis de cada grupo 
    
    for i in range(15):  
        col = grupoUpper[grupoUpper.columns[i]]
        if not col.isna().all():  # Verifica se não são todos NaNs
            df_dados.loc[i, 'p75_Upper'] = np.nanpercentile(col, 75)
        else:
            df_dados.loc[i, 'p75_Upper'] = np.nan  # Atribui NaN se todos os valores forem NaN
    
    for i in range(15):  
        col = grupoLower[grupoLower.columns[i]]
        if not col.isna().all():  # Verifica se não são todos NaNs
            df_dados.loc[i, 'p75_Lower'] = np.nanpercentile(col, 75)
        else:
            df_dados.loc[i, 'p75_Lower'] = np.nan  # Atribui NaN se todos os valores forem NaN
    
    for i in range(15):  
        col = grupoEstag[grupoEstag.columns[i]]
        if not col.isna().all():  # Verifica se não são todos NaNs
            df_dados.loc[i, 'p75_Estag'] = np.nanpercentile(col, 75)
        else:
            df_dados.loc[i, 'p75_Estag'] = np.nan  # Atribui NaN se todos os valores forem NaN
    
    for i in range(15):  
        col = grupoUpper[grupoUpper.columns[i]]
        if not col.isna().all():  # Verifica se não são todos NaNs
            df_dados.loc[i, 'p25_Upper'] = np.nanpercentile(col, 25)
        else:
            df_dados.loc[i, 'p25_Upper'] = np.nan  # Atribui NaN se todos os valores forem NaN
    
    for i in range(15):  
        col = grupoLower[grupoLower.columns[i]]
        if not col.isna().all():  # Verifica se não são todos NaNs
            df_dados.loc[i, 'p25_Lower'] = np.nanpercentile(col, 25)
        else:
            df_dados.loc[i, 'p25_Lower'] = np.nan  # Atribui NaN se todos os valores forem NaN
    
    for i in range(15):  
        col = grupoEstag[grupoEstag.columns[i]]
        if not col.isna().all():  # Verifica se não são todos NaNs
            df_dados.loc[i, 'p25_Estag'] = np.nanpercentile(col, 25)
        else:
            df_dados.loc[i, 'p25_Estag'] = np.nan  # Atribui NaN se todos os valores forem NaN
    
   
       
    
    ############################################################################# 15 dias grafico de linha 

        # Criar o gráfico de linhas
        fig = go.Figure()
        
        # valores superiores 
        fig.add_trace(go.Scatter(
            x=grupoUpperMedian['date'],  # Série de datas
            y=grupoUpperMedian[grupoUpperMedian.columns[0]],  # Série de valores
            mode='lines',
            name='Tendência de Alta',
            line=dict(color='blue'),
            hovertemplate='Data: %{x}<br>Tendência de Alta: %{y}<br><extra></extra>',
            showlegend=True       
        ))
        
        fig.add_trace(go.Scatter(
            x=df_dados['date'],  # Série de datas
            y=df_dados['p75_Upper'],  # Série de valores
            mode='lines',
            name='P75',
            line=dict(color='blue', dash='dot'),
            hovertemplate='Data: %{x}<br>P75: %{y}<br><extra></extra>',
            showlegend=True       
        ))
        
        fig.add_trace(go.Scatter(
            x=df_dados['date'],  # Série de datas
            y=df_dados['p25_Upper'],  # Série de valores
            mode='lines',
            name='P25',
            line=dict(color='blue', dash='dot'),
            hovertemplate='Data: %{x}<br>P25: %{y}<br><extra></extra>',
            showlegend=True       
        ))
        
        # Hachura entre as duas linhas Upper
        fig.add_trace(go.Scatter(
            x=pd.concat([df_dados['date'], df_dados['date'][::-1]]),  # Combine as datas (ida e volta)
            y=pd.concat([df_dados['p25_Upper'], df_dados['p75_Upper'][::-1]]),  # Combine os valores correspondentes
            fill='toself',  # Preencher a área entre as linhas
            fillcolor='rgba(173, 216, 230, 0.4)',  # Azul claro com transparência
            line=dict(color='rgba(255, 255, 255, 0)'),  # Linha invisível
            hoverinfo='skip',  # Ignorar hover na área preenchida
            showlegend=False  # Não mostrar a área preenchida na legenda
            ))
        
        
        # valores inferiores 
        fig.add_trace(go.Scatter(
            x=grupoLowerMedian['date'],  # Série de datas
            y=grupoLowerMedian[grupoLowerMedian.columns[0]],  # Série de valores
            mode='lines',
            name='Tendência de Baixa',
            line=dict(color='red'),
            hovertemplate='Data: %{x}<br>Tendência de Baixa: %{y}<br><extra></extra>',
            showlegend=True       
        ))
        
        
        fig.add_trace(go.Scatter(
            x=df_dados['date'],  # Série de datas
            y=df_dados['p75_Lower'],  # Série de valores
            mode='lines',
            name='P75',
            line=dict(color='red', dash='dot'),
            hovertemplate='Data: %{x}<br>P75: %{y}<br><extra></extra>',
            showlegend=True       
        ))
        
        fig.add_trace(go.Scatter(
            x=df_dados['date'],  # Série de datas
            y=df_dados['p25_Lower'],  # Série de valores
            mode='lines',
            name='P25',
            line=dict(color='red', dash='dot'),
            hovertemplate='Data: %{x}<br>P25: %{y}<br><extra></extra>',
            showlegend=True       
        ))
        
        # Hachura entre as duas linhas Upper
        fig.add_trace(go.Scatter(
            x=pd.concat([df_dados['date'], df_dados['date'][::-1]]),  # Combine as datas (ida e volta)
            y=pd.concat([df_dados['p25_Lower'], df_dados['p75_Lower'][::-1]]),  # Combine os valores correspondentes
            fill='toself',  # Preencher a área entre as linhas
            fillcolor='rgba(255, 0, 0, 0.4)',  # vermelho claro com transparência
            line=dict(color='rgba(255, 255, 255, 0)'),  # Linha invisível
            hoverinfo='skip',  # Ignorar hover na área preenchida
            showlegend=False  # Não mostrar a área preenchida na legenda
            ))
        
        
        
        # valores estagnados 
        fig.add_trace(go.Scatter(
            x=grupoEstagMedian['date'],  # Série de datas
            y=grupoEstagMedian[grupoEstagMedian.columns[0]],  # Série de valores
            mode='lines',
            name='Tendência de Estagnação',
            line=dict(color='yellow'),
            hovertemplate='Data: %{x}<br>Tendência de Estagnação: %{y}<br><extra></extra>',
            showlegend=True       
        ))
        
        
        fig.add_trace(go.Scatter(
            x=df_dados['date'],  # Série de datas
            y=df_dados['p75_Estag'],  # Série de valores
            mode='lines',
            name='P75',
            line=dict(color='yellow', dash='dot'),
            hovertemplate='Data: %{x}<br>P75: %{y}<br><extra></extra>',
            showlegend=True       
        ))
        
        fig.add_trace(go.Scatter(
            x=df_dados['date'],  # Série de datas
            y=df_dados['p25_Estag'],  # Série de valores
            mode='lines',
            name='P25',
            line=dict(color='yellow', dash='dot'),
            hovertemplate='Data: %{x}<br>P25: %{y}<br><extra></extra>',
            showlegend=True       
        ))
        
        # Hachura entre as duas linhas Upper
        fig.add_trace(go.Scatter(
            x=pd.concat([df_dados['date'], df_dados['date'][::-1]]),  # Combine as datas (ida e volta)
            y=pd.concat([df_dados['p25_Estag'], df_dados['p75_Estag'][::-1]]),  # Combine os valores correspondentes
            fill='toself',  # Preencher a área entre as linhas
            fillcolor='rgba(255, 255, 0, 0.4)',  # Amarelo claro com transparência
            line=dict(color='rgba(255, 255, 255, 0)'),  # Linha invisível
            hoverinfo='skip',  # Ignorar hover na área preenchida
            showlegend=False  # Não mostrar a área preenchida na legenda
            ))
        
        # fig.add_trace(go.Scatter(
        #     x=df_dados['date'].iloc[-15:],  # Série de datas
        #     y= dataset0['VWAP'],  # Série de valores
        #     mode='lines',
        #     name='VWAP real',
        #     line=dict(color='black'),
        #     hovertemplate='Data: %{x}<br>VWAP real: %{y}<br><extra></extra>',
        #     showlegend=True       
        # ))
        
        # Configurar o layout do gráfico
        fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    text="Tendências",  # Texto do título
                    xref="paper",  # Referência ao gráfico (em porcentagem da área do gráfico)
                    yref="paper",
                    x=1.02,  # Posição no eixo x (ajuste conforme necessário)
                    y=1,     # Posição no eixo y (ajuste conforme necessário)
                    showarrow=False,
                    font=dict(
                        size=14,
                        color="black"
                    ),
                    xanchor='left',
                    yanchor='bottom'
                )
            ],
            yaxis=dict(
                title='Tendência de VWAP',  # Ajuste o título do eixo y conforme necessário
            ),
            title='Gráfico de Linha',
            hoverlabel=dict(
                bgcolor="white",  # Cor de fundo do box
                font_size=16,     # Tamanho da fonte
                font_family="Rockwell"  # Família da fonte
            )
        )
        
       # Extrair o nome da planilha (remover a extensão do arquivo .xlsx)
    nome_planilha = os.path.splitext(os.path.basename(nomeDaPlanilha))[0]

    # Nome do arquivo para salvar o gráfico
    nome_arquivo = f"15dias_grafico_de_linhas_{nome_planilha}.html"
    
    # Caminho completo do arquivo
    caminho_completo = os.path.join(pasta_destino, nome_arquivo)
    
    # Salve o arquivo no caminho completo
    fig.write_html(caminho_completo)
    print(f"Gráfico salvo em {caminho_completo}")

# Lista de todas as planilhas no diretório
planilhas = [os.path.join(diretorio, arquivo) for arquivo in os.listdir(diretorio) if arquivo.endswith('.xlsx')]

# Loop sobre cada planilha
for nomeDaPlanilha in planilhas:
    analisar_planilha(nomeDaPlanilha)
    
    
   
    