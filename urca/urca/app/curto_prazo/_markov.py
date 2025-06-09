from datetime import datetime, timedelta
import json
import os
import shutil
from flask import current_app, render_template, request, jsonify, Blueprint, session
import matplotlib.pyplot as plt
from hmmlearn import hmm
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_json
from biblioteca.task import Tasker

# Blueprint registration
from app.curto_prazo import b_curto_prazo

script_dir = os.path.dirname(__file__)

@b_curto_prazo.route("/_markov_dash", methods=["POST"])
def _markov_dash():
    pasta = request.form.get("pasta")
    output_dir = os.path.join('web', 'static', 'tasks_saida', 'markov', pasta)
    
    data = request.form.get("data")  # Pega a data fornecida no formato original
    
    # Obter os arquivos iniciados em markov-dash
    files = [f for f in os.listdir(output_dir) if f.startswith("markov-dash_") and f.endswith(".json")]

    if not files:
        return json.dumps({"status": "1", "msg": "Nenhum arquivo encontrado."})

    # Obter as datas dos arquivos (em XX_XX_XXXX)
    datas = [f.replace("markov-dash_", "").split(".")[0] for f in files]

    # Transformar strings de datas em objetos datetime
    datas_datetime = [datetime.strptime(data, "%d_%m_%Y") for data in datas]
    
    # Função para reduzir a data até encontrar um arquivo válido
    def encontrar_arquivo_por_data(data_inicial, arquivos, datas):
        while True:
            formatted_str = data_inicial.strftime('%d_%m_%Y')
            for file, data_str in zip(arquivos, datas):
                if data_str == formatted_str:
                    return file, formatted_str
            # Se não encontrar o arquivo, reduz a data em 1 dia
            data_inicial -= timedelta(days=1)
            # Opcional: limite de redução de datas (ex: até 1 ano atrás)
            if data_inicial < datetime.now() - timedelta(days=365):
                break
        return None, None
    
    # Se uma data específica foi fornecida
    if data:
        formatted_date = datetime.strptime(data, "%d/%m/%Y")
        arquivo, data_arquivo = encontrar_arquivo_por_data(formatted_date, files, datas)
        
        if not arquivo:
            return json.dumps({"status": "1", "msg": "Nenhum arquivo encontrado para a data fornecida ou anterior válida."})
    else:
        # Caso nenhuma data tenha sido fornecida, pegar o arquivo mais recente
        arquivo = files[datas_datetime.index(max(datas_datetime))]
        data_arquivo = arquivo.replace("markov-dash_", "").split(".")[0]

    try:
        # Abrir o arquivo JSON
        with open(os.path.join(output_dir, arquivo), 'r') as f:
            plot_json = json.load(f)
    except:
        return json.dumps({"status": "1", "msg": "Gráfico não encontrado para o modelo Markov."})
    
    return json.dumps({"status": "0", "plot": plot_json, 'data': data_arquivo.replace("_", "/")})


@b_curto_prazo.route("/_markov_obter", methods=["POST"])
def _markov_obter():
    pasta = request.form.get("pasta")
    pasta_dash = f"rolloff suavizado {pasta}SE -> VWAP"  # Para buscar o gráfico do dash
    pasta_saida = f"rolloff suavizado {pasta}SE -> VWAP"  # Para buscar o gráfico markov-saida
    data = request.form.get("data").replace('/', '_')
    
    # Diretórios
    output_dir_dash = os.path.join('web', 'static', 'tasks_saida', 'markov', pasta_dash)
    output_dir_saida = os.path.join('web', 'static', 'tasks_saida', 'markov', pasta_saida)
    
    # -- Processamento para o gráfico 'markov-dash' --
    
    # Obter os arquivos iniciados em markov-dash
    files_dash = [f for f in os.listdir(output_dir_dash) if f.startswith("markov-dash_") and f.endswith(".json")]
    
    # Obter as datas dos arquivos (em XX_XX_XXXX)
    datas_dash = [f.replace("markov-dash_", "").split(".")[0] for f in files_dash if f.endswith(".json")]
    
    # Transformar em datetime
    datas_dash = [datetime.strptime(data_dash, "%d_%m_%Y") for data_dash in datas_dash]
    
    # Procurar o arquivo de dash pela data
    if data and f"markov-dash_{data}.json" in files_dash:
        arquivo_dash = f"markov-dash_{data}.json"
    else:
        # Pegar o arquivo mais recente se data específica não for encontrada
        if datas_dash:
            arquivo_dash = files_dash[datas_dash.index(max(datas_dash))]
        else:
            return json.dumps({"status": "1", "msg": "Nenhum gráfico Markov Dash encontrado."})
    
    # -- Processamento para o gráfico 'markov-saida' --
    
    # Obter os arquivos iniciados em markov-saida
    files_saida = [f for f in os.listdir(output_dir_saida) if f.startswith("markov-saida_")]
    
    # Obter as datas dos arquivos (em XX_XX_XXXX)
    datas_saida = [f.replace('markov-saida_', '').split(".")[0] for f in files_saida]
    
    # Transformar em datetime
    datas_saida = [datetime.strptime(data_saida, "%d_%m_%Y") for data_saida in datas_saida]
    
    # Procurar o arquivo de saida pela data
    if data and any(data in f for f in files_saida):
        arquivo_saida = [f for f in files_saida if data in f][0]
    else:
        if datas_saida:
            arquivo_saida = files_saida[datas_saida.index(max(datas_saida))]
        else:
            return json.dumps({"status": "1", "msg": "Nenhum gráfico Markov Saída encontrado."})
    
    # -- Carregar os dados dos arquivos selecionados --
    
    # Carregar o gráfico do dash
    try:
        with open(os.path.join(output_dir_dash, arquivo_dash), 'r') as f_dash:
            plot_dash_json = json.load(f_dash)
    except Exception as e:
        return json.dumps({"status": "1", "msg": f"Erro ao carregar o gráfico Markov Dash: {str(e)}"})

    # Carregar o gráfico da saida
    try:
        with open(os.path.join(output_dir_saida, arquivo_saida), 'r') as f_saida:
            plot_saida_json = json.load(f_saida)
    except Exception as e:
        return json.dumps({"status": "1", "msg": f"Erro ao carregar o gráfico Markov Saída: {str(e)}"})

    # -- Retornar os dois gráficos no JSON --
    return json.dumps({
        "status": "0",
        "plot": plot_dash_json,   # Gráfico do dash
        "plot2": plot_saida_json, # Gráfico da saída
    })

@b_curto_prazo.route("/_markov", methods=["POST"])
def plot_markov():
    iteracoes = request.form.get("iteracoes")
    arquivo = request.form.get("arquivo")
    Tasker().nova_task(funcao=calcular_markov, usuario_id=session['usuario']['id'], iteracoes=int(iteracoes), arquivo=arquivo)
    return json.dumps({"status": "0"})

def converter_dados(dir_path, output_path):
    # Caminho do diretório contendo os arquivos
    directory = dir_path

    # Listar todos os arquivos no diretório
    files = os.listdir(dir_path)

    # Iterar sobre cada arquivo no diretório
    for file_name in files:
        # Verificar se o arquivo é um CSV
        if file_name.endswith('.csv'):
            # Substituir espaços em branco, '-' e '+' no nome do arquivo
            new_file_name = file_name.replace('.csv', '.xlsx')
        
            # Caminho completo do arquivo CSV antigo e novo arquivo XLSX
            csv_file_path = os.path.join(dir_path, file_name)
            xlsx_file_path = os.path.join(dir_path, new_file_name)
        
            # Ler o arquivo CSV
            df = pd.read_csv(csv_file_path)
        
            # Salvar como arquivo XLSX
            df.to_excel(xlsx_file_path, index=False)

    print("Conversão concluída.")



    # Lista dos arquivos a serem processados
    files = [
        'rolloff suavizado M+0 SE -> VWAP.xlsx',
        'rolloff suavizado M+1 SE -> VWAP.xlsx',
        'rolloff suavizado M+2 SE -> VWAP.xlsx',
        'rolloff suavizado M+3 SE -> VWAP.xlsx'
    ]

    # Nova ordem das colunas
    new_order = ["Unnamed: 0", "produto", "submercado", "expiracao", "data", "volume", "VWAP", "M", "H", "h", "h_cresc"]

    # Processar cada arquivo
    for file in files:
        # Caminho completo do arquivo
        file_path = os.path.join(directory, file)
    
        # Ler o arquivo Excel
        df = pd.read_excel(file_path)
    
        # Excluir as linhas onde a célula da coluna "B" (produto) está vazia
        df = df.dropna(subset=['produto'])
    
        # Reorganizar as colunas
        df = df[new_order]
    
        # Nome do arquivo de saída
        output_file_path = os.path.join(output_path, file)
    
        # Salvar o DataFrame reorganizado em um novo arquivo Excel
        df.to_excel(output_file_path, index=False)
    
        print(f"As colunas do arquivo {file} foram reorganizadas e salvas em {output_file_path}")
    return files

def calcular_markov(iteracoes = None, arquivo = None):
    """
    Funcao para a geracao dos calculos das Cadeias Ocultas de Markov.

    Parametros:
    
    iteracoes: Numero de iteracoes (default: 150)
    arquivo: arquivo a ser processado. Se nao for informado, o sistema le os arquivos da pasta especificada.

    """

    print("Iniciando cálculo do Modelo de Markov...")
    data_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    os.system(f"/bin/echo 'Iniciando cálculo do Modelo de Markov... - {data_hora}' > /tmp/teste.txt")
    
    n_iteracoes = iteracoes if iteracoes else 150
    
    print(f"Quantidade de iterações: {n_iteracoes}")
    
    # Converter para XLSX
    # Define o caminho para a pasta contendo os arquivos CSV
    dir_path = os.path.join(script_dir, 'dados_preco', 'linear interpol', 'raw')
    output_path = os.path.join(script_dir, 'dados_markov')
    
    files = converter_dados(dir_path, output_path)
    
    if arquivo and arquivo != "TODOS":
        files = [arquivo + ".xlsx"]
        
    # inverter a lista
    files = files[::-1]
    
    # Create the directory if it doesn't exist
    output_dir = os.path.join('web', 'static', 'tasks_saida', 'markov')
    
    data_plot = datetime.now().strftime("%d_%m_%Y")
    
    
    for file in files:
        print(file)
        if "M+3" in file and os.path.isfile(output_dir+"/rolloff suavizado M+3 SE -> VWAP/markov-saida_"+data_plot+".json"):
            print("JSON para M+3 já criado para esta data...")
        elif "M+2" in file and os.path.isfile(output_dir+"/rolloff suavizado M+2 SE -> VWAP/markov-saida_"+data_plot+".json"):
            print("JSON para M+2 já criado para esta data...")
        elif "M+1" in file and os.path.isfile(output_dir+"/rolloff suavizado M+1 SE -> VWAP/markov-saida_"+data_plot+".json"):
            print("JSON para M+1 já criado para esta data...")
        elif "M+0" in file and os.path.isfile(output_dir+"/rolloff suavizado M+0 SE -> VWAP/markov-saida_"+data_plot+".json"):
            print("JSON para M+0 já criado para esta data...")
            

        if file.endswith(".xlsx"):
            print(file)
            try:
                df_iteracoes = pd.DataFrame()
                # Lista para armazenar DataFrames temporários de cada iteração
                df_iteracoes_list = []
                path = os.path.join(script_dir, 'dados_markov', file)
                nome_arquivo = file.split('.')[0]
                dados_0 = pd.read_excel(path)
                dados = pd.read_excel(path)
                print(f"[{path}] - Lido.")

                # numero de iterações definido pelo usuario
                for i in range(n_iteracoes):
                    print(f"[ITERAÇÃO {i}] - Iniciando...")
                    # os.system(f"/bin/echo '[ITERAÇÃO {i}] - Iniciando...' >> /tmp/teste.txt")
            
                    # parametros do modelo
                    NUM_TEST = 15
                    K = 50
                    NUM_ITERS = 10000
            
                    dataset0 = dados_0.copy()
                    dataset = dados.copy()
                    dataset = dataset['VWAP'].values
                    dataset = dataset.reshape(-1, 1)

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
                            print(f"Estados: {states} de {STATE_SPACE}")
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
                        # print(f"IDX: {idx} de {str(reversed(range(NUM_TEST)))}")
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

                columns = ['date','lowerp_mark', 'upperp_mark', 'medp_mark', 'high_mark', 'low_mark', 'estag_mark']
                df_dados = pd.DataFrame(columns=columns)


                # ultimo dia negociado do banco de dados
                d0 = dataset0['VWAP'].iloc[-1]

                dmais = d0*1.005
                dmenos = d0*0.995

                for i in range(len(df_iteracoes1)):
            
                    # Contagem de valores na linha i
                    count_values = len(df_iteracoes1.columns) 
                    # Valor d0
                    #d0 = count_values
            
                    # Calcular d0 * 1.005
                    #threshold = d0 * 1.005
            
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
            


                data = dataset0['data'].iloc[-1]
                data = pd.to_datetime(data)


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

                # Salvar o gráfico em um arquivo HTML
                #fig.write_html("15_dias_Teste_grafico_de_linhas.html")


                
                
                # Create the directory if it doesn't exist
                output_dir = os.path.join('web', 'static', 'tasks_saida', 'markov', nome_arquivo)
                
                os.makedirs(output_dir, exist_ok=True)

                # Save the plot to an HTML file
                #fig.write_html(os.path.join(output_dir, 'grafico_de_linhas.html'))

                # Convert the Plotly figure to JSON
                plot_json = fig.to_json()
            
                # obter a data do plot
                data_inicial = df_dados['date'].iloc[0].strftime('%d_%m_%Y')
            
                # Save the JSON file
                output_file = os.path.join(output_dir, f'markov-saida_{data_plot}.json')
                with open(output_file, 'w') as f:
                    json.dump(plot_json, f)
                    
                
                ##############################################################  Figura para a Dash principal
    


                # Certifique-se de que a coluna 'date' esteja no formato datetime
                df_dados['date'] = pd.to_datetime(df_dados['date'])
    
                # Converter valores para strings com uma casa decimal
                high_mark_text = df_dados['high_mark'].apply(lambda x: f'{x:.1f}')
                low_mark_text = df_dados['low_mark'].apply(lambda x: f'{x:.1f}')
                estag_mark_text = df_dados['estag_mark'].apply(lambda x: f'{x:.1f}')
    
                # Converter datas para strings sem horas
                date_text = df_dados['date'].dt.strftime('%d/%m/%y')
    
                # Criar o gráfico de barras
                fig = go.Figure()
    
                # Adicionar barras para 'tendencia de alta', 'tendencia de baixa' e 'estagnação' com o valor no topo
                fig.add_trace(go.Bar(x=df_dados['date'], y=df_dados['high_mark'], name='Tendência de alta', 
                                     text=high_mark_text, textposition='outside', marker_color='blue', showlegend=False, 
                                     customdata=grupoUpperMedian[grupoUpperMedian.columns[0]],  # Use only upperp_mark for this trace
                                     hovertemplate=
                                     'Data: %{x}<br>Probabilidade: %{y:.1f}%<br>' +
                                     'Tendência de alta: %{customdata:.1f}<br>'
                                     ))
                fig.add_trace(go.Bar(x=df_dados['date'], y=df_dados['low_mark'], name='Tendência de baixa', 
                                     text=low_mark_text, textposition='outside', marker_color='red', showlegend=False, 
                                     customdata=grupoLowerMedian[grupoLowerMedian.columns[0]],  # Use only upperp_mark for this trace
                                    hovertemplate=
                                    'Data: %{x}<br>Probabilidade: %{y:.1f}%<br>' +
                                    'Tendência de baixa: %{customdata:.1f}<br>'
                                    ))
                fig.add_trace(go.Bar(x=df_dados['date'], y=df_dados['estag_mark'], name='Tendência de estagnação', 
                                     text=estag_mark_text, textposition='outside', marker_color='yellow', showlegend=False, 
                                     customdata=grupoEstagMedian[grupoEstagMedian.columns[0]],  # Use only upperp_mark for this trace
                                    hovertemplate=
                                    'Data: %{x}<br>Probabilidade: %{y:.1f}%<br>' +
                                    'Tendência de estagnação: %{customdata:.1f}<br>'
                                    ))
    
                # Atualizar layout
                fig.update_layout(
                    title='Previsão Markov',
                    xaxis_title='Data',
                    yaxis_title='Probabilidade (%)',
                    barmode='group',  # agrupar barras lado a lado
                    hovermode='x',
                    xaxis=dict(
                        tickmode='array',
                        tickangle=-30,
                        tickformat='%d/%m',  # Formato para o eixo x
                    ),
                    font=dict(
                        family='Poppins, Helvetica, sans-serif',  # Especifique a família de fontes desejada
                        size=12,  # Tamanho da fonte global
                        color='black'  # Cor da fonte global
                    ),
                )
    
                # Convert the Plotly figure to JSON
                plot_json = fig.to_json()
            
                #fig.write_html(os.path.join(output_dir, f"markov-dash_{data_inicial}.html"))
                
                # current_app.df_dados[file] = {
                #     'df_dados': df_dados,
                # }
            
                # Save the JSON file
                output_file = os.path.join(output_dir, f"markov-dash_{data_plot}.json")
                with open(output_file, 'w') as f:
                    json.dump(plot_json, f)
                
            
                print(f"Gráficos {nome_arquivo} salvo com sucesso!")
            except Exception as e:
                print(f"Erro ao salvar gráficos {nome_arquivo} - {e}")


