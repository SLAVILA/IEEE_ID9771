from datetime import datetime
import os
import re
from flask import current_app, render_template, request, jsonify, Blueprint, session
import pandas as pd
from .dados_analise_tecnica import functions as f
import json
from plotly.io import to_json
from biblioteca.task import Tasker
import plotly.tools as tls
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
import numpy as np
plt.rcParams.update({'font.size': 14})

# Blueprint registration
from app.curto_prazo import b_curto_prazo

data_loaded = None
script_dir = os.path.dirname(__file__)
output_path = os.path.join(script_dir)


def filter_sort_and_unify_data(data):
    # Extract the month and year from the product name
    # month_year = " ".join(product_name.split()[-2:])
    
    # Filter data based on the month in the product name
    #filtered_data = data[data['produto'].str.contains(month_year)]
    
    # Group by the 'produto' and sort by 'data' in ascending order
    grouped_sorted_data = data.sort_values(by='data', ascending=True)
    
    # # Unify rows with the same date value
    # unified_data = data.groupby('data').agg({
    #     'volume': 'mean',
    #     'VWAP': 'mean',
    #     'M': 'max',
    #     'produto': 'first',
    #     'submercado': 'first',
    #     'expiracao': 'first',
    #     'H': 'first',
    #     'h': 'first',
    #     'h_cresc': 'first'
    # }).reset_index()
    
    grouped_sorted_data.rename(columns={'data': 'datetime', 'VWAP': 'close', 'volume':'Soma_Volumes'}, inplace=True)

    # Filter the unified data to only include rows with the selected month in the product name
    #final_data = unified_data[unified_data['produto'].str.contains(month_year, case=False)]
    #final_data = final_data.drop(columns=['missing'])
    return grouped_sorted_data

def filter_consecutive_signals(data, position_col):
    filtered_positions = data[position_col].copy()
    previous_signal = None
    for i in range(len(filtered_positions)):
        current_signal = filtered_positions.iloc[i]
        if current_signal == previous_signal:
            filtered_positions.iloc[i] = np.nan  # Remove consecutive signals
        else:
            previous_signal = current_signal
    return filtered_positions

def plot_model(data, model):
    position_col = f'position_{model}'
    return_col = f'return_{model}'

    # Calcular o retorno acumulado
    data['accumulated_return'] = data[return_col].cumsum()
    
    # Filtrar sinais consecutivos (implementação suposta de filter_consecutive_signals)
    data[f'filtered_position_{model}'] = filter_consecutive_signals(data, position_col)

    # Criar subplots: 3 linhas e 1 coluna
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=('Retorno Logarítmico Acumulado', 'Retornos Logarítmicos Diários', 'Preço de Fechamento com Sinais de Negociação'),
        row_heights=[0.2, 0.2, 0.6]  # 20% para os dois gráficos superiores, 60% para o gráfico inferior
    )

    # Subplot 1: Retorno Logarítmico Acumulado
    fig.add_trace(
        go.Scatter(
            x=data['dt'], y=data['accumulated_return'],
            mode='lines',
            name='Retorno Logarítmico Acumulado',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Adicionar retorno logarítmico no primeiro gráfico
    fig.add_trace(
        go.Scatter(
            x=data['dt'], y=data[return_col],
            mode='lines',
            name='Retorno Logarítmico',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    # Subplot 2: Retornos Logarítmicos Diários
    non_zero_returns = data[data[return_col] != 0]
    fig.add_trace(
        go.Scatter(
            x=non_zero_returns['dt'], y=non_zero_returns[return_col],
            mode='markers',
            name='Retornos Logarítmicos Diários',
            marker=dict(
                color=['blue' if x > 0 else 'red' for x in non_zero_returns[return_col]],
                size=8,
                line=dict(color='black', width=0.5)
            )
        ),
        row=2, col=1
    )

    # Adicionar linha zero em Retornos Logarítmicos Diários
    fig.add_trace(
        go.Scatter(
            x=data['dt'], y=[0] * len(data),
            mode='lines',
            name='Linha Zero',
            line=dict(color='black', width=1)
        ),
        row=2, col=1
    )

    # Subplot 3: Preço de Fechamento com Sinais de Negociação
    fig.add_trace(
        go.Scatter(
            x=data['dt'], y=data['close'],
            mode='lines',
            name='Preço de Fechamento',
            line=dict(color='black', width=1.5)
        ),
        row=3, col=1
    )

    # Sinais de Compra
    buy_signals = data[data[f'filtered_position_{model}'] == 1]
    fig.add_trace(
        go.Scatter(
            x=buy_signals['dt'], y=buy_signals['close'],
            mode='markers',
            name='Sinal de Compra',
            marker=dict(symbol='triangle-up', color='lime', size=10, line=dict(color='black', width=0.5))
        ),
        row=3, col=1
    )

    # Sinais de Venda
    sell_signals = data[data[f'filtered_position_{model}'] == 0]
    fig.add_trace(
        go.Scatter(
            x=sell_signals['dt'], y=sell_signals['close'],
            mode='markers',
            name='Sinal de Venda',
            marker=dict(symbol='triangle-down', color='red', size=10, line=dict(color='black', width=0.5))
        ),
        row=3, col=1
    )

    # Configurar layout e eixo X para todos os subplots
    fig.update_layout(
        title_text=f'Sinais de negociação usando modelo {model}',
        height=800,
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.7,
            xanchor='left',
            x=1.05,
        )
    )

    # Atualizar eixos X
    fig.update_xaxes(
        tickformat='%Y-%m',
        ticks='outside',
        tickangle=45,
        tickfont=dict(size=10)
    )

    # Configurar eixos Y para todos os subplots, deixando os rótulos à esquerda
    fig.update_yaxes(
        tickfont=dict(size=10),
        ticks='outside',
        ticklabelposition="outside",
        side='left'  # Garante que os rótulos do eixo Y fiquem à esquerda
    )
    
    return fig.to_json()


def load_data(m):
    data = data_loaded
    if not data_loaded:
        pd.set_option('display.max_columns', None) # mostra todas as colunas quando mostrando o dataframe
        
        # atribui os dados do ativo à variável de dados
        data=pd.read_csv(os.path.join(script_dir, 'dados_preco', 'linear interpol', 'VWAP pentada', f'rolloff suavizado {m} SE -> VWAP.csv'), index_col=0, parse_dates=True)

        data['data'] = pd.to_datetime(data.data, format='%Y-%m-%d')
        
        print(data)
        
        # Example usage of the function
        dados_agrupados = filter_sort_and_unify_data(data)
        os.makedirs(os.path.join(output_path, 'dados_analise_tecnica', m, 'Data'), exist_ok=True)
        dados_agrupados.to_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'Data', 'dados_agrupados.csv'))
        data = pd.read_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'Data', 'dados_agrupados.csv'), index_col=0)
        print(data)
        data.sort_values(by = 'datetime', inplace = True)
    return data

def execute_backtest():
    for m in ['M+0', 'M+1', 'M+2', 'M+3']:
        dados_agrupados = load_data(m)
        
        print(dados_agrupados)
        # Realiza o backtest da estratégia SMA
        SMA_parameters = f.sma_backtest(dados_agrupados)
        print(SMA_parameters)
        # Salva os dados das iterações da estratégia SMA no arquivo SMA_parameters.csv
        SMA_parameters = SMA_parameters[SMA_parameters['fast_period'] < SMA_parameters['slow_period']]
        SMA_parameters.to_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'Data', 'SMA_parameters.csv'))
        # Realiza o backtest da estratégia EMA 
        EMA_parameters = f.ema_backtest(dados_agrupados)
        print(EMA_parameters)
        # Salva os dados das iterações da estratégia EMA no arquivo EMA_parameters.csv
        EMA_parameters = EMA_parameters[EMA_parameters['fast_period'] < EMA_parameters['slow_period']]
        EMA_parameters.to_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'Data', 'EMA_parameters.csv'))
        # Realiza o backtest da estratégia Bollinger Bands 
        # ATENÇÂO - O PARÂMETRO 'devfactor' está setado para variar como int(), a função np.arange, np.linspace (para float()) não são usadas pois a função backtest falha
        BBANDS_parameters = f.bbands_backtest(dados_agrupados)
        print(BBANDS_parameters)
        # Salva os dados das iterações da estratégia BBANDS no arquivo BBANDS_parameters.csv
        BBANDS_parameters.to_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'Data', 'BBANDS_parameters.csv'))
        
        # Atribui as variáveis dos dataframes salvos (já iterados) na variável STRATEGY_parameters 
        # Permite executar essa parte sem realizar os backtests todos novamente
        SMA_parameters = pd.read_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'Data', 'SMA_parameters.csv'))
        EMA_parameters = pd.read_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'Data', 'EMA_parameters.csv'))
        BBANDS_parameters = pd.read_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'Data', 'BBANDS_parameters.csv'))
        
        # Realiza a estratégia SMA com os paramêtros ótimos
        f.sma_strategy(dados_agrupados, SMA_parameters.iloc[0,20], SMA_parameters.iloc[0,21], m)
        
        # Realiza a estratégia EMA com os paramêtros ótimos
        f.ema_strategy(dados_agrupados, EMA_parameters.iloc[0,20], EMA_parameters.iloc[0,21], m)
        
        # Realiza a estratégia BBANDS com os paramêtros ótimos
        f.bbands_strategy(dados_agrupados, BBANDS_parameters.iloc[0,20], float(BBANDS_parameters.iloc[0,21]), m)
        
        execute_machine_learning(m)
    
    return "Backtest feito"

def execute_machine_learning(m):
    # Unifica os multiplos datasets de cada estratégia em 4 dataframes e filtra dados desnecessários 
    f.create_dataframes(m)
    
    # dataframes unificados
    dataframe_VWAP = pd.read_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'ML_data', 'dataframe_VWAP.csv'))
    dataframe_indicators = pd.read_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'ML_data', 'dataframe_indicators.csv'))
    dataframe_orders = pd.read_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'ML_data', 'dataframe_orders.csv'))
    dataframe_periodic = pd.read_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'ML_data', 'dataframe_periodic.csv'))
    
    
    # Auxiliary functions
    def BBANDS_signal(x):
        if x.iloc[4] > x.iloc[7]:  # Adjust the positions accordingly
            return -1
        if x.iloc[4] < x.iloc[8]:  # Adjust the positions accordingly
            return 1
        return 0

    def BBANDS_signal_2(row):
        mid = row['BBands_mid']
        top = row['BBands_top']
        bot = row['BBands_bot']
        # Calculate the distance of mid from top and bot
        distance_top = np.abs(mid - top)
        distance_bot = np.abs(mid - bot)
        # Normalize the distance
        indicator = (distance_bot - distance_top) / (distance_top + distance_bot)
        return indicator

    def ideal_indicator(x):
        if x['return_shifted'] > 0:
            return 1
        if x['return_shifted'] < 0:
            return -1
        return 0


    # Step 1: Define the new names in the order you want to assign them
    new_names = {
        1: 'SMA_fast',       # First SMA column, e.g., 'SMA_6'
        2: 'SMA_slow',       # Second SMA column, e.g., 'SMA_30'
        3: 'CrossOver_x',    # First CrossOver column
        4: 'EMA_fast',       # First EMA column, e.g., 'EMA_5'
        5: 'EMA_slow',       # Second EMA column, e.g., 'EMA_42'
        6: 'CrossOver_y',    # Second CrossOver column
        7: 'BBands_mid',     # First Bollinger Bands column, e.g., 'BBands_mid_3_1.0'
        8: 'BBands_top',     # Second Bollinger Bands column, e.g., 'BBands_top_3_1.0'
        9: 'BBands_bot'      # Third Bollinger Bands column, e.g., 'BBands_bot_3_1.0'
    }
    
    # Step 2: Get the current column names
    current_columns = list(dataframe_indicators.columns)

    # Step 3: Create a mapping of current column indices to new names
    column_mapping = {current_columns[i]: new_names[i] for i in range(1, len(new_names)+1)}

    # Step 4: Rename the columns
    dataframe_indicators = dataframe_indicators.rename(columns=column_mapping)

    # Exibir o DataFrame com as colunas renomeadas dinamicamente
    print(dataframe_indicators)

    # Only keep relevant columns in dataframe_VWAP
    dataframe_VWAP = dataframe_VWAP[['dt', 'Soma_Volumes', 'close']]
            
    # Calculate returns
    dataframe_VWAP['return'] = np.log(dataframe_VWAP['close']).diff()
    
    print(dataframe_VWAP)

    # Merge dataframes on 'dt'
    merged = pd.merge(dataframe_indicators, dataframe_VWAP, on='dt')
    
    merged['return_shifted'] = merged['return'].shift(-1)
    
    print(f"merged1: {merged}")
    # merged = merged.dropna(axis=0)
    print(f"merged: {merged}")

        # Initialize signals
    merged['SMA_signal'] = 0
    merged['EMA_signal'] = 0
    merged['BBANDS_signal'] = 0

        # Calculate signals
    merged['SMA_signal'] = merged.apply(lambda row: 1 if row['SMA_fast'] > row['SMA_slow'] else -1, axis=1)
    merged['EMA_signal'] = merged.apply(lambda row: 1 if row['EMA_fast'] > row['EMA_slow'] else -1, axis=1)
    merged['BBANDS_signal'] = merged.apply(BBANDS_signal, axis=1)
    merged['BBANDS_signal_2'] = merged.apply(BBANDS_signal_2, axis=1)

        # Apply log transformation
    columns_to_log = ['close', 'Soma_Volumes', 'SMA_fast', 'SMA_slow', 'EMA_fast', 'EMA_slow', 'BBands_mid', 'BBands_top', 'BBands_bot', 'BBANDS_signal', 'BBANDS_signal_2', 'SMA_signal', 'EMA_signal']
    for col in columns_to_log:
        merged[col] = np.log(merged[col])
            
        # Calculate ideal signal
    merged['ideal_signal_(d+1)'] = merged.apply(ideal_indicator, axis=1)
            
        # Keep final columns
    final_columns = ['dt', 'SMA_fast', 'SMA_slow', 'CrossOver_x', 'EMA_fast', 'EMA_slow', 'CrossOver_y', 'BBands_mid', 'BBands_top', 'BBands_bot', 'Soma_Volumes', 'close',  'BBANDS_signal', 'BBANDS_signal_2', 'SMA_signal', 'EMA_signal', 'return_shifted', 'return', 'ideal_signal_(d+1)']
    final_merged = merged[final_columns]

    final_merged.fillna(0, inplace=True)

        # Save to CSV
    final_merged.to_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'ML_data', 'round_2_dataframe.csv'), index=False)

    # data, sinais (0, 1) vs (sell, do nothing, buy), retornos logaritmicos, e volume em log
    #f.round_2_dataframe(dataframe_VWAP, dataframe_indicators)
    round_2 = pd.read_csv(os.path.join(output_path, 'dados_analise_tecnica', m, 'ML_data',  'round_2_dataframe.csv'), index_col='dt')
    
    print(f"round_2: {round_2}")

    # assuming that the dataset size is 849 rows, the dataset is divided into 449 days to train and 400 days to test
    Ntest = int(0.65*len(round_2))
    train = round_2.iloc[1:-Ntest]
    test = round_2.iloc[-Ntest:-1]

    x_columns = ['SMA_fast', 'SMA_slow', 
                 'EMA_fast', 'EMA_slow', 
                 'BBands_mid', 'BBands_top', 'BBands_bot',
                 'return', 'close']

    Xtrain = train[x_columns]
    Ytrain = train['ideal_signal_(d+1)']
    Xtest = test[x_columns]
    Ytest = test['ideal_signal_(d+1)']

    f.linear_regression(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, m)
    f.ridge_regression(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, m)
    f.logistic_regression(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, m)
    f.lgbm_classifier(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, sample_weight = abs(round_2[1:-Ntest]['return_shifted']), m=m)
    f.xgboost_classifier(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, sample_weight = abs(round_2[1:-Ntest]['return_shifted']), m=m)
    f.random_forest_classifier(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, m)
    f.svc_classifier(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, sample_weight = abs(round_2[1:-Ntest]['return_shifted']), m=m)

    f.unify_results(m)
    f.divide_results_train_test(m)

def get_regression_plot(name, file_path):
    r2df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return f.plot_regressions(r2df, name)

def get_classifier_plot(classifier_type, file_path):
    r2df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return f.plot_classifier(r2df, classifier_type)

@b_curto_prazo.route("/_analise_dash", methods=["POST"])
def _analise_dash():
    # Obtenha a data do POST request
    data = request.form.get('data')
    m = request.form.get('m')

    # Load the data from the provided files
    script_dir = os.path.dirname(__file__)
    scores_df = pd.read_csv(os.path.join(script_dir, 'dados_analise_tecnica', m, 'ML_data', 'scores_dataframe.csv'))
    round_2_df = pd.read_csv(os.path.join(script_dir, 'dados_analise_tecnica', m, 'ML_data', 'round_2_dataframe.csv'))
    print(round_2_df)
    # Prepare data for the plot
    round_2_df['dt'] = pd.to_datetime(round_2_df['dt'])

    # Filtrar os dados até a data selecionada
    if data:
        data_selecionada = datetime.strptime(data, '%d/%m/%Y')
        try:
            round_2_df = round_2_df[round_2_df['dt'] <= data_selecionada]
        except:
            return json.dumps({'status': '1', 'msg': 'Data inválida'})
    else:
        data_selecionada = round_2_df['dt'].iloc[-1]
        round_2_df = round_2_df.drop(round_2_df.index[-1])

    # Adjust model position mapping
    model_position_mapping = {
        'linear_regression': 'position_linear_regression',
        'ridge_regression': 'position_ridge_regression',
        'xgboost_classifier': 'position_xgboost_classifier',
        'lgbm_classifier': 'position_lgbm_classifier',
        'logistic_regression': 'position_logistic_regression',
        'random_forest': 'position_random_forest',
        'svc': 'position_svc'
    }

    def calculate_probabilities(scores_df, round_2_df):
        last_positions = round_2_df.iloc[-1]
        accuracies = scores_df.set_index('model')['accuracy_test']
        buy_accuracies = []
        sell_accuracies = []

        for model, position_column in model_position_mapping.items():
            if position_column in last_positions:
                position = last_positions[position_column]
                if pd.notna(position):
                    if position in [1, 'True', '1', True]:
                        buy_accuracies.append(accuracies[model])
                    elif position in [0, 'False', '0', False]:
                        sell_accuracies.append(accuracies[model])
    
        def calculate_probability(accuracies):
            if not accuracies:
                return 0
            return sum(accuracies) / len(accuracies)
    
        buy_probability = calculate_probability(buy_accuracies)
        sell_probability = calculate_probability(sell_accuracies)
        total_probability = buy_probability + sell_probability

        if total_probability > 0:
            buy_probability /= total_probability
            sell_probability /= total_probability

        return buy_probability, sell_probability

    def voting_signal(scores_df, round_2_df):
        buy_probability, sell_probability = calculate_probabilities(scores_df, round_2_df)
        buy_votes = sum(1 for model, position_column in model_position_mapping.items() 
                        if pd.notna(round_2_df.iloc[-1][position_column]) and 
                        (round_2_df.iloc[-1][position_column] in [1, 'True', '1', True]))
        sell_votes = sum(1 for model, position_column in model_position_mapping.items() 
                         if pd.notna(round_2_df.iloc[-1][position_column]) and 
                         (round_2_df.iloc[-1][position_column] in [0, 'False', '0', False]))

        total_votes = buy_votes + sell_votes
        if total_votes > 0 and (buy_votes != sell_votes):
            buy_probability *= total_votes / (total_votes + 1)
            sell_probability *= total_votes / (total_votes + 1)
            hold_probability = 1 / (total_votes + 1)
        else:
            hold_probability = 1
    
        return {
            "most_voted_signal": 'HOLD' if hold_probability == 1 else ('BUY' if buy_votes > sell_votes else 'SELL'),
            "most_voted_probability": max(buy_probability, sell_probability, hold_probability),
            "buy_probability": buy_probability,
            "sell_probability": sell_probability,
            "hold_probability": hold_probability,
            "buy_votes": buy_votes,
            "sell_votes": sell_votes
        }

    

    try:
        # Generate the result using the voting_signal function
        result = voting_signal(scores_df, round_2_df)

    
    
        dates = round_2_df['dt']
        returns = round_2_df['close']
        
        # Filter the last 30 days
        last_30_days = round_2_df.tail(30)
        dates_last_30 = last_30_days['dt']
        returns_last_30 = last_30_days['close']

        # Create the plot with subplot for bars and line chart
        fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], specs=[[{"type": "xy"}, {"type": "bar"}]])

        # Add the line of returns for the last 30 days
        fig.add_trace(go.Scatter(x=dates_last_30, y=returns_last_30, mode='lines', name='Close price'), row=1, col=1)

        # Add a triangle for the most voted signal
        signal_color = 'blue' if result['most_voted_signal'] == 'BUY' else ('red' if result['most_voted_signal'] == 'SELL' else 'gray')
        triangle_marker = dict(symbol='triangle-up' if signal_color == 'blue' else ('triangle-down' if signal_color == 'red' else 'triangle-left'), color=signal_color, size=12)
        fig.add_trace(go.Scatter(x=[dates_last_30.iloc[-1]], y=[returns_last_30.iloc[-1]], mode='markers', 
                                 marker=triangle_marker, name=f"Most Voted Signal: {result['most_voted_signal']}"), row=1, col=1)

        # Add the probability bars for BUY, SELL, and HOLD
        fig.add_trace(go.Bar(
            x=['COMPRAR', 'VENDER', 'SEGURAR'],
            y=[result['buy_probability']*100, result['sell_probability']*100, result['hold_probability']*100],
            text=[f"{result['buy_probability']*100:.2f}%", f"{result['sell_probability']*100:.2f}%", f"{result['hold_probability']*100:.2f}%"],
            marker_color=['blue', 'red', 'yellow'],
            name='Probabilidade'
        ), row=1, col=2)

        # Update the layout of the plot
        fig.update_layout(
            title="Previsão Análise Técnica",
            xaxis_title="Data (Últimos 30 Dias)",
            yaxis_title="Preço de Fechamento",
            showlegend=False,
            xaxis2=dict(
                showticklabels=False,  # Hides the x-axis labels on the second subplot
                showgrid=False,  # Hides the grid lines on the x-axis
                zeroline=False   # Hides the zero line on the x-axis
            ),
            font=dict(
                family='Poppins, Helvetica, sans-serif',  # Especifique a família de fontes desejada
                size=12,  # Tamanho da fonte global
                color='black'  # Cor da fonte global
            ),
        )
    
        plot_json = to_json(fig)
    except:
        return json.dumps({'status': '1', 'msg': 'Erro ao gerar o gráfico Análise Técnica. Data inválida.'})
    
    return json.dumps({'status': '0', 'plot': plot_json, 'data': data_selecionada.strftime('%d/%m/%Y')})


@b_curto_prazo.route("/_backtest", methods=["POST"])
def iniciar_backtest():

    Tasker().nova_task(funcao=execute_backtest, usuario_id=session['usuario']['id'])
    return json.dumps({"status": "0"})

@b_curto_prazo.route("/_machine", methods=["POST"])
def iniciar_machine_learning():    
    
    Tasker().nova_task(funcao=execute_machine_learning, usuario_id=session['usuario']['id'])
    return json.dumps({"status": "0"})

@b_curto_prazo.route("/_plot_regression", methods=["POST"])
def plot_regression():
    name = request.form['name']
    m = request.form['m']
    file_path = os.path.join(script_dir, 'dados_analise_tecnica', m, 'ML_data', 'round_2_dataframe.csv')
    
    
    # Load the CSV file to examine its contents
    data = pd.read_csv(file_path)

    # Load data

    plot_json = plot_model(data, name)
    
    
    
    return json.dumps({'status': '0', 'plot': plot_json})

@b_curto_prazo.route("/_plot_classifier", methods=["POST"])
def plot_classifier():
    name = request.form['name']
    m = request.form['m']
    file_path = os.path.join(script_dir, 'dados_analise_tecnica', m, 'ML_data', 'round_2_dataframe.csv')
    
    
    # Load the CSV file to examine its contents
    data = pd.read_csv(file_path)

    # Load data

    plot_json = plot_model(data, name)
    
    
    return json.dumps({'status': '0', 'plot': plot_json})

