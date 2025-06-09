import os.path
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import plotly.io as pio
from functools import reduce
from fastquant import backtest
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error, accuracy_score, precision_score, f1_score, roc_auc_score

script_dir = os.path.dirname(__file__)

# desliga a geração de gráficos interativa (problema do jupyter que crasha o kernel ao tentar plotar)
plt.ioff()

# função de download dos dados
def download_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start_date, end_date)
    stock_data.to_csv(os.path.join(script_dir, 'Data', f'{ticker}.csv'))
    return stock_data

# prepara o dataset
def preco_bbce(preco):
    # outlier encontrado, acreditamos que o nome deste produto esteja faltando o "MEN"
    preco = preco.replace({"produto": {"SE CON FEV/18": "SE CON MEN FEV/18",
                                       'SE CON MEN FEV/23 DEZ/23': 'SE CON MEN DEZ/23'}})
    preco['datetime'] = preco['data_hora'].dt.date
    preco.drop('data_hora', axis = 1, inplace = True)
    return preco.reindex(columns=['id_bbce', 'produto', 'datetime', 'volume', 'preco', 'tipo'])

# separa os produtos mensais
def preco_bbce_men_preco_fixo(preco):
    # pega apenas os produtos de Preço Fixo
    preco = preco[preco.tipo == 1].drop(["tipo"], axis=1)
    
    # padroniza nomes dos produtos e filtra apenas produtos mensais de fonte não incentivada
    validos = [] # produtos mensais
    rename = {}  # produtos que tem que remover " - Preço Fixo" do final
    for prod in preco.produto.unique():
        partes = prod.strip().replace("- ", "").replace("_", " ").replace("  ", " ").split(" ")
        if "CON" != partes[1]:
            # não é de fonte não incentivada
            continue
        if "MEN" != partes[2]:
            # não é mensal
            continue

        validos.append(prod)
        if prod.endswith(" - Preço Fixo"):
            rename[prod] = prod[:-len(" - Preço Fixo")]

    # seleciona produtos mensais de fonte não incentivada
    linhas_filtradas = [(prod in validos) for prod in preco["produto"]]
    preco = preco[linhas_filtradas]
        
    # renomeia os produtos
    preco = preco.replace({"produto": rename})
    
    # filtra datas que são inferiores à realização do produto
    next_month = {
        "jan": 2, "fev":  3, "mar":  4, "abr":  5,
        "mai": 6, "jun":  7, "jul":  8, "ago":  9,
        "set": 10, "out": 11, "nov": 12, "dez":  1
    }
    
    realizacoes = {}
    for prod in preco.produto.unique():
        # example: JUL/18
        month, year = prod.split(" ")[-1].split("/")
        y = 2000+int(year)
        m = next_month[month.lower()]
        if m == 1:
            # era dezembro, logo é no próximo ano
            y += 1
        realizacoes[prod] = dt.date(y, m, 1)
        
    linhas_filtradas = [row["datetime"] < realizacoes[row["produto"]] for _, row in preco.iterrows()]
    preco = preco[linhas_filtradas]
    
    return preco

# separa os produtos anuais
def preco_bbce_anu_preco_fixo(preco):
    # pega apenas os produtos de Preço Fixo
    preco = preco[preco.tipo == 1].drop(["tipo"], axis=1)
    
    # padroniza nomes dos produtos e filtra apenas produtos mensais de fonte não incentivada
    validos = [] # produtos mensais
    rename = {}  # produtos que tem que remover " - Preço Fixo" do final
    for prod in preco.produto.unique():
        partes = prod.strip().replace("- ", "").replace("_", " ").replace("  ", " ").split(" ")
        if "CON" != partes[1]:
            # não é de fonte não incentivada
            continue
        if "ANU" != partes[2]:
            # não é mensal
            continue

        validos.append(prod)
        if prod.endswith(" - Preço Fixo"):
            rename[prod] = prod[:-len(" - Preço Fixo")]

    # seleciona produtos mensais de fonte não incentivada
    linhas_filtradas = [(prod in validos) for prod in preco["produto"]]
    preco = preco[linhas_filtradas]
        
    # renomeia os produtos
    preco = preco.replace({"produto": rename})
    
    # filtra datas que são inferiores à realização do produto
    next_month = {
        "jan": 2, "fev":  3, "mar":  4, "abr":  5,
        "mai": 6, "jun":  7, "jul":  8, "ago":  9,
        "set": 10, "out": 11, "nov": 12, "dez":  1
    }
    
    realizacoes = {}
    for prod in preco.produto.unique():
        # example: JUL/18
        month, year = prod.split(" ")[-1].split("/")
        y = 2000+int(year)
        m = next_month[month.lower()]
        if m == 1:
            # era dezembro, logo é no próximo ano
            y += 1
        realizacoes[prod] = dt.date(y, m, 1)
        
    linhas_filtradas = [row["datetime"] < realizacoes[row["produto"]] for _, row in preco.iterrows()]
    preco = preco[linhas_filtradas]
    
    return preco

# retira os produtos de Maturidade 0 M0 do dataset para evitar distorções de preços    
def filtro_m0(data):
    # isso é uma preparação para o dataset específico 
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.loc[~(data['datetime'].dt.year < 2015),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/15 DEZ/15') & (data['datetime'].dt.year == 2015)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/16 DEZ/16') & (data['datetime'].dt.year == 2016)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/17 DEZ/17') & (data['datetime'].dt.year == 2017)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/18 DEZ/18') & (data['datetime'].dt.year == 2018)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/19 DEZ/19') & (data['datetime'].dt.year == 2019)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/20 DEZ/20') & (data['datetime'].dt.year == 2020)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/21 DEZ/21') & (data['datetime'].dt.year == 2021)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/22 DEZ/22') & (data['datetime'].dt.year == 2022)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/23 DEZ/23') & (data['datetime'].dt.year == 2023)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/24 DEZ/24') & (data['datetime'].dt.year == 2024)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/25 DEZ/25') & (data['datetime'].dt.year == 2025)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/26 DEZ/26') & (data['datetime'].dt.year == 2026)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/27 DEZ/27') & (data['datetime'].dt.year == 2027)),:]
    data = data.loc[~((data['produto'] == 'SE CON ANU JAN/28 DEZ/28') & (data['datetime'].dt.year == 2028)),:]
    return data

# função que itera os parâmetros para encontrar a estratégia ótima de SMA
def sma_backtest(data):
    SMA = backtest("smac", data, fast_period=range(1, 35, 2), slow_period=range(30, 90, 2), plot=False, return_plot=False)
    return SMA

# função que itera os parâmetros para encontrar a estratégia ótima de EMA
def ema_backtest(data):
    EMA = backtest("emac", data, fast_period=range(1, 35, 2), slow_period=range(30, 90, 2), plot=False, return_plot=False)
    return EMA

# função que itera os parâmetros para encontrar a estratégia ótima de BBANDS
def bbands_backtest(data):
    BBANDS = backtest("bbands", data, period=range(1, 8, 1), devfactor=range(1, 8, 1), plot=False, return_plot=False)
    return BBANDS

# função para rodar a estratégia ótima de trading com base em SMA
def sma_strategy(data, sma_fast_period, sma_slow_period, m):
    sma_optimal, sma_plot = backtest('smac', data, fast_period=sma_fast_period, slow_period=sma_slow_period, return_plot=True)
    sma_plot.savefig(os.path.join(script_dir, m, 'Data', 'sma_optimal_plot.png'))
    sma_save_hist(data, sma_fast_period, sma_slow_period, m)
    return sma_optimal
# função para salvar os datasets de trades
def sma_save_hist(data, sma_fast_period, sma_slow_period, m):
    sma_optimal, sma_hist = backtest('smac', data, fast_period=sma_fast_period, slow_period=sma_slow_period, return_history=True)
    sma_optimal.to_csv(os.path.join(script_dir, m, 'Data', 'sma_optimal.csv'))
    sma_hist['orders'].to_csv(os.path.join(script_dir, m, 'Data', 'sma_orders.csv'),  index=False)
    sma_hist['periodic'].to_csv(os.path.join(script_dir, m, 'Data', 'sma_periodic.csv'),  index=False)
    sma_hist['indicators'].to_csv(os.path.join(script_dir, m, 'Data', 'sma_indicators.csv'),  index=False)
    return 

# função para rodar a estratégia ótima de trading com base em EMA
def ema_strategy(data, ema_fast_period, ema_slow_period, m):
    ema_optimal, ema_plot = backtest('emac', data, fast_period=ema_fast_period, slow_period=ema_slow_period, return_plot=True)
    ema_plot.savefig(os.path.join(script_dir, m, 'Data', 'ema_optimal_plot.png'))
    ema_save_hist(data, ema_fast_period, ema_slow_period, m)
    return ema_optimal
# função para salvar os datasets de trades
def ema_save_hist(data, sma_fast_period, sma_slow_period, m):
    ema_optimal, ema_hist = backtest('emac', data, fast_period=sma_fast_period, slow_period=sma_slow_period, return_history=True)
    ema_optimal.to_csv(os.path.join(script_dir, m, 'Data', 'ema_optimal.csv'))
    ema_hist['orders'].to_csv(os.path.join(script_dir, m, 'Data', 'ema_orders.csv'),  index=False)
    ema_hist['periodic'].to_csv(os.path.join(script_dir, m, 'Data', 'ema_periodic.csv'),  index=False)
    ema_hist['indicators'].to_csv(os.path.join(script_dir, m, 'Data', 'ema_indicators.csv'),  index=False)
    return 

# função para rodar a estratégia ótima de trading com base em BBANDS
def bbands_strategy(data, bbands_period, bbands_devfactor, m):
    bbands_optimal, bbands_plot = backtest('bbands', data, period=bbands_period, devfactor=bbands_devfactor, return_plot=True)
    bbands_plot.savefig(os.path.join(script_dir, m, 'Data','bbands_optimal_plot.png'))
    bbands_save_hist(data, bbands_period, bbands_devfactor, m)
    return bbands_optimal
# função para salvar os datasets de trades
def bbands_save_hist(data, bbands_period, bbands_devfactor, m):
    bbands_optimal, bbands_hist = backtest('bbands', data, period=bbands_period, devfactor=bbands_devfactor, return_history=True)
    bbands_optimal.to_csv(os.path.join(script_dir, m, 'Data', 'bbands_optimal.csv'))
    bbands_hist['orders'].to_csv(os.path.join(script_dir, m, 'Data', 'bbands_orders.csv'),  index=False)
    bbands_hist['periodic'].to_csv(os.path.join(script_dir, m, 'Data', 'bbands_periodic.csv'),  index=False)
    bbands_hist['indicators'].to_csv(os.path.join(script_dir, m, 'Data', 'bbands_indicators.csv'),  index=False)
    return 

# creation of dataframes to ML input
def create_dataframes(m):
    os.makedirs(os.path.join(script_dir, m, 'ML_data'), exist_ok=True)
    dataset_optimal_SMA = pd.read_csv(os.path.join(script_dir, m, 'Data', 'sma_optimal.csv'))
    dataset_optimal_EMA = pd.read_csv(os.path.join(script_dir, m, 'Data', 'ema_optimal.csv'))
    dataset_optimal_BBANDS = pd.read_csv(os.path.join(script_dir, m, 'Data', 'bbands_optimal.csv'))

    dataset_orders_SMA = pd.read_csv(os.path.join(script_dir, m, 'Data', 'sma_orders.csv'))
    dataset_orders_EMA = pd.read_csv(os.path.join(script_dir, m, 'Data', 'ema_orders.csv'))
    dataset_orders_BBANDS = pd.read_csv(os.path.join(script_dir, m, 'Data', 'bbands_orders.csv'))

    dataset_indicators_SMA = pd.read_csv(os.path.join(script_dir, m, 'Data', 'sma_indicators.csv'))
    dataset_indicators_EMA = pd.read_csv(os.path.join(script_dir, m, 'Data', 'ema_indicators.csv'))
    dataset_indicators_BBANDS = pd.read_csv(os.path.join(script_dir, m, 'Data', 'bbands_indicators.csv'))

    dataset_periodic_SMA = pd.read_csv(os.path.join(script_dir, m, 'Data', 'sma_periodic.csv'))
    dataset_periodic_EMA = pd.read_csv(os.path.join(script_dir, m, 'Data', 'ema_periodic.csv'))
    dataset_periodic_BBANDS = pd.read_csv(os.path.join(script_dir, m, 'Data', 'bbands_periodic.csv'))
    
    dataset_indicators_SMA = dataset_indicators_SMA.drop(columns = ['strat_id', 'strat_name', ])
    dataset_indicators_EMA = dataset_indicators_EMA.drop(columns = ['strat_id', 'strat_name', ])
    dataset_indicators_BBANDS = dataset_indicators_BBANDS.drop(columns = ['strat_id', 'strat_name', ])

    dataset_periodic_SMA = dataset_periodic_SMA.drop(columns = ['strat_id', 'strat_name', ])
    dataset_periodic_EMA = dataset_periodic_EMA.drop(columns = ['strat_id', 'strat_name', ])
    dataset_periodic_BBANDS = dataset_periodic_BBANDS.drop(columns = ['strat_id', 'strat_name', ])

    dataset_orders_SMA = dataset_orders_SMA.drop(columns = ['strat_id', 'strat_name', 'commission', 'size', 'order_value', 'pnl'])
    dataset_orders_SMA['dt_sma'] = dataset_orders_SMA['dt']
    dataset_orders_SMA.rename(columns={'type':'type_sma', 'price':'price_sma', 'portfolio_value':'portfolio_value_sma'},inplace=True)
    dataset_orders_EMA = dataset_orders_EMA.drop(columns = ['strat_id', 'strat_name', 'commission', 'size', 'order_value', 'pnl'])
    dataset_orders_EMA['dt_ema'] = dataset_orders_EMA['dt']
    dataset_orders_EMA.rename(columns={'type':'type_ema', 'price':'price_ema', 'portfolio_value':'portfolio_value_ema'},inplace=True)    
    dataset_orders_BBANDS = dataset_orders_BBANDS.drop(columns = ['strat_id', 'strat_name','commission', 'size', 'order_value', 'pnl'])
    dataset_orders_BBANDS['dt_bbands'] = dataset_orders_BBANDS['dt']
    dataset_orders_BBANDS.rename(columns={'type':'type_bbands', 'price':'price_bbands', 'portfolio_value':'portfolio_value_bbands'},inplace=True)
   
    concat_optimal = [dataset_optimal_SMA, dataset_optimal_EMA, dataset_optimal_BBANDS]
    concat_periodic = [dataset_periodic_SMA, dataset_periodic_EMA, dataset_periodic_BBANDS]
    concat_indicators = [dataset_indicators_SMA, dataset_indicators_EMA, dataset_indicators_BBANDS]
    concat_orders = [dataset_orders_SMA, dataset_orders_EMA, dataset_orders_BBANDS]

    # este dataframe é meramente estatístico para as estratégias de AT
    dataframe_optimal = pd.concat(concat_optimal)
    dataframe_optimal = dataframe_optimal.drop(columns=['Unnamed: 0', 'strat_id', 'buy_prop', 'sell_prop', 'fractional', 'slippage', 'single_position', 'commission', 'stop_loss', 'stop_trail', 'take_profit', 'execution_type', 'channel', 'symbol', 'allow_short', 'short_max', 'add_cash_amount', 'add_cash_freq', 'invest_div', 'len', 'max' ], axis = 1)
    dataframe_optimal.to_csv(os.path.join(script_dir, m, 'ML_data', 'dataframe_optimal.csv'), index=False)
    # a variável chama-se concat porém estou mergindo os dfs
    dataframe_indicators = reduce(lambda  left,right: pd.merge(left,right,on=['dt'], how='outer'), concat_indicators)
    dataframe_indicators.to_csv(os.path.join(script_dir, m, 'ML_data', 'dataframe_indicators.csv'), index=False)
    # a variável chama-se concat porém estou mergindo os dfs
    dataframe_periodic = reduce(lambda  left,right: pd.merge(left,right,on=['dt'], how='outer'), concat_periodic)
    dataframe_periodic.to_csv(os.path.join(script_dir, m, 'ML_data', 'dataframe_periodic.csv'), index=False)
    # a variável chama-se concat porém estou mergindo os dfs
    dataframe_orders = reduce(lambda  left,right: pd.merge(left,right,on=['dt'], how='outer'), concat_orders)
    dataframe_orders.to_csv(os.path.join(script_dir, m, 'ML_data', 'dataframe_orders.csv'), index=False)
    # adicionando a série de tempo dos preços VWAP para inclusão no ML
    dataset_VWAP = pd.read_csv(os.path.join(script_dir, m, 'Data', 'dados_agrupados.csv'))
    dataset_VWAP = dataset_VWAP.drop(columns=['Unnamed: 0'])
    dataset_VWAP = dataset_VWAP.rename(columns={'datetime':'dt'})
    dataset_VWAP.to_csv(os.path.join(script_dir, m, 'ML_data', 'dataframe_VWAP.csv'), index=False)
    return 

def bbands_distance_indicator(row):
    mid = row['BBands_mid_5_1.0']
    top = row['BBands_top_5_1.0']
    bot = row['BBands_bot_5_1.0']
    # Calculate the distance of mid from top and bot
    distance_top = np.abs(mid - top)
    distance_bot = np.abs(mid - bot)
    # Normalize the distance
    indicator = (distance_bot - distance_top) / (distance_top + distance_bot)
    return indicator

def market_order_signal(value):
    if value > 0.005:  # Adjust the threshold as needed
        return 1
    elif value < -0.005:  # Adjust the threshold as needed
        return -1
    else:
        return 0

def round_1_dataframe(dataframe_VWAP, dataframe_indicators):
    # Ensure all 'dt' columns are in datetime format
    dataframe_VWAP['dt'] = pd.to_datetime(dataframe_VWAP['dt'])
    dataframe_indicators['dt'] = pd.to_datetime(dataframe_indicators['dt'])
    
    # Merge VWAP and indicators dataframes on 'dt' column
    merged_df = dataframe_VWAP.merge(dataframe_indicators, on='dt', how='outer')
    
    # Ensure 'dt' column is in the first place
    cols = list(merged_df.columns)
    cols.insert(0, cols.pop(cols.index('dt')))
    merged_df = merged_df[cols]
    
    # Apply a 1-day delay to the columns with "crossover" in their names
    crossover_columns = [col for col in merged_df.columns if 'crossover' in col]
    for col in crossover_columns:
        merged_df[col] = merged_df[col].shift(-1)

    # Replace 'buy' with 1, 'sell' with -1, and fill NaNs with 0
    merged_df = merged_df.replace('buy', 1)
    merged_df = merged_df.replace('sell', -1)
    merged_df = merged_df.fillna(0)
    
    # Refer to columns by names to calculate 'log_return' and 'log_volume'
    merged_df['log_return'] = np.log(pd.to_numeric(merged_df['close'], errors='coerce')).diff()
    merged_df['log_volume'] = np.log(pd.to_numeric(merged_df['Soma_Volumes'], errors='coerce'))
    
    log_columns = ['BBands_bot_5_1.0', 'BBands_mid_5_1.0', 'BBands_top_5_1.0', 'EMA_31', 'EMA_75', 'SMA_18', 'SMA_40']
    for col in log_columns:
        merged_df[col] = np.log(pd.to_numeric(merged_df[col], errors='coerce'))
    
    # Generate BBands distance indicator
    merged_df['BBands_distance_indicator'] = merged_df.apply(bbands_distance_indicator, axis=1)

    merged_df['return_shifted'] = merged_df['log_return'].shift(-1)
   
    # Generate market order signal
    merged_df['market_order_signal'] = merged_df['return_shifted'].apply(market_order_signal)
    
    # Drop rows with NaN, inf, or -inf values
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_df.dropna(inplace=True)
    
    # Sort by date
    merged_df.sort_values(by='dt', inplace=True)
    
    # Save to CSV
    merged_df.to_csv(os.path.join(script_dir, 'ML_data', 'round_1_dataframe.csv'), index=False)
    return merged_df


# ML models
def linear_regression(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, m):
    from sklearn.linear_model import LinearRegression
    # assuming that the dataset size is 849 rows, the dataset is divided into 449 days to train and 400 days to test
    df = pd.DataFrame()
    model = LinearRegression()
    model.fit(Xtrain, Ytrain)
    model.score(Xtrain, Ytrain), model.score(Xtest, Ytest)
    Ptrain = model.predict(Xtrain)
    Ptest = model.predict(Xtest)
    
    # Convert numpy arrays to pandas Series
    Ptrain_series = pd.Series(Ptrain, index=Xtrain.index)
    Ptest_series = pd.Series(Ptest, index=Xtest.index)

    # Combine the training and testing predictions
    combined_data = pd.concat([Ptrain_series, Ptest_series])
    print(f"Combined data: {combined_data}")
    # Add the combined data to the round_2 dataframe
    round_2['real_position_linear_regression'] = combined_data

    # Fill any missing values with 0
    round_2['real_position_linear_regression'].fillna(0, inplace=True)

    # accuracy
    df['accuracy_train'] =  np.mean(np.sign(Ptrain) == np.sign(Ytrain)),
    # accuracy
    df['accuracy_test'] = np.mean(np.sign(Ptest) == np.sign(Ytest))
    set(np.sign(Ptrain)), set(np.sign(Ptest))
    train_idx = round_2.index <= train.index[-1]
    test_idx = round_2.index > train.index[-1]
    train_idx[0] = False
    test_idx[-1] = False
    round_2['position_linear_regression'] = 0 # create new column
    round_2.loc[train_idx,'position_linear_regression'] = (Ptrain > 0)
    round_2.loc[test_idx,'position_linear_regression'] = (Ptest > 0)
    round_2['return_linear_regression'] = round_2['position_linear_regression'] * round_2['return_shifted']
    round_2.to_csv(os.path.join(script_dir, m, 'ML_data', 'round_2_dataframe.csv'),)

    # Total algo log return
    df['log_return_train'] = round_2.iloc[1:-Ntest]['return_linear_regression'].sum(),
    df['log_return_test'] = round_2.iloc[-Ntest:-1]['return_linear_regression'].sum(),
    # std dev
    df['std_dev_train'] = round_2.iloc[1:-Ntest]['return_linear_regression'].std(), 
    df['std_dev_test'] = round_2.iloc[-Ntest:-1]['return_linear_regression'].std()
    # SR
    df['sharpe_train'] = round_2.iloc[1:-Ntest]['return_linear_regression'].mean()/round_2.iloc[1:-Ntest]['return_linear_regression'].std(), 
    df['sharpe_test'] = round_2.iloc[-Ntest:-1]['return_linear_regression'].mean()/round_2.iloc[-Ntest:-1]['return_linear_regression'].std()
    # MSE
    df['mse_train'] = mean_squared_error(Ytrain, Ptrain),
    df['mse_test'] = mean_squared_error(Ytest, Ptest)
    # RMSE
    df['rmse_train'] = mean_squared_error(Ytrain, Ptrain, squared=False),
    df['rmse_test'] = mean_squared_error(Ytest, Ptest, squared=False)
    # MAE
    df['mae_train'] = mean_absolute_error(Ytrain, Ptrain),
    df['mae_test'] = mean_absolute_error(Ytest, Ptest) 
    # R^2
    df['r2_train'] = r2_score(Ytrain, Ptrain),
    df['r2_test'] = r2_score(Ytest, Ptest)
    # MAPE
    df['mape_train'] = mean_absolute_percentage_error(Ytrain, Ptrain),
    df['mape_test'] = mean_absolute_percentage_error(Ytest, Ptest)
    # saves the df 
    df.to_csv(os.path.join(script_dir, m, 'ML_data', 'scores_linear_regression.csv'),)
    return
def ridge_regression(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, m):
    from sklearn.linear_model import Ridge
    # assuming that the dataset size is 849 rows, the dataset is divided into 449 days to train and 400 days to test
    df = pd.DataFrame()
    model = Ridge()
    model.fit(Xtrain, Ytrain)
    model.score(Xtrain, Ytrain), model.score(Xtest, Ytest)
    Ptrain = model.predict(Xtrain)
    Ptest = model.predict(Xtest)

    # Convert numpy arrays to pandas Series
    Ptrain_series = pd.Series(Ptrain, index=Xtrain.index)
    Ptest_series = pd.Series(Ptest, index=Xtest.index)

    # Combine the training and testing predictions
    combined_data = pd.concat([Ptrain_series, Ptest_series])

    # Add the combined data to the round_2 dataframe
    round_2['real_position_ridge_regression'] = combined_data

    # Fill any missing values with 0
    round_2['real_position_ridge_regression'].fillna(0, inplace=True)
    
    # accuracy
    df['accuracy_train'] =  np.mean(np.sign(Ptrain) == np.sign(Ytrain)),
    # accuracy
    df['accuracy_test'] = np.mean(np.sign(Ptest) == np.sign(Ytest))
    set(np.sign(Ptrain)), set(np.sign(Ptest))
    train_idx = round_2.index <= train.index[-1]
    test_idx = round_2.index > train.index[-1]
    train_idx[0] = False
    test_idx[-1] = False
    round_2['position_ridge_regression'] = 0 # create new column
    round_2.loc[train_idx,'position_ridge_regression'] = (Ptrain > 0)
    round_2.loc[test_idx,'position_ridge_regression'] = (Ptest > 0)
    round_2['return_ridge_regression'] = round_2['position_ridge_regression'] * round_2['return_shifted']
    round_2.to_csv(os.path.join(script_dir, m, 'ML_data', 'round_2_dataframe.csv'),)

    # Total algo log return
    df['log_return_train'] = round_2.iloc[1:-Ntest]['return_ridge_regression'].sum(),
    df['log_return_test'] = round_2.iloc[-Ntest:-1]['return_ridge_regression'].sum(),
    # std dev
    df['std_dev_train'] = round_2.iloc[1:-Ntest]['return_ridge_regression'].std(), 
    df['std_dev_test'] = round_2.iloc[-Ntest:-1]['return_ridge_regression'].std()
    # SR
    df['sharpe_train'] = round_2.iloc[1:-Ntest]['return_ridge_regression'].mean()/round_2.iloc[1:-Ntest]['return_ridge_regression'].std(), 
    df['sharpe_test'] = round_2.iloc[-Ntest:-1]['return_ridge_regression'].mean()/round_2.iloc[-Ntest:-1]['return_ridge_regression'].std()
    # MSE
    df['mse_train'] = mean_squared_error(Ytrain, Ptrain),
    df['mse_test'] = mean_squared_error(Ytest, Ptest)
    # RMSE
    df['rmse_train'] = mean_squared_error(Ytrain, Ptrain, squared=False),
    df['rmse_test'] = mean_squared_error(Ytest, Ptest, squared=False)
    # MAE
    df['mae_train'] = mean_absolute_error(Ytrain, Ptrain),
    df['mae_test'] = mean_absolute_error(Ytest, Ptest) 
    # R^2
    df['r2_train'] = r2_score(Ytrain, Ptrain),
    df['r2_test'] = r2_score(Ytest, Ptest)
    # MAPE
    df['mape_train'] = mean_absolute_percentage_error(Ytrain, Ptrain),
    df['mape_test'] = mean_absolute_percentage_error(Ytest, Ptest)
    
    # saves the df 
    df.to_csv(os.path.join(script_dir, m, 'ML_data', 'scores_ridge_regression.csv'),)
    return
def logistic_regression(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, m):
    from sklearn.linear_model import LogisticRegression
    df = pd.DataFrame()
    model = LogisticRegression(C=10)
    Ctrain = (Ytrain > 0)
    Ctest = (Ytest > 0)
    model.fit(Xtrain, Ctrain)
    model.score(Xtrain, Ctrain), model.score(Xtest, Ctest)
    
    Ptrain = model.predict(Xtrain)
    Ptest = model.predict(Xtest)
    set(Ptrain), set(Ptest)
    
    train_idx = round_2.index <= train.index[-1]
    test_idx = round_2.index > train.index[-1]
    train_idx[0] = False
    test_idx[-1] = False
    
    round_2.loc[train_idx,'position_logistic_regression'] = Ptrain
    round_2.loc[test_idx,'position_logistic_regression'] = Ptest        
    round_2['return_logistic_regression'] = round_2['position_logistic_regression'] * round_2['return_shifted']

    round_2.to_csv(os.path.join(script_dir, m, 'ML_data', 'round_2_dataframe.csv'),)

    # Total algo log return
    df['log_return_train'] = round_2.iloc[1:-Ntest]['return_logistic_regression'].sum(),
    df['log_return_test'] = round_2.iloc[-Ntest:-1]['return_logistic_regression'].sum(),
    # std dev
    df['std_dev_train'] = round_2.iloc[1:-Ntest]['return_logistic_regression'].std(), 
    df['std_dev_test'] = round_2.iloc[-Ntest:-1]['return_logistic_regression'].std()
    # SR
    df['sharpe_train'] = round_2.iloc[1:-Ntest]['return_logistic_regression'].mean()/round_2.iloc[1:-Ntest]['return_logistic_regression'].std(), 
    df['sharpe_test'] = round_2.iloc[-Ntest:-1]['return_logistic_regression'].mean()/round_2.iloc[-Ntest:-1]['return_logistic_regression'].std()
    
    df['accuracy_train'] = accuracy_score(Ctrain,Ptrain), 
    df['accuracy_test'] = accuracy_score(Ctest,Ptest)
    
    df['precision_train'] = precision_score(Ctrain,Ptrain), 
    df['precision_test'] = precision_score(Ctest,Ptest)
    
    df['f1_train'] = f1_score(Ctrain,Ptrain), 
    df['f1_test'] = f1_score(Ctest,Ptest)
    
    df['roc_auc_train'] = roc_auc_score(Ctrain,Ptrain), 
    df['roc_auc_test'] = roc_auc_score(Ctest,Ptest)
    
    # saves the df 
    df.to_csv(os.path.join(script_dir, m, 'ML_data', 'scores_logistic_regression.csv'),)
    return
def lgbm_classifier(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, sample_weight, m):
    from lightgbm import LGBMClassifier
    df = pd.DataFrame()
    model = LGBMClassifier()
    Ctrain = (Ytrain > 0)
    Ctest = (Ytest > 0)
    model.fit(Xtrain, Ctrain, sample_weight=sample_weight)
    model.score(Xtrain, Ctrain), model.score(Xtest, Ctest)
    
    Ptrain = model.predict(Xtrain)
    Ptest = model.predict(Xtest)
    set(Ptrain), set(Ptest)
    
    train_idx = round_2.index <= train.index[-1]
    test_idx = round_2.index > train.index[-1]
    train_idx[0] = False
    test_idx[-1] = False
    
    round_2.loc[train_idx,'position_lgbm_classifier'] = Ptrain
    round_2.loc[test_idx,'position_lgbm_classifier'] = Ptest        
    round_2['return_lgbm_classifier'] = round_2['position_lgbm_classifier'] * round_2['return_shifted']

    round_2.to_csv(os.path.join(script_dir, m, 'ML_data', 'round_2_dataframe.csv'),)

    # Total algo log return
    df['log_return_train'] = round_2.iloc[1:-Ntest]['return_lgbm_classifier'].sum(),
    df['log_return_test'] = round_2.iloc[-Ntest:-1]['return_lgbm_classifier'].sum(),
    # std dev
    df['std_dev_train'] = round_2.iloc[1:-Ntest]['return_lgbm_classifier'].std(), 
    df['std_dev_test'] = round_2.iloc[-Ntest:-1]['return_lgbm_classifier'].std()
    # SR
    df['sharpe_train'] = round_2.iloc[1:-Ntest]['return_lgbm_classifier'].mean()/round_2.iloc[1:-Ntest]['return_lgbm_classifier'].std(), 
    df['sharpe_test'] = round_2.iloc[-Ntest:-1]['return_lgbm_classifier'].mean()/round_2.iloc[-Ntest:-1]['return_lgbm_classifier'].std()
    
    df['accuracy_train'] = accuracy_score(Ctrain,Ptrain), 
    df['accuracy_test'] = accuracy_score(Ctest,Ptest)
    
    df['precision_train'] = precision_score(Ctrain,Ptrain), 
    df['precision_test'] = precision_score(Ctest,Ptest)
    
    df['f1_train'] = f1_score(Ctrain,Ptrain), 
    df['f1_test'] = f1_score(Ctest,Ptest)
    
    df['roc_auc_train'] = roc_auc_score(Ctrain,Ptrain), 
    df['roc_auc_test'] = roc_auc_score(Ctest,Ptest)
    
    # saves the df 
    df.to_csv(os.path.join(script_dir, m, 'ML_data', 'scores_lgbm_classifier.csv'),)
    return
def xgboost_classifier(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, sample_weight, m):
    from xgboost.sklearn import XGBClassifier
    df = pd.DataFrame()
    model = XGBClassifier()
    Ctrain = (Ytrain > 0)
    Ctest = (Ytest > 0)
    model.fit(Xtrain, Ctrain, sample_weight=sample_weight)
    model.score(Xtrain, Ctrain), model.score(Xtest, Ctest)
    
    Ptrain = model.predict(Xtrain)
    Ptest = model.predict(Xtest)
    set(Ptrain), set(Ptest)
    
    train_idx = round_2.index <= train.index[-1]
    test_idx = round_2.index > train.index[-1]
    train_idx[0] = False
    test_idx[-1] = False
    
    round_2.loc[train_idx,'position_xgboost_classifier'] = Ptrain
    round_2.loc[test_idx,'position_xgboost_classifier'] = Ptest        
    round_2['return_xgboost_classifier'] = round_2['position_xgboost_classifier'] * round_2['return_shifted']

    round_2.to_csv(os.path.join(script_dir, m, 'ML_data', 'round_2_dataframe.csv'),)

    # Total algo log return
    df['log_return_train'] = round_2.iloc[1:-Ntest]['return_xgboost_classifier'].sum(),
    df['log_return_test'] = round_2.iloc[-Ntest:-1]['return_xgboost_classifier'].sum(),
    # std dev
    df['std_dev_train'] = round_2.iloc[1:-Ntest]['return_xgboost_classifier'].std(), 
    df['std_dev_test'] = round_2.iloc[-Ntest:-1]['return_xgboost_classifier'].std()
    
    df['accuracy_train'] = accuracy_score(Ctrain,Ptrain), 
    df['accuracy_test'] = accuracy_score(Ctest,Ptest)
    
    df['precision_train'] = precision_score(Ctrain,Ptrain), 
    df['precision_test'] = precision_score(Ctest,Ptest)
    
    df['f1_train'] = f1_score(Ctrain,Ptrain), 
    df['f1_test'] = f1_score(Ctest,Ptest)
    
    df['roc_auc_train'] = roc_auc_score(Ctrain,Ptrain), 
    df['roc_auc_test'] = roc_auc_score(Ctest,Ptest)
    
    # saves the df 
    df.to_csv(os.path.join(script_dir, m, 'ML_data', 'scores_xgboost_classifier.csv'),)
    return
    return
def gaussnb_classifier(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, sample_weight, m):
    from sklearn.naive_bayes import GaussianNB
    df = pd.DataFrame()
    model = GaussianNB()
    Ctrain = (Ytrain > 0)
    Ctest = (Ytest > 0)
    model.fit(Xtrain, Ctrain, sample_weight=sample_weight)
    model.score(Xtrain, Ctrain), model.score(Xtest, Ctest)

    Ptrain = model.predict(Xtrain)
    Ptest = model.predict(Xtest)
    set(Ptrain), set(Ptest)

    train_idx = round_2.index <= train.index[-1]
    test_idx = round_2.index > train.index[-1]
    train_idx[0] = False
    test_idx[-1] = False

    round_2.loc[train_idx,'position_gnb'] = Ptrain
    round_2.loc[test_idx,'position_gnb'] = Ptest        
    round_2['return_gnb'] = round_2['position_gnb'] * round_2['return_shifted']

    round_2.to_csv(os.path.join(script_dir, m, 'ML_data', 'round_2_dataframe.csv'),)

    # Total algo log return
    df['log_return_train'] = round_2.iloc[1:-Ntest]['return_gnb'].sum(),
    df['log_return_test'] = round_2.iloc[-Ntest:-1]['return_gnb'].sum(),
    # std dev
    df['std_dev_train'] = round_2.iloc[1:-Ntest]['return_gnb'].std(), 
    df['std_dev_test'] = round_2.iloc[-Ntest:-1]['return_gnb'].std()
    # SR
    df['sharpe_train'] = round_2.iloc[1:-Ntest]['return_gnb'].mean()/round_2.iloc[1:-Ntest]['return_gnb'].std(), 
    df['sharpe_test'] = round_2.iloc[-Ntest:-1]['return_gnb'].mean()/round_2.iloc[-Ntest:-1]['return_gnb'].std()
    
    df['accuracy_train'] = accuracy_score(Ctrain,Ptrain), 
    df['accuracy_test'] = accuracy_score(Ctest,Ptest)
    
    df['precision_train'] = precision_score(Ctrain,Ptrain), 
    df['precision_test'] = precision_score(Ctest,Ptest)
    
    df['f1_train'] = f1_score(Ctrain,Ptrain), 
    df['f1_test'] = f1_score(Ctest,Ptest)
    
    df['roc_auc_train'] = roc_auc_score(Ctrain,Ptrain), 
    df['roc_auc_test'] = roc_auc_score(Ctest,Ptest)

    # saves the df 
    df.to_csv(os.path.join(script_dir, m, 'ML_data', 'scores_gnb.csv'),)
    return
def random_forest_classifier(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, m):
    from sklearn.ensemble import RandomForestClassifier
    df = pd.DataFrame()
    model = RandomForestClassifier(random_state=4)
    Ctrain = (Ytrain > 0)
    Ctest = (Ytest > 0)
    model.fit(Xtrain, Ctrain)
    model.score(Xtrain, Ctrain), model.score(Xtest, Ctest)

    Ptrain = model.predict(Xtrain)
    Ptest = model.predict(Xtest)
    set(Ptrain), set(Ptest)

    train_idx = round_2.index <= train.index[-1]
    test_idx = round_2.index > train.index[-1]
    train_idx[0] = False
    test_idx[-1] = False

    round_2.loc[train_idx,'position_random_forest'] = Ptrain
    round_2.loc[test_idx,'position_random_forest'] = Ptest        
    round_2['return_random_forest'] = round_2['position_random_forest'] * round_2['return_shifted']

    round_2.to_csv(os.path.join(script_dir, m, 'ML_data', 'round_2_dataframe.csv'),)

    # Total algo log return
    df['log_return_train'] = round_2.iloc[1:-Ntest]['return_random_forest'].sum(),
    df['log_return_test'] = round_2.iloc[-Ntest:-1]['return_random_forest'].sum(),
    # std dev
    df['std_dev_train'] = round_2.iloc[1:-Ntest]['return_random_forest'].std(), 
    df['std_dev_test'] = round_2.iloc[-Ntest:-1]['return_random_forest'].std()
    # SR
    df['sharpe_train'] = round_2.iloc[1:-Ntest]['return_random_forest'].mean()/round_2.iloc[1:-Ntest]['return_random_forest'].std(), 
    df['sharpe_test'] = round_2.iloc[-Ntest:-1]['return_random_forest'].mean()/round_2.iloc[-Ntest:-1]['return_random_forest'].std()
    
    df['accuracy_train'] = accuracy_score(Ctrain,Ptrain), 
    df['accuracy_test'] = accuracy_score(Ctest,Ptest)
    
    df['precision_train'] = precision_score(Ctrain,Ptrain), 
    df['precision_test'] = precision_score(Ctest,Ptest)
    
    df['f1_train'] = f1_score(Ctrain,Ptrain), 
    df['f1_test'] = f1_score(Ctest,Ptest)
    
    df['roc_auc_train'] = roc_auc_score(Ctrain,Ptrain), 
    df['roc_auc_test'] = roc_auc_score(Ctest,Ptest)

    # saves the df 
    df.to_csv(os.path.join(script_dir, m, 'ML_data', 'scores_random_forest.csv'),)

    return
def svc_classifier(Xtrain, Ytrain, Xtest, Ytest, round_2, Ntest, train, sample_weight, m):
    from sklearn.svm import SVC
    df = pd.DataFrame()
    model = SVC(C=10)
    Ctrain = (Ytrain > 0)
    Ctest = (Ytest > 0)
    model.fit(Xtrain, Ctrain, sample_weight=sample_weight)
    model.score(Xtrain, Ctrain), model.score(Xtest, Ctest)

    Ptrain = model.predict(Xtrain)
    Ptest = model.predict(Xtest)
    set(Ptrain), set(Ptest)

    train_idx = round_2.index <= train.index[-1]
    test_idx = round_2.index > train.index[-1]
    train_idx[0] = False
    test_idx[-1] = False

    round_2.loc[train_idx,'position_svc'] = Ptrain
    round_2.loc[test_idx,'position_svc'] = Ptest        
    round_2['return_svc'] = round_2['position_svc'] * round_2['return_shifted']

    round_2.to_csv(os.path.join(script_dir, m, 'ML_data', 'round_2_dataframe.csv'),)

    # Total algo log return
    df['log_return_train'] = round_2.iloc[1:-Ntest]['return_svc'].sum(),
    df['log_return_test'] = round_2.iloc[-Ntest:-1]['return_svc'].sum(),
    # std dev
    df['std_dev_train'] = round_2.iloc[1:-Ntest]['return_svc'].std(), 
    df['std_dev_test'] = round_2.iloc[-Ntest:-1]['return_svc'].std()
    # SR
    df['sharpe_train'] = round_2.iloc[1:-Ntest]['return_svc'].mean()/round_2.iloc[1:-Ntest]['return_svc'].std(), 
    df['sharpe_test'] = round_2.iloc[-Ntest:-1]['return_svc'].mean()/round_2.iloc[-Ntest:-1]['return_svc'].std()
    
    df['accuracy_train'] = accuracy_score(Ctrain,Ptrain), 
    df['accuracy_test'] = accuracy_score(Ctest,Ptest)
    
    df['precision_train'] = precision_score(Ctrain,Ptrain), 
    df['precision_test'] = precision_score(Ctest,Ptest)
    
    df['f1_train'] = f1_score(Ctrain,Ptrain), 
    df['f1_test'] = f1_score(Ctest,Ptest)
    
    df['roc_auc_train'] = roc_auc_score(Ctrain,Ptrain), 
    df['roc_auc_test'] = roc_auc_score(Ctest,Ptest)

    # saves the df 
    df.to_csv(os.path.join(script_dir, m, 'ML_data', 'scores_svc.csv'),)
    return

# merges the results dataframes 
def unify_results(m):
    linear_regression = pd.read_csv(os.path.join(script_dir, m, 'ML_data', 'scores_linear_regression.csv'), index_col=0)
    ridge_regression = pd.read_csv(os.path.join(script_dir, m, 'ML_data', 'scores_ridge_regression.csv'), index_col=0)
    logistic_regression = pd.read_csv(os.path.join(script_dir, m, 'ML_data', 'scores_logistic_regression.csv'), index_col=0)
    svc = pd.read_csv(os.path.join(script_dir, m, 'ML_data', 'scores_svc.csv'), index_col=0)
    xgboost_classifier = pd.read_csv(os.path.join(script_dir, m, 'ML_data', 'scores_xgboost_classifier.csv'), index_col=0)
    lgbm_classifier = pd.read_csv(os.path.join(script_dir, m, 'ML_data', 'scores_lgbm_classifier.csv'), index_col=0)
    random_forest = pd.read_csv(os.path.join(script_dir, m, 'ML_data', 'scores_random_forest.csv'), index_col=0)
    list = [linear_regression,ridge_regression,logistic_regression, svc, xgboost_classifier, lgbm_classifier, random_forest]
    dataframe_results = pd.concat(list)
    data = ['linear_regression','ridge_regression','logistic_regression', 'svc', 'xgboost_classifier', 'lgbm_classifier', 'random_forest']
    dataframe_results.insert(0, 'model', data)
    dataframe_results.to_csv(os.path.join(script_dir, m, 'ML_data', 'scores_dataframe.csv'),)
    return

# divide in train and test
def divide_results_train_test(m):
    scores = pd.read_csv(os.path.join(script_dir, m, 'ML_data', 'scores_dataframe.csv'), index_col=0)
    scores_test = scores.drop(columns = ['accuracy_train', 'log_return_train', 'std_dev_train', 'sharpe_train', 'mse_train', 'rmse_train', 'mae_train', 'r2_train', 'mape_train', 'precision_train', 'f1_train', 'roc_auc_train'])
    scores_train = scores.drop(columns = ['accuracy_test', 'log_return_test', 'std_dev_test', 'sharpe_test', 'mse_test', 'rmse_test', 'mae_test', 'r2_test', 'mape_test', 'precision_test', 'f1_test', 'roc_auc_test'])
    scores_train.to_csv(os.path.join(script_dir, m, 'ML_data', 'scores_train.csv'),)
    scores_test.to_csv(os.path.join(script_dir, m, 'ML_data', 'scores_test.csv'),)
    return

def plot_regressions(df, model):
    # Dynamic column name based on the model
    position_column = f'position_{model}'
    return_column = f'return_{model}'

    # Convert to boolean and calculate entry/exit points
    df[position_column] = df[position_column] == 'True'
    df[f'entry_point_{model}'] = df[position_column] & ~df[position_column].shift(1).fillna(False)
    df[f'exit_point_{model}'] = ~df[position_column] & df[position_column].shift(1).fillna(False)

    # Calculate accumulated returns
    df[f'accumulated_return_{model}'] = (1 + df[return_column]).cumprod()

    # Create subplots with different heights
    # Adjust these values as needed for your desired plot sizes
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                        subplot_titles=('Price and Trading Points', ),
                        row_heights=[0.85, 0.15])  # 70% height for first plot, 30% for second

    # First subplot for price and trading points
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[df[f'entry_point_{model}']].index, y=df[df[f'entry_point_{model}']]['close'], mode='markers', name='Entry Points', marker=dict(color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[df[f'exit_point_{model}']].index, y=df[df[f'exit_point_{model}']]['close'], mode='markers', name='Exit Points', marker=dict(color='red', size=10)), row=1, col=1)

    # Second subplot for accumulated returns
    fig.add_trace(go.Scatter(x=df.index, y=df[f'accumulated_return_{model}'], mode='lines', name='Accumulated Return'), row=2, col=1)

    # Update layout
    fig.update_layout(title=f'{model} - Price and Accumulated Returns', height=600, width=800)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Accumulated Return', row=2, col=1)

    # Show the figure
    #fig.show()

    # Save the plot as a PNG
    file_name = f"{model}_plot.png"
    pio.write_image(fig, file_name)

    #df.to_csv('dataset_completo.csv')

    print(f"Plot saved as {file_name}")
    return fig

def plot_classifier(df, model):
    # Dynamic column names based on the model
    position_column = f'position_{model}'
    return_column = f'return_{model}'

    # Handle NaN values and convert to boolean
    df[position_column] = df[position_column].fillna(False).astype(bool)

    # Calculate entry and exit points
    df[f'entry_point_{model}'] = df[position_column] & ~df[position_column].shift(1).fillna(False)
    df[f'exit_point_{model}'] = ~df[position_column] & df[position_column].shift(1).fillna(False)

    # Calculate accumulated returns, handle NaN values in return_column
    df[return_column] = df[return_column].fillna(0)  # Assuming 0 return for NaNs
    df[f'accumulated_return_{model}'] = (1 + df[return_column]).cumprod()

    # Create subplots with different heights
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=('Price and Trading Points', 'Accumulated Returns'),
                        row_heights=[0.85, 0.15])

    # First subplot for price and trading points
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[df[f'entry_point_{model}']].index, y=df[df[f'entry_point_{model}']]['close'], mode='markers', name='Entry Points', marker=dict(color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[df[f'exit_point_{model}']].index, y=df[df[f'exit_point_{model}']]['close'], mode='markers', name='Exit Points', marker=dict(color='red', size=10)), row=1, col=1)

    # Second subplot for accumulated returns
    fig.add_trace(go.Scatter(x=df.index, y=df[f'accumulated_return_{model}'], mode='lines', name='Accumulated Return'), row=2, col=1)

    # Update layout
    fig.update_layout(title=f'{model} - Price and Accumulated Returns', height=600, width=800)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Accumulated Return', row=2, col=1)

    # Show and save the figure
    #fig.show()
    file_name = f"{model}_plot.png"
    pio.write_image(fig, file_name)

    # Save the modified DataFrame (optional)
    #df.to_csv('dataset_completo.csv')

    print(f"Plot saved as {file_name}")
    return fig