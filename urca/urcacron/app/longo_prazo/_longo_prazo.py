
import json
import os
from flask import request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotly.io import to_json
from datetime import datetime
from datetime import timedelta
from scipy.optimize import least_squares
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

import plotly.graph_objects as go #Pedro

from app.longo_prazo import b_longo_prazo

script_dir = os.path.dirname(__file__)


### Funções

#ObjFunMRLJumps_20230629Returns
#import numpy as np

def ObjFunMRLJumps_20230629Returns(data,coef,coef0,t,dt,beta0,alpha):
    kappa = coef[0]
    theta = coef[1]
    xi = coef[2]
    lmbda=coef[3]
    mu = coef[4]
    sigma=coef[5]

    xi2 = xi**2
    lmbda2=lmbda**2
    sigma2=sigma**2
    integral = lmbda2*np.exp(1.5*mu**2/sigma2)*(np.exp(8*sigma2+4*mu)-2*np.exp(2*sigma2+2*mu)+1)
    data2 = data**2
    a = xi2+integral

    MeanBeta = beta0
    MeanBetaSqr = beta0**2
    f = np.zeros(t.shape)
    f[0] = np.sqrt(np.abs(MeanBetaSqr -2*data[0]*MeanBeta + data2[0]))  #Alteração_1
    g = np.zeros(t.shape)

    for jj in np.arange(1,len(t)):
        MeanBeta0 = MeanBeta
        MeanBeta = MeanBeta0*np.exp(-kappa*dt) + theta*(1-np.exp(-kappa*dt))
        MeanBetaSqr = np.exp(-2*kappa*dt)*MeanBetaSqr + np.exp(-2*kappa*dt)*(2*theta*MeanBeta0*(1-np.exp(-kappa*dt))+ 2*theta**2 *(kappa*dt-1+np.exp(-kappa*dt))+a*dt)
        f[jj] = np.sqrt(np.abs(MeanBetaSqr - 2*data[jj]*MeanBeta + data2[jj]))
        g[jj] = MeanBetaSqr-MeanBeta**2

    f = np.concatenate((f, alpha*(coef-coef0), alpha*(coef)))
    return f

#ObjFunHeston_20240207Returns

def ObjFunHeston_20240207Returns(data,coef,coef0,t,dt,beta0,alpha):
    kappa = coef[0]
    theta = coef[1]
    xi = coef[2]
    xi2 = xi**2

    data2 = data**2

    MeanBeta = beta0
    MeanBetaSqr = beta0**2
    f = np.zeros(t.shape)
    f[0] = np.sqrt(np.abs(MeanBetaSqr -2*data[0]*MeanBeta + data2[0])) #Alteração_1
    g = np.zeros(t.shape)

    for jj in np.arange(1,len(t)):
        MeanBeta0 = MeanBeta
        MeanBeta = MeanBeta0*np.exp(-kappa*dt) + theta*(1-np.exp(-kappa*dt))
        MeanBetaSqr = np.exp(-2*kappa*dt)*MeanBetaSqr + np.exp(-2*kappa*dt)*(2*theta*MeanBeta0+xi2*dt)
        f[jj] = np.sqrt(np.abs(MeanBetaSqr - 2*data[jj]*MeanBeta + data2[jj]))
        g[jj] = MeanBetaSqr-MeanBeta**2

    f = np.concatenate((f, alpha*(coef-coef0), alpha*(coef)))
    return f

## window

# Features: Concatena os 20 (lenn) tempos anteriores em cada linha. Começa na linha 20 (lenn), concatenando da linha 0 a 19.

def window(lenn, x_features):

    x_window = np.zeros((x_features.shape[0] - lenn,(lenn+1)*x_features.shape[1]))

    for jj in range(lenn, x_features.shape[0]):
        x_window[jj-lenn,:] = x_features[jj-lenn:jj+1,:].ravel()

    return x_window

def model_Ridge(normalized_X_train, y_train, normalized_X_test, y_test):

    ridge = Ridge()

    param_grid = {'alpha': np.linspace(0.1, 2.5, 25)}
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(ridge, param_grid, cv=kfold, scoring='neg_mean_squared_error', verbose=1)
    grid_result = grid_search.fit(normalized_X_train, y_train)

    best_params = grid_result.best_params_
    best_model = grid_result.best_estimator_

    y_pred = best_model.predict(normalized_X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("best_params", best_params, "MSE no conjunto de teste:", mse)

    return 'best_model = ', best_model

#Risco_20240412 => Será usada sempre a mesma. Só mudará a entrada em cada contexto.

def Risco_20240412(sign):

    var05= - np.percentile(sign,5, axis = 1)
    cvar05=np.zeros(len(var05))

    for ii in range(len(var05)):
        aux = sign[ii,:]
        aux = aux[aux<=-var05[ii]]
        cvar05[ii] = - np.sum(aux)/len(aux)

    return 'var05 = ', var05, 'cvar05 = ', cvar05

def show_result(price, predicted):
    return
    plt.figure(figsize=(16, 6))
    plt.grid(True)
    plt.box(True)
    plt.plot(pd.DataFrame(predicted).index, predicted, 'o-', label="Predição", color = 'blue')
    plt.plot(pd.DataFrame(price).index, price, '.-', label="Preço", color = 'red')
    plt.xlabel('Tempo')
    plt.ylabel('Preço Logarítmico')
    plt.legend()

#MRJ_Scenarios_20240424

def MRJ_Scenarios_20240424(coefLMRJ, IpcaPred, lenP, dt, y2, P0, lPld, ll):

    t = np.arange(0,lenP + 1)
    t= t/12
    NSamples= int(1E4)
    kappa = coefLMRJ[0]
    theta = coefLMRJ[1]
    xi = coefLMRJ[2]
    lmbda=coefLMRJ[3]
    mu = coefLMRJ[4]
    sigma=coefLMRJ[5]

    rng = np.random.default_rng()
    A = rng.poisson(lam = dt, size = (int(len(t)), NSamples))
    A = np.concatenate((np.zeros((1,NSamples)), A), axis = 0)

    W = np.sqrt(dt)*np.random.randn(len(t), NSamples)
    W = np.concatenate((np.zeros((1, NSamples)), W), axis=0)

    B = np.random.normal(mu, sigma, (len(t),NSamples))
    B = np.concatenate((np.zeros((1,NSamples)), B), axis = 0)
    B = B*A

    aux=y2

    X= np.zeros((len(t),NSamples))
    X[0,:] = np.log(P0)-aux[0]
    Xtemp = X[0,:]/xi
    X[0,:]=X[0,:]+aux[0]

    ipcaPred = IpcaPred[ll,:]

    tax = 1+ipcaPred/100
    tax = 1/np.cumprod(tax)

    for ii in range(len(t)-1):
        Xtemp = Xtemp + kappa*(theta/xi-Xtemp)*dt + W[ii,:] + lmbda/xi*B[ii,:]
        X[ii+1,:]= np.minimum(lPld[ii,1], np.maximum(lPld[ii,0], xi*Xtemp+aux[ii+1]))


    L = np.percentile(X, 5, axis=1)
    M = np.median(X, axis = 1)
    U = np.percentile(X, 95, axis=1)

    signLMRJ = np.zeros((X.shape[0]-1,X.shape[1]))

    for ii in range(X.shape[0]-1):
        signLMRJ[ii,:] =(tax[ii]*np.exp(X[ii+1,:])-P0)/P0

    return  'signLMRJ = ', signLMRJ, "L_MRJ = ", L, "M_MRJ = ", M, "U_MRJ = ", U

## MR_Scenarios_20240424

def MR_Scenarios_20240424(coefMR, IpcaPred, lenP, dt, y2, P0, lPld, ll):

    t = np.arange(0,lenP + 1)
    t= t/12
    NSamples= int(1E4)
    kappa = coefMR[0]
    theta = coefMR[1]
    xi = coefMR[2]

    W = np.sqrt(dt)*np.random.randn(len(t), NSamples)
    W = np.concatenate((np.zeros((1, NSamples)), W), axis=0)

    aux=y2

    X= np.zeros((len(t),NSamples))
    X[0,:] = np.log(P0)-aux[0]
    Xtemp = X[0,:]
    X[0,:]=X[0,:]+aux[0]


    for ii in range(len(t)-1):
        Xtemp = Xtemp + kappa*(theta-Xtemp)*dt + xi*W[ii,:]
        X[ii+1,:]= np.minimum(lPld[ii,1], np.maximum(lPld[ii,0], xi*Xtemp+aux[ii+1]))


    L = np.percentile(X, 5, axis=1)
    M = np.median(X, axis = 1)
    U = np.percentile(X, 95, axis=1)


    ipcaPred = IpcaPred[ll,:]

    tax = 1+ipcaPred/100
    tax = 1/np.cumprod(tax)

    signMR = np.zeros((X.shape[0]-1,X.shape[1]))

    for ii in range(X.shape[0]-1):
        signMR[ii,:] =(tax[ii]*np.exp(X[ii+1,:])-P0)/P0

    return  'signMR = ', signMR, "L_MR = ", L, "M_MR = ", M, "U_MR = ", U


### MRJ_Error_Evaluation

def MRJ_Error_Evaluation(coefLMRJ, lenP, data, dt, y, y2, lPld):

    t = np.arange(0,lenP + 1)
    t= t/12
    kappa = coefLMRJ[0]
    theta = coefLMRJ[1]
    xi = coefLMRJ[2]
    lmbda=coefLMRJ[3]
    mu = coefLMRJ[4]
    sigma=coefLMRJ[5]

    xi2 = xi**2
    lmbda2=lmbda**2
    sigma2=sigma**2
    integral = lmbda2*np.exp(1.5*mu**2/sigma2)*(np.exp(8*sigma2+4*mu)-2*np.exp(2*sigma2+2*mu)+1)
    data2 = data**2
    a = xi2+integral

    S0=data[-1]
    MeanBeta = np.zeros(t.shape)
    MeanBetaSqr = np.zeros(t.shape)
    MeanBeta[0] = S0
    MeanBetaSqr[0] = S0**2

    for jj in range(1,len(t)):
        MeanBeta0=MeanBeta[jj-1]
        MeanBeta[jj] = MeanBeta0*np.exp(-kappa*dt) + theta*(1-np.exp(-kappa*dt))
        MeanBetaSqr[jj] = np.exp(-2*kappa*dt)*MeanBetaSqr[jj-1] + np.exp(-2*kappa*dt)*(2*theta*MeanBeta0*(1-np.exp(-kappa*dt))+2*theta**2*(kappa*dt-1+np.exp(-kappa*dt))+a*dt)

    logPricesLMRJ = MeanBeta
    logPricesSqrLMRJ = MeanBetaSqr
    stdPrices = np.sqrt(np.abs(MeanBetaSqr-MeanBeta**2))
    PredM = logPricesLMRJ + y[-1]
    PredL = PredM - stdPrices
    PredU = PredM + stdPrices

    for zz in range(len(PredM)):
        PredM[zz] = max(PredM[zz],lPld[zz,0])
        PredM[zz] = min(PredM[zz],lPld[zz,1])
        PredL[zz] = max(PredL[zz],lPld[zz,0])
        PredL[zz] = min(PredL[zz],lPld[zz,1])
        PredU[zz] = max(PredU[zz],lPld[zz,0])
        PredU[zz] = min(PredU[zz],lPld[zz,1])


    PredMb = logPricesLMRJ + y2[-lenP-1:]
    PredLb = PredMb - stdPrices
    PredUb = PredMb + stdPrices

    for zz in range(len(PredM)):
        PredMb[zz] = max(PredMb[zz],lPld[zz,0])
        PredMb[zz] = min(PredMb[zz],lPld[zz,1])
        PredLb[zz] = max(PredLb[zz],lPld[zz,0])
        PredLb[zz] = min(PredLb[zz],lPld[zz,1])
        PredUb[zz] = max(PredUb[zz],lPld[zz,0])
        PredUb[zz] = min(PredUb[zz],lPld[zz,1])

    return  'PredMb = ', PredMb, 'PredUb = ', PredUb, 'PredLb = ', PredLb



def MRJ_plot(datesIbc, AUX, PriceM, datesPred, PredMb, PredUb, PredLb, lPld, ll, LEN, lenP, yy):
    fig = go.Figure()

    # Observado
    fig.add_trace(go.Scatter(
        x=datesIbc[:AUX.shape[0]],
        y=np.minimum(PriceM[:AUX.shape[0]], np.exp(np.max(lPld))*np.ones(len(PriceM[:AUX.shape[0]]))),
        mode='lines',
        name='Observado',
        line=dict(color='black', width=1)
    ))

    # Previsão
    fig.add_trace(go.Scatter(
        x=datesPred,
        y=np.exp(PredMb),
        mode='lines',
        name='Previsão',
        line=dict(color='magenta', width=1)
    ))

    # Desvio Padrão
    fig.add_trace(go.Scatter(
        x=datesPred,
        y=np.exp(PredLb),
        mode='lines',
        fill=None,
        line=dict(color='orange', width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=datesPred,
        y=np.exp(PredUb),
        mode='lines',
        fill='tonexty',
        line=dict(color='orange', width=0),
        name='Desvio Padrão'
    ))

    # Linhas de limite inferior e superior
    if ll < len(datesIbc) - LEN - lenP:
        fig.add_trace(go.Scatter(
            x=datesIbc[:AUX.shape[0]],
            y=np.exp(np.min(lPld)) * np.ones(len(datesIbc[:AUX.shape[0]])),
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=datesIbc[:AUX.shape[0]],
            y=np.exp(np.max(lPld)) * np.ones(len(datesIbc[:AUX.shape[0]])),
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ))
    else:
        date_x = pd.date_range(start=datesIbc.min(), end=datesPred.max(), freq='MS')
        fig.add_trace(go.Scatter(
            x=date_x,
            y=np.exp(np.min(lPld)) * np.ones(len(date_x)),
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=date_x,
            y=np.exp(np.max(lPld)) * np.ones(len(date_x)),
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ))

    fig.update_layout(
        title='Reversão à Média com Saltos, com Regressão de Ridge',
        xaxis_title='Tempo',
        yaxis_title='VWAP',
        showlegend=True,
        xaxis=dict(
            tickmode='array',
            tickangle=-30,
            tickformat='%m/%y',  # Formato para o eixo x
        ),
        font=dict(
            family='Poppins, Helvetica, sans-serif',
            size=12,
            color='black'
        ),
    )

    # Convert Plotly figure to JSON
    plot_json = to_json(fig)

    # Get the date of the start of the prediction
    data_plot = datesPred[0].strftime('%d/%m/%Y')

    output_dir = os.path.join('web', 'static', 'tasks_saida', 'longo_prazo', f"M+{yy}")

    # Save the JSON file
    output_file = os.path.join(output_dir, 'output')
    os.makedirs(output_file, exist_ok=True)
    arquivo = f'longo_prazo_saida_{str(data_plot).replace("/", "_").strip()}_salto.json'
    with open(os.path.join(output_file, arquivo), 'w') as f:
        json.dump(plot_json, f)

    print(f"Gráfico salvo em: {os.path.join(output_file, arquivo)}")

### MR_Error Evaluation

def MR_Error_Evaluation(coefMR, lenP, data, dt, y, y2, lPld):

    t = np.arange(0,lenP + 1)
    t= t/12

    S0=data[-1]
    kappa = coefMR[0]
    theta = coefMR[1]
    xi = coefMR[2]
    xi2 = xi**2

    MeanBeta = np.zeros(t.shape)
    MeanBetaSqr = np.zeros(t.shape)
    MeanBeta[0] = S0
    MeanBetaSqr[0] = S0**2



    for jj in range(1,len(t)):
        MeanBeta0=MeanBeta[jj-1]
        MeanBeta[jj] = MeanBeta0*np.exp(-kappa*dt) + theta*(1-np.exp(-kappa*dt))
        MeanBetaSqr[jj] = np.exp(-2*kappa*dt)*MeanBetaSqr[jj-1] + np.exp(-2*kappa*dt)*(2*theta*MeanBeta0+xi2*dt)


    logPricesMR = MeanBeta
    logPricesSqrMR = MeanBetaSqr

    stdPrices2 = np.sqrt(np.abs(MeanBetaSqr-MeanBeta**2))
    PredM2 = logPricesMR + y[-1]
    PredL2 = PredM2 - stdPrices2
    PredU2 = PredM2 + stdPrices2


    for zz in range(len(PredM2)):
        PredM2[zz] = max(PredM2[zz],lPld[zz,0])
        PredM2[zz] = min(PredM2[zz],lPld[zz,1])
        PredL2[zz] = max(PredL2[zz],lPld[zz,0])
        PredL2[zz] = min(PredL2[zz],lPld[zz,1])
        PredU2[zz] = max(PredU2[zz],lPld[zz,0])
        PredU2[zz] = min(PredU2[zz],lPld[zz,1])


    PredM2b = logPricesMR + y2[-lenP-1:]
    PredL2b = PredM2b - stdPrices2
    PredU2b = PredM2b + stdPrices2

    for zz in range(len(PredM2)):
        PredM2b[zz] = max(PredM2b[zz],lPld[zz,0])
        PredM2b[zz] = min(PredM2b[zz],lPld[zz,1])
        PredL2b[zz] = max(PredL2b[zz],lPld[zz,0])
        PredL2b[zz] = min(PredL2b[zz],lPld[zz,1])
        PredU2b[zz] = max(PredU2b[zz],lPld[zz,0])
        PredU2b[zz] = min(PredU2b[zz],lPld[zz,1])

    return  'PredM2b = ', PredM2b, 'PredU2b = ', PredU2b, 'PredL2b = ', PredL2b

def MR_plot(datesIbc, AUX, PriceM, datesPred, PredM2b, PredU2b, PredL2b, lPld, ll, LEN, lenP, yy):
    fig = go.Figure()

    # Observado
    fig.add_trace(go.Scatter(
        x=datesIbc[:AUX.shape[0]],
        y=np.minimum(PriceM[:AUX.shape[0]], np.exp(np.max(lPld)) * np.ones(len(PriceM[:AUX.shape[0]]))),
        mode='lines',
        name='Observado',
        line=dict(color='black', width=1)
    ))

    # Previsão
    fig.add_trace(go.Scatter(
        x=datesPred,
        y=np.exp(PredM2b),
        mode='lines',
        name='Previsão',
        line=dict(color='red', width=1)
    ))

    # Desvio Padrão
    fig.add_trace(go.Scatter(
        x=datesPred,
        y=np.exp(PredL2b),
        mode='lines',
        fill=None,
        line=dict(color='orange', width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=datesPred,
        y=np.exp(PredU2b),
        mode='lines',
        fill='tonexty',
        line=dict(color='orange', width=0),
        name='Desvio Padrão'
    ))

    # Linhas de limite inferior e superior
    if ll < len(datesIbc) - LEN - lenP:
        fig.add_trace(go.Scatter(
            x=datesIbc[:AUX.shape[0]],
            y=np.exp(np.min(lPld)) * np.ones(len(datesIbc[:AUX.shape[0]])),
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=datesIbc[:AUX.shape[0]],
            y=np.exp(np.max(lPld)) * np.ones(len(datesIbc[:AUX.shape[0]])),
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ))
    else:
        date_x = pd.date_range(start=datesIbc.min(), end=datesPred.max(), freq='MS')
        fig.add_trace(go.Scatter(
            x=date_x,
            y=np.exp(np.min(lPld)) * np.ones(len(date_x)),
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=date_x,
            y=np.exp(np.max(lPld)) * np.ones(len(date_x)),
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ))

    fig.update_layout(
        title='Reversão à Média, com Regressão de Ridge',
        xaxis_title='Tempo',
        yaxis_title='VWAP',
        showlegend=True,
        xaxis=dict(
            tickmode='array',
            tickangle=-30,
            tickformat='%m/%y',  # Formato para o eixo x
        ),
        font=dict(
            family='Poppins, Helvetica, sans-serif',
            size=12,
            color='black'
        ),
    )

    # Convert Plotly figure to JSON
    plot_json = to_json(fig)

    # Get the date of the start of the prediction
    data_plot = datesPred[0].strftime('%d/%m/%Y')

    output_dir = os.path.join('web', 'static', 'tasks_saida', 'longo_prazo', f"M+{yy}")

    # Save the JSON file
    output_file = os.path.join(output_dir, 'output')
    os.makedirs(output_file, exist_ok=True)
    arquivo = f'longo_prazo_saida_{str(data_plot).replace("/", "_").strip()}.json'
    with open(os.path.join(output_file, arquivo), 'w') as f:
        json.dump(plot_json, f)

    print(f"Gráfico salvo em: {os.path.join(output_file, arquivo)}")

def longo_prazo():
    #### Modeling using Prices directly
    objetos_M = {}
    ############
    ### Importando os Dados
    for yy in range(4):

        #### Modeling using Prices directly

        ############
        ### Importando os Dados
        
        # Diretório do script atual
        script_dir = os.path.dirname(__file__)

        # Diretório acima do diretório do script
        parent_dir = os.path.dirname(script_dir)
        
        # ena = pd.read_csv(os.path.join(script_dir, 'dados',  'ena_atual.csv'))
        # carga = pd.read_excel(os.path.join(script_dir, 'dados',  'historico_carga.xlsx'))
        # tna = pd.read_csv(os.path.join(script_dir, 'dados',  'INDICE_TNA_MENSAL.csv'))
        # cmo = pd.read_excel(os.path.join(script_dir, 'dados',  'cmo_semanal.xlsx')) # nao vai ser usado, nao tem previsao
        # ibc = pd.read_csv(os.path.join(script_dir, 'dados',  'ibc_Br.csv'))         # nao vai ser usado, nao tem previsao
        # ipca = pd.read_csv(os.path.join(script_dir, 'dados',  'historico_ipca.csv'))
        # pld= pd.read_excel(os.path.join(script_dir, 'dados',  'Piso_teto_PLD.xlsx'))

        ena = pd.read_csv(os.path.join(script_dir, 'dados',  'ena_atual.csv'))
        carga = pd.read_csv(os.path.join(script_dir, 'dados',  'historico_carga.csv'))
        ipca = pd.read_csv(os.path.join(script_dir, 'dados',  'historico_ipca.csv'))
        pld = pd.read_csv(os.path.join(script_dir, 'dados',  'Piso_teto_PLD.csv'))
        

        #preco = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/IFSC_longo_prazo/rolloff_suavizado_M+1_SE_-_VWAP.csv')
        
        # obter o caminho acima do script dir
        
        
        preco = pd.read_csv(os.path.join('app', 'curto_prazo', 'dados_preco', 'linear interpol', 'interpolacao linear', f'rolloff suavizado M+{str(yy)} SE -> VWAP.csv'))   #Alteração_1
        preco = preco.drop('Unnamed: 0', axis = 1)

        #### tratamento dos dados

        dates = pd.DataFrame(preco, columns=['data'])
        dates['index'] = range(preco.shape[0])
        dates['data'] = pd.to_datetime(dates['data'], format='%Y-%m-%d')
        dates = dates.sort_values(by='data')

        Prices = preco['VWAP'].iloc[dates['index']]
        a = np.isnan(Prices)
        dates = dates['data']
        Prices = np.interp(dates, dates[a == 0], Prices[a == 0])
        prices = Prices

        Returns = np.log(prices)

        dates_price_max = dates.max()
        dates_price_min = dates.min()

        #### ena
        datesB = pd.DataFrame(ena, columns=['data'])
        datesB['data'] = pd.to_datetime(datesB['data'], format='%Y-%m-%d')
        ENA = pd.DataFrame(ena, columns=['01_sudeste'])

        datesB = datesB['data']
        ENA = ENA['01_sudeste']

        datesENA_max = datesB.max()
        datesENA_min = datesB.min()

        ena = np.interp(dates, datesB, ENA)

        #### ipca

        datesB = pd.DataFrame(ipca, columns=['data'])
        datesB['data'] = pd.to_datetime(datesB['data'], format='%d/%m/%Y')
        datesB = datesB['data']
        
        IPCA = ipca['valor']

        dates_IPCA_max = datesB.max()

        dates_IPCA_min = datesB.min()

        ipca = np.interp(dates, datesB, IPCA)

        #### carga

        datesB = pd.DataFrame(carga, columns=['data'])
        datesB['data'] = pd.to_datetime(datesB['data'], format='%d/%m/%Y')

        aux=carga['id_subsistema']

        coord=[]

        for ii in  range(aux.shape[0]):
            if aux[ii] == 1:
                coord= np.concatenate((coord,ii), axis=None)

        datesB = datesB.iloc[coord]

        auxB=carga['valor']

        auxB = auxB.iloc[coord]

        datesB['index'] = range(auxB.shape[0])
        datesB = datesB.sort_values(by='data')
        datesCarga = datesB['data']
        Carga = auxB.iloc[datesB['index']]

        dates_Carga_max = datesCarga.max()
        dates_Carga_min = datesCarga.min()

        dates_min = max(dates_price_min, dates_IPCA_min, datesENA_min, dates_Carga_min)
        dates_max = min(dates_price_max, dates_IPCA_max, datesENA_max, dates_Carga_max)

        datesIbc = pd.date_range(start= dates_min, end= dates_max, freq='MS')           # datas disponíveis em meses

        datesIbc = pd.Series(datesIbc, name="data")

        #### pld

        PLD = np.zeros((len(datesIbc),2))

        for ii in range(len(datesIbc)):
            if datesIbc.iloc[ii].year <= pld.iloc[0,0]:
                PLD[ii,:]= np.log(pld.iloc[0,1:3])
            else:
                ss=1
                while datesIbc.iloc[ii].year > pld.iloc[ss,0] and ss < pld.shape[0] - 1:
                    ss = ss+1
                PLD[ii,:]= np.log(pld.iloc[ss,1:3])

        #### Colocando os dados no mesmo índice e janela temporal

        PriceM = np.interp(datesIbc, dates,prices)
        EnaM = np.interp(datesIbc, dates,ena)
        CargaM = np.interp(datesIbc, datesCarga,Carga)
        IpcaM = np.interp(datesIbc, dates,ipca)

        ################################# Dados com previsao: importar e organizar

        #### Organizacao da ENA prevista

        colnames = ['data'] + [str(i) for i in range(1, 311)]    # Rafael/Pedro: Tem alguma maneira de resolver essa parte? Seria mais fácil se a ena_prevista já viesse com nomes nas colunas.
        aux2 = pd.read_csv(os.path.join(script_dir, 'dados', 'ena_prevista.csv'), header=None, names = colnames)

        datesEnaPred = pd.DataFrame(aux2, columns=['data'])
        datesEnaPred['data'] = pd.to_datetime(datesEnaPred['data'], format='%m/%d/%Y')

        datesEnaPred = datesEnaPred['data']

        EnaPredA = aux2.values[:,1:]

        ll=0
        coord = np.zeros(len(datesIbc))

        for ii in range(len(datesEnaPred)):
            if datesEnaPred[ii] == datesIbc.iloc[ll] +  timedelta(days=1):   #HINT: timedelta
                coord[ll] = ii
                ll = ll+1
            if ll >= len(datesIbc):
                break

        coord = coord.astype(int)

        EnaPredB = EnaPredA[coord,:].astype(float)
        EnaPred = np.zeros((len(datesIbc),10))

        for ii in range(len(datesIbc)):
            datesOrig = pd.Series(pd.date_range(start = datesIbc.iloc[ii], end = datesIbc.iloc[ii] + timedelta(days = EnaPredB[ii, :].shape[0]), freq = 'D'))
            datesPred = pd.Series(pd.date_range(start = datesIbc.iloc[ii], periods = 10, freq = 'MS'))
            EnaPred[ii,:] = np.interp(datesPred, datesOrig, np.concatenate((EnaM[ii],EnaPredB[ii,:]), axis = None))

        # Eliminando NaN

        for ii in range(EnaPred.shape[1]):       #NaN                                          #
            for jj in range(EnaPred.shape[0]):   #NaN
                if np.isnan(EnaPred[jj, ii]) and jj > 12: #NaN
                    EnaPred[jj, ii] = EnaPred[jj - 12, ii] #NaN

        #### Organizacao da Carga Prevista
        ### Sudeste=3

        aux2 = pd.read_csv(os.path.join(script_dir, 'dados', 'carga_prevista.csv'))
        aux = aux2['Submercado']

        CargaPredA = aux2.iloc[:,3:-1]

        coord = []

        for ii in range(aux.shape[0]):
            if aux[ii] == 3:
                coord = np.concatenate((coord,ii), axis = None)

        CargaPredA = CargaPredA.iloc[coord,:]

        CargaPredB = np.zeros((round(CargaPredA.shape[0]/5),5*CargaPredA.shape[1]))

        for ii in range(round(CargaPredA.shape[0]/5)):
            CargaPredB[ii,:] = np.concatenate((CargaPredA.iloc[ii*5,:],CargaPredA.iloc[ii*5+1,:],CargaPredA.iloc[ii*5+2,:],CargaPredA.iloc[ii*5+3,:],CargaPredA.iloc[ii*5+4,:]), axis = 0)

        CargaPredB = CargaPredB[7:,:]

        CargaPred = np.zeros((len(datesIbc),10))

        for ii in range(len(datesIbc)):
            CargaPred[ii,:] = np.concatenate((CargaM[ii],  CargaPredB[datesIbc.iloc[ii].year-2014 - 1, np.arange(datesIbc.iloc[ii].month, datesIbc.iloc[ii].month + 9)]), axis = None)


        ##### Organizacao do IPCA

        aux2 = pd.read_csv(os.path.join(script_dir, 'dados', 'expectativa_ipca.csv'))

        ipcaPred = aux2.iloc[:,1:]

        datesB = pd.DataFrame(aux2, columns=['Data'])
        datesB['Data'] = pd.to_datetime(datesB['Data'], format='%Y-%m-%d')

        datesB = datesB['Data']

        coord = np.zeros(datesIbc.shape)

        for jj in range(len(datesIbc)):
            aux = np.argmin(np.abs(datesB-datesIbc.iloc[jj]))
            coord[jj] = aux+1

        IpcaPred = np.zeros((len(IpcaM),37))
        IpcaPred[:,0] = IpcaM
        IpcaPred[:,1:] = ipcaPred.iloc[coord,0:36]


        Sizes = np.concatenate((CargaPred.shape[1],EnaPred.shape[1],IpcaPred.shape[1]), axis = None)
        Len = min(Sizes)
        CargaPred = CargaPred[:,0:Len]
        EnaPred = EnaPred[:,0:Len]
        IpcaPred = IpcaPred[:,0:Len]


        #################################################### Inicialização das variáveis

        LEN = 30
        lenP = Len-1
        H = np.array([100, 100, 600, 500])

        VAR05 = np.zeros((len(datesIbc)-LEN+1,lenP,2)) #Lucas
        CVAR05 = np.zeros((len(datesIbc)-LEN+1,lenP,2)) #Lucas

        MR_VAR05 = np.zeros((len(datesIbc)-LEN+1,lenP,2)) #Lucas
        MR_CVAR05 = np.zeros((len(datesIbc)-LEN+1,lenP,2)) #Lucas

        L_MRJ = np.zeros((len(datesIbc)-LEN+1, lenP+1)) #Lucas
        M_MRJ = np.zeros((len(datesIbc)-LEN+1, lenP+1)) #Lucas
        U_MRJ = np.zeros((len(datesIbc)-LEN+1, lenP+1)) #Lucas

        L_MR = np.zeros((len(datesIbc)-LEN+1, lenP+1)) #Lucas
        M_MR = np.zeros((len(datesIbc)-LEN+1, lenP+1)) #Lucas
        U_MR = np.zeros((len(datesIbc)-LEN+1, lenP+1)) #Lucas


        for ll in np.arange(lenP,len(datesIbc)-LEN+1): #Lucas

            #### Nondimensional data
            lPld = PLD[ll:ll+lenP + 1,:]

            while lPld.shape[0] < lenP + 1:
                lPld = np.concatenate((lPld, [lPld[-1,:]]), axis=0)

            ipcaPred = IpcaPred[ll,:]

            while len(ipcaPred) < lenP + 1:
                ipcaPred = np.concatenate((ipcaPred, [ipcaPred[-1]]), axis=0)

            datesPred = pd.Series(pd.date_range(start = datesIbc.iloc[LEN-1+ll], periods = 10, freq = 'MS'))
            lenn=20

            ############################################################# Regressão de Ridge

            while len(EnaM) < LEN+ll+lenP:
                EnaM = np.concatenate((EnaM, [EnaM[-1]]), axis=0)

            while len(CargaM) < LEN+ll+lenP:
                CargaM = np.concatenate((CargaM, [CargaM[-1]]), axis=0)

            while len(IpcaM) < LEN+ll+lenP:
                IpcaM = np.concatenate((IpcaM, [IpcaM[-1]]), axis=0)

            AUX = np.zeros((LEN+ll+lenP,len(Sizes)))
            AUX[:,0]= EnaM[0:LEN-1+ll+lenP+1]
            AUX[:,1]= CargaM[0:LEN-1+ll+lenP+1]
            AUX[:,2]= IpcaM[0:LEN-1+ll+lenP+1]

            PriceT = PriceM

            while len(PriceT) < AUX.shape[0]:
                PriceT = np.concatenate((PriceT, [PriceT[-1]]), axis=0)

            ## Preparando dado de treino

            X = AUX[0:-lenP,:]
            T = np.log(PriceT[0:AUX.shape[0]-lenP])
            y_price = T[lenn:]
            x_window = window(lenn, X)

            X_train, X_test, y_train, y_test = train_test_split(x_window, y_price, test_size=0.1, random_state=42)

            scaler_X = MinMaxScaler()
            normalized_X_train = scaler_X.fit_transform(X_train) # Normalizando os dados de treino
            normalized_X_test = scaler_X.transform(X_test) # Aplicando a mesma transformação aos dados de teste

            # Definindo o modelo para ll específico
            print('ll=', ll)
            model = model_Ridge(normalized_X_train, y_train, normalized_X_test, y_test)[1]

            # Definindo y: predição do modelo no dado de treino unido ao dado de teste
            scaler_T = MinMaxScaler()
            normalized_X_T = scaler_T.fit_transform(x_window)  # (OBS: x_window corresponde ao X de treino e ao X de teste juntos)
            y = model.predict(normalized_X_T)
            print('Gráfico 1 = Predição do modelo no dado de treino unido ao dado de teste ')
            print(mean_squared_error(y_price, y))
            show_result(y_price, y)

            ## Definindo y2: Predição para 9 meses (LenP)
            new_p = np.concatenate((EnaPred[LEN+ll-1,1:lenP+1].reshape(-1,1), CargaPred[LEN+ll-1,1:lenP+1].reshape(-1,1), IpcaPred[LEN+ll-1,1:lenP+1].reshape(-1,1)), axis =1)
            X_P = np.concatenate((AUX[lenP:-lenP,:],new_p), axis=0)
            T_P = np.log(PriceT[lenP:AUX.shape[0]])
            y_price_P = T_P[lenn:]
            x_window_P = window(lenn, X_P)
            scaler_p = MinMaxScaler()
            normalized_X_pred = scaler_p.fit_transform(x_window_P)
            y2 = model.predict(normalized_X_pred)
            y2 = y2*y[-1]/y2[-lenP-1]


            if ll < len(datesIbc)-LEN - lenP:

                print('Gráfico 2: Predição de 9 meses')
                print(mean_squared_error(y_price_P, y2))
                show_result(y_price_P, y2)

            else:

                print('Gráfico 2: Predição de 9 meses')
                y_price_M = np.log(PriceM[lenP:AUX.shape[0]])[lenn:]
                print(mean_squared_error(y_price_M, y2[:len(y_price_M)]))
                show_result(y_price_M, y2)


            ############################################ Mean-Reverting (Preparação do dado)

            residual = np.log(PriceT[lenn: AUX.shape[0] - lenP]) - y
            P = PriceT[lenn:AUX.shape[0]-lenP]
            P0 = P[-1]
            data = residual


            ###################################################### Mean-Reverting with Jumps

            #MRJ_Fit_20240412

            S0 = data[0]
            alpha = 1E-3;
            coefLMRJ0= np.array([0.01,0,0.4,0.1,0.1,0.1])
            LBLMRJ = np.array([0,0,0.01,0,-1000,0])
            UBLMRJ = np.array([20,20,100,1000,1000,1000])

            t=np.arange(1,len(residual))
            t = t/12
            dt=1/12

            OF = lambda coef: ObjFunMRLJumps_20230629Returns(data, coef,coefLMRJ0, t, dt, S0, alpha)
            coefLMRJ = least_squares(OF, coefLMRJ0, bounds=(LBLMRJ, UBLMRJ)).x

            ### MRJ_Error Evaluation

            mrj_error = MRJ_Error_Evaluation(coefLMRJ, lenP, data, dt, y, y2, lPld)

            PredMb = mrj_error[1]
            PredUb = mrj_error[3]
            PredLb = mrj_error[5]

            MRJ_plot(datesIbc, AUX, PriceM, datesPred, PredMb, PredUb, PredLb, lPld, ll, LEN, lenP, yy)

            ###################################################### MRJ_Scenarios_20240424

            mrj_scen = MRJ_Scenarios_20240424(coefLMRJ, IpcaPred, lenP, dt, y2, P0, lPld, ll)

            signLMRJ = mrj_scen[1]
            L_MRJ[ll,:] = mrj_scen[3]
            M_MRJ[ll,:] = mrj_scen[5]
            U_MRJ[ll,:] = mrj_scen[7]


            #Risco_20240412

            risco_mrj = Risco_20240412(signLMRJ)

            VAR05[ll,:,0]= risco_mrj[1] #var05
            CVAR05[ll,:,0]= risco_mrj[3] #cvar05

            ############################################################ Mean-Reverting-Type

            # MR_Fit_20240412

            S0 = data[0]
            alpha = 1E-3
            coefMR0= np.array([0.01,0,0.4])
            LBMR = np.array([0,0,0.1])
            UBMR = np.array([20,20,100])

            t=np.arange(1,len(residual))
            t = t/12
            dt=1/12
            data = residual

            ObjFun = lambda coef: ObjFunHeston_20240207Returns(data, coef,coefMR0, t, dt, S0, alpha)
            coefMR = least_squares(ObjFun, coefMR0, bounds=(LBMR, UBMR)).x

            ### MR_Error Evaluation

            mr_error =  MR_Error_Evaluation(coefMR, lenP, data, dt, y, y2, lPld)

            PredM2b = mr_error[1]
            PredU2b = mr_error[3]
            PredL2b = mr_error[5]

            MR_plot(datesIbc, AUX, PriceM, datesPred, PredM2b, PredU2b, PredL2b, lPld, ll, LEN, lenP, yy)

            ##################################################### MR_Scenarios_20240424

            mr_scen = MR_Scenarios_20240424(coefMR, IpcaPred, lenP, dt, y2, P0, lPld, ll)

            signMR = mr_scen[1]
            L_MR[ll,:] = mr_scen[3]
            M_MR[ll,:] = mr_scen[5]
            U_MR[ll,:] = mr_scen[7]


            # RiscoMR_20240412

            risco_mr = Risco_20240412(signMR)

            MR_VAR05[ll,:,0]= risco_mr[1] #var05
            MR_CVAR05[ll,:,0]= risco_mr[3] #cvar05


        objetos_M[yy] = {
            'datesIbc': datesIbc,
            'VAR05': VAR05,
            'CVAR05': CVAR05,
            'MR_VAR05': MR_VAR05,
            'MR_CVAR05': MR_CVAR05
        }

        print(f"Completed iteration for yy = {yy}")

    ##############################################################  Gráfico de Análise de Risco



        # Converter datas para strings sem horas
        #date_text = df_dados['date'].dt.strftime('%Y-%m-%d')

        # Criar o gráfico de barras

    def plot_risk(VAR, CVAR, jj, k, LEN, datesIbc, Title, yy=yy, index=""):
        fig = go.Figure()

        # Converter valores para strings com uma casa decimal
        high_mark_text = [f'{x:.1f}%' for x in VAR[jj,:,k]]
        low_mark_text = [f'{x:.1f}%' for x in CVAR[jj,:,k]]

        datesPred = pd.Series(pd.date_range(start=datesIbc.iloc[LEN-1+jj], periods=10, freq='MS'))

        # Adicionar barras para 'tendencia de alta' e 'tendencia de baixa' com o valor no topo
        fig.add_trace(go.Bar(x=datesPred, y=VAR[jj,:,k], name='VaR 5%',
                             text=high_mark_text, textposition='outside', marker_color='blue'))
        fig.add_trace(go.Bar(x=datesPred, y=CVAR[jj,:,k], name='cVaR 5%',
                             text=low_mark_text, textposition='outside', marker_color='green'))

        # Calcular o valor absoluto máximo para configurar o eixo y simétrico
        max_value = max(abs(VAR[jj, :, k].max()), abs(CVAR[jj, :, k].min())) + 2

        # Atualizar layout
        fig.update_layout(
            title=Title,
            xaxis_title='Data',
            yaxis_title='Valor (%)',
            barmode='group',
            hovermode='x',
            xaxis=dict(
                tickmode='array',
                tickvals=datesPred,
                tickangle=-30,
                zeroline=True,
                zerolinecolor='black',
                tickformat='%d/%m',
            ),
            yaxis=dict(range=[-max_value * 1.2, max_value * 1.2], zeroline=True, zerolinecolor='black'),  # Ajuste dinâmico
            width=1000,
            height=800
        )

        # Get the date of the start of the prediction
        data_plot = datesPred[0].strftime('%d/%m/%Y')

        output_dir = os.path.join('web', 'static', 'tasks_saida', 'longo_prazo', f"M+{yy}", 'risco')
        fig_json = to_json(fig)
        os.makedirs(os.path.join(output_dir), exist_ok=True)
        if index:
            nome_arquivo = f"longo_prazo_{data_plot.replace(' ', '_').replace('-', '_').replace('/', '_')}_{index}.json"
        else:
            nome_arquivo = f"longo_prazo_{data_plot.replace(' ', '_').replace('-', '_').replace('/', '_')}.json"
        with open(os.path.join(output_dir, nome_arquivo), 'w') as f:
            json.dump(fig_json, f)
                
        print(f"Gráfico dash salvo em: {os.path.join(output_dir, nome_arquivo)}")

    # Gráfico da análise de risco para cada data

    for yy in range(4): #Alteração_1

        datesIbc = objetos_M[yy]['datesIbc']     #Alteração_1
        VAR05 = objetos_M[yy]['VAR05'] #Alteração_1
        CVAR05 = objetos_M[yy]['CVAR05'] #Alteração_1
        MR_VAR05 = objetos_M[yy]['MR_VAR05'] #Alteração_1
        MR_CVAR05 = objetos_M[yy]['MR_CVAR05'] #Alteração_1

        for jj in np.arange(lenP,len(datesIbc)-LEN+1):  #Lucas

            print(f"ll: {jj}")

            #MRJ_Scenarios_20240424
            plot_risk(VAR05, CVAR05, jj, 0, LEN, datesIbc, Title = "Análise de Risco (Modelo Estocástico para o Longo Prazo com Saltos)", yy=yy, index="salto")

            ## MR_Scenarios_20240424
            plot_risk(MR_VAR05, MR_CVAR05, jj, 0, LEN, datesIbc, Title = "Análise de Risco (Modelo Estocástico para o Longo Prazo)", yy=yy, index="")

        print(f"Completed iteration for yy = {yy}") # Log successful iteration  #Alteração_1