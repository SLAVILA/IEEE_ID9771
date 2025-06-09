# As funções necessárias estão no arquivo functions.py devido ao fato das bibliotecas não funcionarem corretamente com o Jupyter Notebook
import os.path 
import pandas as pd
import multiprocessing as mp
import functions as f # biblioteca de funções
import streamlit as st
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load and prepare the data (simplified for demonstration)
@st.cache_data  # Cache the data loading for performance
def load_data():
    data = pd.read_csv('rolloff diferenca cumulativa M+2 SE -_ hip_2.csv', index_col="data", parse_dates=True)
    data = data.rename(columns={'VWAP': 'close'})
    return data


def main():
    st.title("Modelo de Análise Técnica e Aprendizado de Máquina (MATAM)")
    data = load_data()

    #Definidas as estratégias: 
    #- Simple Moving Average (SMA),  Bollinger Bands (BBANDS)
    # First two buttons centered
    col1, col2, col3, col4 = st.columns([1,2,2,1])
    with col2:    
        if st.button("1º Passo - Iniciar Backtest de Análise Técnica"):
            f.estrategias_gridsearch(data)
            f.estrategias_otimas(data)
            f.create_dataframes()
            st.write("Backtest iniciado")
    
    with col3:
        if st.button("2º Passo - Iniciar Machine Learning"):
            f.execute_machine_learning(data)
            st.write("Machine Learning iniciado")        
    
    #######################################################################################################################
    #                                                                                                                     #
    #             Lê os dados unificados e divididos                                                                      #
    #             dataframe_results = pd.read_csv(os.path.join('ML_data', 'scores_dataframe.csv'), index_col=1)           #
    #             train_scores = pd.read_csv(os.path.join('ML_data', 'scores_train.csv'), index_col=1)                    #
    #             test_scores = pd.read_csv(os.path.join('ML_data', 'scores_test.csv'), index_col=1)                      #
    #                                                                                                                     #
    #######################################################################################################################

    # Plotting section
    st.title("Gráficos dos trades")

    # Row 1 with 3 buttons
    row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns(5)
    with row1_col1:
        if st.button("Regressão Linear"):
            linear_regression = 'linear_regression'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_regressions(r2df, linear_regression))

    with row1_col2:
        if st.button("Regressão de Crista"):
            ridge_regression = 'ridge_regression'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_regressions(r2df, ridge_regression))

    with row1_col3:
        if st.button("Regressão LGBM"):
            lgbm_regression = 'lgbm_regression'        
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_regressions(r2df, lgbm_regression))

    with row1_col4:
        if st.button("Regressão XGBoost"):
            xgboost_regression = 'xgboost_regression'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_regressions(r2df, xgboost_regression))

    with row1_col5:
        if st.button("Regressão Logistica"):
            logistic_regression = 'logistic_regression'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_classifier(r2df, logistic_regression))

    # Row 3 with 3 buttons
    row3_col1, row3_col2, row3_col3, row3_col4, row3_col5 = st.columns(5)

    with row3_col1:
        if st.button("Classificação LGBM"):
            lgbm_classifier = 'lgbm_classifier'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_classifier(r2df, lgbm_classifier))

    with row3_col2:
        if st.button("Classificação XGBoost"):
            xgboost_classifier = 'xgboost_classifier'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)        
            st.plotly_chart(f.plot_classifier(r2df, xgboost_classifier))

    with row3_col3:
        if st.button("Gaussian Naive Bayes"):
            gnb = 'gnb'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_classifier(r2df, gnb))

    with row3_col4:
        if st.button("Random Forest"):
            random_forest= 'random_forest'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_classifier(r2df, random_forest))

    with row3_col5:  # Use the second column for the first button to center them
        if st.button("Support Vector Classifier"):
            svc = 'svc'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_classifier(r2df, svc))

    # Row 4 with 2 buttons (to center these buttons, use a trick with empty columns)
    row4_col2, row4_col3 = st.columns(2)

    with row4_col2:  # Use the fourth column for the second button to center them
        if st.button("Linear SVC"):
            linear_svc = 'linear_svc'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_classifier(r2df, linear_svc))

    with row4_col3:
        if st.button("Mostrar todos"):
            linear_regression = 'linear_regression'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_regressions(r2df, linear_regression))
            ridge_regression = 'ridge_regression'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_regressions(r2df, ridge_regression))
            lgbm_regression = 'lgbm_regression'        
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_regressions(r2df, lgbm_regression))
            xgboost_regression = 'xgboost_regression'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_regressions(r2df, xgboost_regression))
            logistic_regression = 'logistic_regression'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_classifier(r2df, logistic_regression))
            lgbm_classifier = 'lgbm_classifier'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_classifier(r2df, lgbm_classifier))
            xgboost_classifier = 'xgboost_classifier'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)        
            st.plotly_chart(f.plot_classifier(r2df, xgboost_classifier))
            gnb = 'gnb'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_classifier(r2df, gnb))
            random_forest= 'random_forest'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_classifier(r2df, random_forest))
            svc = 'svc'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_classifier(r2df, svc))
            linear_svc = 'linear_svc'
            r2df = pd.read_csv(os.path.join('ML_data', 'round_2_dataframe.csv'), index_col=0, parse_dates=True)
            st.plotly_chart(f.plot_classifier(r2df, linear_svc))

# Ensure the script runs only when executed directly (not imported as a module)
if __name__ == "__main__":
    main()
