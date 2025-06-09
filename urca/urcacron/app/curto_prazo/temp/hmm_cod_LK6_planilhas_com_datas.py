# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:59:16 2024

@author: Dell
"""

import os
import re
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# tratamento dos arquivos de input 


# Diretório contendo os arquivos CSV
dir_path = r'C:/Users/Dell/Desktop/P&D Urca/dados_preco_atualizado/raw/'
df_iteracoes_list = []

#########################################
# Listar todos os arquivos no diretório
files = os.listdir(dir_path)

# Iterar sobre cada arquivo no diretório
for file_name in files:
    # Verificar se o arquivo é um CSV
    if file_name.endswith('.csv'):
        # Substituir espaços em branco, '-' e '+' no nome do arquivo
        new_file_name = file_name.replace(' ', '_').replace('-', '_').replace('+', 'm').replace('.csv', '.xlsx')
        
        # Caminho completo do arquivo CSV antigo e novo arquivo XLSX
        csv_file_path = os.path.join(dir_path, file_name)
        xlsx_file_path = os.path.join(dir_path, new_file_name)
        
        # Ler o arquivo CSV
        df = pd.read_csv(csv_file_path)
        
        # Salvar como arquivo XLSX
        df.to_excel(xlsx_file_path, index=False)

print("Conversão concluída.")

# Caminho do diretório contendo os arquivos
directory = r'C:/Users/Dell/Desktop/P&D Urca/dados_preco_atualizado/raw/'

# Lista dos arquivos a serem processados
files = [
    'rolloff_suavizado_Mm0_SE____VWAP.xlsx',
    'rolloff_suavizado_Mm1_SE____VWAP.xlsx',
    'rolloff_suavizado_Mm2_SE____VWAP.xlsx',
    'rolloff_suavizado_Mm3_SE____VWAP.xlsx'
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
    output_file_path = os.path.join(directory, file)
    
    # Salvar o DataFrame reorganizado em um novo arquivo Excel
    df.to_excel(output_file_path, index=False)
    
    print(f"As colunas do arquivo {file} foram reorganizadas e salvas em {output_file_path}")

# Diretório onde estão os arquivos Excel
diretorio = r"C:/Users/Dell/Desktop/P&D Urca/dados_preco_atualizado/raw/"
diretorio_com_data = r'C:/Users/Dell/Desktop/P&D Urca/dados_de_preço_com_data/'
# Expressão regular para identificar arquivos já renomeados corretamente
regex_data = re.compile(r'\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}')

# Função para listar e renomear arquivos Excel, ignorando os que já estão no formato correto
def renomear_arquivos_excel_com_data(diretorio):
    # Iterar sobre os arquivos na pasta
    for arquivo in os.listdir(diretorio):
        # Verificar se o arquivo é uma planilha Excel (extensão .xls ou .xlsx)
        if arquivo.endswith('.xls') or arquivo.endswith('.xlsx'):
            # Verificar se o arquivo já está no formato correto (com data no nome)
            if regex_data.search(arquivo):
                print(f"Arquivo já renomeado corretamente: {arquivo} - Ignorado")
                continue
            
            # Verificar se o arquivo correspondente já existe no diretório de saída
            arquivo_base = arquivo[:arquivo.rfind('.')]  # Remover a extensão
            novo_nome_base = f"{arquivo_base}"  # Prefixo
            
            # Listar arquivos existentes no diretório de saída
            arquivos_existentes = os.listdir(diretorio_com_data)
            existe = any(novo_nome_base in existing for existing in arquivos_existentes)

            if existe:
                print(f"Arquivo já existente: {novo_nome_base} - Ignorado")
                continue
            
            caminho_completo = os.path.join(diretorio, arquivo)
            
            # Obter a data de modificação do arquivo
            data_modificacao = os.path.getmtime(caminho_completo)
            data_formatada = datetime.fromtimestamp(data_modificacao).strftime('%d_%m_%Y_%H_%M_%S')
            
            # Separar o nome do arquivo de sua extensão
            nome_sem_extensao, extensao = os.path.splitext(arquivo)
            
            # Criar o novo nome do arquivo com a data de modificação
            novo_nome = f"{nome_sem_extensao}_{data_formatada}{extensao}"
            caminho_novo_nome = os.path.join(diretorio_com_data, novo_nome)
            
            # Renomear o arquivo no diretório
            os.rename(caminho_completo, caminho_novo_nome)
            print(f"Arquivo renomeado: {arquivo} -> {novo_nome}")

# Exemplo de uso
renomear_arquivos_excel_com_data(diretorio)


