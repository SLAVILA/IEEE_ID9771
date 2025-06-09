import time
from flask import session, request
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
import base64
from pg import DB, escape_string
import re
from config.database import DATABASE, DATABASE_URCA
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import os
import json
import random
from time import sleep
import pymssql

# Imports do GeradorPDF
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Image,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from datetime import datetime
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import pandas as pd
import os
from datetime import date


# Define color escape codes
class CoresConsole:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'


class query:
    def sanitizar(self, *valores):
        """
        Sanitiza a query removendo caracteres e expressões perigosas, para evitar injeção SQL.

        MOTIVO: Esta função evita alguns ataques SQL, como:
             - admin' OR '1'='1'; --
             - ') OR ('a'='a') OR ('=')`
             - 1; DELETE FROM users;
             - etc...
          Após a sanitização da query, DELETE, ALTER e mais algumas expressões, junto com alguns caracteres, são limpos

        1. Verifica se o item não é nulo, e se for int, transforma em string
        3. Verifica cada caractere do item para verificar se é um caractere perigoso, e reconstrói a string sanitizada
        4. Verificar se a string contém expressões perigosas
        4. Retorna o item sem os caracteres e expressões perigosas

        """
        caracteres_perigosos = [
            # "'",
            '"',
            "\\",
            ";",
            "-",
            "+",
            ",",
            "*",
            "/",
            # "(",
            # ")",
            "[",
            "]",
            "{",
            "}",
        ]
        expressoes_perigosas = [
            "ALTER",
            "CONNECT",
            "DELETE",
            "UPDATE",
            "SELECT",
            "DROP",
            "CREATE",
        ]
        valores_sanitizados = []
        valores_perigosos = []

        # Transforma os valores em uma lista, se não for
        if not isinstance(valores, list):
            if isinstance(valores, str):
                valores = [valores]
            else:
                valores = list(valores)
        # print(f"Valores iniciais da query: {valores}")

        for valor in valores:
            if valor is not None:
                # Verifica se é int, se for, adiciona aos valores sanitizados e continua
                if isinstance(valor, int):
                    valores_sanitizados.append(valor)
                    continue

                string_sanitizada = valor
                perigoso = False

                valor_interno = ""
                valor_para_sanitizar = ""

                # conta o número de '
                aspas = valor.count("'")

                if aspas == 2:
                    # Se for apenas um par de aspas, verifica seu interior
                    primeira_aspa = valor.find("'")
                    ultima_aspa = valor.rfind("'")

                    if primeira_aspa != -1 and ultima_aspa != -1:
                        valor_para_sanitizar = valor[primeira_aspa + 1 : ultima_aspa]

                        # Verifica cada caractere do valor pra sanitizar para verificar se é um caractere perigoso, e reconstrói a string sanitizada
                        for char in valor_para_sanitizar:
                            if char in caracteres_perigosos:
                                perigoso = True
                            else:
                                valor_interno += char

                        # Adiciona as aspas de novo
                        valor_interno = f"'{valor_interno}'"

                        # Concatena a string sanitizada
                        string_sanitizada = (
                            valor[:primeira_aspa]
                            + valor_interno
                            + valor[ultima_aspa + 1 :]
                        )

                elif aspas > 2:
                    # Se tiver mais de um par de aspas, verifica todos os interiores e concatena
                    string_sanitizada = ""

                    string_separada = valor.split(" OR ")

                    for item in string_separada:
                        primeira_aspa = item.find("'")
                        ultima_aspa = item.rfind("'")
                        valor_interno = ""

                        if primeira_aspa != -1 and ultima_aspa != -1:
                            valor_para_sanitizar = item[primeira_aspa + 1 : ultima_aspa]

                            # Verifica cada caractere do valor pra sanitizar para verificar se é um caractere perigoso, e reconstrói a string sanitizada
                            for char in valor_para_sanitizar:
                                if char in caracteres_perigosos:
                                    perigoso = True
                                else:
                                    valor_interno += char

                            # Adiciona as aspas de novo
                            valor_interno = f"'{valor_interno}'"

                            # Este OR é adicionado pois é removido no split(" OR ")
                            or_adicional = (
                                " OR "
                                if string_separada.index(item)
                                != string_separada.index(string_separada[-1])
                                else ""
                            )

                            # Concatena a string sanitizada
                            item = (
                                item[:primeira_aspa]
                                + valor_interno
                                + item[ultima_aspa + 1 :]
                                + or_adicional
                            )
                            string_sanitizada += (
                                f" {item}" if not item.startswith(" ") else item
                            )

                # Verifica se a string contém expressões perigosas usando regex, procurando apenas pelas paravras chaves (ex: alter não é mais detectado em: alterado)
                for expressao in expressoes_perigosas:
                    if re.search(
                        r"\b" + expressao.lower() + r"\b", string_sanitizada.lower()
                    ):
                        perigoso = True
                        break

                # Se for um valor perigoso, provavelmente é uma tentativa de ataque de injeção SQL
                if perigoso:
                    print(
                        f"Perigo. Expressão desconhecida. Provavelmente uma tentativa de ataque de injeção SQL. String do ataque: '{valor_para_sanitizar}'"
                    )
                    string_sanitizada = ""
                    valores_perigosos.append(perigoso)

                try:
                    # Utilizando a função escape_string do pg, adiciona mais uma camada de proteção.
                    # Remove as aspas duplas se houver (pois o escape_strings as vezes as adiciona).
                    string_sanitizada = escape_string(string_sanitizada).replace(
                        "''", "'"
                    )
                except:
                    # Utilizando try-except pois apenas com acentuação (á, é, í, ó, ú) que causa problemas na função escape_string do pg
                    pass

                valores_sanitizados.append(string_sanitizada)

        # Verifica se não há nenhum valor perigoso na lista de valores
        if any(valor_perigoso for valor_perigoso in valores_perigosos):
            perigoso = True

        # print(f"Strings sanitizadas: {valores_sanitizados}")
        return tuple(valores_sanitizados), perigoso


class Banco:
    def __init__(self):
        try:
            self.conn = pymssql.connect(
                server=DATABASE_URCA["DB_SERVER"],
                user=DATABASE_URCA["DB_USER"],
                password=DATABASE_URCA["DB_PASSWORD"],
                database=DATABASE_URCA["DB"]
            )
        except Exception as e:
            print("Erro ao conectar ao banco de dados:", e)
            self.conn = None

    def sql(self, query, values, dic=1, log_tempo=False):
        if self.conn is None:
            return [{"erro": 1}]
        
        start_time = time.time()  # Record the start time
        
        # Define a regular expression pattern to match harmful SQL keywords
        harmful_keywords_pattern = re.compile(r'\b(UPDATE|DELETE|DROP|ALTER|TRUNCATE)\b', re.IGNORECASE)
        
        # Check if the query contains any harmful keywords
        if harmful_keywords_pattern.search(query):
            print(f"Operação não permitida. Query: {query}")
            return [{"erro": "Operação não permitida."}]
        
        try:
            with self.conn.cursor(as_dict=(dic == 1)) as cursor:
                cursor.execute(query, values)
                resposta = cursor.fetchall()
        except Exception as e:
            print("Erro ao executar a query:", e)
            resposta = None
        
        if log_tempo:
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f"Query demorou: {elapsed_time} segundos")
        
        return resposta
    
    def get_tables(self):
        # Consulta para recuperar todas as tabelas do banco de dados
        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        # Executar a consulta
        return [row['TABLE_NAME'] for row in self.sql(query, ())]
    
    def get_table(self, table: str):
        # Consulta para recuperar todas as tabelas do banco de dados
        query = f"SELECT * FROM urca.{table};"
        # Executar a consulta
        return self.sql(query, ())

    def close(self):
        if self.conn is not None:
            self.conn.close()
    
    def get_precos(self):
        from biblioteca.db_api import UrcaDb 
    
        URCA = UrcaDb()
        preco_h = URCA.get_table("preco_bbce_men_preco_fixo_diario")
        
        preco_r = preco_h
        preco = URCA.get_table('preco_bbce_men_preco_fixo')
        preco.sort_values(by = 'data', inplace = True)
        preco_h.sort_values(by = 'data', inplace = True)
        preco = preco[preco['produto'].str.contains("NO|SU|NE|12|13|14") == False]
        preco_h = preco_h[(preco_h['submercado'] == 'SE') &
                          (preco_h['data'] >= pd.to_datetime(date(2015, 1, 1)))]
        precobbce = URCA.get_table('preco_bbce_teste')
        subsistema = URCA.get_table('subsistema')
        historico_hidrologia = URCA.get_table('historico_hidrologia')
        return preco, preco_h, preco_r, precobbce, subsistema, historico_hidrologia
    
    def get_pld(self):
        from biblioteca.db_api import Local
        
        local = Local()
        
        pld_piso_teto = local.get_table('piso_teto_pld')
        PLD = local.get_table('PLD_mensal')
        PLD['data'] = pd.to_datetime(PLD['data'], format='%Y-%d-%m')
        pld = PLD[PLD['data'].dt.year >= 2015]
        pld['data'] = pld['data'].dt.strftime('%Y-%m-%d')
        return pld, pld_piso_teto, PLD


class banco:
    def __init__(self):
        try:
            self.db = DB(
                dbname=DATABASE["db"],
                host=DATABASE["host"],
                user=DATABASE["user"],
                passwd=DATABASE["password"],
            )
        except:
            self.db = None

    def sql(self, query, values, dic=1, log_tempo=False):
        start_time = time.time()  # Record the start time
        # print(f"Query: {query}")
        if self.db == None:
            return [{"erro": 1}]
        q = self.db.query(query, values)
        if dic == 1:
            resposta = q.dictresult()
        else:
            resposta = None
        self.db.close()
        
        if log_tempo:
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f"Query demorou: {elapsed_time} segundos")
        return resposta

    
    
    def mudar_banco(self, database, url, host=None):
        # Update the database parameters
        self.db = DB(
            dbname=database,
            host=DATABASE["host"] if host is None else host,
            user=DATABASE["user"],
            passwd=DATABASE["password"],
        )
        
        # Define the new database configuration
        new_config = {
            "db": database,
            "host": DATABASE["host"] if host is None else host,
            "user": DATABASE["user"],
            "password": DATABASE["password"]
        }

        # Path to the database configuration file
        config_file_path = "config/database.py"
        
        # Write the new configuration to the file
        with open(config_file_path, "w") as f:
            f.write("DATABASE = " + json.dumps(new_config, indent=4))

        # Optionally, you can execute additional actions after updating the file
        os.system(f"/etc/rc.d/{url} reload")


class bancocustom(banco):
    def __init__(self, b):
        try:
            self.db = DB(
                dbname=b,
                host=DATABASE["host"],
                user=DATABASE["user"],
                passwd=DATABASE["password"],
            )
        except:
            self.db = None

    def sql(self, query, values, dic=1):
        if self.db == None:
            return [{"erro": 1}]
        q = self.db.query(query, values)
        if dic == 1:
            resposta = q.dictresult()
        else:
            resposta = None
        self.db.close()
        return resposta


class bancocliente:
    def __init__(self):
        try:
            self.db = DB(
                dbname=session["usuario"]["db"],
                host=session["usuario"]["host"],
                user=session["usuario"]["user"],
                passwd=session["usuario"]["password"],
            )
        except:
            self.db = None

    def sql(self, query, values, dic=1):
        if self.db == None:
            return [{"erro": 1}]
        q = self.db.query(query, values)
        if dic == 1:
            resposta = q.dictresult()
        else:
            resposta = None
        self.db.close()
        return resposta



class check:
    def check_modulos(self):
        pass


class config:
    def cliente(self, fk_id_cliente=0, chave=""):
        if chave == "":
            resposta = banco().sql(
                "select * from config_cliente where fk_id_cliente=$1", (fk_id_cliente)
            )
        else:
            resposta = banco().sql(
                "select * from config_cliente where fk_id_cliente=$1 and str_nome=$2",
                (fk_id_cliente, chave),
            )
        config_cliente = {}
        for i in resposta:
            config_cliente[i["str_nome"]] = i["str_valor"]
        if not chave == "":
            resposta = banco().sql(
                "select str_nome from cliente where id=$1", (fk_id_cliente)
            )
            config_cliente["nome_cliente"] = resposta[0]["str_nome"]
        return config_cliente

    def verificar_config(self, id_cliente):
        variaveis = [
            "auth_app"
        ]

        variaveis_txt = [
            "nome_custom_1"
        ]

        query = (
            "INSERT INTO config_cliente (str_nome, str_valor, fk_id_cliente) (VALUES"
        )

        # Adiciona as variáveis ao novo cliente, em loop, para fazer apenas um query ao banco de dados
        for variavel in variaveis:
            query += f"('{variavel}', '0', $1),"

        for variavel in variaveis_txt:
            query += f"('{variavel}', '', $1)"

            query += (
                "," if variavel != variaveis_txt[-1] else ""
            )  # Adiciona uma virgula depois de cada variável, menos na ultima

        query += ") ON CONFLICT (str_nome, fk_id_cliente) DO NOTHING"

        #banco().sql(query, (id_cliente), dic=0)


class criptografia:
    def __init__(self):
        self.key = b"pRmgMa8T0INjEAfksaq2aafzoZXEuwKI7wDe4c1F8AY="
        self.cipher_suite = Fernet(self.key)

    def encriptar(self, senha):
        senha_encriptada = self.cipher_suite.encrypt(str.encode(senha))
        return senha_encriptada.decode("utf-8")

    def descriptar(self, hash_senha):
        ciphered_text = str.encode(hash_senha)
        hash_descriptado = self.cipher_suite.decrypt(ciphered_text)
        return hash_descriptado


class hash:
    def codificar(self, valor):
        hash = base64.b64encode(str.encode(valor))
        return hash.decode("utf-8")

    def decodificar(self, hash):
        valor = base64.b64decode(str.encode(hash))
        return valor.decode("utf-8")


class senha:
    def gerar(self, caracteres="0123456789", quant=6, especial=False):
        # caracteres ='0123456789'
        senha = "".join(random.sample(caracteres, quant))
        if especial:
            tmps = list(senha)
            tmp = ["#", "@", "_", "+", "!", "*"]
            tmp2 = random.randint(0, quant - 1)
            tmp3 = random.randint(0, 5)
            tmps[tmp2] = tmp[tmp3]
            tmp3 = random.randint(0, 5)
            tmps.append(tmp[tmp3])
            s = ""
            senha = s.join(tmps)
        return senha

class string:
    def limpar_mascara(self, string):
        return (
            string.replace("(", "")
            .replace(")", "")
            .replace("-", "")
            .replace(" ", "")
            .replace(".", "")
            .replace("-", "")
            .replace("/", "")
        )


class permissao:
    def verifica_permissao(self, usuario, modulo, tipo):
        resposta = banco().sql(
            "select fk_id_permissao from usuario where id=$1", (usuario)
        )
        if not resposta:
            return {"MENSAGEM": "USUÁRIO NÃO ENCONTRADO..."}
        else:
            id_perm = resposta["0"]["fk_id_permissao"]
        if tipo == "s":
            resposta = banco().sql(
                "select id from une_grupo_a_module where bol_s and fk_id_modulo=$1 and id=$2",
                (modulo, id_perm),
            )
            if resposta:
                return True
            else:
                return False
        elif tipo == "i":
            resposta = banco().sql(
                "select id from une_grupo_a_module where bol_i and fk_id_modulo=$1 and id=$2",
                (modulo, id_perm),
            )
            if resposta:
                return True
            else:
                return False
        elif tipo == "u":
            resposta = banco().sql(
                "select id from une_grupo_a_module where bol_u and fk_id_modulo=$1 and id=$2",
                (modulo, id_perm),
            )
            if resposta:
                return True
            else:
                return False
        elif tipo == "d":
            resposta = banco().sql(
                "select id from une_grupo_a_module where bol_d and fk_id_modulo=$1 and id=$2",
                (modulo, id_perm),
            )
            if resposta:
                return True
            else:
                return False

    def lista_permissao(self, usuario):
        resposta = banco().sql(
            "select fk_id_permissao,bol_admin from usuario where id=$1", (usuario)
        )
        if not resposta:
            return {"MENSAGEM": "USUÁRIO NÃO ENCONTRADO..."}
        else:
            # Lista modulos. Caso um modulo seja adicionado, inclui no dicionario, para salvar no banco.
            modulos = banco().sql("select id from modulos", ())
            if resposta[0]["bol_admin"]:
                dic = {}
                for j in modulos:
                    dic[j["id"]] = [True, True, True, True]
                return dic
            id_perm = resposta[0]["fk_id_permissao"]
            resposta = banco().sql(
                "select fk_id_modulo,bol_s,bol_i,bol_u,bol_d from une_grupo_a_modulo where fk_id_grupo=$1",
                (id_perm),
            )
            if resposta:
                dic = {}
                for i in resposta:
                    dic[i["fk_id_modulo"]] = [
                        i["bol_s"],
                        i["bol_i"],
                        i["bol_u"],
                        i["bol_d"],
                    ]
                # print(dic)
                for j in modulos:
                    if not j["id"] in dic.keys():
                        dic[j["id"]] = [False, False, False, False]
                return dic


