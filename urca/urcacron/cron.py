import sys

if not len(sys.argv) == 2:
	print("Escolha uma das funcoes: python cron.py <nome da funcao>")
	sys.exit(1)

import os
import multiprocessing


class funcoes():
    
    def __init__(self, *args, **kwargs):

        self.dados_historicos = multiprocessing.Manager().dict()  # Shared dictionary for historical data
        self.estocastico_dash = multiprocessing.Manager().dict()  # Shared dictionary for historical data
        #self.markov_dash = multiprocessing.Manager().dict()  # Shared dictionary for historical data
        self.estocastico_risco = multiprocessing.Manager().dict()  # Shared dictionary for historical data
        self.dados_obtidos = multiprocessing.Manager().Value('i', False)  # Shared boolean for data obtained status
        
    def atualizar_banco(self):

        print(f"[PROCESSANDO] - Obtendo dados históricos...")
        dados = self.obter_dados_historicos()
        # Store the result in the shared dictionary
        for key, value in dados.items():
            self.dados_historicos[key] = value
        self.dados_obtidos.value = True
        print(f"[FINALIZADO] Dados históricos obtidos com sucesso!")
        # self.notify_clients("Dados históricos prontos para uso.")
        self.gerar_csv()
        dados['preco_h'].to_csv(os.path.join('app', 'curto_prazo', 'dados_analise_tecnica', 'Data', 'preco_bbce.csv'))
    
    def iniciar_markov(self):
        print(f"[PROCESSANDO] - Calculando markov...")
        markov.calcular_markov()
    
    def iniciar_longo_prazo(self):
        print(f"[PROCESSANDO] - Calculando longo prazo...")
        generate_long_data.generate_all(os.path.join("app", "longo_prazo", "dados"))
        longo_prazo.longo_prazo()
    
    def iniciar_estocastico(self):
        print(f"[PROCESSANDO] - Calculando estocastico...")
        estocastico.estocastico(None, 15, 0.005, make_plot=False)
    
    
    def iniciar_analise(self):
        print(f"[PROCESSANDO] - Calculando analise backtest...")
        analise.execute_backtest()
    
    def gerar_csv(self):
    
        def get_precos(preco):
            price_dict = PricesFillMissing.get_all(preco)

            preco_clean = price_dict["raw"]
            preco_pent = price_dict["VWAP pentada"]
            preco_lin = price_dict["linear"]
            return preco_clean, preco_pent, preco_lin

        preco = self.obter_dados_historicos()['preco_h']
        preco_clean, preco_pent, preco_lin = get_precos(preco)

        varible_hip = Variable_hip.implemented()
        varible_hip["VWAP"] = lambda x: x

        directory = os.path.join('app', 'curto_prazo', 'dados_preco', 'linear interpol')
        for process_type, processado in (("raw", preco_clean), ("VWAP pentada", preco_pent), ("interpolacao linear", preco_lin)): 
            os.makedirs(os.path.join(directory, process_type), exist_ok=True)
            dados = {}
            count = 0
            for prod_hip in list(Product_hip.implemented().keys()):
                print(prod_hip)

                data = Product_hip.implemented()[prod_hip](processado.copy())
                idx = pd.Index(pd.date_range(data.data.min(), data.data.max()), name='data')

                data = data.set_index("data").reindex(idx, fill_value=None).reset_index()
                data["missing"] = None
                data["ok"] = None
                data.loc[data.VWAP.isna(), "missing"] = count-0.25
                data.loc[False==(data.VWAP.isna()), "ok"] = count
                count += 1

                for var_hip in list(varible_hip.keys()):
                    if var_hip in ("hip_12", "hip_15"):
                        if prod_hip.startswith("rolloff diferenca cumulativa"):
                            continue
                    name = f"{prod_hip} -> {var_hip}"
                    print(name)
                    df = varible_hip[var_hip](data.copy())
                    idx = pd.Index(pd.date_range(df.data.min(), df.data.max()), name='data')
                    df = df.set_index("data").reindex(idx, fill_value=None).reset_index()
                    dados[name] = df
                    print("\t", df.VWAP.isna().sum(), df[df.VWAP.isna()].data.max())
                    df.to_csv(os.path.join(directory, process_type, name+".csv"))
                    print(f"\t{name} salvo em: '{os.path.join(directory, process_type, name)}.csv'")
    
    def obter_dados_historicos(self):
        preco, preco_h, preco_r, precobbce, subsistema, historico_hidrologia = modulos.Banco().get_precos()
        pld, pld_piso_teto, PLD = modulos.Banco().get_pld()
        self.dados_historicos["preco"] = preco
        self.dados_historicos["preco_h"] = preco_h
        self.dados_historicos["preco_r"] = preco_r
        self.dados_historicos["produtos"] = preco_h["produto"].unique()
        self.dados_historicos["pld"] = pld
        self.dados_historicos["PLD"] = PLD
        self.dados_historicos["pld_piso_teto"] = pld_piso_teto
        self.dados_historicos["precobbce"] = precobbce
        self.dados_historicos['subsistema'] = subsistema
        self.dados_historicos['historico_hidrologia'] = historico_hidrologia
        self.dados_historicos['ultimo_update'] = datetime.now()
        return self.dados_historicos

   

if sys.argv[1]=='atualizar_banco':
	from biblioteca.pre_processing.prices import PricesFillMissing
	from biblioteca.pre_processing import Product_hip, Variable_hip
	from datetime import datetime, timedelta
	from biblioteca import modulos
	import pandas as pd

elif sys.argv[1]=='iniciar_estocastico':
	from app.curto_prazo import _estocastico as estocastico
elif sys.argv[1]=='iniciar_analise':
	from app.curto_prazo import _analise_tecnica as analise
elif sys.argv[1]=='iniciar_markov':
	from app.curto_prazo import _markov as markov
elif sys.argv[1]=='iniciar_longo_prazo':
	from app.longo_prazo import _longo_prazo as longo_prazo
	from biblioteca import generate_long_data


else:
	print("Escolha uma das funcoes: python cron.py <nome da funcao>")
	sys.exit(1)

funcao="funcoes()."+sys.argv[1]+"()"
exec(funcao)