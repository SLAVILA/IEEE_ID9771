from datetime import datetime
import json
from flask import request, current_app, session
import tensorflow as tf
import pandas as pd 
import numpy as np
import pickle as pk
import os
from matplotlib import pyplot as plt
from .dados_rede_neural.Experiment import ExperimentAbstract, FuzzyTendency
from .dados_rede_neural.AutoregressiveModel import AutoregressiveModel
from plotly.io import to_json
import numpy as np
import pandas as pd
import os
from biblioteca import modulos
import plotly.graph_objects as go
import plotly.io as pio

# Blueprint registration
from app.curto_prazo import b_curto_prazo

script_dir = os.path.dirname(__file__)

@b_curto_prazo.route("/_rede_neural_dash", methods=["POST"])
def _rede_neural_dash():
    
    m = request.form.get('m')
    
    data = request.form.get('data')
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'dados_preco', 'linear interpol', 'VWAP pentada', f'rolloff suavizado {m} SE -> VWAP.csv')

    # Carrega o arquivo CSV
    VWAP = pd.read_csv(csv_path, index_col=1, parse_dates=True)
    
    path_best = os.path.join(script_dir, 'dados_rede_neural', "diff_fuzzy_UNI_autoregressive_suavizado", "teste_exp_block_diff_VWAP_064_units_1_layers_vanilla")
    test = TesteURCA(path_best, load=True)

    # Número de passos de entrada e saída
    inp_steps = test._kwargs['model_kwargs']['inp_steps']
    out_steps = test._kwargs['model_kwargs']['out_steps']

    # Gera uma lista de possíveis datas
    possible_dates = VWAP.index[inp_steps:]
    
    print(f"Possible dates: {possible_dates}")
    
    # Converte as datas para strings no formato brasileiro
    date_strings = possible_dates.sort_values(ascending=False).strftime('%d/%m/%Y').tolist()

    # Função para converter string de data brasileira para datetime
    def str_to_date(date_str):
        return datetime.strptime(date_str, '%d/%m/%Y')

    # Verifica se a data está definida e é válida
    if data and data in date_strings:
        selected_date = str_to_date(data)
        
    else:
        # Pega a última data disponível
        selected_date = possible_dates[-1]
        
    print(f"Selected date: {selected_date}")

    
    # Ajusta a lógica de retorno usando selected_date
    selected_index = possible_dates.get_loc(selected_date)
    
    print(f"Selected index: {selected_index} ")

    try:
        rede_retorno = rede(selected_index + 1, m)
    except:
        return json.dumps({"status": "1", "msg": "Erro ao carregar o arquivo."})
    return rede_retorno
        
        
        
    

def gerar_csv():

    from biblioteca.pre_processing.prices import PricesFillMissing
    from biblioteca.pre_processing import Product_hip, Variable_hip

    
    def get_precos(preco):
        price_dict = PricesFillMissing.get_all(preco)

        preco_clean = price_dict["raw"]
        preco_pent = price_dict["VWAP pentada"]
        preco_lin = price_dict["linear"]
        return preco_clean, preco_pent, preco_lin

    preco = current_app.obter_dados_historicos()['preco_h']
    preco_clean, preco_pent, preco_lin = get_precos(preco)

    varible_hip = Variable_hip.implemented()
    varible_hip["VWAP"] = lambda x: x

    directory = os.path.join(script_dir, 'dados_preco', 'linear interpol')
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

class TesteURCA(ExperimentAbstract):
    def __init__(self, *args, load=False, **kwargs):
        super().__init__(*args, **kwargs, load=load)
        self.dataset = None
        if not load:
            self.process_data()

    @property
    def models_dict(self):
        return {
            "AutoregressiveModel": AutoregressiveModel,
        }

    def process_data(self):
        with open(self._kwargs["data_kwargs"]["pca.pkl"],"rb") as f:
            pca = pk.load(f)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        threshold = np.searchsorted(cumvar, self._kwargs["data_kwargs"]["pca_var_threshold"], side="left")
        #print(threshold)
        all_data_pca = pd.read_csv(self._kwargs["data_kwargs"]["all_data_pca.csv"], index_col=0)
        data = all_data_pca[all_data_pca.columns[:threshold+1]].copy()
        #print(data.shape)

        price = pd.read_csv(self._kwargs["data_kwargs"]["VWAP.csv"], index_col=1)
        
        data["VWAP"] = None
        data.update(price["VWAP"])
        self._kwargs["VWAP_transform"] = {"mean": data["VWAP"].mean(), "std": data["VWAP"].std(), "scale": 1}
        self.save()
        data["VWAP"] = self.transform_VWAP_to_y(data["VWAP"])
        self.data = data.astype(float).copy()

        input_vars = list(self.data.columns)
        input_vars.remove("VWAP")
        
        if "div" not in self._kwargs["data_kwargs"]:
            self._kwargs["data_kwargs"]["div"] = {}
            
        self.dataset = self.reamostrate(
            self.data, ["VWAP"], ["VWAP"], 
            input_days=self._kwargs["model_kwargs"]["inp_steps"],
            output_days=self._kwargs["model_kwargs"]["out_steps"],
            **self._kwargs["data_kwargs"]["div"])
        
    def transform_VWAP_to_y(self, VWAP):
        return VWAP/self._kwargs["VWAP_transform"]["std"]*self._kwargs["VWAP_transform"]["scale"]
    
    def transform_y_inp(self, x):
        return x*self._kwargs["VWAP_transform"]["std"]/self._kwargs["VWAP_transform"]["scale"]
    
    def transform_y(self, x, y):
        return (y*np.expand_dims(x[:, -1, :], 1))*self._kwargs["VWAP_transform"]["std"]/self._kwargs["VWAP_transform"]["scale"]
    
    def transform_y_prop(self, x, y):
        return y
        
    @staticmethod
    def resampler_inp(df, target, input_days=30, diff=False):
        inp = []
        for shift in range(-input_days+1, 1):
            inp.append(df[target].shift(-shift).to_numpy())
        if diff:
            return ((np.stack(inp)-inp[-1]).T, np.stack([inp[-1] for i in range(input_days)]).T)
        return np.stack(inp).T

    @staticmethod
    def resampler_out(df, target, output_days=15):
        out = []
        for shift in range(output_days+1):
            out.append(df[target].shift(-shift).to_numpy())
        return (np.divide(np.stack(out[1:])-out[0], out[0])).T

    @staticmethod
    def reamostrate(df, inp_cols, out_cols, input_days=30, output_days=15, 
                  train_proportion=0.6, validation_proportion=0.2, test_proportion=0.2):
        if train_proportion+validation_proportion+test_proportion != 1:
            raise ValueError("train_proportion+validation_proportion+test_proportion must be equal to 1")

        result = {"input": [], "output": []}
        nan_mask = np.full(len(df), True)
        for d in inp_cols:
            resampled = TesteURCA.resampler_inp(df, d, input_days)
            nan_mask &= ~np.isnan(resampled).any(axis=1)
            result["input"].append(resampled)

        for d in out_cols:
            resampled = TesteURCA.resampler_out(df, d, output_days)
            nan_mask &= ~np.isnan(resampled).any(axis=1)
            result["output"].append(resampled)

        result["input"] = np.stack(result["input"], axis=-1)[nan_mask, :, :]
        result["output"] = np.stack(result["output"], axis=-1)[nan_mask, :, :]
        index = df.index[nan_mask]

        n_amostras = result["input"].shape[0]
        #print(f"De {df.shape[0]} datas para {n_amostras} amostras")

        len_tr  = int(np.rint(train_proportion*n_amostras))
        len_val = int(np.rint(validation_proportion*n_amostras))
        len_tst = int(n_amostras - (len_tr+len_val))
        #print(f"len_tr: {len_tr}")
        #print(f"len_val: {len_val}")
        #print(f"len_tst: {len_tst}")

        tr_index = np.random.permutation(len_tr)
        data_tr  = {}
        data_val = {}
        data_tst = {}
        for d in result:
            data_tr[d]  = result[d][tr_index, :, :]
            data_val[d] = result[d][len_tr:-len_tst, :, :]
            data_tst[d] = result[d][-len_tst:, :, :]

        return {"tr": data_tr,
                "tr_index": index[tr_index],
                "val": data_val,
                "val_index": index[len_tr:-len_tst],
                "tst": data_tst,
                "tst_index": index[-len_tst:],
                "input_cols": inp_cols.copy(),
                "output_cols": out_cols.copy()}

    @property
    def train_data(self):
        if not self.dataset:
            self.process_data()
        return (self.dataset["tr"]["input"], self.dataset["tr"]["output"])

    @property
    def val_data(self):
        if not self.dataset:
            self.process_data()
        return (self.dataset["val"]["input"], self.dataset["val"]["output"])

    @property
    def test_data(self):
        if not self.dataset:
            self.process_data()
        return (self.dataset["tst"]["input"], self.dataset["tst"]["output"])
    
    def plot_sample(self, dataset="tr", models=None, raw=False):
        if not self.dataset:
            self.process_data()
        idx = np.random.choice(len(self.dataset[dataset]["input"]))
        date = self.dataset[dataset+"_index"][idx]
        
        input_val = self.dataset[dataset]["input"][idx, :, :]
        output_val = self.dataset[dataset]["output"][idx, :, :]
        
        x = np.array([input_val])
        
        y_input = self.transform_y_inp(input_val[:, -1]) #+ self.VWAP_transform["mean"]
        y_true = self.transform_y(x, output_val.reshape(-1)) + y_input[-1] # + self.VWAP_transform["mean"]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        #ax.plot([pd.Timedelta(days=i) + pd.to_datetime(date) for i in range(-len(input_val)+1, 1)], 
        #         input_val - input_val[-1, :])
        #ax.plot([pd.Timedelta(days=i) + pd.to_datetime(date) for i in range(1, len(output_val)+1)], output_val)
        
        if raw:
            ax.plot([pd.Timedelta(days=i) + pd.to_datetime(date) for i in range(-len(y_input)+1, y_true.size+1)], 
                     np.concatenate((input_val - input_val[-1, :], output_val)), "--",  label="true")
        else:
            ax.plot([pd.Timedelta(days=i) + pd.to_datetime(date) for i in range(-len(y_input)+1, y_true.size+1)], 
                     np.concatenate((y_input, y_true.reshape(-1))), "--",  label="true")
        if models is None:
            models = range(self._kwargs["times"])
        
        for i in models:
            model_dir = os.path.join(self.experiment_path, str(i))
            model = self.load_model(model_dir)

            y_pred = self.transform_y(x, model.predict(x, verbose=0)[0, :].reshape(-1)) + y_input[-1]# + self.VWAP_transform["mean"]
            if raw:
                ax.plot([pd.Timedelta(days=i) + pd.to_datetime(date) for i in range(1, y_pred.size+1)],
                         model.predict([input_val], verbose=0)[0, :].reshape(-1), label=f"predicted model {i}")
            else:
                ax.plot([pd.Timedelta(days=i) + pd.to_datetime(date) for i in range(1, y_pred.size+1)],
                         y_pred.reshape(-1), label=f"predicted model {i}")
        ax.set_title("Prediction from "+date)
        ax.set(xlabel='days', ylabel="price defference (R$)")
        ax.legend()
        return fig

class TesteURCA_mult(ExperimentAbstract):
    def __init__(self, *args, load=False, **kwargs):
        super().__init__(*args, **kwargs, load=load)

        self.dataset = None
        if not load:
            self.process_data()
        else:
            print(kwargs)
            if kwargs:
                if "data_kwargs" in kwargs:
                    self._kwargs["data_kwargs"].update(kwargs["data_kwargs"])
            self.reprocess_data()

    @property
    def models_dict(self):
        return {
            "AutoregressiveModel": AutoregressiveModel,
        }
    
    @property
    def dados_tipo_dict(self):
        return {
            'ampere_ENA_atual_REE': "ENA",
            'ampere_ena_prevista_rees_CFSV2-REFORECAST': "ENA",
            'ampere_climatologia_ena_ree': "ENA",
            'ampere_precipitacao_historica_rees': "precipitação",
            'ampere_climatologia_precipitacao_ree': "precipitação",
            'previsao_historica_rees_cfsv2': "precipitação",
            'cmo': "cmo",
            'intercambio_internacional': 'intercambio_internacional',
            'intercambio_nacional': "intercambio_nacional",
            'ampere_ASAS': "ampere_ASAS",
            'ampere_Frentes_frias': "ampere_Frentes_frias",
            'ampere_JBN': "ampere_JBN",
            'ampere_ZCAS': "ampere_ZCAS",
            'ampere_indice_AOO_diario': "ampere_indice_AOO_diario",
            'ampere_indice_IOD': "ampere_indice_IOD",
            'ampere_indice_MJO_diario': "ampere_indice_MJO_diario",
            'ampere_indice_AMO_mensal': "ampere_indice_AMO_mensal",
            'ampere_indice_ONI_mensal': "ampere_indice_ONI_mensal",
            'energia_vertida_turbinavel': "energia_vertida_turbinavel",
            'demanda_maxima': "demanda_maxima",
            'geracao_usinas_despachadas': "geracao_usinas_despachadas"
        }
    
    def reprocess_data(self):
        input_vars = self._kwargs["input_vars"].copy()
        data = pd.read_csv(self._kwargs["data_kwargs"]["all_data.csv"], index_col=0, header=[0, 1])
        data.index = pd.to_datetime(data.index)
        data = data.rename(columns={"REE MANAUS": "REE MANAUS AMAPA"})
        price = pd.read_csv(self._kwargs["data_kwargs"]["VWAP.csv"], index_col=1)
        price.index = pd.to_datetime(price.index)
        
        data["VWAP", "VWAP"] = None
        data["VWAP", "VWAP"] = data["VWAP", "VWAP"].astype(float)
        update_idx = price.index.intersection(data.index)
        data.loc[update_idx, ("VWAP", "VWAP")] = price.loc[update_idx, "VWAP"].values
        self.data = data.astype(float).copy()
        # dropa as colunas que são constantes
        self.data = self.data.drop(self.data.columns[(self.data.max()-self.data.min()) == 0], axis=1)
        
        # climatologia rename
        clim = ["ampere_climatologia_precipitacao_ree", "ampere_climatologia_ena_ree"]
        rename = {}
        for c in clim:
            if c not in input_vars:
                continue
            for c2 in self.data[c]:
                rename[c2] = "-".join(c2.split()[1:]).lower()
        self.data = self.data.rename(columns=rename)
        
        # Normalize
        self.data["VWAP"] = self.transform_VWAP_to_y(self.data["VWAP"])
        
        for p in ['PLD_mensal', 'piso_teto_pld']:
            if p not in input_vars:
                continue
            self.data[p] = self.transform_VWAP_to_y(self.data[p])
    
        
        preds = ['ampere_ena_prevista_rees_CFSV2-REFORECAST', 'previsao_historica_rees_cfsv2',
                 "ampere_climatologia_precipitacao_ree", "ampere_climatologia_ena_ree"]
        self.data = self.transform_input_data(
            self.data, [c for c in data.columns.get_level_values(0).unique() if c not in preds])

        # climatologia
        dfs = [self.data]
        for col in clim:
            if c not in input_vars:
                continue
            input_vars.append(col+"_adj")
            for i in range(1, 16):
                df = self.data[col].copy()
                df = df.rename(columns={c: (col+"_adj", "A "+"-".join(c.split())+f" d+{i}") for c in df.columns})
                dfs.append(df.shift(-i))
        self.data = pd.concat(dfs, axis=1)

        # previsões
        self.data_predictions = {}
        preds = [("ampere_climatologia_precipitacao_ree", "ampere_climatologia_precipitacao_ree_adj"),
                 ("ampere_ENA_atual_REE", "ampere_ena_prevista_rees_CFSV2-REFORECAST"), 
                 ("ampere_precipitacao_historica_rees", "previsao_historica_rees_cfsv2"),
                 ("ampere_climatologia_ena_ree", "ampere_climatologia_ena_ree_adj")]
        for hist, pred in preds:
            if hist not in input_vars: continue
            if pred not in input_vars: continue
            columns_dict = {
                c: (c.split()[1], int(c.split()[2].split('+')[-1])) 
                    for c in self.data[pred].columns
            }
            temp_df = pd.DataFrame(index=pd.MultiIndex.from_product([range(1, 31), self.data.index]))
            for col in columns_dict:
                gr, dt = columns_dict[col]
                temp_df.loc[dt, gr] = self.data[pred][col].values
            temp_df = temp_df.swaplevel().sort_index()
            temp_df.columns = pd.MultiIndex.from_product([[hist], temp_df.columns])
            
            temp_df = self.transform_input_data(temp_df)
            
            temp_df.columns = temp_df.columns.get_level_values(1)
            self.data_predictions[hist] = temp_df
            
        for hist, pred in preds:
            if pred in input_vars:
                input_vars.remove(pred)
        if "VWAP" in input_vars:
            input_vars.remove("VWAP")
        
        self.dataset = self.reamostrate(
            self.data, self.data_predictions, self._kwargs["input_var_order"], ["VWAP"], 
            input_days=self._kwargs["model_kwargs"]["inp_steps"],
            output_days=self._kwargs["model_kwargs"]["out_steps"],
            **self._kwargs["data_kwargs"]["div"])

    def process_data(self):
        input_vars = self._kwargs["input_vars"].copy()
        data = pd.read_csv(self._kwargs["data_kwargs"]["all_data.csv"], index_col=0, header=[0, 1])
        data.index = pd.to_datetime(data.index)
        data = data.rename(columns={"REE MANAUS": "REE MANAUS AMAPA"})
        price = pd.read_csv(self._kwargs["data_kwargs"]["VWAP.csv"], index_col=1)
        price.index = pd.to_datetime(price.index)
        
        data["VWAP", "VWAP"] = None
        data["VWAP", "VWAP"] = data["VWAP", "VWAP"].astype(float)
        update_idx = price.index.intersection(data.index)
        data.loc[update_idx, ("VWAP", "VWAP")] = price.loc[update_idx, "VWAP"].values
        self.data = data.astype(float).copy()
        # dropa as colunas que são constantes
        self.data = self.data.drop(self.data.columns[(self.data.max()-self.data.min()) == 0], axis=1)
        
        # climatologia rename
        clim = ["ampere_climatologia_precipitacao_ree", "ampere_climatologia_ena_ree"]
        rename = {}
        for c in clim:
            if c not in input_vars:
                continue
            for c2 in self.data[c]:
                rename[c2] = "-".join(c2.split()[1:]).lower()
        self.data = self.data.rename(columns=rename)
        
        # Normalize
        self._kwargs["VWAP_transform"] = {
            "mean": data["VWAP", "VWAP"].mean(), "std": data["VWAP", "VWAP"].std(), "scale": 1}
        self.save()
        self.data["VWAP"] = self.transform_VWAP_to_y(self.data["VWAP"])
        
        for p in ['PLD_mensal', 'piso_teto_pld']:
            if p not in input_vars:
                continue
            self.data[p] = self.transform_VWAP_to_y(self.data[p])
    
        self._kwargs["normalize_params"] = {}
        self._kwargs["normalize_params"]['ENA'] = {
            "min": data["ampere_ENA_atual_REE"].min().to_dict(),
            "max": data["ampere_ENA_atual_REE"].max().to_dict(),
        }
        self._kwargs["normalize_params"]['precipitação'] = {
            "min": data["ampere_precipitacao_historica_rees"].min().to_dict(),
            "max": data["ampere_precipitacao_historica_rees"].max().to_dict(),
        }
        
        for col in data.columns.get_level_values(0).unique():
            if not col in self.dados_tipo_dict: continue
            if self.dados_tipo_dict[col] in self._kwargs["normalize_params"]: continue
            self._kwargs["normalize_params"][self.dados_tipo_dict[col]] = {
                "min": data[col].min().to_dict(),
                "max": data[col].max().to_dict(),
            }
        self.save()
        
        preds = ['ampere_ena_prevista_rees_CFSV2-REFORECAST', 'previsao_historica_rees_cfsv2',
                 "ampere_climatologia_precipitacao_ree", "ampere_climatologia_ena_ree"]
        self.data = self.transform_input_data(
            self.data, [c for c in data.columns.get_level_values(0).unique() if c not in preds])

        # climatologia
        dfs = [self.data]
        for col in clim:
            input_vars.append(col+"_adj")
            for i in range(1, 16):
                df = self.data[col].copy()
                df = df.rename(columns={c: (col+"_adj", "A "+"-".join(c.split())+f" d+{i}") for c in df.columns})
                dfs.append(df.shift(-i))
        self.data = pd.concat(dfs, axis=1)

        # previsões
        self.data_predictions = {}
        preds = [("ampere_climatologia_precipitacao_ree", "ampere_climatologia_precipitacao_ree_adj"),
                 ("ampere_ENA_atual_REE", "ampere_ena_prevista_rees_CFSV2-REFORECAST"), 
                 ("ampere_precipitacao_historica_rees", "previsao_historica_rees_cfsv2"),
                 ("ampere_climatologia_ena_ree", "ampere_climatologia_ena_ree_adj")]
        for hist, pred in preds:
            if hist not in input_vars: continue
            if pred not in input_vars: continue
            columns_dict = {
                c: (c.split()[1], int(c.split()[2].split('+')[-1])) 
                    for c in self.data[pred].columns
            }
            temp_df = pd.DataFrame(index=pd.MultiIndex.from_product([range(1, 31), self.data.index]))
            for col in columns_dict:
                gr, dt = columns_dict[col]
                temp_df.loc[dt, gr] = self.data[pred][col].values
            temp_df = temp_df.swaplevel().sort_index()
            temp_df.columns = pd.MultiIndex.from_product([[hist], temp_df.columns])
            
            temp_df = self.transform_input_data(temp_df)
            
            temp_df.columns = temp_df.columns.get_level_values(1)
            self.data_predictions[hist] = temp_df

        
        if "div" not in self._kwargs["data_kwargs"]:
            self._kwargs["data_kwargs"]["div"] = {}
            
        for hist, pred in preds:
            if pred in input_vars:
                input_vars.remove(pred)
        if "VWAP" in input_vars:
            input_vars.remove("VWAP")
        
        self._kwargs["input_var_order"] = input_vars+["VWAP"]
        self.save()
        
        n_columns = len(self.data[self._kwargs["input_var_order"]].columns)
        self._kwargs["model_kwargs"]["input_features"] = n_columns
        self._kwargs["model_kwargs"]["output_features"] = n_columns
        
        self.dataset = self.reamostrate(
            self.data, self.data_predictions, self._kwargs["input_var_order"], ["VWAP"], 
            input_days=self._kwargs["model_kwargs"]["inp_steps"],
            output_days=self._kwargs["model_kwargs"]["out_steps"],
            **self._kwargs["data_kwargs"]["div"])
        
    def transform_input_data(self, data, columns=None):
        if columns is None:
            columns = data.columns.get_level_values(0).unique()
        for col1 in columns:
            if col1 not in self.dados_tipo_dict: continue
            params = self._kwargs["normalize_params"][self.dados_tipo_dict[col1]]
            for col2 in data[col1].columns:
                data[(col1, col2)] = (data[(col1, col2)]-params["min"][col2])/(params["max"][col2]-params["min"][col2])
        return data
        
    def transform_VWAP_to_y(self, VWAP):
        return VWAP/self._kwargs["VWAP_transform"]["std"]*self._kwargs["VWAP_transform"]["scale"]
    
    def transform_y_inp(self, y):
        return y*self._kwargs["VWAP_transform"]["std"]/self._kwargs["VWAP_transform"]["scale"]
    
    def transform_y(self, x, y):
        return (y*x[0][:, -1, -1])*self._kwargs["VWAP_transform"]["std"]/self._kwargs["VWAP_transform"]["scale"]
    
    def transform_y_prop(self, x, y):
        return y
        
    @staticmethod
    def resampler_inp(df, target, input_days=30):#, diff=False):
        inp = []
        for shift in range(-input_days+1, 1):
            inp.append(df[target].shift(-shift).to_numpy())
        
        stacked = np.stack(inp)
        #stacked.shape = (input_days, samples, *features)
        stacked = np.moveaxis(stacked, 0, 1)
        #stacked.shape = (samples, input_days, *features)
        stacked = stacked.reshape((stacked.shape[0], stacked.shape[1], -1))
        #stacked.shape = (samples, input_days, features)
        return stacked

    @staticmethod
    def resampler_out(df, target, output_days=15):
        out = []
        for shift in range(output_days+1):
            out.append(df[target].shift(-shift).to_numpy())
        stacked = np.stack(out)
        #print("resampler_out: stacked")
        #print(stacked)
        #stacked.shape = (output_days, samples, *features)
        stacked = np.moveaxis(stacked, 0, 1)
        #stacked.shape = (samples, output_days, *features)
        stacked = stacked.reshape((stacked.shape[0], stacked.shape[1], -1))
        #stacked.shape = (samples, output_days, features)
        
        first_day = stacked[:, 0, :].reshape((stacked.shape[0], 1, stacked.shape[2]))
        #print("resampler_out: first_day")
        #print(first_day)
        #print("resampler_out: result")
        #print(np.divide(stacked[:, 1:, :]-first_day, first_day))
        return np.divide(stacked[:, 1:, :]-first_day, first_day)

    @staticmethod
    def reamostrate(df, pred_df, inp_cols, out_cols, input_days=30, output_days=15, 
                  train_proportion=0.6, validation_proportion=0.2, test_proportion=0.2):
        if train_proportion+validation_proportion+test_proportion != 1:
            raise ValueError("train_proportion+validation_proportion+test_proportion must be equal to 1")

        result = {"input": [], "output": []}
        nan_mask = np.full(len(df), True)
        #print("reamostrate: sum mask, len mask")
        #print(sum(nan_mask), len(nan_mask))
        for d in inp_cols:
            resampled = TesteURCA_mult.resampler_inp(df, d, input_days)
            a = np.isnan(resampled)
            b = a.any(axis=(1, 2))
            nan_mask &= ~b
            #print("reamostrate (inp for): sum mask, len mask")
            #print(sum(nan_mask), len(nan_mask))
            result["input"].append(resampled)
        
        for d in out_cols:
            #print(df[d])
            resampled = TesteURCA_mult.resampler_out(df, d, output_days)
            #print(resampled.shape)
            #print(resampled.shape)

            nan_mask &= ~np.isnan(resampled).any(axis=(1, 2))
            #print("reamostrate (out for): sum mask, len mask")
            #print(sum(nan_mask), len(nan_mask))
            result["output"].append(resampled)
        
        index = df.index[nan_mask]-pd.Timedelta(input_days-1, "d")
        print(index)
        #print(len(index), len(df.index))
        pred = []
        mask = []
        val_shape = (len(index), output_days)
        for i, d in enumerate(inp_cols):
            if d in pred_df:
                #print("em predictions: ", d)
                for dp in pred_df[d].columns:
                    pred.append(pred_df[d].loc[index, dp].unstack(1).to_numpy()[:, :output_days])
                    #print("-", pred[-1].shape)
                    mask.append(np.ones(val_shape))
            else:
                print(d, df[d].shape)
                for dp in range(df[d].shape[1]):
                    #print("n tem em predictions: ", d)
                    pred.append(np.zeros(val_shape))
                    mask.append(np.zeros(val_shape))
        
        result["input"] = np.concatenate(result["input"], axis=-1)[nan_mask, :, :]
        result["pred"] = np.stack(pred, axis=-1)
        result["mask"] = np.stack(mask, axis=-1)
        result["output"] = np.concatenate(result["output"], axis=-1)[nan_mask, :, :]
        
        print(result["input"].shape)
        print(result["pred"].shape)
        print(result["mask"].shape)
        print(result["output"].shape)
        
        n_amostras = result["input"].shape[0]
        
        len_tr  = int(np.rint(train_proportion*n_amostras))
        len_val = int(np.rint(validation_proportion*n_amostras))
        len_tst = int(n_amostras - (len_tr+len_val))
        
        tr_index = np.random.permutation(len_tr)
        data_tr  = {}
        data_val = {}
        data_tst = {}
        for d in result:
            data_tr[d]  = result[d][tr_index, :, :]
            data_val[d] = result[d][len_tr:-len_tst, :, :]
            data_tst[d] = result[d][-len_tst:, :, :]

        return {"tr": data_tr,
                "tr_index": index[tr_index],
                "val": data_val,
                "val_index": index[len_tr:-len_tst],
                "tst": data_tst,
                "tst_index": index[-len_tst:],
                "input_cols": inp_cols.copy(),
                "output_cols": out_cols.copy(),
                "all": result,
                "all_index": index
               }

    @property
    def train_data(self): # inputs, inputs_pred, mask
        if not self.dataset:
            self.process_data()
        return ((self.dataset["tr"]["input"],
                 self.dataset["tr"]["pred"],
                 self.dataset["tr"]["mask"]
                ), self.dataset["tr"]["output"])

    @property
    def val_data(self):
        if not self.dataset:
            self.process_data()
        return ((self.dataset["val"]["input"],
                 self.dataset["val"]["pred"],
                 self.dataset["val"]["mask"]
                ), self.dataset["val"]["output"])

    @property
    def test_data(self):
        if not self.dataset:
            self.process_data()
        return ((self.dataset["tst"]["input"],
                 self.dataset["tst"]["pred"],
                 self.dataset["tst"]["mask"]
                ), self.dataset["tst"]["output"])
    
    @property
    def all_data(self):
        if not self.dataset:
            self.process_data()
        return ((self.dataset["all"]["input"],
                 self.dataset["all"]["pred"],
                 self.dataset["all"]["mask"]
                ), self.dataset["all"]["output"])

def rede(day, m):
    
    #gerar_csv()
    
    ####################### carrega o modelo
    path_best = os.path.join(script_dir, 'dados_rede_neural', "diff_fuzzy_UNI_autoregressive_suavizado", "teste_exp_block_diff_VWAP_064_units_1_layers_vanilla")
    best_model = 0
    test = TesteURCA(path_best, load=True)
    model = test.load_model(os.path.join(path_best, f"{best_model:d}"))
    
    inp_steps = test._kwargs['model_kwargs']['inp_steps']
    out_steps = test._kwargs['model_kwargs']['out_steps']
    
    VWAP_path = os.path.join(script_dir, 'dados_preco', 'linear interpol', 'interpolacao linear', f'rolloff suavizado {m} SE -> VWAP.csv')
    all_data_path = os.path.join(script_dir, 'dados_rede_neural', 'filtered_ree_data_updated.csv')
    
    path_best_mult = os.path.join(script_dir, 'dados_rede_neural', "497_units_3_layers_LSTM")
    best_model_mult = 0
    data_kwargs = {}
    data_kwargs["VWAP.csv"] = VWAP_path
    data_kwargs["all_data.csv"] = all_data_path
    print("before loading test")
    test_mult = TesteURCA_mult(path_best_mult, load=True, data_kwargs=data_kwargs)
    print("before loading model")
    model_mult = test.load_model(os.path.join(path_best_mult, f"{best_model_mult:d}"))
    print(model_mult)
    #######################
    
    
    
    # escolhe aleatóriamente 1 data e faz a previsão do modelo para esta data
    # pode dar erro caso tenha algum valor faltante no dataset
    #######################
    
    VWAP = pd.read_csv(VWAP_path, index_col=1, parse_dates=True)
    print(VWAP)
    
    def to_fuzzy(prop, stag_trig=0.1, var_trig=0.15):
        f = FuzzyTendency(stag_trig, var_trig)
        percentage = f(prop).mean(axis=1)[0, 0]
        return {name: p for p, name in zip(percentage, ["Aumento", "Estagnação", "Diminuição"])}
    
    def predict_day(from_day):
        i = VWAP.index.get_loc(from_day)-inp_steps+1
        idx = pd.date_range(start=from_day-pd.Timedelta(inp_steps-1, "d"), periods=inp_steps+out_steps)
        empty_series = pd.Series(index=idx, name="VWAP")
        
        all_data = empty_series.copy()
        all_data.update(VWAP.VWAP.iloc[i: i+inp_steps])
        
        out = empty_series.copy()
        out.update(VWAP.VWAP.iloc[i+inp_steps-1: min(len(VWAP), i+inp_steps+out_steps)])
        
        inp = all_data.copy().dropna()
        all_data.update(out.dropna())
        
        inp_processed_raw = test.resampler_inp(test.transform_VWAP_to_y(inp), slice(None), input_days=inp_steps)
        
        # remove as amostras com dados faltantes
        inp_mask = ~np.isnan(inp_processed_raw).any(axis=1)
        inp_processed = np.expand_dims(inp_processed_raw[inp_mask], 2)
        # index do dia em que a previsão está sendo feita
        test_index = inp.index[inp_mask]
        
        # faz a previsão
        predicted_processed = model.predict(inp_processed)
        predicted = test.transform_y(inp_processed, predicted_processed)
        return all_data, inp, out, predicted, test_index
    
    # escolhe uma data aleatória
    i = day
    print(f"Index: {i}")
    day = VWAP.index[i+inp_steps-1]
    data_plot = str(VWAP.index[i+inp_steps-1].strftime('%d/%m/%Y'))
    all_data, inp, out, predicted, test_index = predict_day(day)

    # plota os resultados
    pred = all_data.copy()
    pred[:-out_steps-1] = None
    pred.iloc[-out_steps:] = predicted[0, :, 0] + inp[-1]
    
    def to_fuzzy(prop, stag_trig=0.1, var_trig=0.15):
        f = FuzzyTendency(stag_trig, var_trig)
        percentage = f(prop).mean(axis=1)[0, 0]
        return {name: p for p, name in zip(percentage, ["Aumento", "Estagnação", "Diminuição"])}
    prop = predicted/inp[-1]
    percentage = to_fuzzy(prop, 0.05, 0.15)

    # Usando Plotly para plotar
    fig = go.Figure()

    # Adiciona dados históricos
    fig.add_trace(go.Scatter(
        x=all_data.index,
        y=all_data,
        mode='lines',
        name='Histórico'
    ))

    # Adiciona dados verdadeiros
    fig.add_trace(go.Scatter(
        x=out.index,
        y=out,
        mode='lines',
        name='VWAP Real'
    ))

    # Adiciona dados previstos
    fig.add_trace(go.Scatter(
        x=pred.index,
        y=pred,
        mode='lines',
        name='VWAP Previsto Univariado'
    ))
    
    def predict_day_mult(test, model, from_day):
        if from_day not in test.dataset["all_index"]:
            return None
        i = test.dataset["all_index"].get_loc(from_day)
        from_day = pd.Timestamp(from_day)
        idx = pd.date_range(start=from_day-pd.Timedelta(inp_steps-1, "d"), periods=inp_steps+out_steps)
        all_data = pd.Series(index=idx, name="VWAP")
        
        input_data = tuple([np.expand_dims(test.all_data[0][k][i, :], 0) for k in range(3)])
        output = np.expand_dims(test.all_data[1][i, :, :], 0)
        
        inp = all_data.copy()
        inp.iloc[:inp_steps] = test.transform_y_inp(input_data[0][0, :, -1])
        print(inp)
        out = all_data.copy()
        x = test.transform_y(input_data, output)
        out.iloc[inp_steps:] = x[0, :, 0]
        out +=  inp.dropna().iloc[-1]
        all_data = inp.copy()
        all_data.update(out)
    
        # faz a previsão
        predicted_processed = model.predict(input_data)
        predicted = test.transform_y(input_data, predicted_processed)
        
        pred = all_data.copy()
        pred[:-out_steps-1] = None
        pred[-out_steps:] = predicted[0, :, 0] + inp.dropna().iloc[-1]
        out.loc[day] = all_data.loc[day]
        
        def to_fuzzy(prop, stag_trig=0.1, var_trig=0.15):
            f = FuzzyTendency(stag_trig, var_trig)
            percentage = f(prop).mean(axis=1)[0, 0]
            return {name: p for p, name in zip(percentage, ["Aumento", "Estagnação", "Diminuição"])}
        prop = predicted/inp.dropna().iloc[-1]
        percentage = to_fuzzy(prop, 0.05, 0.15)
        return inp, out, pred, percentage
    
    mult_result = predict_day_mult(test_mult, model_mult, day)
    if mult_result:
        fig.add_trace(go.Scatter(
            x=mult_result[2].index,
            y=mult_result[2],
            mode='lines',
            name='VWAP Previsto Multivariado'
        ))
    
    # Adiciona linhas verticais
    for date in test_index:
        fig.add_shape(type="line", x0=date, y0=0, x1=date, y1=600, line=dict(color="Red", dash="dash"))

    fig.update_layout(
        yaxis=dict(range=[0, 600]),
        xaxis_title='Data',
        yaxis_title='VWAP (R$)'
    )
    
    output_dir = os.path.join('web', 'static', 'tasks_saida', 'rede_neural', str(session['usuario']['id']), 'saidas_univariado')
    os.makedirs(output_dir, exist_ok=True)

    # Exporta o gráfico para JSON
    plot_json = pio.to_json(fig)
    
    # Save the JSON file
    output_file = os.path.join(output_dir)
    os.makedirs(output_file, exist_ok=True)
    
    with open(os.path.join(output_file, f'rede_neural_{str(VWAP.index[i+inp_steps-1].strftime("%d/%m/%Y")).replace("/", "_").strip()}.json'), 'w') as f:
        json.dump(plot_json, f)
        
    

    text_uni = "Univariada"
    text_mult = "Multivariada"
    percentage_uni = percentage
    if mult_result:
        percentage_mult = mult_result[-1]
        results = {text_uni: percentage_uni, text_mult: percentage_mult}
        models = [text_mult, text_uni]
    else:
        results = {text_uni: percentage_uni}
        models = [text_uni]

    text_dim  = "Diminuição"
    text_stag = "Estagnação"
    text_aum  = "Aumento"
    color_dim  = "#FF0000"  # red
    color_stag = "#FFFF00"  # yellow
    color_aum  = "#0000FF"  # blue

    categories = [text_dim, text_stag, text_aum]
    colors = {
        text_dim: color_dim, 
        text_stag: color_stag, 
        text_aum: color_aum
    }
    # Criar a figura
    fig = go.Figure(data=[
        go.Bar(
            x=[results[model][cat]*100 for model in models], 
            y=models, 
            name=cat, 
            orientation='h', 
            marker_color=colors[cat], 
            showlegend=False,
            text=[f'{results[model][cat]*100:.1f}%' for model in models],  # Adicionar o texto
            textposition='inside',  # Posição do texto
            insidetextanchor='middle'  # Centralizar o texto dentro da barra
        ) for cat in categories
    ])
    fig.update_layout(barmode='stack', title='Previsão Rede Neural')# (Univarida e Multivariada)',)
    plot_json2 = pio.to_json(fig)
    
    #######################
    
    
    
    
    #######################
    
    # Carrega dados
    # VWAP = pd.read_csv(os.path.join(script_dir, 'dados_rede_neural', 'dados_preco', 'linear interpol', 'VWAP pentada', 'rolloff suavizado M+1 SE -> VWAP.csv'),
    #                    index_col=1, parse_dates=True)
    
    # inp_processed_raw = test.resampler_inp(test.transform_VWAP_to_y(VWAP.VWAP), slice(None), input_days=inp_steps)
    
    # # Remove amostras com dados faltantes
    # inp_mask = ~(np.isnan(inp_processed_raw).any(axis=1))
    # inp_processed = np.expand_dims(inp_processed_raw[inp_mask], 2)
    
    # # Índice do dia em que a previsão está sendo feita
    # test_index = VWAP.index[inp_mask]
    
    # # Faz a previsão
    # predicted_processed = model.predict(inp_processed)
    # predicted_diff = test.transform_y(inp_processed, predicted_processed)
    # predicted = VWAP.loc[test_index, "VWAP"].to_numpy().reshape((-1, 1)) + predicted_diff[:, :, 0]

    # predictions = pd.DataFrame(predicted, index=test_index, 
    #                            columns=[f"d+{i}" for i in range(1, out_steps+1)])

    # # Plotando os resultados com Plotly
    # fig = go.Figure()

    # # Adiciona dados históricos
    # fig.add_trace(go.Scatter(
    #     x=VWAP.index,
    #     y=VWAP['VWAP'],
    #     mode='lines',
    #     name='VWAP'
    # ))

    # # Adiciona previsões para cada passo de tempo
    # for i in range(out_steps):
    #     fig.add_trace(go.Scatter(
    #         x=predictions.index,
    #         y=predictions.iloc[:, i],
    #         mode='lines',
    #         name=f'Predicted d+{i+1}'
    #     ))

    # fig.update_layout(
    #     title='VWAP Prediction',
    #     xaxis_title='Date',
    #     yaxis_title='Price (R$)',
    #     showlegend=True
    # )
    
    # Convert the Plotly figure to JSON
    #plot_json = to_json(plt)
    
    print(plot_json2)
    
    return json.dumps({'status': '0', 'plot': plot_json, 'plot2': plot_json2, 'data': data_plot})

@b_curto_prazo.route("/_rede_neural", methods=["POST"])
def _rede_neural():
    day = int(request.form.get('date'))
    return rede(day) 
