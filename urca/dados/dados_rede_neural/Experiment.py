import os
import sys
import json
import shutil
#sys.path.insert(0, '..') # to import modules from parent folder 

import pandas as pd
import numpy as np

# +
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

def AjustMeanSquaredError(y_true, y_pred):  
    # https://dx.doi.org/10.2139/ssrn.3973086
    diff_squared = np.square(y_pred - y_true)
    signal_err = (np.multiply(y_pred, y_true) < 0).astype(float)
    wrong = diff_squared * signal_err * 2.5
    right = diff_squared * (1 - signal_err)
    return np.mean(wrong + right)

def fuzzy_accuracy(y_true, y_pred):
    #print(f'y_true {y_true.shape}: sum {})
    return np.mean(np.stack([y_true, y_pred]).min(axis=0).sum(axis=-1))

class FuzzyTendency():
    def __init__(self, stag_trig=0.1, var_trig=0.15):
        self.stag_trig = stag_trig
        self.var_trig = var_trig
        
    def __call__(self, y_prop):
        stag_cresc = np.clip((y_prop-self.stag_trig)/(self.var_trig-self.stag_trig), 0, 1)
        descr_stag = np.clip((-y_prop-self.stag_trig)/(self.var_trig-self.stag_trig), 0, 1)
        
        return np.stack([
            stag_cresc,
            1-(stag_cresc+descr_stag),
            descr_stag,
        ], axis=-1)

# deve:
#   executar diversos modelos diferentes com configurações diferentes, sem precisar de mt adaptação
#   salvar progresso e retomar, tratando erros ocorridos
class ExperimentAbstract(ABC):
    def __init__(self, experiment_path, load=False, **experiment_kwargs):
        self.experiment_path = experiment_path
        if not load:
            os.makedirs(experiment_path)
            
            self._kwargs = experiment_kwargs.copy()
            self.save()
        else:
            with open(os.path.join(experiment_path, "kwargs.json"), "r") as f:
                self._kwargs = json.load(f)
        for kwg in ["eval_kwargs", "train_kwargs"]:
            if kwg not in self._kwargs:
                self._kwargs[kwg] = {}
                
    def save(self, file=None):
        if file is None:
            file = os.path.join(self.experiment_path, "kwargs.json")
        with open(file, "w") as f:
                json.dump(self._kwargs, f)
    
    def create_model(self, model_dir):
        return self.models_dict[self._kwargs["model_type"]](model_dir, **self._kwargs["model_kwargs"])
    
    def load_model(self, model_dir):
        return self.models_dict[self._kwargs["model_type"]].load(model_dir)
    
    @property
    @abstractmethod 
    def models_dict(self):
        pass
    
    @property
    @abstractmethod
    def train_data(self):
        pass
    
    @property
    @abstractmethod
    def val_data(self):
        pass
    
    @property
    @abstractmethod
    def test_data(self):
        pass
    
    def transform_y(self, x, y):
        return y
    
    def transform_y_prop(self, x, y):
        return y/x
    
    @staticmethod
    def evaluate_complete(model, x, y, verbose=1, transform_y=None, transform_prop=None, *args, **kwargs):
        #print("x:", [i.shape for i in x])
        #print("y:", y.shape)
        y_predicted_orig = model.predict(x, *args, **kwargs)#.reshape((*y.shape[:-2], -1))
        y_orig = y.copy()#.reshape((*y.shape[:-2], -1))
        
        #print("y_predicted_orig:", y_predicted_orig.shape)
        #print("y_orig:", y_orig.shape)
        
        y_predicted = y_predicted_orig.copy()
        y = y_orig.copy()
            
        if transform_y:
            y_predicted = transform_y(x, y_predicted)
            y = transform_y(x, y)
            
        y_predicted = y_predicted[:, :, 0]
        y = y[:, :, 0]
        #print(f"y_predicted.shape: {y_predicted.shape}")
        #print(f"y.shape: {y.shape}")
            
        metrics_func = {
          "mae" : mean_absolute_error,
          "mape (%)" :lambda *a: mean_absolute_percentage_error(*a)*100,
          "mse" : mean_squared_error,
          "rmse" : lambda *a: mean_squared_error(*a, squared=False),
          "r2_score" : r2_score,
          "mean_bias_error" : lambda y, y_pred: np.mean(y_pred - y),
          "AdjMSE" : AjustMeanSquaredError  
        }
        metrics = {}
        for metric in metrics_func:
            #try:
            metrics[metric] = float(metrics_func[metric](y, y_predicted))
            if verbose:
                print(f'{metric}: {metrics[metric]}')
            #except Exception as error:
                #metrics[metric] = None
                #if verbose:
                    #print(f"Error for computing {metric}", error)
        
        y_predicted = y_predicted_orig.copy()
        y = y_orig.copy()
        if transform_prop:
            #print(f"y_predicted.shape: {y_predicted.shape}")
            #print(f"y.shape: {y.shape}")
            y_predicted_prop = transform_prop(x, y_predicted)
            y_prop = transform_prop(x, y)
            #print(f"y_predicted_prop {y_predicted_prop.shape}: {y_predicted_prop[0]}")
            #print(f"y_prop.shape{y_prop.shape}: {y_prop[0]}")
        
        fuzzy_set = FuzzyTendency()
        metrics["fuzzy_accuracy (%)"] = fuzzy_accuracy(fuzzy_set(y_predicted_prop), fuzzy_set(y_prop)) * 100
        #print(f"y_predicted_prop {fuzzy_set(y_predicted_prop).shape}: {fuzzy_set(y_predicted_prop)[0]}")
        #print(f"y_prop.shape{fuzzy_set(y_prop).shape}: {fuzzy_set(y_prop)[0]}")
        if verbose:
            print(f'fuzzy_accuracy (%): {metrics["fuzzy_accuracy (%)"]}')
        return metrics
    
    def run(self, verbose=False):
        info_file = os.path.join(self.experiment_path, "info.json")
        if os.path.isfile(info_file):
            with open(info_file, "r") as f:
                info = json.load(f)
        else:
            info = {}
            with open(info_file, "w") as f:
                json.dump(info, f)
        
        excs = {}
        for i in range(self._kwargs["times"]):
            if i in info:
                continue
            
            model_dir = os.path.join(self.experiment_path, str(i))
            os.mkdir(model_dir)
            model = self.create_model(model_dir)
            model.train(self.train_data, self.val_data, verbose=verbose, **self._kwargs["train_kwargs"])
            try:
                pass
            except Exception as e:
                excs[i] = str(e)
                with open(os.path.join(self.experiment_path, "excs.json"), "w") as f:
                    json.dump(excs, f)
                shutil.rmtree(model_dir)
            else:
                info[i] = {
                    "tr_err":  self.evaluate_complete(
                        model, *self.train_data, transform_y=self.transform_y, 
                        transform_prop=self.transform_y_prop, verbose=verbose, **self._kwargs["eval_kwargs"]),
                    "val_err": self.evaluate_complete(
                        model, *self.val_data, transform_y=self.transform_y, 
                        transform_prop=self.transform_y_prop, verbose=verbose, **self._kwargs["eval_kwargs"]),
                    "tst_err": self.evaluate_complete(
                        model, *self.test_data, transform_y=self.transform_y, 
                        transform_prop=self.transform_y_prop, verbose=verbose, **self._kwargs["eval_kwargs"])
                }
                with open(info_file, "w") as f:
                    json.dump(info, f)
        err = {
            "tr_err": {},
            "val_err": {},
            "tst_err": {},
        }
        for i in info:
            for e in err:
                err[e][i] = info[i][e]
        for e in err:
            err[e] = pd.DataFrame(err[e]).T
            err[e].to_csv(os.path.join(self.experiment_path, e+".csv"))
        
        return err
    
    def reevaluate(self, verbose=False):
        info = {}
        for i in range(self._kwargs["times"]):
            model_dir = os.path.join(self.experiment_path, str(i))
            #print("Antes de carregar")
            model = self.load_model(model_dir)
            #print("Dps de carregar")
            #print(model)
            info[i] = {
                    "tr_err":  self.evaluate_complete(
                        model, *self.train_data, transform_y=self.transform_y, 
                        transform_prop=self.transform_y_prop, verbose=verbose, **self._kwargs["eval_kwargs"]),
                    "val_err": self.evaluate_complete(
                        model, *self.val_data, transform_y=self.transform_y, 
                        transform_prop=self.transform_y_prop, verbose=verbose, **self._kwargs["eval_kwargs"]),
                    "tst_err": self.evaluate_complete(
                        model, *self.test_data, transform_y=self.transform_y, 
                        transform_prop=self.transform_y_prop, verbose=verbose, **self._kwargs["eval_kwargs"])
            }
        err = {
            "tr_err": {},
            "val_err": {},
            "tst_err": {},
        }
        for i in info:
            for e in err:
                err[e][i] = info[i][e]
        for e in err:
            err[e] = pd.DataFrame(err[e]).T
            #err[e].to_csv(os.path.join(self.experiment_path, e+".csv"))
        
        return err
    
    @property
    def errors(self):
        err = {
            "tr_err": {},
            "val_err": {},
            "tst_err": {},
        }
        for e in err:
            err[e] = pd.read_csv(os.path.join(self.experiment_path, e+".csv"), index_col=0)
        
        return err


# -


if __name__ == "__main__": 
    y = np.arange(-0.30, 0.31, 0.01)
    lst_x = 1

    ft = FuzzyTendency(0.1, 0.15)

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    r = ft(y/lst_x)
    ax.plot(y*100, r[:, 0], label="cresc")
    ax.plot(y*100, r[:, 1], label="stag")
    ax.plot(y*100, r[:, 2], label="descr")
    ax.set_ylabel("pertinência")
    ax.set_xlabel("diferença percentual do preço (%)")
    ax.legend()

if __name__ == "__main__": 
    y0 = ft(np.array([[[0.10, 0.11, 0.12, 0.13, 0.14, 0.15]]]))
    y1 = ft(np.array([[[0.15, 0.14, 0.13, 0.12, 0.11, 0.10]]]))
    print(y0)
    print(y1)
    print(np.stack([y0, y1]).min(axis=0).sum(axis=-1))

if __name__ == "__main__":
    np.array([[1, 1, 1], [1, 1, 1]]).sum(axis=1)

if __name__ == "__main__": 
    file = "/home/lucasyuki/Documents/dados_preco/linear interpol/rolloff suavizado M+1 SE -> VWAP.csv"
    VWAP = pd.read_csv(file, index_col=0).set_index("data").VWAP#.iloc[:100]
    VWAP.index = VWAP.index.astype('datetime64[ns]')
    VWAP_prop = ((VWAP.shift(-15)-VWAP)/VWAP.shift(-15)).dropna()
    ft = FuzzyTendency(0.1, 0.15)

    from matplotlib import pyplot as plt
    r = np.array(ft(VWAP_prop))
    dates = np.array(VWAP_prop.index)

if __name__ == "__main__": 
    #import matplotlib.dates as mdates
    import matplotlib.ticker as ticker
    
    k = pd.DataFrame(index=VWAP.index, columns=["cresc", "stag", "descr"])
    k.update(pd.DataFrame(data=r, index=VWAP_prop.index, columns=["cresc", "stag", "descr"]))
    k = k[["descr", "stag", "cresc"]]
    
    
    fig, ax = plt.subplots(2, figsize=(16, 9), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    fig.subplots_adjust(hspace=0)
    
    color={"cresc":"g", "stag":"b", "descr":"r"}
    #k.plot.bar(stacked=True, width=1, ax=ax[1], )
    k_index = k.index.to_numpy()
    bottom = np.zeros(len(k_index))
    for col in ["descr", "stag", "cresc"]:
        p = ax[1].bar(k_index, k[col].to_numpy(), k_index[1]-k_index[0], 
                      color=color[col], alpha=0.7, label=col, bottom=bottom)
        bottom += k[col]
    ax[1].legend(loc='center left')
    
    #ax[0].set_ylabel("Difference 15 days ahead (% truncated)")
    ax[0].set_ylabel("Diferença percentual 15 dias a frente (% truncado)")
    prop_percent = ((VWAP.shift(-15)-VWAP)/VWAP.shift(-15)*100).clip(-100, 100)
    ax[0].plot(prop_percent.index.to_numpy(), prop_percent.to_numpy(), label="diff d+15 (%)")
    ax[0].fill_between(VWAP.index, -10, 10, facecolor='blue', alpha=0.3)
    ax[0].fill_between(VWAP.index, 15, 100, facecolor='green', alpha=0.3)
    ax[0].fill_between(VWAP.index, -15, -100, facecolor='red', alpha=0.3)
    
    axVWAP = ax[0].twinx()
    axVWAP.set_ylabel("VWAP rolloff suavizado M+1 SE (R$)")
    axVWAP.plot(VWAP.index.to_numpy(), VWAP.to_numpy(), "k--", label="VWAP")
    
    ax[0].vlines(["2021-09-15", "2022-05-02", "2022-12-17"], ymin=-100, ymax=100, colors="r")
    plt.xlim(k_index.min(), k_index.max())
    ax[0].legend(loc='upper left')
    axVWAP.legend(loc='upper right')
    
    """
    # Make most of the ticklabels empty so the labels don't get too crowded
    days = k.index.to_series()
    ticklabels = np.array([""]*len(k.index), dtype=object)
    # Every 4th ticklable shows the month and day
    day1 = (days.dt.day == 1).values
    ticklabels[day1] = days[day1].dt.month
    # Every 12th ticklabel includes the year
    month1 = ((days.dt.day == 1) & (days.dt.month == 1)).values
    ticklabels[month1] = days[month1].dt.year
    
    ax[1].xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
    plt.gcf().autofmt_xdate()
    """

    fig.show()
