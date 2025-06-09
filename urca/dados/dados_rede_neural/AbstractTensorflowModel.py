# # +
from abc import ABC, abstractmethod
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, ConvLSTM1D

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.losses import Loss
import matplotlib.pyplot as plt


class AjustMeanSquaredError(Loss):
    # https://dx.doi.org/10.2139/ssrn.3973086
    def __init__(self, alpha=2.5, name="AdjMSE", *args, **kwargs):
        self.alpha = alpha
        super().__init__(name=name, *args, **kwargs)
    
    def call(self, y_true, y_pred, dtype=tf.dtypes.float32):
        diff_squared = tf.math.square(y_pred - y_true)
        a = tf.math.multiply(y_pred, y_true) < 0
        signal_err = tf.cast(a, dtype)
        wrong = diff_squared * signal_err * tf.cast(self.alpha, dtype)
        right = diff_squared * (1 - signal_err)
        return tf.reduce_mean(wrong + right, axis=-1)
    
    def get_config(self):
        return {
            "alpha": self.alpha,
            "name": self.name,
        }
get_custom_objects().update({'AdjMSE': AjustMeanSquaredError})


class AjustMeanSquaredErrorFuzzy(Loss):
    def __init__(self, 
                 alpha=1.5, 
                 stag_trig=0.1,  # valor máximo para stagnação
                 var_trig=0.15,  # valor mínimo para variação
                 name="AdjMSEFuzzy", 
                 *args, **kwargs):
        
        self.alpha = alpha
        self.stag_trig = stag_trig
        self.var_trig = var_trig
        super().__init__(name=name, *args, **kwargs)
    
    def call(self, y_true, y_pred, dtype=tf.dtypes.float32):
        diff_squared = tf.math.square(y_pred - y_true)
        
        y_true_fuzzy = self.convert_fuzzy(y_true)
        y_pred_fuzzy = self.convert_fuzzy(y_pred)
        accuracy = tf.reduce_sum(tf.math.minimum(y_true_fuzzy, y_pred_fuzzy), -1)
        error = tf.cast(1-accuracy, dtype) 
        
        return tf.reduce_mean(diff_squared * (error * tf.cast(self.alpha, dtype) + 1), axis=-1)
    
    def convert_fuzzy(self, y_prop):
        _stag_cresc = (y_prop-self.stag_trig)/(self.var_trig-self.stag_trig)
        stag_cresc = tf.clip_by_value(_stag_cresc, 0, 1)
        _descr_stag = (-y_prop-self.stag_trig)/(self.var_trig-self.stag_trig)
        descr_stag = tf.clip_by_value(_descr_stag, 0, 1)
        
        return tf.stack([
            stag_cresc,
            1-(stag_cresc+descr_stag),
            descr_stag,
        ], axis=-1)
    
    def get_config(self):
        return {
            "alpha": self.alpha,
            "name": self.name,
            "stag_trig": self.stag_trig,
            "var_trig": self.var_trig
        }
get_custom_objects().update({'AdjMSEFuzzy': AjustMeanSquaredErrorFuzzy})


class AbstractTensorflowModel(ABC):
    def __init__(self,
            model_dir,
            out_steps, 
            output_features,
            layers=2,
            units=32,
            noise=0.1,
            dropout=0.2,
            batch_size=32,
            loss="mse",
            loss_criteria="min",
            patience=10,
            epochs=100,
            load_model=False,
            **kwargs,
        ):
        self.model_dir = model_dir
        self.out_steps = out_steps
        self.output_features = output_features
        self.layers = layers
        self.units = units
        self.noise = noise
        self.dropout = dropout
        self.batch_size = batch_size
        self.loss = loss
        self.loss_criteria = loss_criteria
        self.patience = patience
        self.epochs = epochs
        self.kwargs = kwargs
        
        # initialize model
        self.model = self.create_model()
        #self.summary()
        if load_model:
            self.model.load_weights(os.path.join(self.model_dir, "model"))
        else:
            if os.path.exists(os.path.join(self.model_dir, "model")):
                raise FileExistsError(
                    f"Already exists a model in '{os.path.join(self.model_dir, 'model')}'." + \
                    " If you are trying to load an existing model, use the AutoregressiveModel.load(model_dir) method.")
        
            with open(os.path.join(self.model_dir, "model_structure.json"), "w") as f:
                f.write(self.model.to_json())
        self.save()
    
    def summary(self):
        try:
            summary = self.model.str_summary()
        except:
            summary = []
            self.model.summary(print_fn=lambda x: summary.append(x))
            summary = "\n".join(summary)
        print(summary)
        
    def train(self, train_data, val_data, verbose=1, transform=True):
        if transform:
            train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(self.batch_size)
            val_data = tf.data.Dataset.from_tensor_slices(val_data).batch(self.batch_size)

        callbacks = {}
        cl = CSVLogger(os.path.join(self.model_dir, "train_history.csv"), 
                       separator=',', append=False)
        mc = ModelCheckpoint(os.path.join(self.model_dir, "model"), monitor='val_loss', 
                             mode='min', verbose=verbose, save_best_only=True, save_weights_only=True)

        hist = History()

        callbacks["cl"] = cl
        callbacks["mc"] = mc
        callbacks["hist"] = hist

        if self.patience>=1:
            es = EarlyStopping(monitor='val_loss', mode=self.loss_criteria, verbose=verbose, 
                               patience=self.patience, restore_best_weights=True)
            callbacks["es"] = es

        history = self.model.fit(train_data, epochs=self.epochs, validation_data=val_data, 
                  callbacks=list(callbacks.values()), verbose=verbose)

        plt.plot(callbacks["hist"].history["loss"], label="train")
        plt.plot(callbacks["hist"].history["val_loss"], label="validation")
        plt.legend()
        plt.savefig(os.path.join(self.model_dir, "train_hist.png"))
        plt.close()
        return history
    
    def evaluate(self, x, transform=True, *args, **kwargs):
        if transform:
            x = tf.data.Dataset.from_tensor_slices(x).batch(self.batch_size)
        return self.model.evaluate(x, *args, **kwargs)
    
    def predict(self, x, verbose=1, transform=True):
        #print("[AbstractTensorflowModel]Transform:", transform)
        #print("[AbstractTensorflowModel]predict:", x)
        if transform:
            x = tf.data.Dataset.from_tensor_slices(x).batch(self.batch_size)
            #print("[AbstractTensorflowModel]predict:", x)
            
        y_batches = []
        for x_batch in x.as_numpy_iterator():
            #print(x_batch)
            y_batch = self.model(x_batch).numpy()
            y_batches.append(y_batch)
        y = np.concatenate(y_batches)
        #print("[AbstractTensorflowModel] y:", y.shape)
        return y
        #return self.model.predict(x, verbose=verbose)
    
    def save(self):
        config = {}
        config["out_steps"] = self.out_steps
        config["output_features"] = self.output_features
        config["layers"] = self.layers
        config["units"] = self.units
        config["noise"] = self.noise
        config["dropout"] = self.dropout
        config["batch_size"] = self.batch_size
        config["loss"] = self.loss
        config["loss_criteria"] = self.loss_criteria
        config["patience"] = self.patience
        config["epochs"] = self.epochs
        config["kwargs"] = self.kwargs
        with open(os.path.join(self.model_dir, "config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, model_dir):
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        config.update({"model_dir": model_dir})
        kwargs = config.pop("kwargs")
        return cls(**config, **kwargs, load_model=True)

    @property
    def layers_types(self):
        return {
            "LSTM": LSTM,
            "GRU": GRU,
            "vanilla": SimpleRNN
        }
