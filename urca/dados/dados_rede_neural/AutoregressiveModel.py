import tensorflow as tf
from tensorflow.keras.layers import Dense, GaussianNoise, Dropout, LSTM, Reshape, GRU, SimpleRNN
from tensorflow.keras import Model, Sequential

from .Autoregressive import Autoregressive, AutoregressiveMasked
from .AbstractTensorflowModel import AbstractTensorflowModel

import os
import json


# +
class PropAutoregressive(Autoregressive):
    def call(self, inputs, training=None, prop=True):
        lst_observation = inputs[:, -1, :][:, tf.newaxis, :]
        out = super().call(inputs, training)
        if prop:
            return out/lst_observation - 1  # (out-lst_observation)/lst_observation
        else:
            return out-lst_observation
        
class PropAutoregressiveMasked(AutoregressiveMasked):
    def call(self, inputs, training=None, prop=True):
        #print("PropAutoregressiveMasked x:", inputs)
        #print("PropAutoregressiveMasked x spliced:", inputs[0].shape, inputs[1].shape, inputs[2].shape)
        if len(inputs[0].shape) == 2:
            lst_observation = inputs[0][-1, -1][tf.newaxis, tf.newaxis]
        else:
            lst_observation = inputs[0][:, -1, -1][:, tf.newaxis, tf.newaxis]
        out = super().call(inputs, training)
        if prop:
            return out/lst_observation - 1  # (out-lst_observation)/lst_observation
        else:
            return out-lst_observation

# +
class DiffAutoregressive(Autoregressive):
    def call(self, inputs, training=None):
        lst_observation = inputs[:, -1, :][:, tf.newaxis, :]
        out = super().call(inputs, training)
        return out-lst_observation
    
class DiffAutoregressiveMasked(AutoregressiveMasked):
    def call(self, inputs, training=None):
        lst_observation = inputs[0][:, -1, -1][:, tf.newaxis, tf.newaxis]
        out = super().call(inputs, training)
        return out-lst_observation


# -

class AutoregressiveModel(AbstractTensorflowModel):
    def create_model(self):        
        layers = [GaussianNoise(self.noise)]
        layer_type = LSTM
        if "layer_type" in self.kwargs:
            layer_type = self.layers_types[self.kwargs["layer_type"]]
        if "layer_kwargs" in self.kwargs:
            old = layer_type
            layer_type = lambda *args, **kwargs: old(*args, **self.kwargs["layer_kwargs"], **kwargs)
            
        for l in range(1, self.layers):
            layers.append(layer_type(self.units, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))
        layers.append(layer_type(self.units, return_sequences=False, recurrent_dropout=self.dropout, dropout=self.dropout))
        layers.append(Dropout(self.dropout))
        
        layers.append(Dense(self.output_features, activation="linear"))
        
        if "masked" in self.kwargs and self.kwargs["masked"]: 
            #print("MASKED!!!!!!!!")
            if "prop" in self.kwargs and self.kwargs["prop"]: 
                #print("PropAutoregressiveMasked")
                model = PropAutoregressiveMasked(layers, self.out_steps, self.kwargs["input_features"], self.output_features)
                #print(model)
            elif "diff" in self.kwargs and self.kwargs["diff"]: 
                model = DiffAutoregressiveMasked(layers, self.out_steps, self.kwargs["input_features"], self.output_features)
            else:
                model = AutoregressiveMasked(layers, self.out_steps, self.kwargs["input_features"], self.output_features)
        else:
            if "prop" in self.kwargs and self.kwargs["prop"]: 
                model = PropAutoregressive(layers, self.out_steps, self.output_features)
            elif "diff" in self.kwargs and self.kwargs["diff"]: 
                model = DiffAutoregressive(layers, self.out_steps, self.output_features)
            else:
                model = Autoregressive(layers, self.out_steps, self.output_features)
        model.compile(loss=self.loss, optimizer='Adam')
        return model


if __name__ == "__main__": 
    # https://www.tensorflow.org/tutorials/structured_data/time_series
    import os
    import datetime

    import IPython
    import IPython.display
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import tensorflow as tf

    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False

    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)
    
    df = pd.read_csv(csv_path)
    # Slice [start:stop:step], starting from index 5 take every 6th record.
    df = df[5::6]

    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    df['wv (m/s)'][df['wv (m/s)'] == -9999.0] = 0.0
    df['max. wv (m/s)'][df['max. wv (m/s)'] == -9999.0] = 0.0

    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')
    wd_rad = df.pop('wd (deg)')*np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv*np.cos(wd_rad)
    df['Wy'] = wv*np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv*np.cos(wd_rad)
    df['max Wy'] = max_wv*np.sin(wd_rad)
    
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24*60*60
    year = (365.2425)*day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    
    class WindowGenerator():
        def __init__(self, input_width, label_width, shift,
                       train_df=train_df, val_df=val_df, test_df=test_df,
                       label_columns=None):
            # Store the raw data.
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df

            # Work out the label column indices.
            self.label_columns = label_columns
            if label_columns is not None:
                self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
            self.column_indices = {name: i for i, name in
                                   enumerate(train_df.columns)}

            # Work out the window parameters.
            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift

            self.total_window_size = input_width + shift

            self.input_slice = slice(0, input_width)
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]

            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        def __repr__(self):
            return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'multi_window.plot(model)Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])
        
        def split_window(self, features):
            inputs = features[:, self.input_slice, :]
            labels = features[:, self.labels_slice, :]
            if self.label_columns is not None:
                labels = tf.stack(
                    [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                    axis=-1)

            # Slicing doesn't preserve static shape information, so set the shapes
            # manually. This way the `tf.data.Datasets` are easier to inspect.
            inputs.set_shape([None, self.input_width, None])
            labels.set_shape([None, self.label_width, None])

            return inputs, labels
        
        def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
            inputs, labels = self.example
            plt.figure(figsize=(12, 8))
            plot_col_index = self.column_indices[plot_col]
            max_n = min(max_subplots, len(inputs))
            for n in range(max_n):
                plt.subplot(max_n, 1, n+1)
                plt.ylabel(f'{plot_col} [normed]')
                plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                         label='Inputs', marker='.', zorder=-10)

                if self.label_columns:
                    label_col_index = self.label_columns_indices.get(plot_col, None)
                else:
                    label_col_index = plot_col_index

                if label_col_index is None:
                    continue

                plt.scatter(self.label_indices, labels[n, :, label_col_index],
                            edgecolors='k', label='Labels', c='#2ca02c', s=64)
                if model is not None:
                    predictions = model(inputs)
                    plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                              marker='X', edgecolors='k', label='Predictions',
                              c='#ff7f0e', s=64)

                if n == 0:
                    plt.legend()

            plt.xlabel('Time [h]')
            
        def make_dataset(self, data):
            data = np.array(data, dtype=np.float32)
            ds = tf.keras.utils.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=self.total_window_size,
              sequence_stride=1,
              shuffle=True,
              batch_size=128,)

            ds = ds.map(self.split_window)
            return ds

        @property
        def train(self):
            return self.make_dataset(self.train_df)

        @property
        def val(self):
            return self.make_dataset(self.val_df)

        @property
        def test(self):
            return self.make_dataset(self.test_df)

        @property
        def example(self):
            """Get and cache an example batch of `inputs, labels` for plotting."""
            result = getattr(self, '_example', None)
            if result is None:
                # No example batch was found, so get one from the `.train` dataset
                result = next(iter(self.train))
                # And cache it for next time
                self._example = result
            return result


    OUT_STEPS = 24
    output_features = 19

    multi_window = WindowGenerator(input_width=24,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS)    

    class MultiStepLastBaseline(tf.keras.Model):
        def call(self, inputs):
            return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

    last_baseline = MultiStepLastBaseline()
    last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanAbsoluteError()])

    multi_val_performance = {}
    multi_performance = {}

    multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val, verbose=0)
    multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)
    multi_window.plot(last_baseline)
    
    dirname = "test_autoregressive"
    
    os.mkdir(dirname)
    model = AutoregressiveModel(dirname,
            OUT_STEPS, 
            output_features,
            layers=1,
            units=32,
            noise=0.15,
            dropout=0.2,
            batch_size=128,
            loss="mse",
            loss_criteria="min",
            patience=2,
            epochs=20)
    model.train(multi_window.train, multi_window.val, transform=False)
    print(model.summary())
    
    inp = []
    out = []
    for x, y in multi_window.test:
        inp.append(x)
        out.append(y)
    inp = np.concatenate(inp, axis=0)
    out = np.concatenate(out, axis=0)
    #model.save()
    model2 = AutoregressiveModel.load(dirname)
    multi_val_performance['AR LSTM'] = model2.evaluate(multi_window.val, transform=False)
    multi_performance['AR LSTM'] = model2.evaluate(multi_window.test, verbose=0, transform=False)
    multi_window.plot(model2.model)
    
    print(model2.evaluate_complete(inp, out))
    print(multi_performance)
