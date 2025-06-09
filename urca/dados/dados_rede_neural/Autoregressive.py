# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:09:32 2020

@author: lucas
"""

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import RNN
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.keras.utils import get_custom_objects

@tf.keras.utils.register_keras_serializable(package="Autoregressive")
class Autoregressive(Model):
    """Model class for Autoregressive models.
    Arguments:
      layers: List of tensorflow layers.
      out_steps: Positive integer, number of steps predicted.
      output_features: Positive integer, number of output features.

    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    """
    def __init__(self, model_layers, out_steps, output_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_layers = model_layers
        self.out_steps = out_steps
        self.output_features = output_features
        
        self.initialize_model()
    
    def initialize_model(self):
        self.recurrent_layers = []
        for i in range(len(self.custom_layers)):
            is_recurrent = False
            if isinstance(self.custom_layers[i], DropoutRNNCellMixin):
                is_recurrent = True
                self.custom_layers[i] = RNN(self.custom_layers[i], return_state=False)
            elif isinstance(self.custom_layers[i], RNN):
                is_recurrent = True
                self.custom_layers[i].return_state = False
            self.recurrent_layers.append(is_recurrent)
            
        self._summary = []
        inp = Input(shape=(None, self.output_features))
        out = inp
        for layer in self.custom_layers:
            out = layer(out)
        temp_model = Model(inputs=inp, outputs=out)
        temp_model.summary(print_fn=lambda x: self._summary.append(x))
        del temp_model
        self._summary = "\n".join(self._summary)
        
        for i in range(len(self.custom_layers)):
            if self.recurrent_layers[i]:
                self.custom_layers[i].return_state = True
        
    
    def process(self, inputs, initial_state=None, training=None):
        # inputs.shape => (batch, time, features)
        # output.shape => (batch, time, units) or (batch, units)
        
        mode_RNN = initial_state is None
        #if mode_RNN is True, use the cell warped with RNN layer
        #if mode_RNN is False, use the raw cell        
        
        states = []
        outputs = []
        for i in range(len(self.custom_layers)):
            #print(i, "input  ", in_serialized_attributesputs.shape)
            if self.recurrent_layers[i]:
                if mode_RNN:
                    #print(self.custom_layers[i][1])
                    out, *state = self.custom_layers[i](inputs,
                                                    training=training)
                else:
                    out, state = self.custom_layers[i].cell(inputs,
                                                   states=initial_state[i], 
                                                   training=training)
            else:
                out = self.custom_layers[i](inputs, training=training)
                state = [None]
            states.append(state)
            outputs.append(out)
            inputs = out
            
        return outputs[-1], states
    
    def call(self, inputs, training=None):
        predictions = []
        state = None
        prediction = inputs

        for n in range(self.out_steps):
            # Use the last prediction as input.
            prediction, state = self.process(prediction, state, training)
            # Add the prediction to the output
            predictions.append(prediction)
        
        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        
        return predictions

    def get_config(self):
        base_config = super().get_config()
        config = {
            "model_layers": self.custom_layers,
            "out_steps": self.out_steps,
            "output_features": self.output_features
        }
        return {**base_config, **config}
    
    def str_summary(self):
        return self._summary
    """
    @classmethod
    def from_config(cls, config):
        model_layers_config = config.pop("model_layers")
        model_layers = model_layers_config
        return cls(model_layers, **config)
    """

@tf.keras.utils.register_keras_serializable(package="AutoregressiveMasked")
class AutoregressiveMasked(Autoregressive):
    """Model class for Autoregressive models.
    Arguments:
      layers: List of tensorflow layers.
      out_steps: Positive integer, number of steps predicted.

    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    """
    def __init__(self, model_layers, out_steps, input_features, output_features, *args, **kwargs):
        self.input_features = input_features
        super().__init__(model_layers, out_steps, output_features, *args, **kwargs)
        #print(self._summary)
        
    def initialize_model(self):
        self.recurrent_layers = []
        for i in range(len(self.custom_layers)):
            is_recurrent = False
            if isinstance(self.custom_layers[i], DropoutRNNCellMixin):
                is_recurrent = True
                self.custom_layers[i] = RNN(self.custom_layers[i], return_state=False)
            elif isinstance(self.custom_layers[i], RNN):
                is_recurrent = True
                self.custom_layers[i].return_state = False
            self.recurrent_layers.append(is_recurrent)
            
        self._summary = []
        inp = Input(shape=(None, self.input_features))
        out = inp
        for layer in self.custom_layers:
            out = layer(out)
        temp_model = Model(inputs=inp, outputs=out)
        temp_model.summary(print_fn=lambda x: self._summary.append(x))
        del temp_model
        self._summary = "\n".join(self._summary)
        
        for i in range(len(self.custom_layers)):
            if self.recurrent_layers[i]:
                self.custom_layers[i].return_state = True
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "input_features": self.input_features
        }
        return {**base_config, **config}
    
    
    def call(self, inp_tuple, training=None):
        inputs = inp_tuple[0]
        inputs_pred = inp_tuple[1]
        mask = inp_tuple[2]
        
        #if len(inputs.shape) == 2:
        #    inputs = inp_tuple[tf.newaxis, :, :]
        #    inputs_pred = inp_tuple[tf.newaxis, :, :]
        #    mask = inp_tuple[tf.newaxis, :, :]
        #print(inp_tuple)
        mask = tf.cast(mask, bool)

        # inputs.shape => (batch, time, features)
        # inputs_pred.shape => (batch, output_time, features)
        # mask.shape => (batch, output_time, features)
        
        # inputs_pred.shape => (output_time, batch, features)
        inputs_pred = tf.transpose(inputs_pred, [1, 0, 2])
        # mask.shape => (output_time, batch, features)
        mask = tf.transpose(mask, [1, 0, 2])
        
        prediction, state = self.process(inputs, None, training)
        predictions = [prediction]

        for n in range(1, self.out_steps):
            prediction = tf.where(mask[n-1, :, :], inputs_pred[n, :, :], prediction)
            # Use the last prediction as input.
            prediction, state = self.process(prediction, state, training)
            # Add the prediction to the output
            predictions.append(prediction)
            
        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        #print(predictions.shape)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])[:, :, -1][:, :, tf.newaxis]
        #print(predictions.shape)
        
        return predictions

get_custom_objects().update({
    'Autoregressive': Autoregressive,
    'AutoregressiveMasked': AutoregressiveMasked
})

# +
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
    from tensorflow.keras.layers import Dense, LSTM

    input_t = 30
    output_t = 15
    n_inputs = 5
    layers = [LSTM(2, return_sequences=True), 
              LSTM(2, return_sequences=False), 
              Dense(n_inputs)]
    model = Autoregressive(layers, output_t, n_inputs)
    print(model.str_summary())
    
    
    mc = ModelCheckpoint("model.weights.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True)
    
    model.compile(loss='mse', optimizer='Adam')
    
    model.fit(tf.ones((4, 30, n_inputs)), tf.ones((4, 15, n_inputs)), epochs=5, callbacks=mc)
    
    out = model(tf.ones((4, 30, n_inputs)))
    print(out)
    
    model.save_weights('my_model.weights.h5', save_format='tf')
# -

if __name__ == '__main__':
    layers = [LSTM(2, return_sequences=True), 
              LSTM(2, return_sequences=False), 
              Dense(n_inputs)]
    model2 = Autoregressive(layers, 15, 5)
    model2.compile(loss='mse', optimizer='Adam')
    
    model2.load_weights('my_model.weights.h5')
    print(model2.optimizer.variables)
    print(model.optimizer.variables)
    
    x, y = tf.ones((4, 30, n_inputs)), tf.ones((4, 15, n_inputs))
    model.fit(x, y, epochs=100)
    model2.fit(x, y, epochs=100)
    
    out = model(x)
    out2 = model2(x)
    print(out2-out)

if __name__ == '__main__':
    from IPython.display import display
    import numpy as np
    n_vars = 1
    x = [np.zeros((100, 30, n_vars)),
         np.zeros((100, 15, n_vars)),
         np.zeros((100, 15, n_vars))]
    y = np.zeros((100, 15, n_vars))
    for sample in range(100):
        for var in range(1):
            x[0][sample,   :, var] = sample+np.arange(30)*(var+1)
            x[1][sample,   :, var] = sample+(np.arange(15)+30)*(var+1)+100
            x[2][sample,   9:, var] = True
            y[sample, :10, var] = sample+(np.arange(10)+30)*(var+1)
            y[sample, 10:, var] = sample+(np.arange(5)+40)*(var+1)+100
    x[0] /= 100
    x[1] /= 100
    y = y/100
    print("input:")
    display(x[0][0, :, 0])
    print("pred:")
    display(x[1][0, :, 0])
    print("mask:")
    display(x[2][0, :, 0])
    print("output:")
    display(y[0, :, 0])

if __name__ == '__main__':
    from tensorflow.keras.callbacks import ModelCheckpoint

    from tensorflow.keras.layers import Dense, LSTM, Reshape, SimpleRNN

    n_inputs = n_vars
    layers = [SimpleRNN(n_vars, activation="linear", recurrent_dropout=0.999999999999),
             ]
    model = AutoregressiveMasked(layers, 15, n_inputs)
    print(model.str_summary())
    
    mc = ModelCheckpoint("model.weights.h5", monitor='val_loss', mode='min', 
                         verbose=1, save_best_only=True, save_weights_only=True)
    
    model.compile(loss='mse', optimizer='Adam')
    
    
    model.fit(x, y, epochs=1000, callbacks=mc)
    
    out = model(x, training=True)
    print(out)
    
    model.save_weights('my_model.weights.h5', save_format='tf')

if __name__ == '__main__':
    model.fit(x, y, epochs=100, callbacks=mc)
    out = model(x, training=True)

if __name__ == '__main__':
    from IPython.display import display
    display(y[50, :, :])

if __name__ == '__main__':
    display(np.round(out[50], 4))
