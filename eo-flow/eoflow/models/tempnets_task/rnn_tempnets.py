import logging
import tensorflow as tf
from marshmallow import fields
from marshmallow.validate import OneOf

from keras.layers import TimeDistributed
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from tensorflow.python.keras.utils.layer_utils import print_summary

from eoflow.models.layers import ResidualBlock
from eoflow.models.tempnets_task.tempnets_base import BaseTempnetsModel

from eoflow.models import transformer_encoder_layers
from eoflow.models import pse_tae_layers

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

rnn_layers = dict(rnn=SimpleRNN, gru=GRU, lstm=LSTM)


class BiRNN(BaseTempnetsModel):
    """ Implementation of a Bidirectional Recurrent Neural Network

    This implementation allows users to define which RNN layer to use, e.g. SimpleRNN, GRU or LSTM
    """

    class BiRNNModelSchema(BaseTempnetsModel._Schema):
        rnn_layer = fields.String(required=True, validate=OneOf(['rnn', 'lstm', 'gru']),
                                  description='Type of RNN layer to use')

        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)

        rnn_units = fields.Int(missing=64, description='Size of the convolution kernels.')
        rnn_blocks = fields.Int(missing=1, description='Number of LSTM blocks')
        bidirectional = fields.Bool(missing=True, description='Whether to use a bidirectional layer')

        activation = fields.Str(missing='relu', description='Activation function for fully connected layers')
        kernel_initializer = fields.Str(missing='he_normal', description='Method to initialise kernel parameters.')
        kernel_regularizer = fields.Float(missing=1e-6, description='L2 regularization parameter.')
        nb_fc_stacks = fields.Int(missing=0, description='Number of fully connected layers.')
        nb_fc_neurons = fields.Int(missing=0, description='Number of fully connected neurons.')

        layer_norm = fields.Bool(missing=True, description='Whether to apply layer normalization in the encoder.')
        batch_norm = fields.Bool(missing=False, description='Whether to use batch normalisation.')

    def _rnn_layer(self, net, last=False):
        """ Returns a RNN layer for current configuration. Use `last=True` for the last RNN layer. """
        RNNLayer = rnn_layers[self.config.rnn_layer]
        dropout_rate = 1 - self.config.keep_prob

        layer = RNNLayer(
            units=self.config.rnn_units,
            dropout=dropout_rate,
            return_sequences=not last,
        )

        # Use bidirectional if specified
        if self.config.bidirectional:
            layer = tf.keras.layers.Bidirectional(layer)

        return layer(net)


    def _fcn_layer(self, net):
        dropout_rate = 1 - self.config.keep_prob
        layer_fcn = Dense(units=self.config.nb_fc_neurons,
                          kernel_initializer=self.config.kernel_initializer,
                          kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)
        layer_fcn = tf.keras.layers.Activation(self.config.activation)(layer_fcn)
        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)

        return layer_fcn

    def build(self, inputs_shape):
        """ Creates the RNN model architecture. """

        x = tf.keras.layers.Input(inputs_shape[1:])
        net = x

        if self.config.layer_norm:
            net = tf.keras.layers.LayerNormalization(axis=-1)(net)

        for _ in range(self.config.rnn_blocks -1):
            net = self._rnn_layer(net)
        net = self._rnn_layer(net, last=True)

        if self.config.layer_norm:
            net = tf.keras.layers.LayerNormalization(axis=-1)(net)

        for _ in range(self.config.nb_fc_stacks):
            net = self._fcn_layer(net)

        net = tf.keras.layers.Dense(units=1,
                                      activation='linear',
                                      kernel_initializer=self.config.kernel_initializer,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)

        self.net = tf.keras.Model(inputs=x, outputs=net)

        print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)



#https://www.sciencedirect.com/science/article/pii/S0034425721003205


class ConvLSTM(BaseTempnetsModel):
    """ Implementation of a Bidirectional Recurrent Neural Network

    This implementation allows users to define which RNN layer to use, e.g. SimpleRNN, GRU or LSTM
    """


    class ConvLSTMShema(BaseTempnetsModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)
        kernel_size = fields.Int(missing=5, description='Size of the convolution kernels.')
        nb_conv_filters = fields.Int(missing=16, description='Number of convolutional filters.')
        nb_conv_stacks = fields.Int(missing=3, description='Number of convolutional blocks.')
        nb_conv_strides = fields.Int(missing=1, description='Value of convolutional strides.')
        nb_fc_neurons = fields.Int(missing=256, description='Number of Fully Connect neurons.')
        nb_fc_stacks = fields.Int(missing=1, description='Number of fully connected layers.')

        padding = fields.String(missing='SAME', validate=OneOf(['SAME','VALID', 'CAUSAL']),
                                description='Padding type used in convolutions.')
        activation = fields.Str(missing='relu', description='Activation function used in final filters.')
        kernel_initializer = fields.Str(missing='he_normal', description='Method to initialise kernel parameters.')
        kernel_regularizer = fields.Float(missing=1e-6, description='L2 regularization parameter.')
        enumerate = fields.Bool(missing=False, description='Increase number of filters across convolution')
        batch_norm = fields.Bool(missing=True, description='Whether to use batch normalisation.')

        rnn_layer = fields.String(required=True, validate=OneOf(['rnn', 'lstm', 'gru']),
                                  description='Type of RNN layer to use')
        rnn_units = fields.Int(missing=64, description='Size of the convolution kernels.')
        rnn_blocks = fields.Int(missing=1, description='Number of LSTM blocks')
        bidirectional = fields.Bool(missing=False, description='Whether to use a bidirectional layer')
        layer_norm = fields.Bool(missing=True, description='Whether to apply layer normalization in the encoder.')

    def _cnn_layer(self, net, i = 0):

        dropout_rate = 1 - self.config.keep_prob
        filters = self.config.nb_conv_filters
        kernel_size = self.config.kernel_size

        if self.config.enumerate:
            filters = filters * (2**i)
            kernel_size = kernel_size * (i+1)

        layer = tf.keras.layers.Conv1D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=self.config.nb_conv_strides,
                                       padding=self.config.padding,
                                       kernel_initializer=self.config.kernel_initializer,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer = tf.keras.layers.BatchNormalization(axis=-1)(layer)

        layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        layer = tf.keras.layers.Activation(self.config.activation)(layer)

        return layer

    def _rnn_layer(self, net, last=False):
        """ Returns a RNN layer for current configuration. Use `last=True` for the last RNN layer. """
        RNNLayer = rnn_layers[self.config.rnn_layer]
        dropout_rate = 1 - self.config.keep_prob

        layer = RNNLayer(
            units=self.config.rnn_units,
            dropout=dropout_rate,
            return_sequences=not last,
        )

        # Use bidirectional if specified
        if self.config.bidirectional:
            layer = tf.keras.layers.Bidirectional(layer)

        return layer(net)

    def _fcn_layer(self, net):
        dropout_rate = 1 - self.config.keep_prob
        layer_fcn = Dense(units=self.config.nb_fc_neurons,
                          kernel_initializer=self.config.kernel_initializer,
                          kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)

        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)
        layer_fcn = tf.keras.layers.Activation(self.config.activation)(layer_fcn)


        return layer_fcn

    def build(self, inputs_shape):
        """ Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(shape = inputs_shape[1:])
        print(x.shape)
        net = x
        for i, _ in enumerate(range(self.config.nb_conv_stacks)):
            net = self._cnn_layer(net, i)

        for i, _ in range(self.config.rnn_blocks-1):
            net = self._rnn_layer(net)
        net = self._rnn_layer(net, last=True)

        for _ in range(self.config.nb_fc_stacks):
            net = self._fcn_layer(net)

        net = Dense(units = 1,
                    activation = 'linear',
                    kernel_initializer=self.config.kernel_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)

        self.net = tf.keras.Model(inputs=x, outputs=net)

        print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)

    def get_feature_map(self, inputs, training=None):
        return self.backbone(inputs, training)

