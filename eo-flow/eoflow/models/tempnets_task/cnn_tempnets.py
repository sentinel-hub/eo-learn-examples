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


class TCNModel(BaseTempnetsModel):
    """ Implementation of the TCN network taken form the keras-TCN implementation

        https://github.com/philipperemy/keras-tcn
    """

    class TCNModelSchema(BaseTempnetsModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)

        kernel_size = fields.Int(missing=2, description='Size of the convolution kernels.')
        nb_filters = fields.Int(missing=64, description='Number of convolutional filters.')
        nb_conv_stacks = fields.Int(missing=1)
        dilations = fields.List(fields.Int, missing=[1, 2, 4, 8, 16, 32], description='Size of dilations used in the '
                                                                                      'covolutional layers')
        padding = fields.String(missing='CAUSAL', validate=OneOf(['CAUSAL', 'SAME']),
                                description='Padding type used in convolutions.')
        use_skip_connections = fields.Bool(missing=True, description='Flag to whether to use skip connections.')
        return_sequences = fields.Bool(missing=False, description='Flag to whether return sequences or not.')
        activation = fields.Str(missing='linear', description='Activation function used in final filters.')
        kernel_initializer = fields.Str(missing='he_normal', description='method to initialise kernel parameters.')
        kernel_regularizer = fields.Float(missing=0, description='L2 regularization parameter.')

        batch_norm = fields.Bool(missing=False, description='Whether to use batch normalisation.')
        layer_norm = fields.Bool(missing=False, description='Whether to use layer normalisation.')

    def _cnn_layer(self, net):

        dropout_rate = 1 - self.config.keep_prob

        layer = tf.keras.layers.Conv1D(filters= self.config.nb_filters,
                                       kernel_size=self.config.kernel_size,
                                       padding=self.config.padding,
                                       kernel_initializer=self.config.kernel_initializer,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer = tf.keras.layers.BatchNormalization(axis=-1)(layer)

        layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        layer = tf.keras.layers.Activation(self.config.activation)(layer)
        return layer

    def build(self, inputs_shape):
        """ Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(inputs_shape[1:])

        dropout_rate = 1 - self.config.keep_prob

        net = x

        net = self._cnn_layer(net)

        # list to hold all the member ResidualBlocks
        residual_blocks = []
        skip_connections = []

        total_num_blocks = self.config.nb_conv_stacks * len(self.config.dilations)
        if not self.config.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for _ in range(self.config.nb_conv_stacks):
            for d in self.config.dilations:
                net, skip_out = ResidualBlock(dilation_rate=d,
                                              nb_filters=self.config.nb_filters,
                                              kernel_size=self.config.kernel_size,
                                              padding=self.config.padding,
                                              activation=self.config.activation,
                                              dropout_rate=dropout_rate,
                                              use_batch_norm=self.config.batch_norm,
                                              use_layer_norm=self.config.layer_norm,
                                              kernel_initializer=self.config.kernel_initializer,
                                              last_block=len(residual_blocks) + 1 == total_num_blocks,
                                              name=f'residual_block_{len(residual_blocks)}')(net)
                residual_blocks.append(net)
                skip_connections.append(skip_out)


        # Author: @karolbadowski.
        output_slice_index = int(net.shape.as_list()[1] / 2) \
            if self.config.padding.lower() == 'same' else -1
        lambda_layer = tf.keras.layers.Lambda(lambda tt: tt[:, output_slice_index, :])

        if self.config.use_skip_connections:
            net = tf.keras.layers.add(skip_connections)

        if not self.config.return_sequences:
            net = lambda_layer(net)

        net = tf.keras.layers.Dense(1, activation='linear')(net)

        self.net = tf.keras.Model(inputs=x, outputs=net)

        #print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)



class TempCNNModel(BaseTempnetsModel):
    """ Implementation of the TempCNN network taken from the temporalCNN implementation

        https://github.com/charlotte-pel/temporalCNN
    """

    class TempCNNModelSchema(BaseTempnetsModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)
        kernel_size = fields.Int(missing=5, description='Size of the convolution kernels.')
        nb_conv_filters = fields.Int(missing=16, description='Number of convolutional filters.')
        nb_conv_stacks = fields.Int(missing=3, description='Number of convolutional blocks.')
        nb_conv_strides = fields.Int(missing=1, description='Value of convolutional strides.')
        nb_fc_neurons = fields.Int(missing=256, description='Number of Fully Connect neurons.')
        nb_fc_stacks = fields.Int(missing=1, description='Number of fully connected layers.')
        final_layer = fields.String(missing='Flatten', validate=OneOf(['Flatten','GlobalAveragePooling1D', 'GlobalMaxPooling1D']),
                                    description='Final layer after the convolutions.')
        padding = fields.String(missing='SAME', validate=OneOf(['SAME','VALID', 'CAUSAL']),
                                description='Padding type used in convolutions.')
        activation = fields.Str(missing='relu', description='Activation function used in final filters.')
        fc_activation = fields.Str(missing=None, description='Activation function used in final FC layers.')
        kernel_initializer = fields.Str(missing='he_normal', description='Method to initialise kernel parameters.')
        kernel_regularizer = fields.Float(missing=1e-6, description='L2 regularization parameter.')
        enumerate = fields.Bool(missing=False, description='Increase number of filters across convolution')
        batch_norm = fields.Bool(missing=False, description='Whether to use batch normalisation.')

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

    def _embeddings(self,net):

        name = "embedding"
        if self.config.final_layer == 'Flatten':
            net = tf.keras.layers.Flatten(name=name)(net)
        elif self.config.final_layer == 'GlobalAveragePooling1D':
            net = tf.keras.layers.GlobalAveragePooling1D(name=name)(net)
        elif self.config.final_layer == 'GlobalMaxPooling1D':
            net = tf.keras.layers.GlobalMaxPooling1D(name=name)(net)

        return net

    def _fcn_layer(self, net):
        dropout_rate = 1 - self.config.keep_prob
        layer_fcn = Dense(units=self.config.nb_fc_neurons,
                          kernel_initializer=self.config.kernel_initializer,
                          kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer_fcn = tf.keras.layers.BatchNormalization(axis=-1)(layer_fcn)

        layer_fcn = tf.keras.layers.Dropout(dropout_rate)(layer_fcn)
        if self.config.fc_activation:
            layer_fcn = tf.keras.layers.Activation(self.config.activation)(layer_fcn)

        return layer_fcn


    def build(self, inputs_shape):
        """ Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(inputs_shape[1:])

        net = x
        for i, _ in enumerate(range(self.config.nb_conv_stacks)):
            net = self._cnn_layer(net, i)

        net = self._embeddings(net)
        self.backbone = tf.keras.Model(inputs=x, outputs=net)

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



class HistogramCNNModel(BaseTempnetsModel):
    """ Implementation of the CNN2D with histogram time series

        https://cs.stanford.edu/~ermon/papers/cropyield_AAAI17.pdf
    """

    class HistogramCNNModel(BaseTempnetsModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)
        kernel_size = fields.List(fields.Int, missing=2, description='Size of the convolution kernels.')
        nb_conv_filters = fields.Int(missing=16, description='Number of convolutional filters.')
        nb_conv_stacks = fields.Int(missing=3, description='Number of convolutional blocks.')
        nb_conv_strides = fields.List(fields.Int, missing=2, description='Value of convolutional strides.')
        nb_fc_neurons = fields.Int(missing=256, description='Number of Fully Connect neurons.')
        nb_fc_stacks = fields.Int(missing=1, description='Number of fully connected layers.')
        final_layer = fields.String(missing='Flatten', validate=OneOf(['Flatten','GlobalAveragePooling2D', 'GlobalMaxPooling2D']),
                                    description='Final layer after the convolutions.')
        padding = fields.String(missing='SAME', validate=OneOf(['SAME','VALID', 'CAUSAL']),
                                description='Padding type used in convolutions.')
        activation = fields.Str(missing='relu', description='Activation function used in final filters.')
        kernel_initializer = fields.Str(missing='he_normal', description='Method to initialise kernel parameters.')
        kernel_regularizer = fields.Float(missing=1e-6, description='L2 regularization parameter.')
        enumerate = fields.Bool(missing=False, description='Increase number of filters across convolution')
        batch_norm = fields.Bool(missing=False, description='Whether to use batch normalisation.')

    def _cnn_layer(self, net, i = 1, last = False):

        dropout_rate = 1 - self.config.keep_prob
        filters = self.config.nb_conv_filters
        s_i, s_j = self.config.nb_conv_strides.copy()

        if self.config.enumerate:
            filters = filters * (2**i)
            if last:
                strides = self.config.nb_conv_strides.copy()
            else:
                strides = (s_i * (i + 1), s_i * (i + 1))

            print(strides)

        layer = tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=self.config.kernel_size,
                                       strides=strides,
                                       padding=self.config.padding,
                                       kernel_initializer=self.config.kernel_initializer,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
        if self.config.batch_norm:
            layer = tf.keras.layers.BatchNormalization(axis=-1)(layer)

        #if self.config.enumerate: layer = tf.keras.layers.MaxPool1D()(layer)

        layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        layer = tf.keras.layers.Activation(self.config.activation)(layer)
        return layer

    def _embeddings(self,net):

        name = "embedding"
        if self.config.final_layer == 'Flatten':
            net = tf.keras.layers.Flatten(name=name)(net)
        elif self.config.final_layer == 'GlobalAveragePooling2D':
            net = tf.keras.layers.GlobalAveragePooling2D(name=name)(net)
        elif self.config.final_layer == 'GlobalMaxPooling2D':
            net = tf.keras.layers.GlobalMaxPooling2D(name=name)(net)

        return net

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
        x = tf.keras.layers.Input(inputs_shape[1:])

        net = x
        for i, _ in enumerate(range(self.config.nb_conv_stacks)):
            net = self._cnn_layer(net, i)
            print(net.shape)
            net = self._cnn_layer(net, i, True)
            print(net.shape)

        net = self._embeddings(net)
        self.backbone = tf.keras.Model(inputs=x, outputs=net)

        for _ in range(self.config.nb_fc_stacks):
            net = self._fcn_layer(net)

        net = Dense(units = 1,
                    activation = 'linear',
                    kernel_initializer=self.config.kernel_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)

        self.net = tf.keras.Model(inputs=x, outputs=net)

        #print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)

    def get_feature_map(self, inputs, training=None):
        return self.backbone(inputs, training)
