from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
from keras.models import Model
from keras.layers import MaxPooling3D, Conv2D, GlobalAveragePooling2D, Concatenate, Lambda, ConvLSTM2D, Conv3D
from keras.layers import TimeDistributed, Multiply, Add, UpSampling2D, BatchNormalization, ReLU, Dropout
from configs import *
from keras.layers import Input, Layer, Dense
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K


class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907

    x=[batch, node, C], adj = [batch, n, n] --> [batch, node, OutChannel]
    """

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        super(GraphConvolution, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        output_shape = features_shape[:-1] + (self.units,)
        return output_shape  # (batch_size, node,  output_dim)

    def build(self, input_shape):
        features_shape = input_shape[0]
        assert len(features_shape) == 3
        input_dim = features_shape[2]

        self.kernel = self.add_weight(shape=(input_dim,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1]
        supports = K.batch_dot(basis, features)
        output = K.dot(supports, self.kernel)

        if self.use_bias:
            output = output + self.bias
        return self.activation(output)


class SGcn(Layer):
    def __init__(self, out_channels, **kwargs):
        self.out_channels = out_channels

        # self.bn1 = BatchNormalization()
        super(SGcn, self).__init__(**kwargs)

    def build(self, input_shape):
        self.size = input_shape
        self.sim_embed1 = Dense(input_shape[-1])
        self.sim_embed2 = Dense(input_shape[-1])

        self.graph1 = GraphConvolution(input_shape[-1], activation='relu')
        self.graph2 = GraphConvolution(input_shape[-1], activation='relu')
        self.graph3 = GraphConvolution(input_shape[-1], activation='relu')

        super(SGcn, self).build(input_shape)

    def call(self, inputs, **kwargs):
        n, h, w, c = self.size
        inputs = tf.reshape(inputs, [n, h * w, c])
        adj = self.get_adj(inputs)

        outs = self.graph1([inputs, adj])
        outs = self.graph2([outs, adj])
        outs = self.graph3([outs, adj])

        # outs = self.bn1(outs)

        outs = tf.reduce_mean(outs, 1)
        outs = tf.expand_dims(outs, -2)
        outs = tf.expand_dims(outs, -2)  # [N,T,1,1,C]

        return outs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1, self.out_channels)

    def get_adj(self, x):
        sim1 = self.sim_embed1(x)
        sim2 = self.sim_embed2(x)
        adj = tf.matmul(sim1, tf.transpose(sim2, [0, 2, 1]))  # d x d mat.
        adj = tf.nn.softmax(adj)
        return adj


def feature_extractor(shapes=(batch_size, input_t, input_shape[0], input_shape[1], 3)):
    inputs = Input(batch_shape=shapes)

    x = Conv3D(filters=64, kernel_size=3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling3D((1, 2, 2))(x)

    x = Conv3D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling3D((1, 2, 2))(x)

    x = Conv3D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling3D((1, 2, 2))(x)

    x = Conv3D(filters=512, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(filters=512, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(filters=512, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    model = Model(inputs=inputs, outputs=x)

    return model


def my_net(x, y, stateful=False):
    encoder = feature_extractor()
    seg_encoder = feature_extractor()

    x = encoder(x)
    y = seg_encoder(y)
    y = TimeDistributed(SGcn(512))(y)

    outs = Multiply()([x, y])

    outs = ConvLSTM2D(filters=256, kernel_size=3, padding='same', stateful=stateful)(outs)

    outs = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(outs)
    outs = BatchNormalization()(outs)
    outs = ReLU()(outs)
    outs = UpSampling2D(4, interpolation='bilinear')(outs)
    outs = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid')(outs)
    outs = UpSampling2D(2, interpolation='bilinear')(outs)

    return outs, outs, outs


if __name__ == '__main__':
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    session = tf.Session(config=config)  # 设置session KTF.set_session(sess)

    batch_size, input_t, input_shape = [4, 5, (256, 192)]

    x = Input(batch_shape=(32, 5, 256, 192, 3))
    y = Input(batch_shape=(32, 5, 256, 192, 3))
    m = Model(inputs=[x, y], outputs=my_net(x, y, ))
    print("Compiling MyNet")
    m.summary()
