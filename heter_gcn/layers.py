from gcn.inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.featureless = featureless
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)])
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class SkipGramLayer(Layer):
    def __init__(self, input_dim, placeholders, heter_weights=1.0, neg_sample_weights=1.0, **kwargs):
        super(SkipGramLayer, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.heter_weights = heter_weights
        self.neg_sample_weights = neg_sample_weights

    def affinity(self, inputs1, inputs2):
        # element-wise production
        # 1-D tensor of shape (batch_size, )
        result = tf.reduce_sum(inputs1 * inputs2, axis=1)
        return result

    def neg_cost(self, inputs1, neg_samples):
        # neg sample size: (batch_size, num_neg_samples, input_dim)
        # (batch_size, 1, input_dim)
        inputs1_reshaped = tf.expand_dims(inputs1, axis=1)
        # tensor of shape (batch_size, 1, num_neg_samples)
        neg_aff = tf.matmul(inputs1_reshaped, tf.transpose(neg_samples, perm=[0, 2, 1]))
        # squeeze
        neg_aff = tf.squeeze(neg_aff, [1])
        return neg_aff

    def loss(self, inputs1, inputs2_u, inputs2_l, neg_samples):
        return self._skipgram_loss(inputs1, inputs2_u, inputs2_l, neg_samples)

    def _skipgram_loss(self, inputs1, inputs2_u, inputs2_l, neg_samples):
        aff_1 = self.affinity(inputs1, inputs2_u)
        aff_2 = self.affinity(inputs1, inputs2_l)
        neg_cost = self.neg_cost(inputs1, neg_samples)
        true_1_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff_1), logits=aff_1)
        true_2_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff_2), logits=aff_2)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_cost), logits=neg_cost)
        loss = tf.reduce_mean(true_1_xent) + self.heter_weights * tf.reduce_mean(true_2_xent) + \
               self.neg_sample_weights * tf.reduce_mean(tf.reduce_sum(negative_xent, axis=1))
        return loss


class SemiSupSkipGramLayer(Layer):
    def __init__(self, input_dim, placeholders, num_users,
                semi_sup_weights=1.0, heter_weights=1.0, neg_sample_weights=1.0, **kwargs):
        super(SemiSupSkipGramLayer, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.heter_weights = heter_weights
        self.neg_sample_weights = neg_sample_weights
        self.semi_sup_weights = semi_sup_weights
        self.num_users = num_users

    def affinity(self, inputs1, inputs2):
        # element-wise production
        # 1-D tensor of shape (batch_size, )
        result = tf.reduce_sum(inputs1 * inputs2, axis=1)
        return result

    def neg_cost(self, inputs1, neg_samples):
        # neg sample size: (batch_size, num_neg_samples, input_dim)
        # (batch_size, 1, input_dim)
        inputs1_reshaped = tf.expand_dims(inputs1, axis=1)
        # tensor of shape (batch_size, 1, num_neg_samples)
        neg_aff = tf.matmul(inputs1_reshaped, tf.transpose(neg_samples, perm=[0, 2, 1]))
        # squeeze
        neg_aff = tf.squeeze(neg_aff, [1])
        return neg_aff

    def loss(self, inputs1, inputs2_u, inputs2_l, neg_samples, semi_sup_samples):
        return self._skipgram_loss(inputs1, inputs2_u, inputs2_l, neg_samples, semi_sup_samples)

    def _skipgram_loss(self, inputs1, inputs2_u, inputs2_l, neg_samples, semi_sup_samples):
        aff_1 = self.affinity(inputs1, inputs2_u)
        aff_2 = self.affinity(inputs1, inputs2_l)
        aff_semi_sup = self.affinity(inputs1[:self.num_users], semi_sup_samples)
        neg_cost = self.neg_cost(inputs1, neg_samples)
        true_1_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff_1), logits=aff_1)
        true_2_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff_2), logits=aff_2)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_cost), logits=neg_cost)
        semi_sup_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff_semi_sup), logits=aff_semi_sup)
        loss = tf.reduce_mean(true_1_xent) + self.heter_weights * tf.reduce_mean(true_2_xent) + \
               self.neg_sample_weights * tf.reduce_mean(tf.reduce_sum(negative_xent, axis=1)) + \
               self.semi_sup_weights * tf.reduce_mean(semi_sup_xent)
        return loss