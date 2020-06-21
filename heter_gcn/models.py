from gcn.layers import *
from gcn.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    # base model builder class
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        # set model name
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        # stores layers and activations
        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        # put all underlying nodes in a model-layer variable_scope
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        # _build only defines the layers
        # have not create nodes in computation graph until this step
        self.activations.append(self.inputs)
        for layer in self.layers:
            # take previous layer's outputs as inputs
            hidden = layer(self.activations[-1])
            # store the output node of each layer in a list
            self.activations.append(hidden)
        # final output of the model
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        # create a mapping dictionary
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        # create optimize operation
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        # restore parameters, but still have to recreate computation graph
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GeneralizedModel(Model):
    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)
        
    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self._loss()

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)


class RGCN(GeneralizedModel):
    def __init__(self, placeholders, hidden_1, emb_size, neg_sample_size, learning_rate, **kwargs):
        super(RGCN, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = self.inputs.get_shape().as_list()[1]
        self.output_dim = emb_size
        self.hidden_1_dim = hidden_1
        self.neg_sample_size = neg_sample_size
        self.placeholders = placeholders
        self.logging = False

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.build()

    def _build(self):
        self.layer_1 = GraphConvolution(input_dim=self.input_dim,
                                    output_dim=self.hidden_1_dim,
                                    placeholders=self.placeholders,
                                    act=tf.nn.relu,
                                    dropout=True,
                                    logging=self.logging)
        
        self.layer_2 = GraphConvolution(input_dim=self.hidden_1_dim,
                                   output_dim=self.output_dim,
                                   placeholders=self.placeholders,
                                   act=lambda x:x,
                                   dropout=True,
                                   logging=self.logging)
        
        self.hidden_1 = self.layer_1(self.inputs)
        self.outputs = self.layer_2(self.hidden_1)
        self.outputs = tf.nn.l2_normalize(self.outputs, 1)

        homo_sampling_idx = self.placeholders['homo_samples']
        heter_sampling_idx = self.placeholders['heter_samples']
        neg_sampling_idx = self.placeholders['neg_samples']

        self.homo_samples = tf.stop_gradient(tf.gather(self.outputs, homo_sampling_idx))
        self.heter_samples = tf.stop_gradient(tf.gather(self.outputs, heter_sampling_idx))

        neg_samples_list = list()
        for i in range(self.neg_sample_size):
            neg_samples_list.append(tf.stop_gradient(tf.gather(self.outputs, neg_sampling_idx[:, i])))
        # shape: (num_nodes, neg_sample_size, input_dim)
        self.neg_samples = tf.stack(neg_samples_list, axis=1)
        
        self.skip_gram = SkipGramLayer(self.output_dim, self.placeholders, 
                         heter_weights=FLAGS.heter_weights, neg_sample_weights=FLAGS.neg_sample_weights)

    def _loss(self):
        # l2 loss for first conv layer
        for var in self.layer_1.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss += self.skip_gram.loss(self.outputs, self.homo_samples, self.heter_samples, self.neg_samples)


class SemiSupRGCN(GeneralizedModel):
    def __init__(self, placeholders, hidden_1, emb_size, neg_sample_size, learning_rate, num_users, **kwargs):
        super(SemiSupRGCN, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = self.inputs.get_shape().as_list()[1]
        self.output_dim = emb_size
        self.hidden_1_dim = hidden_1
        self.neg_sample_size = neg_sample_size
        self.placeholders = placeholders
        self.num_users = num_users
        self.logging = False

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.build()

    def _build(self):
        self.layer_1 = GraphConvolution(input_dim=self.input_dim,
                                    output_dim=self.hidden_1_dim,
                                    placeholders=self.placeholders,
                                    act=tf.nn.relu,
                                    dropout=True,
                                    logging=self.logging)
        
        self.layer_2 = GraphConvolution(input_dim=self.hidden_1_dim,
                                   output_dim=self.output_dim,
                                   placeholders=self.placeholders,
                                   act=lambda x:x,
                                   dropout=True,
                                   logging=self.logging)
        
        self.hidden_1 = self.layer_1(self.inputs)
        self.outputs = self.layer_2(self.hidden_1)
        self.outputs = tf.nn.l2_normalize(self.outputs, 1)

        homo_sampling_idx = self.placeholders['homo_samples']
        heter_sampling_idx = self.placeholders['heter_samples']
        neg_sampling_idx = self.placeholders['neg_samples']

        self.homo_samples = tf.stop_gradient(tf.gather(self.outputs, homo_sampling_idx))
        self.heter_samples = tf.stop_gradient(tf.gather(self.outputs, heter_sampling_idx))

        neg_samples_list = list()
        for i in range(self.neg_sample_size):
            neg_samples_list.append(tf.stop_gradient(tf.gather(self.outputs, neg_sampling_idx[:, i])))
        # shape: (num_nodes, neg_sample_size, input_dim)
        self.neg_samples = tf.stack(neg_samples_list, axis=1)
        
        semi_sup_sampling_idx = self.placeholders['semi_sup_samples']
        self.semi_sup_samples = tf.stop_gradient(tf.gather(self.outputs, semi_sup_sampling_idx))

        self.skip_gram = SemiSupSkipGramLayer(self.output_dim, self.placeholders, 
                         self.num_users, semi_sup_weights=FLAGS.semi_sup_weights,
                         heter_weights=FLAGS.heter_weights, neg_sample_weights=FLAGS.neg_sample_weights)

    def _loss(self):
        # l2 loss for first conv layer
        for var in self.layer_1.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss += self.skip_gram.loss(self.outputs, self.homo_samples, 
                     self.heter_samples, self.neg_samples, self.semi_sup_samples)