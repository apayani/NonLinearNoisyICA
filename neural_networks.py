"""Reusable neural network layers"""

import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np
import tensorflow.contrib.layers
from utils import *


NP_DTYPE = np.float32
DTYPE= tf.float32
NPC = np.log2(np.exp(1))

dense = tf.layers.dense


DTYPE = tf.float32

def get_nb_params_shape(shape):
        nb_params = 1
        for dim in shape:
            nb_params = nb_params*int(dim)
        return nb_params 
def count_number_trainable_params():
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params


def FC(inputs,sizes,activations=None,name='fc'):
    if isinstance(inputs,(list,tuple)):
        inputs = tf.concat(inputs,-1)
    if not isinstance(sizes,(list,tuple)):
        sizes=[sizes]
    X = inputs
    if not isinstance(activations,(list,tuple)):
        activations = [activations] * len(sizes)

    for i in range(len(sizes)):
        if len(activations) < i-1:
            act = None
        else:
            act = activations[i]
        X = tf.layers.dense( X, sizes[i] , act , name=name+'_%d'%(i+1) )
         
    return X
        
def fc_layer(input_tensor, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer."""
    input_dim = input_tensor.get_shape()[-1].value
    with tf.variable_scope(layer_name):
        with tf.variable_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
        with tf.variable_scope('bias'):
            biases = bias_variable([output_dim])
        with tf.variable_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        if act is not None:
            activations = act(preactivate, name='activation')
        else:
            activations = preactivate
        return activations

def get_made_masks(dim_in, dim_out):
        """See eqns. (8), (9) of Germain 2015. These assume a single hidden layer auto-encoder."""
        # msh[k] is max number of input units that the k'th hidden dimension can be connected to.
        msh = np.random.randint(1, dim_in, size=dim_out)
        # Eqn (8). An element is 1 when msh[k] >= d, for d in {1 ... dim_in}
        mask_in = (msh[:, np.newaxis] >= (np.tile(range(0, dim_in), [dim_out, 1]) + 1)).astype(np.float).T
        # Eqn (9). An element is 1 when d > msh[k]
        mask_out = ((np.tile(np.arange(0, dim_in)[:, np.newaxis], [1, dim_out])+1) > msh[np.newaxis, :]).astype(np.float).T
        return mask_in, mask_out

def made_layer(input_tensor, output_dim, layer_name, act=tf.nn.relu,made_hidden_dim=300):

    

    input_dim = input_tensor.get_shape()[-1].value
            
    with tf.variable_scope(layer_name):
        with tf.variable_scope('weight_in'):
            weights_in = weight_variable([input_dim, made_hidden_dim])
        with tf.variable_scope('weight_out'):
            weights_out = weight_variable([made_hidden_dim, input_dim])
        with tf.variable_scope('masks'):
            mask_in, mask_out = get_made_masks(input_dim, made_hidden_dim)
        with tf.variable_scope('bias_in'):
            biases_in = bias_variable([made_hidden_dim])
        with tf.variable_scope('bias_out'):
            biases_out = bias_variable([input_dim])

        with tf.variable_scope('transformations'):
            hidden = tf.nn.relu(tf.matmul(input_tensor, weights_in * mask_in) + biases_in)
            preactivate = tf.matmul(hidden, weights_out * mask_out) + biases_out
            if act is not None:
                activations = act(preactivate, name='activation')
            else:
                activations = preactivate
        return activations
def made_layer2(input_tensor, output_dim, layer_name, made_hidden_dim,mask_in,mask_out, act=tf.nn.relu):
    
    

    input_dim = input_tensor.get_shape()[-1].value
    with tf.variable_scope(layer_name):
        with tf.variable_scope('weight_in'):
            weights_in = weight_variable([input_dim, made_hidden_dim])
        with tf.variable_scope('weight_out'):
            weights_out = weight_variable([made_hidden_dim, input_dim])
        with tf.variable_scope('bias_in'):
            biases_in = bias_variable([made_hidden_dim])
        with tf.variable_scope('bias_out'):
            biases_out = bias_variable([input_dim])
        # mask_i= tf.constant( np.tril( np.ones((input_dim,input_dim),dtype=np.float32) ),dtype=tf.float32)
        with tf.variable_scope('weight_i'):
            weights_i = weight_variable([input_dim, input_dim])
        

        with tf.variable_scope('transformations'):
            hidden = tf.nn.relu(tf.matmul(input_tensor, weights_in * mask_in) + biases_in)
            preactivate = tf.matmul(hidden, weights_out * mask_out) + biases_out
            if act is not None:
                activations = act(preactivate, name='activation')
            else:
                activations = preactivate
        
        return activations
def nf_layer(input_tensor, output_dim, layer_name):
    # See equations (10), (11) of Kingma 2016
    input_dim = input_tensor.get_shape()[-1].value
    with tf.variable_scope(layer_name):
        with tf.variable_scope('u'):
            u = weight_variable(input_dim)
        with tf.variable_scope('w'):
            w = weight_variable(input_dim)
        with tf.variable_scope('b'):
            b = bias_variable(1)

        with tf.variable_scope('transformations'):
            z = input_tensor
            temp = tf.expand_dims(tf.nn.tanh(tf.reduce_sum(w * z, 1) + b), 1)
            temp = tf.tile(temp, [1, output_dim])
            z = z + tf.mul(u, temp)

            temp = tf.expand_dims(dtanh(tf.reduce_sum(w * z, 1) + b), 1)
            temp = tf.tile(temp, [1, output_dim])
            log_detj = tf.log(tf.abs(1. + tf.reduce_sum(tf.mul(u, temp * w), 1)))

        return z, log_detj
def dtanh(tensor):
    return 1.0 - tf.square(tf.nn.tanh(tensor))
def norm_flow(z0,K,dim):
    z = z0
    log_detj = 0.0
     

    
    for k in range(K):
            # See equations (10), (11) of Kingma 2016
        u = tf.get_variable( 'nf_u_%d'%(k), [dim], dtype=DTYPE )
        w = tf.get_variable( 'nf_w_%d'%(k), [dim], dtype=DTYPE )
        b = tf.get_variable( 'nf_b_%d'%(k), [1], dtype=DTYPE )

        temp = tf.nn.tanh( tf.reduce_sum( tf.multiply(w, z), -1, keep_dims=True)+b)
        z = z + temp*u
        temp = dtanh( tf.reduce_sum( tf.multiply(w, z), -1, keep_dims=True) + b)
        psi = temp * w
        log_detj += tf.log(tf.abs(1. + tf.reduce_sum(u* psi, -1)))

    return z, log_detj 
def iaf(z0, K, mu_dim, log_std_dim):
    z = z0
    log_detj = 0.0
    for k in range(K):
        # See equations (10), (11) of Kingma 2016
        mu = made_layer(z, mu_dim, 'flow_mu_%d' % k)
        log_std = made_layer(z, log_std_dim, 'flow_log_std_%d' % k)
        z = (z - mu) / tf.exp(log_std)
        log_detj += -tf.reduce_sum(log_std, 1)
    return z, log_detj   
def create_gmm_1( d,K,name='gmm', reuse=False, scale_act=tf.nn.softplus,zero_mean=False):
    with tf.variable_scope( name , reuse):
        probs = tf.nn.softmax( tf.get_variable('probs', shape=[d,K] , dtype=DTYPE),axis=-1)
        locs = tf.get_variable('locs', shape=[d,K] , dtype=DTYPE)
        if zero_mean:
            locs=tf.zeros_like(locs)
        
        scales = tf.get_variable('scales', shape=[d,K] , dtype=DTYPE)

        pis = tfd.Categorical( probs = probs )
        ps = tfd.Normal( loc = locs , scale =scale_act(scales))
        p = tfd.MixtureSameFamily( pis, ps)
        p = tfd.Independent(p,1)

    return p
def create_lmm_1( d,K,name='gmm', reuse=False, scale_act=tf.nn.softplus,zero_mean=False):
    with tf.variable_scope( name , reuse):
        probs = tf.nn.softmax( tf.get_variable('probs', shape=[d,K] , dtype=DTYPE),axis=-1)
        locs = tf.get_variable('locs', shape=[d,K] , dtype=DTYPE)
        if zero_mean:
            locs=tf.zeros_like(locs)
        
        scales = tf.get_variable('scales', shape=[d,K] , dtype=DTYPE)

        pis = tfd.Categorical( probs = probs )
        ps = tfd.Laplace( loc = locs , scale =scale_act(scales))
        p = tfd.MixtureSameFamily( pis, ps)
        p = tfd.Independent(p,1)

    return p
def create_gmm_1h( h,d,K,name='gmm', reuse=False, scale_act=tf.nn.softplus,zero_mean=False):
    with tf.variable_scope( name , reuse):
        ps = tf.layers.dense( h, d*K, name='pis')
        ps = tf.reshape( ps, [-1,d,K])
        
        probs = tf.nn.softmax( ps,axis=-1)
        locs = tf.layers.dense( h, d*K, name='locs')
        locs = tf.reshape( locs, [-1,d,K])
        if zero_mean:
            locs=tf.zeros_like(locs)

        scales = tf.layers.dense( h, d*K, name='scales')
        scales = tf.reshape( scales, [-1,d,K])
        
        pis = tfd.Categorical( probs = probs )
        ps = tfd.Normal( loc = locs , scale =scale_act(scales))
        p = tfd.MixtureSameFamily( pis, ps)
        p = tfd.Independent(p,1)

    return p
def create_gmm_2( d,K,name='gmm', reuse=False, scale_act=tf.nn.softplus,zero_mean=False):
    with tf.variable_scope( name , reuse):
        probs = tf.nn.softmax( tf.get_variable('probs', shape=[d,K] , dtype=DTYPE),axis=-1)
        pis = tfd.Categorical( probs = probs )
        locs = tf.get_variable('locs', shape=[d,K] , dtype=DTYPE)
        scales = tf.get_variable('scales', shape=[d,K] , dtype=DTYPE)
        if zero_mean:
            locs=tf.zeros_like(locs)
        ps=[]
        for i in range(K):
            psi = tfd.Normal( loc = locs[:,i] , scale = scale_act(scales[:,i]) )
            ps.append( psi )
        p = tfd.Mixture( pis, ps)
        p = tfd.Independent(p,1)
    return p
 
def create_gmm_mv( d,K,name='gmm', reuse=False, scale_act=tf.nn.softplus,zero_mean=False):
    with tf.variable_scope( name , reuse):
        probs = tf.nn.softmax( tf.get_variable('probs', shape=[K] , dtype=DTYPE))
        pis = tfd.Categorical( probs = probs )
        
        locs = tf.get_variable('locs', shape=[d,K] , dtype=DTYPE)
        if zero_mean:
            locs=tf.zeros_like(locs)
        scales = tf.get_variable('scales', shape=[d,K] , dtype=DTYPE)
        ps=[]
        for i in range(K):
            psi = tfd.MultivariateNormalDiag( loc = locs[:,i], scale_diag=scale_act(scales[:,i]))
            ps.append(psi)
        
        p = tfd.Mixture( pis, ps)
        
    return p



class MLP(object):

    def __init__(self, dims, activations, stddev=1., bias_value=0.0):
        self.dims = dims
        self.activations = activations
        self.layers = []
        previous_dim = dims[0]
        for i, dim, activation in zip(range(len(activations)),
                                      dims[1:], activations):
            with tf.variable_scope('layer' + str(i)):
                weights = weight_variable(shape=(previous_dim, dim),
                                           stddev= stddev / np.sqrt(previous_dim))
                if i < len(activations) - 1:
                    biases = bias_variable((dim,), value=bias_value)
                else:
                    biases = bias_variable((dim,), value=0.0)

            self.layers.append((weights, biases, activation))
            previous_dim = dim

    def __call__(self, x, add_bias=True, return_activations=False):
        h = x
        hidden = []
        for weights, biases, activation in self.layers:
            h = tf.matmul(h, weights)
            if add_bias:
                h += biases
            if activation:
                h = activation(h)
            hidden.append(h)
        self.hidden = hidden
        if return_activations:
            return hidden
        else:
            return h

    def get_forward_derivative(self, x, fprimes):
        h = x
        for layer, fprime in zip(self.layers, fprimes):
            weights, biases, activation = layer
            h = tf.matmul(h, weights)
            h *= fprime
        return h


class MLPBlock(object):
    """Applies a separate MLP to each dimension of the input.

    The output dimensionality is assumed to be identical to the input.
    """

    def __init__(self, input_dim, hidden_dim, n_layers=1,
            stddev=1., bias_value=0.0):
        # bias value will only be applied to the hidden layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = []
        with tf.variable_scope('block_mlp'):
            self.w_in_var = weight_variable((input_dim, input_dim * hidden_dim),
                                   stddev / np.sqrt(hidden_dim),
                                   name='w_in')
            self.w_out_var = weight_variable((input_dim * hidden_dim, input_dim),
                                    stddev / np.sqrt(hidden_dim),
                                    name='w_out')
            mask = np.zeros((input_dim, input_dim * hidden_dim),
                            dtype='float32')
            hid_to_hid_mask = np.zeros((input_dim * hidden_dim,
                                        input_dim * hidden_dim),
                                       dtype='float32')
            self.bias_hid = bias_variable((hidden_dim * input_dim,),
                                          value=bias_value,
                                          name='bias_first_hid')
            self.bias_out = bias_variable((input_dim,),
                                          name='bias_out')
            for i, row in enumerate(mask):
                row[i * hidden_dim:(i + 1) * hidden_dim] = 1.0

            for i in range(0, input_dim * hidden_dim, hidden_dim):
                hid_to_hid_mask[i:i + hidden_dim, i:i + hidden_dim] = 1.0

            self.hid_to_hid_mask = tf.convert_to_tensor(hid_to_hid_mask)
            self.in_out_mask = tf.convert_to_tensor(mask)
            self.w_in = self.w_in_var * self.in_out_mask
            self.w_out = self.w_out_var * tf.transpose(self.in_out_mask)
            for i in range(n_layers - 1):
                with tf.variable_scope('layer_' + str(i)):
                    w_hid = weight_variable((input_dim * hidden_dim,
                                             input_dim * hidden_dim),
                                             stddev / np.sqrt(hidden_dim))
                    b_hid = bias_variable((hidden_dim * input_dim,),
                                          value=bias_value)
                    self.hidden_layers.append((w_hid * self.hid_to_hid_mask,
                                               b_hid))

    def __call__(self, y, **kwargs):
        return self.forward(y, **kwargs)

    def forward(self, y, activation=None):
        h = tf.matmul(y, self.w_in) + self.bias_hid
        if activation is not None:
            h = activation(h)
        for w_hid, b_hid in self.hidden_layers:
            h = tf.matmul(h, w_hid) + b_hid
            if activation is not None:
                h = activation(h)
        x = tf.matmul(h, self.w_out) + self.bias_out
        return x


def get_synth_data(seed=101, mix_dim=6, task_type='linear', samples=4000):
    # 10kHz, 4000 samples by default
    # This version of the task adds laplacian noise as a source and uses a
    # non-linear partially non-invertible, possibly overdetermined,
    # transformation.
    np.random.seed(seed)
    t = np.linspace(0, samples * 1e-4, samples)
    two_pi = 2 * np.pi
    s0 = np.sign(np.cos(two_pi * 155 * t))
    s1 = np.sin(two_pi * 800 * t)
    s2 = np.sin(two_pi * 300 * t + 6 * np.cos(two_pi * 60 * t))
    s3 = np.sin(two_pi * 90 * t)
    # s3=np.random.chisquare(4, (samples,))
    s4 = np.random.uniform(-1, 1, (samples,))
    s5 = np.random.laplace(0, 1, (samples,))
    x = np.stack([s0, s1, s2, s3, s4, s5])
    mix_mat = np.random.uniform(-.5, .5, (mix_dim, 6))
    y = np.dot(mix_mat, x)
    if task_type in ['mlp', 'pnl']:
        y = np.tanh(y)
        if task_type == 'mlp':
            mix_mat2 = np.random.uniform(-.5, .5, (mix_dim, mix_dim))
            y = np.tanh(np.dot(mix_mat2, y))
    return x.T, y.T, mix_mat

def get_random_batch(x, n):
    indices = np.random.randint(x.shape[0], size=(n,))
    return x[indices]
