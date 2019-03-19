import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import tensorflow.contrib.distributions as tfd
from munkres import Munkres

NP_DTYPE = np.float32
DTYPE= tf.float32
NPC = np.log2(np.exp(1))

activations = dict()
activations['tanh']=tf.tanh
activations['tan']=tf.tan
activations['sin']=tf.sin
activations['cos']=tf.cos
activations['exp']=tf.exp
activations['logabs']=lambda x:tf.log( tf.abs(x)+1e-10)
activations['abs']=tf.abs

activations['myhardtanh']=lambda x: tf.nn.leaky_relu( 1.0- tf.nn.leaky_relu(1.0-x) )

activations['softplus']=tf.nn.softplus
activations['relu']=tf.nn.relu
activations['lrelu']=tf.nn.leaky_relu
activations['relu6']=tf.nn.relu6
activations['elu']=tf.nn.elu
activations['crelu']=tf.nn.crelu
activations['selu']=tf.nn.selu
activations['softsign']=tf.nn.softsign
activations['sigmoid']=tf.nn.sigmoid
activations['hardsigmoid']=tf.keras.activations.hard_sigmoid
activations['identity']=tf.identity

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
def weight_variable(shape, stddev=0.01, name='weight'):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)
def bias_variable(shape, value=0.0, name='bias'):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name=name)
def get_data(seed=101, mix_dim=6, task_type='linear', samples=4000,noise_std=0,whiten=True):
    
    np.random.seed(seed)
    t = np.linspace(0, samples * 1e-4, samples)
    two_pi = 2 * np.pi
    s0 = np.sign(np.cos(two_pi * 155 * t))
    s1 = np.sin(two_pi * 800 * t)
    s2 = np.sin(two_pi * 300 * t + 6 * np.cos(two_pi * 60 * t))
    s3 = np.sin(two_pi * 90 * t)
    s4 = np.random.uniform(-1, 1, (samples,))
    s5 = np.random.laplace(0, 1, (samples,))
    x = np.stack([s0, s1, s2, s3, s4, s5])
    
    mix_mat = np.random.uniform(-.5, .5, (mix_dim, 6))
   
   
    
    y = np.dot(mix_mat, x)
    if task_type in ['sin']:
        y = np.sin(y)
    if task_type in ['cos']:
        y = np.cos(y)
    if task_type in ['axbx2']:
        mix_mat2 = np.random.uniform(-.5, .5, (mix_dim, mix_dim))
        y1 = np.dot(mix_mat2, x*x*x)
        y = y+y1 

    if task_type in ['mlp', 'pnl']:
        y = np.tanh(y)
        if task_type in ['mlp', 'mlp3']:
            mix_mat2 = np.random.uniform(-.5, .5, (mix_dim, mix_dim))
            y = np.tanh(np.dot(mix_mat2, y))
            if task_type in ['mlp3', 'mlp4']:
                mix_mat3 = np.random.uniform(-.5, .5, (mix_dim, mix_dim))
                y = np.tanh(np.dot(mix_mat3, y))
    
    if noise_std>0:
        y = y + np.random.multivariate_normal([0]*6,noise_std*np.eye(6), (samples,)).T
    pca = PCA(whiten=True)
    if whiten:
       
        y=pca.fit_transform(y.T)
        y=y.T    
    return  x.T, y.T ,pca
def FC(inputs,sizes,activations=None,name='fc',ki=None):
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
        X = tf.layers.dense( X, sizes[i] , act , name=name+'_%d'%(i+1) ,kernel_initializer=ki)
         
    return X
def create_gmm_1( d,K,name='gmm', reuse=False, scale_act=tf.nn.softplus,zero_mean=False,ki=None):
    with tf.variable_scope( name , reuse):
        #tf.random_uniform_initializer(0.,3.)
        probs = tf.nn.softmax( tf.get_variable('probs', shape=[d,K] , dtype=DTYPE,initializer=None),axis=-1)
        #tf.random_uniform_initializer(-.5,.5)
        locs = tf.get_variable('locs', shape=[d,K] , dtype=DTYPE,initializer=None)
        if zero_mean:
            locs=tf.zeros_like(locs)
        
        scales = tf.get_variable('scales', shape=[d,K] , dtype=DTYPE,initializer=None)

        pis = tfd.Categorical( probs = probs )
        ps = tfd.Normal( loc = locs , scale =scale_act(scales))
        p = tf.contrib.distributions.MixtureSameFamily( pis, ps)
        p = tf.contrib.distributions.Independent(p,1)

    return p
def get_random_batch(x, n):
    indices = np.random.randint(x.shape[0], size=(n,))
    return x[indices]
def mycorrelation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
         method: correlation method ('Pearson' or 'Spearman')
     """

    # print("Calculating correlation...")

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    if method=='Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
    elif method=='Spearman':
        corr, pvalue = sp.stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    

    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    # Re-calculate correlation --------------------------------
    if method=='Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]
    elif method=='Spearman':
        corr_sort, pvalue = sp.stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort
def sample_integers(n, shape):
    sample = tf.random_uniform(shape, minval=0, maxval=tf.cast(n, 'float32'))
    sample = tf.cast(sample, 'int32')
    return sample
def resample_rows_per_column(x):
    """Permute all rows for each column independently."""
    n_batch = tf.shape(x)[0]
    n_dim = tf.shape(x)[1]
    row_indices = sample_integers(n_batch, (n_batch * n_dim,))
    col_indices = tf.tile(tf.range(n_dim), [n_batch])
    indices = tf.transpose(tf.stack([row_indices, col_indices]))
    x_perm = tf.gather_nd(x, indices)
    x_perm = tf.reshape(x_perm, (n_batch, n_dim))
    return x_perm
def get_gan_cost(logits):
    return tf.reduce_mean(tf.nn.softplus(logits)) 
def cross_entropy_with_logits_loss(logits, targets):
    log_q = tf.nn.log_softmax(logits)
    return -tf.reduce_sum(targets * log_q, 1)
