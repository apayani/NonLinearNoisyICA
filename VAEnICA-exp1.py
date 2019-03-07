# this model assumes simple posterior and mixture prior

sss='********************======================*************************\n'
import numpy as np
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
from neural_networks import *
import scipy.io as sio
import tensorflow.contrib.distributions as tfd
tfb = tfd.bijectors
from datetime import datetime
from itertools import permutations
from tensorflow.python.client import timeline

NP_DTYPE = np.float32
DTYPE= tf.float32
NPC = np.log2(np.exp(1))
dense = tf.layers.dense
N = 4000
validation = 500

 
 
def get_data(seed=101, mix_dim=6, task_type='linear', samples=4000):
    
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
        y1 = np.dot(mix_mat2, x*x)
        y = y+y1 

    if task_type in ['mlp', 'pnl']:
        y = np.tanh(y)
        if task_type == 'mlp':
            mix_mat2 = np.random.uniform(-.5, .5, (mix_dim, mix_dim))
            y = np.tanh(np.dot(mix_mat2, y))
     
    from sklearn.decomposition import PCA
    pca = PCA(whiten=True)
    y=pca.fit_transform(y.T)
    y=y.T    
    return  x.T, y.T, mix_mat

parser = argparse.ArgumentParser()

parser.add_argument('--EPOCHS', default=500000, help='',type=int)
parser.add_argument('--OPTI', default='adam', help='',type=str)
parser.add_argument('--LR', default=.0031, help='',type=float)
parser.add_argument('--BS', default=100, help='',type=int)
parser.add_argument('--SEEDNP', default=101, help='',type=int)
parser.add_argument('--SEEDTF', default=0, help='',type=int)
parser.add_argument('--GPU', default=1, help='',type=int)

parser.add_argument('--NX', default=6, help='',type=int)
parser.add_argument('--NZ', default=6, help='',type=int)

parser.add_argument('--NKZ', default=2, help='',type=int)
parser.add_argument('--NK', default=100, help='',type=int)


parser.add_argument('--ZM', default=0, help='',type=int)
parser.add_argument('--ONLYLINEAR', default=0, help='',type=int)
parser.add_argument('--SINGLEVAR', default=0, help='',type=int)
parser.add_argument('--ZEROMEANPX', default=1, help='',type=int)



parser.add_argument('--NH', default=100, help='',type=int)
parser.add_argument('--SF', default='exp', help='',type=str)
parser.add_argument('--SFPX', default='softplus', help='',type=str)
parser.add_argument('--ACT', default='tanh', help='',type=str)


parser.add_argument('--NHFZ', default=100, help='',type=int)
parser.add_argument('--ACTFZ', default='tanh', help='',type=str)
parser.add_argument('--FZHL', default=2, help='',type=int)


parser.add_argument('--TASK', default='pnl', help='linear/pnl/mlp/sin/cos/axbx2',type=str)
args = parser.parse_args()

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
activations['relu6']=tf.nn.relu6
activations['elu']=tf.nn.elu
activations['crelu']=tf.nn.crelu
activations['selu']=tf.nn.selu
activations['softsign']=tf.nn.softsign
activations['sigmoid']=tf.nn.sigmoid
activations['hardsigmoid']=tf.keras.activations.hard_sigmoid
activations['identity']=tf.identity


only_linear = args.ONLYLINEAR > 0

HL=[args.NH,args.NH]
print('displaying config setting...')
for arg in vars(args):
        sys.stdout.write( '{}-{} **** \n'.format ( arg, getattr(args, arg) ) )


############################################################################################
 


S,X,_=get_data(seed=args.SEEDNP, mix_dim=args.NX, task_type=args.TASK, samples=N)
sio.savemat('sx.mat',{'S':S,'X':X})

 

j = N-validation
Xtr = X[:j,:]
Xte = X[j:,:]
Str=S[:j,:]
Ste=S[j:,:]
interop = Xtr.shape[0] // args.BS

 


############################################################################################
# input 
############################################################################################
config=tf.ConfigProto( device_count = {'GPU': args.GPU} )
config.log_device_placement = False
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.set_random_seed(args.SEEDTF)
x = tf.placeholder(tf.float32, (None, args.NX) ,'x')
xb=x
  
############################################################################################
# prior 
############################################################################################

prior = create_gmm_1(args.NZ,args.NK,'prior',scale_act=activations[args.SF],zero_mean=args.ZM>0)

############################################################################################
# posterior 
############################################################################################
y_ = tf.fill(tf.stack([tf.shape(xb)[0], args.NKZ]), 0.0)
predictions = [None] *args.NKZ
# L1j=[None]*nKZ
with tf.variable_scope('fz', reuse=False):
    fzModel1 =  MLP([args.NZ, args.NX], [ None], stddev=1.)
   
   
loss = tf.constant(0.0)
L1 = tf.constant(0.0)
L2 = tf.constant(0.0)
L3 = tf.constant(0.0)
zm=tf.constant(0.0)
prediction_final = tf.constant(0.0)

with tf.variable_scope( 'posterior', reuse=False):
    qy_logit = FC( xb  , [args.NH,args.NH,args.NKZ] , [activations[args.ACT],activations[args.ACT],None], name = 'qy')
    qy = tf.nn.softmax(qy_logit)

for j in range(args.NKZ):
   
    y = tf.add(y_, tf.constant(np.eye(args.NKZ)[j],'float32'))
    with tf.variable_scope( 'posterior', reuse=j>0):
        
        hid1 = FC( (xb,y) , [args.NH,args.NH] , activations[args.ACT], name = 'hid1')
        loc =  FC( hid1 , [args.NZ] , [None], name = 'u')
        scale =  FC( hid1 , [args.NZ] , [activations[args.SF]], name = 'sig')
        posterior = tfd.MultivariateNormalDiag( loc = loc , scale_diag =  scale )

    predictions[j] = loc 
    prediction_final = prediction_final +  qy[:,j:j+1]*loc
    z = posterior.sample()


    
    ############################################################################################
    # Px
    ############################################################################################
    with tf.variable_scope('px', reuse=j>0):
        
        loc = bias_variable([args.NX], value=0.0, name='loc') 
        if args.ZEROMEANPX==1:
            loc = tf.zeros_like(loc)
        
        if args.SINGLEVAR==1:
            scale = activations[args.SFPX]( bias_variable([], value=-1.0, name='scale')  )
            scale=tf.stack( [scale]*args.NX,-1)
        else:
            scale = activations[args.SFPX](bias_variable([args.NX], value=-1., name='scale') )
       
        loc =  FC( z , [600,args.NX] , [tf.tanh,tf.identity], name = 'u')
        pxIz = tfd.MultivariateNormalDiag( loc = loc, scale_diag=  scale   )
    
    ############################################################################################
    # loss
    ############################################################################################
    
    L1j = -posterior.entropy()
    L2j = prior.log_prob(z)
    L3j = pxIz.log_prob(xb)
   
    loss = loss+ qy[:,j] * ( tf.log(1.0e-17+qy[:,j]) + L1j - L2j - L3j  ) 
    L1 += qy[:,j]*L1j
    L2 += qy[:,j]*L2j
    L3 += qy[:,j]*L3j
    
    loss = tf.reduce_mean(loss)

############################################################################################
# find correlation
############################################################################################

def get_corr_perm(x, y,perm):
    x_centered = x - np.mean(x, 0, keepdims=True)
    y_centered = y - np.mean(y, 0, keepdims=True)
    sd_x = np.sqrt(np.mean(x_centered**2, 0, keepdims=True))
    x_norm = x_centered / sd_x
    sd_y = np.sqrt(np.mean(y_centered**2, 0, keepdims=True))
    corrs = []
        
    cov_diag = np.mean(perm.T * y_centered, 0) / (1e-20+sd_y)
    corrs = np.mean(np.abs(cov_diag))
    
    return corrs
def get_max_corr_perm(x, y):
    x_centered = x - np.mean(x, 0, keepdims=True)
    y_centered = y - np.mean(y, 0, keepdims=True)
    sd_x = np.sqrt(np.mean(x_centered**2, 0, keepdims=True))
    sd_y = np.sqrt(np.mean(y_centered**2, 0, keepdims=True))
    
    x_norm = x_centered / sd_x
    y_norm = y_centered / sd_y
    # print(x.shape)
    # print(y.shape)
    
    corrs = []
    all_perms=[]
    covdiags=[]
    for y_perm in permutations(y_norm.T):
        y_perm = np.stack(y_perm)
        all_perms.append(y_perm)
        
        
        ytp = y_perm.T
        ytp=ytp[:,:x.shape[-1]]
        # print(ytp.shape)
        # print(x_centered.shape)
        cov_diag = np.mean(ytp * x_centered, 0) / (1e-20+sd_x)
        covdiags.append(cov_diag)
        corrs.append(np.mean(np.abs(cov_diag)))
    ind = np.argmax    (corrs)
    return corrs[ind],all_perms[ind],covdiags[ind]

############################################################################################
# training
############################################################################################



print( '***********************')
print( 'number of trainable parameters : {}'.format(count_number_trainable_params()))
print( '***********************')


if args.OPTI == 'adam':
    train_op = tf.train.AdamOptimizer(args.LR).minimize(loss)
    # train_op2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2)
elif args.OPTI == 'prop':
    train_op = tf.train.RMSPropOptimizer(args.LR).minimize(loss)



print('--------------------------------------------------------------------')
print('--------------------------------------------------------------------')

sess.run(tf.global_variables_initializer())
corr=0.0
def get_random_batch(x, n):
    indices = np.random.randint(x.shape[0], size=(n,))
    return x[indices]

start_time = datetime.now()
max_c=0
tr_loss_ar=[]
te_loss_ar=[]    

for epoch in range (args.EPOCHS):
    
    
    
    xt = get_random_batch(Xtr,args.BS)
    _,tr_loss = sess.run([train_op,loss], feed_dict={'x:0': xt})
    tr_loss_ar.append(tr_loss.mean())

    if ( (epoch+1) % 1000 == 0) :

        now = datetime.now()
        te_loss,test_z,a1,a2,a3 = sess.run([loss,prediction_final,L1,L2,L3],feed_dict={'x:0': Xte})
        train_z  = sess.run(prediction_final,feed_dict={'x:0': Xtr})
        
        corr,perm,cov = get_max_corr_perm( Ste, test_z)
        corrtrain,aa1,aa2 = get_max_corr_perm( Str, train_z)
        
        if corr>max_c:
            max_c = corr
            try:
                sio.savemat( 'estimated.mat',{'ztr':train_z,'zte':test_z})
            except:
                print( "error saving")

        te_loss_ar.append(te_loss.mean())
        print( 'ep=%d, loss = [%.2f , %.2f ], C=[%.4f,%.4f], max_c : %.4f ,DL=[%.2f,%.2f,%.2f], T=%s'%(epoch+1,NPC*np.mean(tr_loss_ar),NPC*np.mean(te_loss_ar) ,corrtrain, corr,max_c, NPC*np.mean(a1),NPC*np.mean(a2),NPC*np.mean(a3), str(now-start_time)))
        print('cov = ', np.round( 100*cov ) )
        start_time = now
        tr_loss_ar=[]
        te_loss_ar=[]    
