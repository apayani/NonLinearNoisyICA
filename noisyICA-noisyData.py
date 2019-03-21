#like 3 but added clustering factor, w_clustering which controls the entropy of q(y)

import numpy as np
import sys
import tensorflow as tf
import scipy.io as sio
import tensorflow.contrib.distributions as tfd
tfb = tfd.bijectors  
from datetime import datetime
from itertools import permutations
from tensorflow.python.client import timeline
import argparse
import scipy as sp

from ICAlib import *

N = 4000
validation = 500




parser = argparse.ArgumentParser()


parser.add_argument('--WHITEN', default= True, help='',type=bool)
parser.add_argument('--NOISE_STD', default= .05, help='',type=float)
parser.add_argument('--LR_GAN', default=.001  , help='',type=float)
parser.add_argument('--LAMBDA', default=0., help='',type=float)
parser.add_argument('--LAMBDA1', default=0, help='',type=float)
parser.add_argument('--init_sigma', default= -1.5, help='',type=float)
parser.add_argument('--w_clustering', default= 0., help='',type=float)


parser.add_argument('--EPOCHS', default=500000, help='',type=int)
parser.add_argument('--OPTI', default='adam', help='',type=str)
parser.add_argument('--LR', default=.001, help='',type=float)

parser.add_argument('--BS', default=500, help='',type=int)
parser.add_argument('--SEEDNP', default=101, help='',type=int)
parser.add_argument('--SEEDTF', default=0, help='',type=int)
parser.add_argument('--GPU', default=1, help='',type=int)

parser.add_argument('--L', default=1, help='',type=int)
parser.add_argument('--NX', default=6, help='',type=int)
parser.add_argument('--NZ', default=6, help='',type=int)

parser.add_argument('--POSTERIOR_NK', default=4, help='',type=int)
parser.add_argument('--PRIOR_NK', default=10 , help='',type=int)


parser.add_argument('--ZM', default=0, help='',type=int)    
parser.add_argument('--SINGLEVAR', default=0, help='',type=int)
parser.add_argument('--ZEROMEANPX', default=0, help='',type=int)


parser.add_argument('--SEPARATE_POSTERIORS', default=0, help='',type=int)

parser.add_argument('--NH', default=200, help='',type=int)
parser.add_argument('--NH_PX', default=200, help='',type=int)

parser.add_argument('--SF', default='exp', help='',type=str)
parser.add_argument('--SFPX', default='softplus', help='',type=str)
parser.add_argument('--ACT', default='tanh', help='',type=str)

parser.add_argument('--NHFZ', default=2, help='',type=int)
parser.add_argument('--ACTFZ', default='tanh', help='',type=str)
parser.add_argument('--FZHL', default=200, help='',type=int)

parser.add_argument('--ACT_DISC', default='tanh', help='',type=str)

parser.add_argument('--POSTERIOR_IAF', default=0, help='',type=int)
parser.add_argument('--POSTERIOR_IAF_TYPE', default='iaf', help='affine/iaf/nvp/masked',type=str)
parser.add_argument('--POSTERIOR_IAF_NH', default=128*4, help='',type=int)



parser.add_argument('--TASK', default='linear', help='linear/pnl/mlp/sin/cos/axbx2',type=str)
args = parser.parse_args()

print('displaying config setting...')
for arg in vars(args):
        sys.stdout.write( '{}-{} **** \n'.format ( arg, getattr(args, arg) ) )

 
S,X,_=get_data(seed=args.SEEDNP, mix_dim=args.NX, task_type=args.TASK, samples=N, noise_std=args.NOISE_STD, whiten=args.WHITEN)
 
j = N-validation
Xtr = X[:j,:]
Xte = X[j:,:]
Str=S[:j,:]
Ste=S[j:,:]
interop = Xtr.shape[0] // args.BS


###########################################################################################
def get_discreminator(X):
    with tf.variable_scope('disc', reuse= tf.AUTO_REUSE) :  
        logit =  FC( X , [200,200,1] , [activations[args.ACT_DISC],activations[args.ACT_DISC],tf.identity], name = 'u')
        return logit[:,0]

###########################################################################################
def get_posterior(xb,j):
    with tf.variable_scope( 'posterior', reuse=tf.AUTO_REUSE):
        y_ = tf.fill(tf.stack([tf.shape(xb)[0], args.POSTERIOR_NK]), 0.0)
        y = tf.add(y_, tf.constant(np.eye(args.POSTERIOR_NK)[j],'float32'))
        
        if args.SEPARATE_POSTERIORS==1:
            indj = j
        else:
            indj = 0

        hid1 = FC( (xb,y) , [args.NH,args.NH] , activations[args.ACT], name = 'hid_%d'%indj)
        loc =  FC( hid1 , [args.NZ] , [None], name = 'u_%d'%indj)
        scale =  FC( hid1 , [args.NZ] , [activations[args.SF]], name = 'sig_%d'%indj)
        posterior = tfd.MultivariateNormalDiag( loc = loc , scale_diag =  scale  ) 
         

        if args.POSTERIOR_IAF == 0:
            return posterior,loc
        bjs=[]
        for ii in range(args.POSTERIOR_IAF ):
            
            if args.POSTERIOR_IAF_TYPE=='affine':

                bj   =tfb.Affine(shift= tf.Variable(tf.zeros(6)), scale_tril=tfd.fill_triangular(tf.Variable(tf.ones(6*(6+1)/2 ))), name='aff_%d_%d'%(ii,j) )
            
            if args.POSTERIOR_IAF_TYPE=='iaf':
                bj = tfb.Invert(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(hidden_layers=[args.POSTERIOR_IAF_NH, args.POSTERIOR_IAF_NH]) ),name='floaw_%d_%d'%(ii,j) ) 

            if args.POSTERIOR_IAF_TYPE=='nvp':
                bj = tfb.RealNVP(        num_masked=3,        shift_and_log_scale_fn=tfb.real_nvp_default_template(            hidden_layers=[args.POSTERIOR_IAF_NH, args.POSTERIOR_IAF_NH]))
            
            if args.POSTERIOR_IAF_TYPE=='masked':
                bj = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(hidden_layers=[args.POSTERIOR_IAF_NH, args.POSTERIOR_IAF_NH]) ,name='floaw_%d_%d'%(ii,j) ) 
            bjs.append(bj)

        bj = tfb.Chain( bjs )
        maf = tfd.TransformedDistribution(
            distribution=posterior,
            bijector=bj)

        return maf,loc
###########################################################################################
def get_fz(z):
   
    inps = [z,]
    with tf.variable_scope('fz', reuse= tf.AUTO_REUSE) :  
        for i in range(args.NHFZ):
            inps.append(  tf.layers.dense( tf.concat(inps,-1),args.FZHL, activation=activations[args.ACTFZ] , kernel_initializer=None,name= 'fz_layers_%d'%i ))
        # fz = tf.layers.dense( tf.concat(inps,-1),args.NX, activation=None , kernel_initializer=None,name= 'fz_last' )
        fz = tf.layers.dense( tf.concat(inps,-1),args.NX, activation=None , kernel_initializer=None,name= 'fz_last' )
        # fz = tf.layers.dense( tf.concat(inps,-1),args.NX, activation=None , kernel_initializer=tf.constant_initializer( np.eye(args.NZ).astype(np.float32)),name= 'fz_last' )
    return fz
###########################################################################################
with tf.variable_scope('pdfError', reuse= tf.AUTO_REUSE) :  


    if args.SINGLEVAR==1:
        scale_px =  activations[args.SFPX]( bias_variable([], value=float(args.init_sigma), name='scale')  )
        scale_px=tf.stack( [scale_px]*args.NX,-1)
        loc = bias_variable([args.NX], value=0., name='loc') 
        if args.ZEROMEANPX == 1:
            loc = tf.zeros_like(loc)
        pdfError = tfd.MultivariateNormalDiag( loc = loc , scale_diag=  scale_px   )

    if args.SINGLEVAR==0:
        scale_px = activations[args.SFPX](bias_variable([args.NX], value=[ float(args.init_sigma)]*args.NZ, name='scale') )
        loc = bias_variable([args.NX], value=0., name='loc') 
        if args.ZEROMEANPX == 1:
            loc = tf.zeros_like(loc)
        pdfError = tfd.MultivariateNormalDiag( loc = loc, scale_diag=  scale_px   )


    if args.SINGLEVAR==2:
        scale_px = activations[args.SFPX](bias_variable([args.NX*(args.NX+1)//2], value=-1., name='scale') )
        loc = bias_variable([args.NX], value=0., name='loc') 
        if args.ZEROMEANPX == 1:
            loc = tf.zeros_like(loc)
        pdfError = tfd.MultivariateNormalTriL( loc = loc, scale_tril=  tfd.fill_triangular(scale_px)   )
###########################################################################################
with tf.variable_scope('prior', reuse= tf.AUTO_REUSE) :  
    prior = create_gmm_1(args.NZ,args.PRIOR_NK,'prior',scale_act=activations[args.SF],zero_mean=args.ZM>0)
    # prior = tfd.Independent( tfd.Laplace( loc = [ [0.]*args.NZ , ], scale=[ [1.]*args.NZ ,] ))

############################################################################################
config=tf.ConfigProto( device_count = {'GPU': args.GPU} )
config.log_device_placement = False
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# sess = tf.Session()
tf.set_random_seed(args.SEEDTF)
x = tf.placeholder(tf.float32, (None, args.NX) ,'x')
xb=x


loss = tf.constant(0.0)
L1 = tf.constant(0.0)
L2 = tf.constant(0.0)
L3 = tf.constant(0.0)
prediction_final = tf.constant(0.0)
predictions=[]


var_py = bias_variable([args.POSTERIOR_NK], value=np.random.uniform(1.0,10.,args.POSTERIOR_NK).astype(np.float32), name='py')
pys = np.random.uniform(-20.0,-1.0,args.POSTERIOR_NK).astype(np.float32)
pys = np.exp(pys)/np.sum( np.exp(pys))
# py = tfd.Dirichlet( np.random.uniform(1.0,12.,args.POSTERIOR_NK).astype(np.float32))

# py = tfd.Dirichlet( [1.0]*args.POSTERIOR_NK)
with tf.variable_scope( 'posteriorY', reuse=False):
    qy_logit = FC( xb  , [args.NH,args.NH,args.POSTERIOR_NK] , [activations[args.ACT],activations[args.ACT],None], name = 'qy')
    qy = tf.nn.softmax(qy_logit,-1)

for j in range(args.POSTERIOR_NK):

    posterior,posterior_mean = get_posterior(xb,j)
    prediction_final = prediction_final +  qy[:,j:j+1]*posterior_mean
    predictions.append(posterior_mean)
    L3jL=0
    L2jL=0
    L1jL=0

    for l in range(args.L):

        z = posterior.sample()
        e = xb - get_fz(z)
       
        L3jL += pdfError.log_prob( e )
        L2jL += prior.log_prob(z)
        L1jL += posterior.log_prob(z) 
        # L1jL -= posterior.entropy()
    L1j = L1jL/args.L
    L2j = L2jL/args.L
    L3j = L3jL/args.L

    # loss = loss - qy[:,j] * ( tf.log(1.0/args.POSTERIOR_NK) -L1j + L2j + L3j  ) 
    loss = loss - qy[:,j] * ( pys[j] -L1j + L2j + L3j  ) 
    # loss -= py.log_prob(qy)
    L1 += qy[:,j]*L1j
    L2 += qy[:,j]*L2j
    L3 += qy[:,j]*L3j



 
qy_entropy = cross_entropy_with_logits_loss(qy_logit,qy)

predictions=tf.stack( predictions, -1)
ind = tf.argmax( qy, -1)
ind = tf.expand_dims(ind,-1)
prediction_final_2 = tf.reduce_sum( tf.one_hot(ind,depth=args.POSTERIOR_NK, on_value=1.0,off_value =0.0) * predictions, axis=-1)
# prediction_final_2 = prediction_final

loss = tf.reduce_mean(loss - qy_entropy)


 
zs = prior.sample(args.BS) 

# zs = resample_rows_per_column(prediction_final)
# zs=tf.reshape( zs, [-1,args.NZ])

r_logits = get_discreminator(zs)
f_logits = get_discreminator(prediction_final)

disc_loss = get_gan_cost(r_logits) + get_gan_cost(-f_logits)
gen_loss =  get_gan_cost(f_logits)

total_loss = loss+ disc_loss * args.LAMBDA1 +  tf.reduce_mean(qy_entropy*args.w_clustering)

prediction_perm_mean = tf.reduce_mean(prediction_final, 0)
prediction_perm_norm = prediction_final - prediction_perm_mean
prediction_perm_sd = tf.sqrt(tf.reduce_mean(prediction_perm_norm**2, 0) + 1e-4)
prediction_perm_norm /= prediction_perm_sd

cov_pred = tf_cov(prediction_perm_norm)
pen_cov1  = tf.square( cov_pred-tf.eye(args.NZ))
pen_cov = tf.reduce_sum(pen_cov1)

# total_loss += pen_cov*.01
e = xb - get_fz(prediction_final)
# e_loss = -pdfError.log_prob( e )
e_loss = pen_cov
############################################################################################
# training
############################################################################################



print( '***********************')
print( 'number of trainable parameters : {}'.format(count_number_trainable_params()))
print( '***********************')
vs=tf.global_variables()
for v in vs:
    if v is not None:
        print(v.name,v.shape)
# opt = tf.train.AdamOptimizer(learning_rate)
# clipped_grads_and_vars, global_grad_norm = tensorbayes.tfutils.clip_gradients(opt,loss,max_clip=1000,max_norm=10)
# train_op = opt.apply_gradients(clipped_grads_and_vars)


all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES )
disc_vars  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="disc")
e_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES )
for item in disc_vars:
    all_vars.remove(item)



if args.OPTI == 'adam':
    train_op = tf.train.AdamOptimizer(args.LR).minimize(total_loss ,var_list=all_vars)
    train_op_gan = tf.train.AdamOptimizer(args.LR_GAN ).minimize(disc_loss,var_list=disc_vars)
    train_op_e = tf.train.AdamOptimizer(args.LR_GAN ).minimize(e_loss,var_list=e_vars)

if args.OPTI == 'prop':
    train_op = tf.train.RMSPropOptimizer(args.LR).minimize(total_loss ,var_list=all_vars)
    train_op_gan = tf.train.RMSPropOptimizer(args.LR_GAN ).minimize(disc_loss,var_list=disc_vars)


print('--------------------------------------------------------------------')

sess.run(tf.global_variables_initializer())
corr=0.0


start_time = datetime.now()

max_c=0
tr_loss_ar=[]
tr_d_loss_ar=[]
tr_g_loss_ar=[]
te_loss_ar=[]    

for epoch in range (args.EPOCHS):
    
    
    
    L = Xtr.shape[0] // args.BS
    inds=np.random.permutation( Xtr.shape[0] )
    
    for i in range(100):
        xt = get_random_batch(Xtr,args.BS)
        
        if True or i%2==0: 
            _,tr_loss,g,d = sess.run([train_op,loss,gen_loss,disc_loss], feed_dict={'x:0': xt})
            tr_loss_ar.append(tr_loss.mean())
            tr_g_loss_ar.append(g.mean())
            if args.LAMBDA==0:
                tr_d_loss_ar.append(d.mean()) 
        
        if args.LAMBDA != 0  and i%2==1: 
           
            _,d = sess.run( [train_op_gan,disc_loss], feed_dict={'x:0': Xtr})
            tr_d_loss_ar.append(d.mean()) 
            
        # if i%2==1: 
        #     sess.run( [train_op_e], feed_dict={'x:0': Xtr})

        
    
        
    if True or ( (epoch+1) % 100 == 0) :

        now = datetime.now()
        
        te_loss=[]
        test_z=[]
        test_z1=[]
        test_allc_z=[]
        a1=[]
        a2=[]
        a3=[]

        for j in range( Xte.shape[0]//args.BS):
            sc,jte_loss,allc_z,jtest_z,jtest_z1,ja1,ja2,ja3 = sess.run(['pdfError/scale:0', loss,predictions,prediction_final,prediction_final_2,L1,L2,L3],feed_dict={'x:0': Xte[j*args.BS:(j+1)*args.BS,:] })
            te_loss.append(te_loss)
            test_allc_z.append(allc_z)
            test_z.append(jtest_z)
            test_z1.append(jtest_z1)
            a1.append(ja1)
            a2.append(ja2)
            a3.append(ja3)
        te_loss = np.mean(jte_loss)
        a1=np.mean(a1)
        a2=np.mean(a2)
        a3=np.mean(a3)
        
        test_allc_z = np.concatenate( test_allc_z, axis=0)
        test_z = np.concatenate( test_z, axis=0)
        test_z1 = np.concatenate( test_z1, axis=0)
        
        cov,_,_=mycorrelation(test_z.T,Ste.T,'Pearson')
        cov1,_,_=mycorrelation(test_z1.T,Ste.T,'Pearson')
        
        cov = np.diag(cov)
        cov1 = np.diag(cov1)

        corr=np.mean( np.abs(cov))
        corr1=np.mean( np.abs(cov1))
        if corr>max_c:
            max_c = corr
        
        te_loss_ar.append(te_loss.mean())
        print( 'ep=%d, loss = [%.2f , %.2f ], genloss=%.2f, discloss=%.2f, C=[%.3f], max_c : %.3f ,DL=[%.2f,%.2f,%.2f], T=%s'%(epoch+1,NPC*np.mean(tr_loss_ar),NPC*np.mean(te_loss_ar) ,np.mean(tr_g_loss_ar), np.mean(tr_d_loss_ar), corr,max_c, NPC*np.mean(a1),NPC*np.mean(a2),NPC*np.mean(a3), str(now-start_time)))
        print('cov = ', np.round( 100*cov ) )
        
        if args.SFPX=='exp':
            print('sc',sc,np.exp(sc))
        if args.SFPX=='softplus':
            print('sc',sc,np.log(1.0+np.exp(sc)) )
        
       
        start_time = now
        tr_loss_ar=[]
        te_loss_ar=[]    
