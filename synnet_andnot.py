from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # using specific GPU
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

from compatible.likelihoods import MultiClass, Gaussian
from compatible.kernels import RBF, White
from gpflow.models.svgp import SVGP
from gpflow.training import AdamOptimizer, ScipyOptimizer
from scipy.stats import mode
from scipy.cluster.vq import kmeans2
import gpflow
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow import autoflow, params_as_tensors, ParamList
import pandas as pd
import itertools
pd.options.display.max_rows = 999
import gpflow_monitor

from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.io import loadmat
from gpflow_monitor import *
print('tf_ver:', tf.__version__, 'gpflow_ver:', gpflow.__version__)
from tensorflow.python.client import device_lib
print('avail devices:\n'+'\n'.join([x.name for x in device_lib.list_local_devices()]))
from jack_utils.common import time_it
import sys
import gpflow.training.monitor as mon

# our impl
from dgp_graph import *


import argparse

parser = argparse.ArgumentParser(description='main.')
parser.add_argument('--kern', type=str)


args = parser.parse_args()

cc=5
loc=60
tts=200
inc=False
M = 10

nss=0

# # build data

# 0-2 1-2 

# g2 = g0 * g1

nodes=3
ndata=200

data=np.zeros((ndata, nodes))

tns=0.01

data[:ndata, 0]=np.random.rand(ndata)
data[:ndata, 1]=np.random.rand(ndata)
data[:ndata, 2]=data[:ndata, 0]*(1-data[:ndata, 1]) + tns*np.random.randn(ndata)

adj = np.ones((nodes,nodes)) - np.eye(nodes)
# plt.imshow(adj); plt.colorbar()

trX1=data[:, :]
trY1=data[:, :]

trX1=trX1[:,:,None]
trY1=trY1[:,:,None]

trX=trX1
trY=trY1

Z = np.stack([kmeans2(trX[:,i], M, minit='points')[0] for i in range(nodes)],axis=1)  # (M=s2=10, n, d_in=5)
# print('inducing points Z: {}'.format(Z.shape))

# build model
# adj_identity = np.identity(adj.shape[0])  # without nb information
adj = adj.astype('float64')
input_adj = adj # adj  / np.identity(adj.shape[0]) /  np.ones_like(adj)

# Poly 1

time_vec=np.ones(trX.shape[0], )

with gpflow.defer_build():
    m_dgpg = DGPG(trX, trY, Z, time_vec, [1], Gaussian(), input_adj,
                  agg_op_name='concat3d', ARD=True,
                  is_Z_forward=True, mean_trainable=True, out_mf0=True,
                  num_samples=20, minibatch_size=10,
                  #kern_type='Matern32', 
                  #kern_type='RBF', 
                  #kern_type='Poly1', 
                  #wfunc='logi'
                  kern_type=args.kern, 
                  wfunc='krbf'
                 )
    # m_sgp = SVGP(X, Y, kernels, Gaussian(), Z=Z, minibatch_size=minibatch_size, whiten=False)
m_dgpg.compile()
model = m_dgpg

session = m_dgpg.enquire_session()
optimiser = gpflow.train.AdamOptimizer(0.0001)
# optimiser = gpflow.train.ScipyOptimizer()
global_step = mon.create_global_step(session)


Zcp = model.layers[0].feature.Z.value.copy()

model.X.update_cur_n(0,cc=cc,loc=loc)
model.Y.update_cur_n(0,cc=cc,loc=loc)

pred_res = []

maxiter=10000

exp_path="./exp/tmp-cc%d" % int(cc)
#exp_path="./exp/temp"

print_task = mon.PrintTimingsTask()\
    .with_name('print')\
    .with_condition(mon.PeriodicIterationCondition(10))\

checkpoint_task = mon.CheckpointTask(checkpoint_dir=exp_path)\
        .with_name('checkpoint')\
        .with_condition(mon.PeriodicIterationCondition(15))\

with mon.LogdirWriter(exp_path) as writer:
    tensorboard_task = mon.ModelToTensorBoardTask(writer, model)\
        .with_name('tensorboard')\
        .with_condition(mon.PeriodicIterationCondition(100))\
        .with_exit_condition(True)
    monitor_tasks = [tensorboard_task] # [print_task, tensorboard_task]

    with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
        model.layers[0].feature.Z.assign(Zcp.copy())
        model.layers[0].kern.lengthscales.assign(np.ones((nodes, nodes)))
        optimiser.minimize(model, step_callback=monitor, global_step=global_step, maxiter=maxiter)
        #optimiser.minimize(model, step_callback=monitor, maxiter=maxiter)

res = []

for i in range(nodes):
    for j in range(nodes):
        res.append((i,j,model.layers[0].kern.lengthscales.value[j][i]))

res.sort(key=lambda x: -x[2])

outfile = './res/synthetic_andnot.txt'

with open(outfile, 'w') as f:
    for (i, j, v) in res:
        if i != j:
            f.write('g%d\tg%d\t%.4f\n'%(i,j,v))