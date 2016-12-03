# Assume that trainVal.prototxt was already defined.
# Note :
# 1) notify the number of test images
# 2) notify train.prototxt path
# My thinking : Why don't you improve this program for more general type ? such as input argument is job name, train.prototxt

from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import datetime

import math
import os
import shutil
import stat
import subprocess
import sys


### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.
caffe_root = os.getcwd()
# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = True
# If true, Remove old model files.
remove_old_models = False

train_data = "/home/youngwan/data/cifar10/cifar10-gcn-leveldb-splits/cifar10_full_train_leveldb_padded"
test_data = "/home/youngwan/data/cifar10/cifar10-gcn-leveldb-splits/cifar10_test_leveldb"
train_mean_file = "/home/youngwan/data/cifar10/cifar10-gcn-leveldb-splits/paddedmean.binaryproto"
test_mean_file = "/home/youngwan/data/cifar10/cifar10-gcn-leveldb-splits/mean.binaryproto"

#backend = leveldb


#resize_width = 384#300
#resize_height = 384#300
#resize = "{}x{}".format(resize_width, resize_height)

#job_name = "resnet_50_{}".format(resize)#job_name = "SSD_{}".format(resize)
job_name = "inception_v3_2_PreactRes_cifar10"
# The name of the model. Modify it if you want.
#[LYW]
model_name = "cifar_10_{}".format(job_name) #model_name = "VGG_KITTI_{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "models/CIFAR-10/inception_v3/{}".format(job_name)#save_dir = "models/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the snapshot of models.
#[LYW]
snapshot_dir = "models/CIFAR-10/inception_v3/{}".format(job_name)#snapshot_dir = "models/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the job script and log file.
#[LYW]
job_dir = "jobs/CIFAR-10/inception_v3/{}".format(job_name)#job_dir = "jobs/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the detection results.
#[LYW]
#output_result_dir = "{}/data/KITTIdevkit/results/kitti/{}/Main".format(os.environ['HOME'], job_name)#output_result_dir = "{}/data/KITTIdevkit/results/kitti/{}/Main".format(os.environ['HOME'], job_name)

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
#train_net_file = "examples/nvidia/Res_50/Res_50_trainval.prototxt"
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
#solver_file = "examples/nvidia/Res_50/food_resnet_50L_solver.prototxt"
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)
#pretrain_model = "models/ResNet/ResNet-50-model.caffemodel" #[LYW]
#pretrain_model = "models/ResNet/ResNet-50-model.caffemodel" #[LYW]
pretrain_model = False;

#gpus = "0,1"
gpus = "0,1"
gpulist = gpus.split(",")
num_gpus = len(gpulist)

# Divide the mini-batch to different GPUs.
batch_size = 128
accum_batch_size = 128
solver_mode = P.Solver.CPU
device_id = 0
batch_size_per_device = batch_size
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])


# Evaluate on whole test set.

# iter : ~64k : 0.05
# iter : 64k ~ 121k : 0.005
base_lr = 0.05

num_test_image = 10000
test_batch_size = 100 #1
test_batch_size_per_device = int(math.ceil(float(test_batch_size) / num_gpus))
test_iter = num_test_image / test_batch_size
max_iter = 64000  
#test_iter = 100

#src/caffe/proto/caffe.proto 
solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0001, #0.0005
    'lr_policy': "poly",
    'power':3,
    'stepsize': 32000,    
    #'stepvalue': 48000,
    'gamma': 0.01,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': max_iter,
    'snapshot': max_iter,#40000,
    'display': 10,
    'average_loss': 10,
    'type': "SGD",#SGD
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': 250,#10000,
    'eval_type': "classification",
    'test_initialization': False,
    }


train_transform_param = {
        'mirror': True,
        'mean_file':train_mean_file,
        'crop_size':32
        }

test_transform_param = {
        'mirror': False,
        'mean_file':test_mean_file,
        'crop_size':32        
        }


### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(train_data)
check_if_exist(test_data)
#check_if_exist(pretrain_model)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)    





############################################################################################
# Create train net.

net = caffe.NetSpec()

net.data, net.label = CreateAnnotatedDataLayerLEVELDB(train_data, batch_size=batch_size_per_device,
        train=True, output_label=True,
        transform_param=train_transform_param)

Inception_v3_2_PreActRes_Conv3x3_basic_SSD(net,from_layer='data',global_pool=True)

net.loss = L.SoftmaxWithLoss(net.fc10, net.label)


with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

############################################################################################
# Create test net.

net = caffe.NetSpec()

net.data, net.label = CreateAnnotatedDataLayerLEVELDB(test_data, batch_size=test_batch_size_per_device,
        train=False, output_label=True,
        transform_param=test_transform_param)

Inception_v3_2_PreActRes_Conv3x3_basic_Cifar10(net,from_layer='data',global_pool=True)

net.accuracy = L.Accuracy(net.fc10, net.label)

with open(test_net_file, 'w') as f:
    print('name: "{}_test"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

#############################################################################################

# Create deploy net.
# Remove the first and last layer from test net.
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, 32, 32])])
    print(net_param, file=f)


############################################################################################
# # Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

#if pretrain_model is not None:
#    train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
train_src_param=''
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models: # False
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
############################################################################################

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  if resume_training:
      if max_iter > 0:
          f.write(train_src_param)

  now = datetime.datetime.now()
  #if solver_param['solver_mode'] == P.Solver.GPU:
  f.write('--gpu {} 2>&1 | tee {}/{}_{}-{}-{}-{}:{}:{}.log\n'.format(gpus, job_dir, model_name,now.year,now.month, now.day, now.hour, now.minute, now.second))
  # else:
  #   f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)