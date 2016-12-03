# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 22:29:31 2016

@author: youngwan
"""

import os
import sys
import subprocess
import pandas as pd

#matplotlib.use('Agg')
import matplotlib.pylab as plt

#plt.style.use('ggplot')
def load_data(data_file, field_idx0, field_idx1):
    data = [[], []]
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()  
            if line[0] != '#':
                fields = line.split()
                data[0].append(float(fields[field_idx0].strip()))
                data[1].append(float(fields[field_idx1].strip()))
    return data


caffe_path = '/home/youngwan/caffe/'

#model_log_path = caffe_path + sys.argv[1]
#learning_curve_path = caffe_path + sys.argv[2]


#model_log_path = caffe_path + 'jobs/New/KITTI/inception_v3/SSD_Inception_v3_4_preActRes_OriginASP_BN_ImageNet_pretrained_unFreezed300x300/KITTI_SSD_Inception_v3_4_preActRes_OriginASP_BN_ImageNet_pretrained_unFreezed300x300_2016-11-6-17:41:14.log'
#model_log_path_2 = caffe_path + 'jobs/New/KITTI/inception_v3/SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_No_pretrained_300x300/KITTI_SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_No_pretrained_300x300_2016-11-14-23:26:29.log'

model_log_path = caffe_path + 'jobs/New/KITTI/inception_v3/SSD_Inception_v3_4_preActRes_OriginASP_BN_ImageNet_pretrained_unFreezed_5th_300x300/KITTI_SSD_Inception_v3_4_preActRes_OriginASP_BN_ImageNet_pretrained_unFreezed_5th_300x300_2016-11-19-15:56:0.log'
model_log_path_2 = caffe_path + 'jobs/New/KITTI/inception_v3/SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_No_pretrained_4th300x300/KITTI_SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_No_pretrained_4th300x300_2016-11-17-23:47:35.log'
model_log_path_3 = caffe_path + 'jobs/New/KITTI/inception_v3/SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_cifar_pretrained_3rd_300x300/KITTI_SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_cifar_pretrained_3rd_300x300_2016-11-18-10:56:17.log'
model_log_path_4 = caffe_path + 'jobs/New/KITTI/inception_v3/SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_cifar100_pretrained_1st_300x300/KITTI_SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_cifar100_pretrained_1st_300x300_2016-11-23-17:53:56.log'

learning_curve_path = caffe_path + 'jobs/figures/Total_dpi5000.png'


#Get directory where the model logs is saved, and move to it
model_log_dir_path = os.path.dirname(model_log_path)
os.chdir(model_log_dir_path)
command = caffe_path + 'tools/extra/parse_log.sh ' + model_log_path
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()
#Read training and test logs
train_log_path = model_log_path + '.train'
test_log_path = model_log_path + '.test'
train_log = pd.read_csv(train_log_path, delim_whitespace=True)
test_log = pd.read_csv(test_log_path, delim_whitespace=True)



model_log_dir_path_2 = os.path.dirname(model_log_path_2)
os.chdir(model_log_dir_path_2)

#Parsing training/validation logs
command_2 = caffe_path + 'tools/extra/parse_log.sh ' + model_log_path_2
process_2 = subprocess.Popen(command_2, shell=True, stdout=subprocess.PIPE)
process_2.wait()


train_log_path_2 = model_log_path_2 + '.train'
test_log_path_2 = model_log_path_2 + '.test'
train_log_2 = pd.read_csv(train_log_path_2, delim_whitespace=True)
test_log_2 = pd.read_csv(test_log_path_2, delim_whitespace=True)



model_log_dir_path_3 = os.path.dirname(model_log_path_3)
os.chdir(model_log_dir_path_3)

#Parsing training/validation logs
command_3 = caffe_path + 'tools/extra/parse_log.sh ' + model_log_path_3
process_3 = subprocess.Popen(command_3, shell=True, stdout=subprocess.PIPE)
process_3.wait()


train_log_path_3 = model_log_path_3 + '.train'
test_log_path_3 = model_log_path_3 + '.test'
train_log_3 = pd.read_csv(train_log_path_3, delim_whitespace=True)
test_log_3 = pd.read_csv(test_log_path_3, delim_whitespace=True)



model_log_dir_path_4 = os.path.dirname(model_log_path_4)
os.chdir(model_log_dir_path_4)
#Parsing training/validation logs
command_4 = caffe_path + 'tools/extra/parse_log.sh ' + model_log_path_4
process_4 = subprocess.Popen(command_4, shell=True, stdout=subprocess.PIPE)
process_4.wait()


train_log_path_4 = model_log_path_4 + '.train'
test_log_path_4 = model_log_path_4 + '.test'
train_log_4 = pd.read_csv(train_log_path_4, delim_whitespace=True)
test_log_4 = pd.read_csv(test_log_path_4, delim_whitespace=True)


#Parsing
#train_data = load_data(train_log_path,1,3)
#test_data = load_data(test_log_path,1,3)


train_loss =  train_log['TrainingLoss']
train_iter = train_log['#Iters']

test_iter = test_log['#Iters']
test_loss = test_log['TestLoss']
test_acc = test_log['TestAccuracy']
#test_error = 1- test_accuracy


train_loss_2 =  train_log_2['TrainingLoss']
train_iter_2 = train_log_2['#Iters']

test_iter_2 = test_log_2['#Iters']
test_loss_2 = test_log_2['TestLoss']
test_acc_2 = test_log_2['TestAccuracy']
#test_error_2 = 1- test_accuracy_2


train_loss_3 =  train_log_3['TrainingLoss']
train_iter_3 = train_log_3['#Iters']

test_iter_3 = test_log_3['#Iters']
test_loss_3 = test_log_3['TestLoss']
test_acc_3 = test_log_3['TestAccuracy']
#test_error_3 = 1- test_accuracy_3


train_loss_4 =  train_log_4['TrainingLoss']
train_iter_4 = train_log_4['#Iters']

test_iter_4 = test_log_4['#Iters']
test_loss_4 = test_log_4['TestLoss']
test_acc_4 = test_log_4['TestAccuracy']
#test_error_3 = 1- test_accuracy_3

'''
Making learning curve
'''
fig, ax1 = plt.subplots()

#Plotting training and test losses
#train_loss, = ax1.plot(train_log['#Iters'], train_log['TrainingLoss'], color='red',linewidth=2,  alpha=.5)
#test_loss, = ax1.plot(test_log['#Iters'], test_log['TestLoss'], linewidth=2, color='green')


train_loss = plt.plot(train_iter,train_loss, label='Loss ImageNet', color='red',linewidth=1)
train_loss_2, = plt.plot(train_iter, train_loss_2, label='Loss No Pretrained', linewidth=1, color='blue')
train_loss_3, = plt.plot(train_log_3['#Iters'], train_loss_3,label='Loss cifar-10', linewidth=1, color='green',)
train_loss_4, = plt.plot(train_log_4['#Iters'], train_loss_4,label='Loss cifar-100', linewidth=1, color='black',)

plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.yscale('log')
plt.tick_params(labelsize=10)
plt.legend(bbox_to_anchor=(0.5, 1), loc=2, borderaxespad=0.)

#Plotting Accuracy
ax2 = plt.twinx()
test_accuracy, = plt.plot(test_log['#Iters'], test_acc, label='Acc. ImageNet',linewidth=3, color='red', linestyle=':')
test_accuracy_2, = plt.plot(test_log_2['#Iters'], test_acc_2, label='Acc. No Pretrained', linewidth=3, color='blue',linestyle=':')
test_accuracy_3, = plt.plot(test_log_3['#Iters'], test_acc_3, label='Acc. cifar-10', linewidth=3, color='green',linestyle=':')
test_accuracy_4, = plt.plot(test_log_4['#Iters'], test_acc_4, label='Acc. cifar-100', linewidth=3, color='black',linestyle='-')

ax2.set_ylim(ymin=0, ymax=1)
ax2.set_ylabel('Accuracy', fontsize=15)
ax2.tick_params(labelsize=10)
plt.legend(bbox_to_anchor=(0.03, 1), loc=2, borderaxespad=0.)
plt.title('Training Curve', fontsize=18)

#Saving learning curve
plt.savefig(learning_curve_path,dpi=2000)
plt.show()


'''
Deleting training and test logs
'''
command = 'rm ' + train_log_path
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()

command = command = 'rm ' + test_log_path
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()





