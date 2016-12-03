import numpy as np
import matplotlib.pyplot as plt
import time
from caffe.model_libs import *
import datetime

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
voc_labelmap_file = '/home/youngwan/caffe/data/kitti_ssd/labelmap_kitti.prototxt'
file = open(voc_labelmap_file, 'r')
voc_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), voc_labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


model_def = '/home/youngwan/caffe/models/New/KITTI/inception_v2/SSD_Inception_v2_8_preActRes_ASP4_BN_cifar10_pretrained_350x250/deploy.prototxt'
model_weights = '/home/youngwan/caffe/models/New/KITTI/inception_v2/SSD_Inception_v2_8_preActRes_ASP4_BN_cifar10_pretrained_350x250/KITTI_SSD_Inception_v2_8_preActRes_ASP4_BN_cifar10_pretrained_350x250_iter_100000.caffemodel'



#variable
cntFrame = 0
sumPTime = 0
avgPTime = 0
resize_width = 350
resize_height = 250
resize = "{}x{}".format(resize_width, resize_height)


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([96,99,94])) # mean pixel
#transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


net.blobs['data'].reshape(1,3,resize_height,resize_width)
conf_th = 0.0



now = datetime.datetime.now()
job_name = "SSD_inception_v2_8_ASP4_PED3_BN_cifar_pretrained_trainVal_test_{}".format(resize)
save_dir = "/home/youngwan/caffe/jobs/KITTI/validation_result/inception_v2/{}".format(job_name)
save_file = "{}/{}_runTime_{}-{}-{}-{}:{}.txt".format(save_dir,job_name,now.year,now.month,now.day,now.hour,now.minute)
make_if_not_exist(save_dir)
fp = open(save_file, 'w')

for c in xrange(1,2,1) :

    conf_th = c / 100.0



    print "conf:%f --- saving ---" % conf_th

    result_path = '{}/{}/Result_full_lmdb_text_{}'.format(save_dir,job_name,conf_th)
    make_if_not_exist(result_path)
    
    test_image_path = "/home/youngwan/data/KITTI/kitti_test_imgs"

    image_list = os.listdir(test_image_path)
    sorted_list = sorted(image_list)

    for s in sorted_list:
        loadingfile = test_image_path+'/' + s
        textfile = result_path+'/' + s
        textfile = textfile.replace("png", "txt")
        image = caffe.io.load_image(loadingfile)
        f = open(textfile, 'w')
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        # Forward pass.
        cntFrame += 1
        start = time.time()
        detections = net.forward()['detection_out']
        end = time.time() - start
    	sumPTime += end
    	#print "frame: %d | runTime: %f\n" % (cntFrame, end)
        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_th]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(voc_labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        for i in xrange(top_conf.shape[0]):
            if 'DontCare' in top_labels[i]:
                continue
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = top_labels[i]
            f.write("%s -1 -1 -1 %d %d %d %d -1 -1 -1 -1 -1 -1 -1 %.2f\n" %(label, xmin, ymin, xmax, ymax, score))
        f.close()

avgPTime = sumPTime/cntFrame
print "### The end ###\n ### Average run time : %f" %(avgPTime)
fp.write("#############\n### Average run time ###\n%f"%(avgPTime))
print "--- End ---"