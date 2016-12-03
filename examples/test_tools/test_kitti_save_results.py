import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from caffe.model_libs import *

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

#model_def = '/home/youngwan/caffe/models/New/KITTI/SSD_Res_15L_l4_ASP4_350x250/deploy.prototxt'
#model_weights = '/home/youngwan/caffe/models/New/KITTI/SSD_Res_15L_l4_ASP4_350x250/KITTI_SSD_Res_15L_l4_ASP4_350x250_iter_150000.caffemodel'

#model_def = '/home/youngwan/caffe/models/New/KITTI/SSD_Inception_Res_l2_ASP4_350x250/deploy.prototxt'
#model_weights = '/home/youngwan/caffe/models/New/KITTI/SSD_Inception_Res_l2_ASP4_350x250/KITTI_SSD_Inception_Res_l2_ASP4_350x250_iter_100000.caffemodel'

model_def = '/home/youngwan/caffe/models/New/KITTI/SSD_Inception_v2_Res_l2_ASP4_350x250/deploy.prototxt'
model_weights = '/home/youngwan/caffe/models/New/KITTI/SSD_Inception_v2_Res_l2_ASP4_350x250/KITTI_SSD_Inception_v2_Res_l2_ASP4_350x250_iter_100000.caffemodel'


#model_def = '/home/youngwan/caffe/models/VGGNet/KITTI/deploy.prototxt'
#model_weights = '/home/youngwan/caffe/models/VGGNet/KITTI/VGG_KITTI_SSD_300x300_iter_60000.caffemodel'

#model_def = '/home/youngwan/caffe/models/ResNet/KITTI/SSD_ResNet_ASP4_350x250_19L_conv4350x250/deploy.prototxt'
#model_weights = '/home/youngwan/caffe/models/ResNet/KITTI/SSD_ResNet_ASP4_350x250_19L_conv4350x250/ResNet_KITTI_SSD_ResNet_ASP4_350x250_19L_conv4350x250_iter_449000.caffemodel'




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

resize_width = 350
resize_height = 250
resize = "{}x{}".format(resize_width, resize_height)

net.blobs['data'].reshape(1,3,resize_height,resize_width)
cntcnt = 0
#image = caffe.io.load_image('examples/images/fish-bike.jpg')



now = datetime.datetime.now()
job_name = "SSD_incep_v2_res_l2_test_{}".format(resize)
#save_dir = "/home/youngwan/caffe/jobs/KITTI/test_KITTI_RAW_PED_results_imgs/{}".format(job_name)
save_dir = "/home/youngwan/caffe/jobs/KITTI/test_bechmark_results_imgs/{}".format(job_name)
save_file = "{}/{}_runTime_{}-{}-{}-{}:{}.txt".format(save_dir,job_name,now.year,now.month,now.day,now.hour,now.minute)
make_if_not_exist(save_dir)



img_path ="/home/youngwan/data/KITTI/kitti_test_benchmark_img/"
#img_path ="/home/youngwan/data/KITTI/raw/2011_09_28/2011_09_28_drive_0021_sync/image_02/"
image_list = os.listdir(img_path)
sorted_list = sorted(image_list)
for s in sorted_list:
    cntcnt += 1
    #loadingfile = '/home/cvlab/caffe/Narrow/' + s
    loadingfile = img_path + s
    image = caffe.io.load_image(loadingfile)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    # Forward pass.
    detections = net.forward()['detection_out']
    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]
    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(voc_labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    #print end  
    colors = ['m', 'y', 'c', 'g', 'r', 'b', 'k', 'w']
    temp = plt.imshow(image)
    currentAxis = plt.gca()
    #currentAxis = plt.figure()
    for i in xrange(top_conf.shape[0]):
    	if 'DontCare' in top_labels[i]:
    		continue
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = top_labels[i]
        name = '%s: %.2f'%(label, score)
        #name = '%s'%(label)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        if 'Car' in top_labels[i]:
            color = colors[0]
        if 'Van' in top_labels[i]:
            color = colors[1]
        if 'Truck' in top_labels[i]:
            color = colors[2]
        if 'Pedestrian' in top_labels[i]:
            color = colors[3]
        if 'Person_sitting' in top_labels[i]:
            color = colors[4]         
        if 'Cyclist' in top_labels[i]:
            color = colors[5]      
        if 'Tram' in top_labels[i]:
            color = colors[6]      
        if 'Misc' in top_labels[i]:
            color = colors[7]      
            
        currentAxis.text(xmin, ymin, name, fontsize = 5,  color = 'white', fontweight = 'bold', bbox={'facecolor':color, 'alpha':0.5})
        
        # if 'Tram' in top_labels[i]:
        #     currentAxis.text(xmin, ymin, name, fontsize = 5,  color = 'black', fontweight = 'bold', bbox={'facecolor':color, 'alpha':0.5})
        # if 'Cyclist' in top_labels[i]:
        #     currentAxis.text(xmin, ymin, name, fontsize = 5,  color = 'black', fontweight = 'bold', bbox={'facecolor':color, 'alpha':0.5})
          
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        #plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2)
        plt.text(xmin, ymin, name, size = 8, bbox={'facecolor':'white', 'alpha':0.5})
        #currentAxis.text(xmin, ymin, name, size = 10, bbox={'facecolor':'white', 'alpha':0.5})
        currentAxis.patch.set_visible(True)
        currentAxis.axis('off')
   

    img_name = save_dir +'/'+ str(cntcnt) +'.png'
    plt.savefig(img_name)
    #currentAxis.canvas.print_png(img_name)
    #plt.show(block=False)
    #plt.pause(0.001)
    plt.clf()
