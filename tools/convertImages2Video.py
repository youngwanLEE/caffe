import cv2
import os

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')


# img_path ="/home/youngwan/caffe/jobs/KITTI/test_KITTI_RAW_PED_results_imgs/SSD_ResNet_19L_test_350x250/" # 1.png, 2.png, 3.png, ...
# out_path ="/home/youngwan/caffe/jobs/KITTI/test_KITTI_RAW_PED_results_imgs/SSD_ResNet_19L_test_350x250/video_res_19L.avi"

# img_path ="/home/youngwan/caffe/jobs/KITTI/test_bechmark_results_imgs/SSD_Res_19L_ASP4_test_350x250/" # 1.png, 2.png, 3.png, ...
# out_path ="/home/youngwan/caffe/jobs/KITTI/test_bechmark_results_imgs/SSD_Res_19L_ASP4_test_350x250/video_res_19L.avi"

img_path ="/home/youngwan/caffe/jobs/KITTI/test_bechmark_results_imgs/SSD_incep_v2_res_l2_test_350x250/" # 1.png, 2.png, 3.png, ...
out_path ="/home/youngwan/caffe/jobs/KITTI/test_bechmark_results_imgs/SSD_incep_v2_res_l2_test_350x250/video_new2.avi"



img_type = ".png" #default
img_size = (1000,1000)
out_fps = 5


image_list = os.listdir(img_path)
#print len(image_list)
sorted_list=[None]*len(image_list)

for rnd_img in image_list:
    index = rnd_img.split(img_type)[0]
    #print int(index)
    sorted_list[int(index)-1] = int(index)
#print sorted_list

def make_video(images, outvid=None, fps=5, size=None,
               is_color=True, format="MJPG"):
               #is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    #size = width, height
    vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
    #print outvid, size, vid

    print "--encoding--"
    for image in images:
        img_name = img_path+str(image)+img_type
        if not os.path.exists(img_name):
            print image+" not exist"
            break
        img = imread(img_name)

        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

make_video(sorted_list, out_path, out_fps, img_size)
print "--the end--"
