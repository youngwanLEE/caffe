

#new preActiv
#./build/examples/kitti_ssd/ssd_detect_kitti_lyw_v1.bin models/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/deploy.prototxt models/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/KITTI_SSD_PreActiv_Res_15L_l4_ASP4_350x250_iter_61500.caffemodel /home/youngwan/data/demos/demo_0225.txt jobs/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/

#KITTI
#./build/examples/kitti_ssd/ssd_detect_kitti_lyw_v1.bin models/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/deploy.prototxt models/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/KITTI_SSD_PreActiv_Res_15L_l4_ASP4_350x250_iter_61500.caffemodel /home/youngwan/data/KITTI/test_kitti_images.txt jobs/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/

# Res_15_350x250
./build/examples/kitti_ssd/ssd_detect_kitti_lyw_v1.bin models/New/KITTI/SSD_Res_15L_l4_ASP4_350x250/deploy.prototxt models/New/KITTI/SSD_Res_15L_l4_ASP4_350x250/KITTI_SSD_Res_15L_l4_ASP4_350x250_iter_150000.caffemodel /home/youngwan/data/KITTI/test_kitti_images.txt jobs/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/

#Res_19_350x250
#./build/examples/kitti_ssd/ssd_detect_kitti_lyw_v1.bin models/ResNet/KITTI/SSD_ResNet_ASP4_19L_350x250/deploy.prototxt models/ResNet/KITTI/SSD_ResNet_ASP4_19L_350x250/ResNet_KITTI_SSD_ResNet_ASP4_350x250_19L_conv4350x250_iter_450000.caffemodel /home/youngwan/data/demos/demo_0225.txt jobs/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/

#KITTI
#./build/examples/kitti_ssd/ssd_detect_kitti_lyw_v1.bin models/ResNet/KITTI/SSD_ResNet_ASP4_19L_350x250/deploy.prototxt models/ResNet/KITTI/SSD_ResNet_ASP4_19L_350x250/ResNet_KITTI_SSD_ResNet_ASP4_350x250_19L_conv4350x250_iter_450000.caffemodel /home/youngwan/data/KITTI/test_kitti_images.txt jobs/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/


#Pascal VOC
#./build/examples/kitti_ssd/ssd_detect_kitti_lyw_v1.bin models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel /home/youngwan/data/demos/demo_0223.txt jobs/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/


# video
#./build/examples/kitti_ssd/ssd_detect_lyw.bin models/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/deploy.prototxt models/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/KITTI_SSD_PreActiv_Res_15L_l4_ASP4_350x250_iter_61500.caffemodel data/kitti_ssd/test_kitti_images.txt jobs/New/KITTI/SSD_PreActiv_Res_15L_l4_ASP4_350x250/