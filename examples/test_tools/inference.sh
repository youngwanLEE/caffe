# ./build/examples/test_tools/ssd_detect_kitti_inference.bin 실행파일
# argv[0] : deploy.prototxt 경로
# argv[1] : .caffemodel 경로
# argv[2] : 테스트 영상 경로가 담긴.txt
# argv[3] : 평균실행시간.txt 저장경로

./build/examples/test_tools/ssd_detect_kitti_inference.bin \
models/New/KITTI/SSD_Inception_v2_Res_l2_ASP4_350x250/deploy.prototxt \
models/New/KITTI/SSD_Inception_v2_Res_l2_ASP4_350x250/KITTI_SSD_Inception_v2_Res_l2_ASP4_350x250_iter_100000.caffemodel \
/home/youngwan/data/KITTI/KITTI_raw_ped.txt \
jobs/New/KITTI/SSD_Inception_v2_Res_l2_ASP4_350x250/test.txt
#/home/youngwan/data/demos/Narrow.txt
#/home/youngwan/data/KITTI/KITTI_raw_ped.txt
