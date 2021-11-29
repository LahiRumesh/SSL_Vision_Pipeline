import argparse
from model_utils import Yolov4Inference



if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/prefix50.pth', help='saved checkpoint weight file path')
    parser.add_argument('--class_file', type=str, default='class.names', help='class file which contains class names i.e class.names')
    parser.add_argument('--dir_name', type=str, default='test_set', help='folder path which contians unlabeld images')
    parser.add_argument('--conf_thresh', type=float, default=0.6, help='Confident threshold value')
    parser.add_argument('--iou_thresh', type=float, default=0.2, help='IOU threshold value')
    parser.add_argument('--image_size', nargs='+', type=int, default=(416, 416), help='test image size (should be multiply by 32  i.e (416,416),(512,512) or (608,608) )')

    args = parser.parse_args()

    infer_yolov4 = Yolov4Inference(args.class_file,args.checkpoint)
    infer_yolov4.pseduolabel(args.dir_name,args.image_size,args.conf_thresh,args.iou_thresh)
