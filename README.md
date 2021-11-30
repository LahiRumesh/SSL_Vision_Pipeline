## Semi-Supervised Pseudo Labeling with Object Detection Models

- Images Capturing Pipeline

    - [Images Capture with Augmentaion ](https://github.com/LahiRumesh/SSL_Vision_Pipeline/tree/main/image_capture) image_capture/capture_images.py
        ```bash
        cap = Image_Capture(0) #camera device id

        '''
        args : 
                img_size : Capture image size (i.e 608x608)
                img_dir : Image Folder
                rotate90 : rotate image by 90 degree angle
                rotate180 : rotate image by 180 degree angle
                rotate270 : rotate image by 270 degree angle
                scale : Scale the image
                scale_val : Scale value of the image
        '''
        cap.capture_image(img_size, 
                          img_dir,
                          rotate90=True,rotate180=True,rotate270=True,
                          scale=True,scale_val=0.2)
   
        ```

- Data Augmentaion Pipeline for Object Detection 

    - [Data Augmentation For Object Detection](https://github.com/LahiRumesh/Object-Detection_Data-Augmentation)
        ```bash
        git clone git@github.com:LahiRumesh/Object-Detection_Data-Augmentation.git
        cd Object-Detection_Data-Augmentation/
         ```


### Pytorch YOLO models Train 
- Data Prepare
    - annotation file should be in vott csv format

    | image | xmin | ymin | xmax | ymax | label 
    | :---: | :---: | :---: | :---: | :---: | :---:
    | image1.jpg | 50 | 150 | 288 | 328 | label1 | 
    | image1.jpg | 300 | 263 | 410 | 333 | label2 | 
    | image2.jpg | 88 | 63 | 110 | 223 | label1 |
    | image3.jpg | 22 | 190 | 150 | 250 | label3 |

    - Data Folder - >  

        -       image1.jpg 
                image2.jpg
                image3.jpg
                annotation.csv
- Use the **train_models.py** script to  train YOLOv3 and YOLOv4 models 

- Training arguments for the model training.

    - --model : object detection model, i.e. YOLOv4 or YOLOv3 

    - --data_dir : Image folder which contains images and the csv file

    - --weights : pre-trained weights path 

    - --validation : validation data split

    - --epochs : number of training epochs

    - --batch_size : train and validation batch size 

    - --image_size : train and test image size (should be multiply by 32  i.e (416,416),(512,512) or (608,608) )

    - --learning_rate: learning rate

    - --device : cuda device, i.e. 0 or 0,1,2,3 or cpu


### Pytorch model Inference with Pseudo Labeling   

- Use the **pseudo_label.py** script to for the pseudo labeling 

- Inference arguments for the pseudo labeling.

    - --checkpoint : saved checkpoint weight file path 

    - --class_file : class file which contains class names i.e class.names

    - --dir_name : folder path which contians unlabeld images

    - --conf_thresh : confident threshold value

    - --iou_thresh : IOU threshold value

    - --image_size : test image size (should be multiply by 32  i.e (416,416),(512,512) or (608,608) )



### Reference:

- [https://github.com/ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)
- [https://github.com/eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- [https://github.com/Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
```

@article{yolov4,
  title={YOLOv4: YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  journal = {arXiv},
  year={2020}
}
```
