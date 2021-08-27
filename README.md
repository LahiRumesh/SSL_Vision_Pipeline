## Pseudo Labeling with Object Detection

- Images Capturing Pipeline

    - [Images Capture with Augmentaion ](https://github.com/LahiRumesh/SSL_Vision_Pipeline/tree/main/image_capture) image_capture/capture_images.py
    - ```bash
        cap = Image_Capture(0) #camera device id
        
        cap.capture_image(img_size, 
                         img_dir,
                         rotate90=True,rotate180=True,rotate270=True,
                         scale=True,scale_val=0.2)

        ```

- Data Augmentaion Pipeline for Object Detection 

    - [Data Augmentation For Object Detection](https://github.com/LahiRumesh/Object-Detection_Data-Augmentation)
    - ```bash
        git clone git@github.com:LahiRumesh/Object-Detection_Data-Augmentation.git
        cd Object-Detection_Data-Augmentation/
         ```


- Pytorch YOLO model Trainer 

    - [YOLO Trainer](https://github.com/LahiRumesh/SSL_Vision_Pipeline/tree/main/yolo_train)