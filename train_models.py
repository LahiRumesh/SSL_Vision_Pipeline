import os
import argparse
from model_utils import YoloV4model,YoloV3model
from data_utils.data_process import ImagePreProcess


class ModelTrainer:

    def __init__(self,model):
        self.model = model

    def run(self,
            data_dir,  
            weights,
            val_split=0.1,
            batch_size=2,
            image_size=(416,416),
            learning_rate=0.001,
            epochs=50,device=0):

        data_prepare = ImagePreProcess(val_split)
        data_prefix = data_prepare.csv_data_process(folder_path = data_dir)
        if self.model == "YOLOv3":
            yolov3 = YoloV3model(weights)
        
        elif self.model == "YOLOv4":
            yolov4 = YoloV4model(weights)
            yolov4.train_model(data_prefix,
                    classes_path = os.path.join(data_prefix,'class.names'),
                    batch_size=batch_size,
                    image_size=image_size,
                    learning_rate=learning_rate,
                    epochs=epochs,device=device)

if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='YOLOv4', help='Object detection model, i.e. YOLOv4 or YOLOv3')
    parser.add_argument('--data_dir', type=str, default='/image_data', help='Image folder which contains images and the csv file')
    parser.add_argument('--weights', type=str, default='yolov4.conv.137.pth', help='pre-trained weights path')
    parser.add_argument('--validation', type=float, default=0.1, help='validation data split')
    parser.add_argument('--epochs', type=int, default=50 ,help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--image_size', nargs='+', type=int, default=(416, 416), help='train and test image size (should be multiply by 32  i.e (416,416),(512,512) or (608,608) )')
    parser.add_argument('--learning_rate', type=float, default=0.00261, help='learning rate')
    parser.add_argument('--device', type=int, default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_args()

    model_ = ModelTrainer(args.model)
    model_.run(args.data_dir, 
               args.weights,
               args.validation,
               args.batch_size,
               args.image_size,
               args.learning_rate,
               args.epochs,
               args.device
                )
