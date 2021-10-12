from torch.utils.data import DataLoader

#import yolov4 utils
from yolo4 import cfg
from yolov4.train import Yolo_loss,collate
from yolov4.dataset import Yolo_dataset
from yolov4.models import Yolov4
from yolov4.tool.tv_reference.utils import collate_fn as val_collate

def get_classes(classes_path):

    '''
    Read the class file and return class names as an array
    args : class file (classes.txt)

    '''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


class YoloV4model():

    def __init__(self, image_folder, csv_file):
        self.image_folder = image_folder
        self.csv_file = csv_file


    def train_model(self,pre_traind,
                        epochs = 100, 
                        batch_size = 8,
                        device = 0):

        classes = get_classes(classes_path)
        cfg.max_batches = classes * 2000
        cfg.steps = [int(cfg.max_batches*0.8), int(cfg.max_batches*0.9)]


        train_dataset = Yolo_dataset(lable_path, cfg, train=True)
        valid_dataset = Yolo_dataset(lable_path, cfg, train=False)

        model = Yolov4(pre_traind,n_classes=len(classes))
        model.to(device=device)

        train_loader = DataLoader(train_dataset,batch_size=batch_size // cfg.subdivisions,
                                    shuffle=True,num_workers=8, pin_memory=False,
                                    drop_last=True, collate_fn=collate)  


        valid_loader = DataLoader(valid_dataset,batch_size=batch_size // cfg.subdivisions,
                                    shuffle=True,num_workers=8, pin_memory=False,
                                    drop_last=True, collate_fn=val_collate )


        #learning rate setup from YOLOV4 train
        def burnin_schedule(i):
            if i < cfg.burn_in:
                factor = pow(i / cfg.burn_in, 4)
            elif i < cfg.steps[0]:
                factor = 1.0
            elif i < cfg.steps[1]:
                factor = 0.1
            else:
                factor = 0.0
            return factor

        if cfg.TRAIN_OPTIMIZER.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.learning_rate / batch_size,
                betas=(0.9, 0.999),
                eps=1e-08,
            )
        elif cfg.TRAIN_OPTIMIZER.lower() == 'sgd':
            optimizer = optim.SGD(
                params=model.parameters(),
                lr=cfg.learning_rate / batch_size,
                momentum=cfg.momentum,
                weight_decay=cfg.decay,
            )
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

        criterion = Yolo_loss(device=device, image_size=cfg.width, batch=batch_size // cfg.subdivisions, n_classes=len(classes))