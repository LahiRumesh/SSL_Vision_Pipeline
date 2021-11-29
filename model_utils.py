import os
import torch
from torch.utils.data import DataLoader
from torch import mode, optim
from tqdm.auto import tqdm
from collections import deque
import cv2
import numpy as np
import pandas as pd
#import yolov4 utils
from yolov4.cfg import Cfg as cfg
from yolov4.train import Yolo_loss,collate,evaluate
from yolov4.dataset import Yolo_dataset
from yolov4.models import Yolov4
from yolov4.tool.tv_reference.utils import collate_fn as val_collate
from yolov4.tool.torch_utils import do_detect
from yolov4.tool.utils import load_class_names
from tensorboardX import SummaryWriter
import wandb

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
    '''

    Train the yolov4 model with pretrained .pth weights
    YOLOv4 pytorch training reference : https://github.com/Tianxiaomo/pytorch-YOLOv4

    '''
    def __init__(self, pre_traind):
        self.pre_traind = pre_traind
        

    def train_model(self,data_prefix, classes_path=None,
                        image_size = (416,416),
                        learning_rate = 0.00261,
                        epochs = 100, 
                        batch_size = 8,
                        device = 0,
                        log_step=20,save_checkpoint=True,keep_weights=5):


        '''
        train the model with pre process data, train reference https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/train.py

        args:
            data_prefix : image pre process dir (all the log files and checkpoints will be save in this)
            classes_path : file contains the classes names (class.names)
            image_size : training image size (should be multiply by 32  i.e (416,416),(512,512) or (608,608) )
            learning_rate : learning rate
            epochs : number of training epochs
            batch_size : batch size
            device : cuda device (cuda device, i.e. 0 or 0,1,2,3 or cpu)
            log_step : logging steps
            save_checkpoint : save .pth weights
            keep_weights : maximum number of weights keep in the checkpoints 

        '''

        train_data = os.path.join(data_prefix,'train.txt')
        val_data = os.path.join(data_prefix,'val.txt')
        classes = get_classes(classes_path)
        cfg.max_batches = len(classes) * 2000
        cfg.steps = [int(cfg.max_batches*0.8), int(cfg.max_batches*0.9)]
        cfg.learning_rate = learning_rate
        cfg.width, cfg.height = image_size[0],image_size[1]
        cfg.keep_checkpoint_max = keep_weights

        train_dataset = Yolo_dataset(train_data, cfg, train=True)
        valid_dataset = Yolo_dataset(val_data, cfg, train=False)
        n_train = len(train_dataset)
        model = Yolov4(self.pre_traind,n_classes=len(classes))
        model.to(device=device)

        train_loader = DataLoader(train_dataset,batch_size=batch_size // cfg.subdivisions,
                                    shuffle=True,num_workers=8, pin_memory=False,
                                    drop_last=True, collate_fn=collate)  


        val_loader = DataLoader(valid_dataset,batch_size=batch_size // cfg.subdivisions,
                                    shuffle=True,num_workers=8, pin_memory=False,
                                    drop_last=True, collate_fn=val_collate )

        writer = SummaryWriter(log_dir=os.path.join(data_prefix, 'log'),
                               filename_suffix=f'OPT_{cfg.TRAIN_OPTIMIZER}_LR_{cfg.learning_rate}_BS_{batch_size}_Sub_{cfg.subdivisions}_Size_{cfg.width}',
                               comment=f'OPT_{cfg.TRAIN_OPTIMIZER}_LR_{cfg.learning_rate}_BS_{cfg.batch}_Sub_{cfg.subdivisions}_Size_{cfg.width}')

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
        global_step = 0
        saved_models = deque()
        checkpoints_dir = os.path.join(data_prefix, 'checkpoints')
        project_prefix = os.path.basename(data_prefix)

        wandb.login()
        wandb.init(project=f"yolo-{project_prefix}".replace("/", "-"), config=cfg)
        model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_step = 0

            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=50) as pbar:
                for i, batch in enumerate(train_loader):
                    global_step += 1
                    epoch_step += 1
                    images = batch[0]
                    bboxes = batch[1]
                    images = images.to(device=device, dtype=torch.float32)
                    bboxes = bboxes.to(device=device)

                    bboxes_pred = model(images)
                    loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                    loss.backward()

                    epoch_loss += loss.item()

                    if global_step % cfg.subdivisions == 0:
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()

                    if global_step % (log_step * cfg.subdivisions) == 0:
                        writer.add_scalar('train/Loss', loss.item(), global_step)
                        writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                        writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                        writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                        writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                        writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                        writer.add_scalar('lr', scheduler.get_lr()[0] * cfg.batch, global_step)
                        pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                                            'loss_wh': loss_wh.item(),
                                            'loss_obj': loss_obj.item(),
                                            'loss_cls': loss_cls.item(),
                                            'loss_l2': loss_l2.item(),
                                            'lr': scheduler.get_lr()[0] * cfg.batch
                                            })

                    pbar.update(images.shape[0])


                if save_checkpoint:
                    try:
                        os.makedirs(checkpoints_dir, exist_ok=True)
                    except OSError:
                        pass
                    save_path = os.path.join(checkpoints_dir, f'{project_prefix}{epoch + 1}.pth')
                    torch.save(model.state_dict(), save_path)

                    saved_models.append(save_path)
                    if len(saved_models) > cfg.keep_checkpoint_max > 0:
                        model_to_remove = saved_models.popleft()
                        try:
                            os.remove(model_to_remove)
                        except:
                            print(f'failed to remove {model_to_remove}')
                    #validation
                    eval_model = Yolov4(self.pre_traind, n_classes=len(classes), inference=True)
                    eval_model.to(device=device)
                    eval_model.load_state_dict(model.state_dict())
                    evaluator = evaluate(eval_model, val_loader, cfg, device)
                    del eval_model

                    stats = evaluator.coco_eval['bbox'].stats
                    
                    wandb.log({"train_loss": loss.item(),
                       "train_loss_object" : loss_obj.item(),
                       "train_loss_class" : loss_cls.item(),
                       "train_loss_l2" : loss_l2.item(),
                       "train_AP" :  stats[0],
                       "train_AP50" : stats[1],
                       "train_AP75" : stats[2]
                        })
                    

        writer.close()



class Yolov4Inference():
    '''
    YOLOV4 inference with PyTorch weights which use for pseudo labeling
    
    '''
    
    def __init__(self,
                class_file,
                weightfile,
                use_cuda = True) -> None:
        

        self.class_file = class_file
        self.weightfile = weightfile
        self.use_cuda = use_cuda


    def getImageList(self,
                        dirName,
                        endings=['.jpg','.jpeg','.png','.JPG']):

        listOfFile = os.listdir(dirName)
        allFiles = list()

        for i,ending in enumerate(endings):
            if ending[0]!='.':
                endings[i] = '.'+ending
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.getImageList(fullPath,endings)
            else:
                for ending in endings:
                    if entry.endswith(ending):
                        allFiles.append(fullPath)               
        return allFiles  
        

    def getDetails(self,
                        img,
                        boxes,
                        class_names=None):

        img = np.copy(img)   
        width = img.shape[1]
        height = img.shape[0]

        out_prediction = []
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = int(box[0] * width) if int(box[0] * width) > 0 else 0
            y1 = int(box[1] * height) if int(box[1] * height) > 0 else 0#int(box[1] * height)
            x2 = int(box[2] * width) if int(box[2] * width) > 0 else 0#int(box[2] * width)
            y2 = int(box[3] * height) if int(box[3] * height) > 0 else 0#int(box[3] * height)
            out_prediction.append([x1,y1,x2,y2,class_names[box[6]]])

        return out_prediction


    def pseduolabel(self,dirName,img_size,
                            conf_thresh = 0.4,
                            iou_thresh=0.2):

        n_classes = len(get_classes(self.class_file))
        model = Yolov4(yolov4conv137weight=None, n_classes=n_classes,inference=True)
        pretrained_dict = torch.load(self.weightfile, map_location=torch.device('cuda'))
        model.load_state_dict(pretrained_dict)

        if self.use_cuda:
            model.cuda()

        input_paths = self.getImageList(dirName)
        input_image_paths = [] 
    
        for img in input_paths:
            input_image_paths.append(img)

        out_df = pd.DataFrame(columns=['image','xmin', 'ymin', 'xmax', 'ymax', 'label'])

        for imgfile in tqdm(input_image_paths):
            img = cv2.imread(imgfile)
            sized = cv2.resize(img, (img_size[0], img_size[1]))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            boxes = do_detect(model, sized, conf_thresh, iou_thresh, self.use_cuda)
            class_names = load_class_names(self.class_file)
            out_prediction = self.getDetails(img,boxes[0],class_names=class_names)

            for pred in out_prediction:
                out_df = out_df.append(pd.DataFrame([[os.path.basename(imgfile)]+pred],columns=['image','xmin', 'ymin', 'xmax', 'ymax', 'label']))

        out_df.to_csv(os.path.join(dirName,'psedo_data.csv'),index=False)



class YoloV3model:
    def __init__(self)->None:
         self.cfg = Cfg
