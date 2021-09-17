import torch
from torchvision import transforms
import os
import argparse
from darknet import Darknet, parse_cfg
from util import *
from data_aug.data_aug import Sequence
from preprocess import *
import numpy as np
import cv2
import pickle as pkl
import matplotlib.pyplot as plt
from bbox import bbox_iou, corner_to_center, center_to_corner
import pickle 
from customloader import custom_transforms, CustomDataset
import torch.optim as optim
import torch.autograd.gradcheck
from tensorboardX import SummaryWriter
import sys 


writer = SummaryWriter()

random.seed(0)


class YOLOv3_model():
    def __init__(self,cfgfile):
        self.cfgfile = cfgfile
    
    def YOLO_loss(self,ground_truth, output,num_classes):

        total_loss = 0

        #get the objectness loss
        loss_inds = torch.nonzero(ground_truth[:,:,-4] > -1)
        objectness_pred = output[loss_inds[:,0],loss_inds[:,1],4]
        target = ground_truth[loss_inds[:,0],loss_inds[:,1],4]
        objectness_loss = torch.nn.MSELoss(size_average=False)(objectness_pred, target)
        #print("Obj Loss", objectness_loss)
        #Only objectness loss is counted for all boxes
        object_box_inds = torch.nonzero(ground_truth[:,:,4] > 0).view(-1, 2)

        try:
            gt_ob = ground_truth[object_box_inds[:,0], object_box_inds[:,1]]
        except IndexError:
            return None

        pred_ob = output[object_box_inds[:,0], object_box_inds[:,1]]

        #get centre x and centre y 
        centre_x_loss = torch.nn.MSELoss(size_average=False)(pred_ob[:,0], gt_ob[:,0])
        centre_y_loss = torch.nn.MSELoss(size_average=False)(pred_ob[:,1], gt_ob[:,1])

        #print("Center_x_loss", float(centre_x_loss))
        #print("Center_y_loss", float(centre_y_loss))

        total_loss += centre_x_loss 
        total_loss += centre_y_loss 

        #get w,h loss
        w_loss = torch.nn.MSELoss(size_average=False)(pred_ob[:,2], gt_ob[:,2])
        h_loss = torch.nn.MSELoss(size_average=False)(pred_ob[:,3], gt_ob[:,3])

        total_loss += w_loss 
        total_loss += h_loss 

        print("w_loss:", float(w_loss))
        print("h_loss:", float(h_loss))

        cls_labels = torch.zeros(gt_ob.shape[0], num_classes).to(device)
        cls_labels[torch.arange(gt_ob.shape[0]).long(), gt_ob[:,5].long()] = 1
        cls_loss = 0    

        for c_n in range(num_classes):
            targ_labels = pred_ob[:,5 + c_n].view(-1,1)
            targ_labels = targ_labels.repeat(1,2)
            targ_labels[:,0] = 1 - targ_labels[:,0]
            cls_loss += torch.nn.CrossEntropyLoss(size_average=False)(targ_labels, cls_labels[:,c_n].long())

        #print(cls_loss)
        total_loss += cls_loss

        return total_loss

    def train_model(self,weights,data_file,epochs=50,
                    num_classes=10,batch_size=10,image_size=416,
                    lr=0.001,momentum=0.9,wd=0.0005,unfreeze=2
                    ):
        #Load the model
        model = Darknet(self.cfgfile, train=True)

        # "unfreeze" refers to the last number of layers to tune (allow gradients to be tracked)
        p_i = 1
        p_len = len(list(model.parameters()))
        unfreeze = args.unfreeze
        stop_layer = p_len - unfreeze

        model.load_weights(weights, stop=stop_layer)

        # Freeze all weights before layer "stop_layer" from "unfreeze" argument
        for p in model.parameters():
            if p_i < stop_layer:
                p.requires_grad = False
            else:
                p.requires_grad = True
            p_i += 1

        # select the device 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.train()
        model = model.to(device)

        # Overloading custom data transforms from customloader (may add more here)
        custom_transforms = Sequence([YoloResizeTransform(args.image_size)])

        # Data instance and loader
        data = CustomDataset(root="data", num_classes=num_classes,ann_file=data_file, det_transforms=custom_transforms)
        train_loader = DataLoader(data, 
                         batch_size=batch_size,
                         shuffle=False,
                         collate_fn=data.collate_fn)

        # Use this optimizer calculation for training loss
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

        ### TRAIN MODEL ###

        itern = 0
        lr_update_step = 0.8 * epochs
        lr_updated = False
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_step = 0
            for image, ground_truth in train_loader:
                if len(ground_truth) == 0:
                    continue
                
                # # Track gradients in backprop
                image = image.to(device)
                ground_truth = ground_truth.to(device)

                output = model(image)

                # Clear gradients from optimizer for next iteration
                optimizer.zero_grad()

                print("\n\n")
                print('Iteration ', itern)

                if (torch.isnan(ground_truth).any()):
                    print("Nans in Ground_truth")
                    assert False

                if (torch.isnan(output).any()):
                    print("Nans in Output")
                    assert False

                if (ground_truth == float("inf")).any() or (ground_truth == float("-inf")).any():
                    print("Inf in ground truth")
                    assert False


                if (output == float("inf")).any() or (output == float("-inf")).any():
                    print("Inf in output")
                    assert False

                loss  = self.YOLO_loss(ground_truth, output,num_classes)

                # Update LR
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                print(epoch_loss)
                for param_group in optimizer.param_groups:
                    if itern >= lr_update_step and lr_updated == False:
                        optimizer.param_groups[0]["lr"] = (lr*pow((itern / lr_update_step),4))
                        lr_updated = True
                print('lr: ', optimizer.param_groups[0]["lr"])

                #if loss:
                #    print("Loss for iter no: {}: {}".format(itern, float(loss)/batch_size))
                #    writer.add_scalar("Loss/vanilla", float(loss), itern)
                #    loss.backward()
                #    optimizer.step()

                itern += 1

        ### FINE TUNE MODEL ON MORE LAYERS ###

        # "unfreeze" refers to the last number of layers to tune (allow gradients to be tracked)
        p_i = 1
        p_len = len(list(model.parameters()))
        stop_layer = 5 # Unfreeze all but this number of layers at the beginning

        # Unfreeze more layers for fine-tuning
        for p in model.parameters():
            if p_i < stop_layer:
                p.requires_grad = False
            else:
                p.requires_grad = True
            p_i += 1

        # New iteration counter and make sure LR is set correctly
        itern_fine = 0
        lr = optimizer.param_groups[0]["lr"] / 10
        lr_updated = False
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

        # Reset data loader
        print('Batch size ', bs)
        # Data instance and loader
        data = CustomDataset(root="data", num_classes=num_classes, 
                             ann_file="annt.txt", 
                             det_transforms=custom_transforms)
        #print ("*************************")
        data_loader = DataLoader(data, batch_size=bs,
                                 shuffle=False,
                                 collate_fn=data.collate_fn)

        for image, ground_truth in data_loader:
            if len(ground_truth) == 0:
                continue
            
            # # Track gradients in backprop
            image = image.to(device)
            ground_truth = ground_truth.to(device)

            output = model(image)

            # Clear gradients from optimizer for next iteration
            optimizer.zero_grad()

            print("\n\n")
            print('Iteration ', itern)

            if (torch.isnan(ground_truth).any()):
                print("Nans in Ground_truth")
                assert False

            if (torch.isnan(output).any()):
                print("Nans in Output")
                assert False

            if (ground_truth == float("inf")).any() or (ground_truth == float("-inf")).any():
                print("Inf in ground truth")
                assert False

            if (output == float("inf")).any() or (output == float("-inf")).any():
                print("Inf in output")
                assert False

            loss  = YOLO_loss(ground_truth, output)

            # Update learning rate (decrease) at lr_update_step specified above
            for param_group in optimizer.param_groups:
                if itern_fine >= lr_update_step and lr_updated == False:
                    optimizer.param_groups[0]["lr"] = (lr*pow((itern_fine / lr_update_step),4))
                    lr_updated == True

            print('lr: ', optimizer.param_groups[0]["lr"])
            if loss:
                print("Loss for iter no: {}: {}".format(itern, float(loss)/bs))
                writer.add_scalar("Loss/vanilla", float(loss), itern)
                if itern_fine % 5 == 0:
                    torch.save(model.state_dict(), os.path.join('logs', 'epoch{0}-bs{1}-loss{2:.4f}.pth'.format(itern, bs, float(loss)/bs)))
                loss.backward()
                optimizer.step()

            itern_fine += 1
            itern += 1

        writer.close()

        # Save final model in pytorch format (the state dictionary only, i.e. parameters only)
        torch.save(model.state_dict(), os.path.join('logs', 'epoch{0}-final-bs{1}-loss{2:.4f}.pth'.format(itern, bs, float(loss)/bs)))    





if __name__ == '__main__':


    def arg_parse():
        """
        Parse arguements to the detect module

        """
        parser = argparse.ArgumentParser(description='YOLO v3 Training Module')
        parser.add_argument("--cfg", dest = 'cfgfile', help ="Config file", default = "cfg/yolov3.cfg", type = str)
        parser.add_argument("--weights", dest = 'weightsfile', help ="weightsfile",default = "weights/yolov3.weights", type = str)
        #parser.add_argument("--datacfg", dest = "datafile", help = "cfg file containing the configuration for the dataset",type = str, default = "data/obj.data")
        parser.add_argument("--epochs", dest = "epochs", help = "number of epochs",type = int, default = 60)
        parser.add_argument("--classes", dest = "classes", help = "number of classes",type = int, default = 15)
        parser.add_argument("--batch_size", dest = "batch_size", help = "batch size",type = int, default = 10)
        parser.add_argument("--image_size", dest = "image_size", help = "Input image size",type = int, default = 416)
        parser.add_argument("--lr", dest = "lr", type = float, default = 0.001)
        parser.add_argument("--mom", dest = "mom", type = float, default = 0)
        parser.add_argument("--wd", dest = "wd", type = float, default = 0)
        parser.add_argument("--unfreeze", dest = "unfreeze", type = int, default = 2, help="Last number of layers to unfreeze for training")
        parser.add_argument("--datafile", dest = "datafile", help = "cfg file containing the configuration for the dataset",type = str, default = "data/train.txt")

        return parser.parse_args()

    args = arg_parse()


    yolo = YOLOv3_model(args.cfgfile)
    yolo.train_model(args.weightsfile,args.datafile,
                    args.epochs,args.classes,args.batch_size,
                    args.image_size,args.lr,args.mom,args.wd,args.unfreeze)