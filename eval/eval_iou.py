# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)
    #model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    t_values = [0.01, 0.04, 0.05, 0.08, 0.1] 
    if float(args.temperature) == -1:  # Se temperature è -1, cerca il miglior valore tra t_values
        best_temperature = find_best_temperature(loader, model, args.cpu, t_values, args.method)
        print(f"Best temperature found: {best_temperature}")
        args.temperature = best_temperature
    else :
      
      iouEvalVal = iouEval(NUM_CLASSES)

      start = time.time()

      for step, (images, labels, filename, filenameGt) in enumerate(loader):
          if (not args.cpu):
              images = images.cuda()
              labels = labels.cuda()

          inputs = Variable(images)
          with torch.no_grad():
              outputs = model(inputs)

          # Seleziona le previsioni del modello in base al metodo specificato dalla riga di comando
          if args.method == 'msp':
              softmax_output = F.softmax(outputs / float(args.temperature), dim=1)
              predicted_labels = torch.argmax(softmax_output, dim=1).unsqueeze(1).data
          elif args.method == 'maxLogit':
              predicted_labels = torch.argmax(outputs, dim=1).unsqueeze(1).data
          elif args.method == 'maxEntr':
              predicted_labels = torch.argmax(F.softmax(outputs, dim=1), dim=1).unsqueeze(1).data

          iouEvalVal.addBatch(predicted_labels, labels)

          filenameSave = filename[0].split("leftImg8bit/")[1] 

          #print (step, filenameSave)


      iouVal, iou_classes = iouEvalVal.getIoU()

      iou_classes_str = []
      for i in range(iou_classes.size(0)):
        #iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iouStr = '{:0.2f}'.format(iou_classes[i]*100)
        iou_classes_str.append(iouStr)

      if not os.path.exists('mIoU_results.txt'):
        open('mIoU_results.txt', 'w').close()
    file = open('mIoU_results.txt', 'a')

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    file.write("================================ Model: ERFNET ================================\n")
    #print("TOTAL IOU: ", iou * 100, "%")
    file.write("Per-Class IoU:\n")
    file.write("Road -----> " + iou_classes_str[0])
    file.write("\nsidewalk -----> " + iou_classes_str[1])
    file.write("\nbuilding -----> " + iou_classes_str[2])
    file.write("\nwall -----> " + iou_classes_str[3])
    file.write("\nfence -----> " + iou_classes_str[4])
    file.write("\npole -----> " + iou_classes_str[5])
    file.write("\ntraffic light -----> " + iou_classes_str[6])
    file.write("\ntraffic sign -----> " + iou_classes_str[7])
    file.write("\nvegetation -----> " + iou_classes_str[8])
    file.write("\nterrain -----> " + iou_classes_str[9])
    file.write("\nsky -----> " + iou_classes_str[10])
    file.write("\nperson -----> " + iou_classes_str[11])
    file.write("\nrider -----> " + iou_classes_str[12])
    file.write("\ncar -----> " + iou_classes_str[13])
    file.write("\ntruck -----> " + iou_classes_str[14])
    file.write("\nbus -----> " + iou_classes_str[15])
    file.write("\ntrain -----> " + iou_classes_str[16])
    file.write("\nmotorcycle -----> " + iou_classes_str[17])
    file.write("\nbicycle -----> " + iou_classes_str[18])
    file.write("\n=======================================\n")
    #iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    iouStr = '{:0.2f}'.format(iouVal*100)
    file.write ("MEAN IoU: "+iouStr+"% with method: "+str(args.method) + " with temperature: "+ str(args.temperature))
    print ("MEAN IoU: "+iouStr+"% with method: "+str(args.method) + " with temperature: "+ str(args.temperature))
    
def find_best_temperature(loader, model, cpu, t_values, method):
    best_temperature = None
    best_miou = -1

    for temperature in t_values:
        print(f"Evaluating with temperature = {temperature}")
        iouVal, iouClasses = evaluate_model(loader, model, temperature, cpu)
        print(f"Mean IoU with temperature = {temperature}: {iouVal}")

        if iouVal > best_miou:
            best_miou = iouVal
            best_temperature = temperature
            best_classes=iouClasses
    

    best_class = []
    for i in range(best_classes.size(0)):
        #iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iouStr = '{:0.2f}'.format(best_classes[i]*100)
        best_class.append(iouStr)
    if not os.path.exists('mIoU_results.txt'):
      open('mIoU_results.txt', 'w').close()
    file = open('mIoU_results.txt', 'a')

    file.write("================================ Model: ERFNET ================================\n")
    #print("TOTAL IOU: ", iou * 100, "%")
    file.write("Per-Class IoU:\n")
    file.write("Road -----> " + best_class[0])
    file.write("\nsidewalk -----> " + best_class[1])
    file.write("\nbuilding -----> " + best_class[2])
    file.write("\nwall -----> " + best_class[3])
    file.write("\nfence -----> " + best_class[4])
    file.write("\npole -----> " + best_class[5])
    file.write("\ntraffic light -----> " + best_class[6])
    file.write("\ntraffic sign -----> " + best_class[7])
    file.write("\nvegetation -----> " + best_class[8])
    file.write("\nterrain -----> " + best_class[9])
    file.write("\nsky -----> " + best_class[10])
    file.write("\nperson -----> " + best_class[11])
    file.write("\nrider -----> " + best_class[12])
    file.write("\ncar -----> " + best_class[13])
    file.write("\ntruck -----> " + best_class[14])
    file.write("\nbus -----> " + best_class[15])
    file.write("\ntrain -----> " + best_class[16])
    file.write("\nmotorcycle -----> " + best_class[17])
    file.write("\nbicycle -----> " + best_class[18])
    file.write("\n=======================================\n")
    #iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    iouStr = '{:0.2f}'.format(iouVal*100)
    file.write ("MEAN IoU: "+iouStr+"% with method: "+str(method) + " with temperature: "+ str(best_temperature))
    print ("MEAN IoU: "+iouStr+"% with method: "+str(method) + " with temperature: "+ str(best_temperature))
    

    return best_temperature

def evaluate_model(loader, model, temperature, cpu):
    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        # Seleziona le previsioni del modello in base al metodo specificato dalla riga di comando
        softmax_output = F.softmax(outputs / temperature, dim=1)
        predicted_labels = torch.argmax(softmax_output, dim=1).unsqueeze(1).data

        iouEvalVal.addBatch(predicted_labels, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 

        #print (step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()
    print(f"Took {time.time()-start} seconds")

    return iouVal, iou_classes

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default='msp')  # Aggiunge l'argomento method con valore predefinito 'msp'
    parser.add_argument('--temperature', type=float, default=1.0)  # Aggiunge l'argomento temperature con valore predefinito -1
    
    main(parser.parse_args())
