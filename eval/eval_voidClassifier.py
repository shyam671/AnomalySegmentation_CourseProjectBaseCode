# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
#from otherModel.ENet import ENet 
from otherModel.BiSeNetV1 import BiSeNetV1
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

#custom function to load model when not all dict elements
def load_my_state_dict(model, state_dict, model_name):
    if model_name == 'erfnet':
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
    else:
        model = model.load_state_dict(state_dict)
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="bisenetv1.pth")
    parser.add_argument('--loadModel', default="./models/BiSeNetV1.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--model', default='ERFNet', help="choose which model load between ERFNet, ENet, BiSeNet")
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []
    pathes = []
    if not os.path.exists('results_voidClassifier.txt'):
        open('results_voidClassifier.txt', 'w').close()
    file = open('results_voidClassifier.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)
    print("Model you choose : ", str(args.model)) 
    if args.model == 'ERFNet':
        model = ERFNet(NUM_CLASSES)
   # elif args.model == 'ENet':
        #model = ENet(NUM_CLASSES)
    elif args.model =='BiSeNet': 
        model = BiSeNetV1(NUM_CLASSES)
    else:
      raise Exception("Model Not found")
    

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()
    else:
        if args.model != 'ERFNet':
            raise Exception("Impossible to eval this model without cuda")


    Dataset_string = "LostAndFound"
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage), args.model)
    print("Model and weights LOADED successfully")
    model.eval()

    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        #print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)

        #########################
        # si fa questa cosa perché il modello BiSeNet produce due output: uno a bassa risoluzione e uno ad alta risoluzione. 
        # Il primo output ha una dimensione di 1/32 rispetto all’input, mentre il secondo ha una dimensione di 1/8. 
        # Per ottenere il risultato finale, si usa il secondo output, che è il primo elemento della lista restituita dal modello.
        # Quindi, si usa model(images)[0] per selezionare il secondo output1
        # Poi,si usa result.squeeze(0).data.cpu().numpy()[19,:,:] per ottenere il risultato della classe void, 
        # che è l’ultima classe tra le 20 classi del dataset Cityscapes. 
        # Si usa squeeze(0) per rimuovere la dimensione del batch, data.cpu().numpy() 
        # per convertire il tensore in un array numpy, e [19,:,:] per selezionare la ventesima fetta lungo la dimensione dei canali. 
        # Questo array numpy rappresenta il punteggio di anomalia per ogni pixel dell’immagine2
        # Infine, si usa path.replace("images", "labels_masks") per ottenere il percorso della maschera di verità (ground truth mask) 
        # corrispondente all’immagine di input. Questa maschera indica quali pixel appartengono alla classe void e quali no. 
        # Si usa questa maschera per calcolare le metriche di valutazione, come AuPRC e FPR95, confrontando il punteggio di anomalia con la verità.
        with torch.no_grad():
            if str(args.model) == 'BiSeNet':
                result = model(images)[0]
            else:
                result = model(images)
        anomaly_result = result.squeeze(0).data.cpu().numpy()[19,:,:]   #we are using the last channel for anomaly_result which is the background
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)
    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write('############################### ' + str(Dataset_string) + ' ###############################\n')
    file.write(('Model:' + str(args.model) + '   AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0)))
    file.write('\n\n')
    file.close()

if __name__ == '__main__':
    main()