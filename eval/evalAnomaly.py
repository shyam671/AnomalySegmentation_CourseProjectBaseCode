# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
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


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  # can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    # ADDED FOR THE PROJECT
    parser.add_argument('--method', default='msp')
    parser.add_argument('--temperature', default=1.0)
    ####
    args = parser.parse_args()
    #print(float(args.temperature)==-1.0)
    if float(args.temperature) != -1.0:  # Check if temperature argument is set to 1
        evaluate_model(args)  # If temperature is !=-1, evaluate the model with default temperature
    else:
        print(f"Finding best temperature scaling ")
        # Initialize temperature values for grid search
        t_values = [0.01, 0.04, 0.05, 0.08, 0.1]
        best_t = None
        best_score = -np.inf
        best_fpr = 0.0
        # Perform grid search over temperature values
        for t in t_values:
            args.temperature = t  # Set current temperature value
            prc_auc, fpr, temperature = evaluate_model(args)  # Evaluate model with current temperature
            print(f'Evaluation with Temperature={temperature}:')
            print(f'AUPRC score: {prc_auc*100.0}')
            print(f'FPR@TPR95: {fpr*100.0}')

            # Update best temperature and score if necessary
            if prc_auc > best_score:
                best_t = temperature
                best_score = prc_auc
                best_fpr = fpr*100.0

        print(f'Best temperature found: {best_t}')
        print(f'Corresponding AUPRC score: {best_score*100.0}')
        print(f'Corresponding FPR95 score: {best_fpr}')

def evaluate_model(args):
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
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
    Dataset_string = "LostAndFound"
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully")
    model.eval()
    temperature = float(args.temperature)
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        #print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0, 3, 1, 2)
        with torch.no_grad():
            result = model(images)
        # ADDED FOR THE PROJECT
        if args.method == 'msp':
            softmax_probs = torch.nn.functional.softmax(result.squeeze(0) / temperature, dim=0)
            anomaly_result = 1.0 - np.max(softmax_probs.data.cpu().numpy(), axis=0)
        elif args.method == 'maxLogit':
            anomaly_result = -(np.max(result.squeeze(0).data.cpu().numpy(), axis=0))
        elif args.method == 'maxEntr':
            softmax_probs = torch.nn.functional.softmax(result.squeeze(0), dim=0)
            log_softmax_probs = torch.nn.functional.log_softmax(result.squeeze(0), dim=0)
            anomaly_result = -torch.sum(softmax_probs * log_softmax_probs, dim=0).data.cpu().numpy()
        ####
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            Dataset_string = "Road Obstacle 21"
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            Dataset_string = "FS Static"
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        ood_gts = np.array(mask)
        ### da controllare: nell'if del LostAndFound non entra mai
        if "RoadAnomaly" in pathGT:
            Dataset_string = "Road Anomaly"
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            Dataset_string = "Lost & Found"
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue
        else:
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write("\n")

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
    print(f'Temperature : {temperature}')
    # SOME EXTRA WRITING ON THE FILE IN ORDER TO BE MORE READABLE
    file.write('############################### ' + str(Dataset_string) + ' ###############################\n')
    file.write(('Method:' + str(args.method) + '   AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0)+ " with Temperature: " + str(temperature)))
    file.write('\n\n')
    file.close()

    return prc_auc, fpr, temperature


if __name__ == '__main__':
    main()
