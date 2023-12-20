# Code for evaluating IoU 
# Nov 2017
# Eduardo Romera
#######################

import torch

class iouEval:

    def __init__(self, nClasses, ignoreIndex=19):
        # define the number of classes
        self.nClasses = nClasses
        # define the class to ignore
        # if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.ignoreIndex = ignoreIndex if nClasses>ignoreIndex else -1
        # reset the evalutation
        self.reset()

    def reset (self):
        # define how many classes to consider
        classes = self.nClasses if self.ignoreIndex==-1 else self.nClasses-1
        # define the number of true positives per class
        self.tp = torch.zeros(classes).double()
        # define the number of false positives per class
        self.fp = torch.zeros(classes).double()
        # define the number of false negatives per class
        self.fn = torch.zeros(classes).double()        

    def addBatch(self, x, y):  # x=preds, y=targets
        #sizes should be "batch_size x nClasses x H x W"
        
        #print ("X is cuda: ", x.is_cuda)
        #print ("Y is cuda: ", y.is_cuda)

        # if indicated, move the predictions and targets to the CUDA device
        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        # if predictions size is "batch_size x 1 x H x W" scatter to onehot
        if (x.size(1) == 1):
            # transform the predictions to one-hot encodings
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))  
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, x, 1).float()
        else:
            # do not make any changes, otherwise
            x_onehot = x.float()

        # if targets size is "batch_size x 1 x H x W" scatter to onehot
        if (y.size(1) == 1):
            # transform the targets to one-hot encodings
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            # do not make any changes, otherwise
            y_onehot = y.float()

        # if there is a class to ignore
        if (self.ignoreIndex != -1):
            # extract its targets
            ignores = y_onehot[:,self.ignoreIndex].unsqueeze(1)
            # discard its predictions
            x_onehot = x_onehot[:, :self.ignoreIndex]
            # discard its targets
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            # do not ignore any class, otherwise
            ignores=0

        #print(type(x_onehot))
        #print(type(y_onehot))
        #print(x_onehot.size())
        #print(y_onehot.size())

        # compute the true positives per class
        tpmult = x_onehot * y_onehot  # times prediction and gt coincide is 1
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()  # tp has shape (nClasses,)
        # compute the false positives per class
        fpmult = x_onehot * (1-y_onehot-ignores)  # times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()  # fp has shape (nClasses,)
        # compute the false negatives per class
        fnmult = (1-x_onehot) * (y_onehot)  # times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()  # fn has shape (nClasses,)

        # update the current number of true positives
        self.tp += tp.double().cpu()
        # update the current number of false positives
        self.fp += fp.double().cpu()
        # update the current number of false negatives
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        # compute the intersection over union per class
        iou = num / den
        return torch.mean(iou), iou  # returns "iou mean", "iou per class"

# Class for colors
class colors:
    # define a few colours just for a nice print
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# Colored value output if colorized flag is activated.
def getColorEntry(val):
    # if the value is not a float, perform text reset
    if not isinstance(val, float):
        return colors.ENDC
    # set a color, otherwise
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN

