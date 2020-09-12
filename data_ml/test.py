# script 
# downloads test sets from the podcast data archive
# model
# runs inference on the test set

import argparse
import os

import numpy as np

from torch.utils.data import DataLoader
import src.dataloader
import src.model

from sklearn import metrics

import matplotlib.pyplot as plt

def infer_and_evaluate(test_set, model_path, use_cuda=False):
    """

    """

    # paths
    wav_folder = os.path.join(test_set, "wav")
    mean_file = os.path.join(model_path, "mean64.txt")
    invstd_file = os.path.join(model_path, "invstd64.txt")
    tsv_file = os.path.join(test_set, "test.tsv")
    test_dataset = src.dataloader.AudioFileDataset(wav_folder, tsv_file, mean=mean_file,invstd=invstd_file)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # load and instantiate net
    net, curr_epoch = src.model.get_model_or_checkpoint("AudioSet_fc_all", model_path, 2, use_cuda=use_cuda)
    net = net.eval()

    # infer
    scores = []
    preds = []
    targets = []
    for i, (data,target) in enumerate(test_dataloader):
        if use_cuda:
            data, target = data.unsqueeze(1).float().cuda(), target.cuda()
        else:
            data, target = data.unsqueeze(1).float(), target
        logposterior, _ = net(data)
        scores.append(np.exp(logposterior.detach().cpu().numpy())[0,-1])
        targets.append(target.cpu().item())
        
    # evaluate
    threshold = 0.7
    
    preds = [ 1 if s > threshold else 0 for s in scores ]
    fpr, tpr, thresholds = metrics.roc_curve(targets,scores)
    auc = metrics.auc(fpr,tpr)
    print("AUC: {:.3f}".format(auc))
    print(metrics.classification_report(targets,preds))
    print(metrics.confusion_matrix(targets,preds))

    plt.plot(fpr,tpr)
    plt.plot(fpr,thresholds)
    plt.ylim((-0.1,1.1))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("AUC: {:.3f}".format(auc))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default=None, type=str, required=True)
    parser.add_argument('--model_path', default=None, type=str, required=True)

    args = parser.parse_args()
    infer_and_evaluate(args.test_path, args.model_path)

