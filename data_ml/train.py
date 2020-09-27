# imports 
import argparse, glob, os, time, contextlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import src.params as params
import pandas as pd 
import numpy as np

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataloader import AudioFileDataset
from src.model import ResNet_slim, AverageMeter, PredScorer, set_logger, get_model_or_checkpoint, get_finetune_model
from src.augment import SpecAug

# deal with a known bug in sklearn that pollutes stdout: https://stackoverflow.com/questions/52596204/the-imp-module-is-deprecated
with contextlib.redirect_stderr(None):
    from sklearn.model_selection import train_test_split

# train() 
#TODO: Refactor a bit and remove any custom/internal references
def train(iteration, train_dataloader, model, optimizer, records, print_freq, epoch, batchsize, logger, summary_writer, scheduler=None):
    prev = time.time()
    predscorer, epoch_loss = records

    print("\nTraining epoch:", epoch)
    for i, (data, target) in enumerate(tqdm(train_dataloader)):

        iteration += 1

        # measure data loading time
        data_time = time.time() - prev
        prev = time.time()

        # forward - data:(b x 1 x N x d), target:(b), pred:(b x C)
        data, target = data.unsqueeze(1).float().cuda(), target.cuda()  
        pred, embed = model(data)

        # compute classification error
        pred_id = torch.argmax(pred, dim=1)
        accuracy = (torch.sum(pred_id == target).item())/batchsize*100
        # predscorer.update(target,pred_id)

        # compute gradient and do SGD step
        loss = F.nll_loss(pred, target)
        # losses.update(loss.data.item())
        epoch_loss.update(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure computation time
        batch_time = time.time() - prev
        prev = time.time()
    
        if scheduler is not None:
            scheduler.step()

        #TODO: Add tensorboard
        # print progress
        if iteration % print_freq == 0:
            # log to tensorboard 
            summary_writer.add_scalar("perf/data_time", data_time, global_step=iteration)
            summary_writer.add_scalar("perf/batch_time", batch_time, global_step=iteration)
            summary_writer.add_scalar("train/loss", loss.data.item(), global_step=iteration)
            summary_writer.add_scalar("train/accuracy", accuracy, global_step=iteration)
            summary_writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], global_step=iteration)
        #     print('Epoch:', epoch, '\t', 'Iter:', iteration, '\t',
        #           'Data:', '%.2f' % data_time.sum, '\t',
        #           'Batch:', '%.2f' % batch_time.sum, '\t',
        #           'Lr:', '%.4f' % optimizer.param_groups[0]['lr'], '\t',
        #           'Loss:', '%.4f' % losses.avg, '\t',
        #           'Accuracy:', '%.2f' % (correct.sum*100.0/(print_freq*batchsize)), '\t',
        #           'F1 Global:', '%.2f' % (predscorer.F1_global)
        #           )
        #     predscorer.log_classification_report(logger,iteration,epoch)
        #     data_time.reset()
        #     batch_time.reset()
        #     losses.reset()
        #     correct.reset()
        #     predscorer.reset()
    
    # print some per-epoch info 

    return iteration


def validate(val_dataloader, model, iteration, epoch, summary_writer, records, logger):

    predscorer, epoch_loss = records
    model.eval()

    print("Validating epoch:", epoch)
    for i, (data, target) in enumerate(tqdm(val_dataloader)):
        # forward - data:(b x 1 x N x d), target:(b), pred:(b x C)
        data, target = data.unsqueeze(1).float().cuda(), target.cuda()  
        pred, embed = model(data)

        # compute classification error
        pred = pred.detach()
        pred_id = torch.argmax(pred, dim=1).detach()
        # accuracy = (torch.sum(pred_id == target).item())/batchsize*100
        with torch.no_grad():
            loss = F.nll_loss(pred, target).data.item()

        predscorer.update(target, pred_id, pred[:,1])
        epoch_loss.update(loss)
    
    predscorer.log_classification_report(logger, iteration, epoch)
    summary_writer.add_scalar('validation/loss', epoch_loss.avg, global_step=iteration)
    summary_writer.add_scalar('validation/accuracy', predscorer.accuracy, global_step=iteration)
    summary_writer.add_pr_curve(
        'validation/pr_curve', 
        np.asarray(predscorer.targets_list), np.asarray(predscorer.scores_list),
        global_step=iteration
        )


def split_annotations_train_val(annotations_tsv, split=0.1):
    data_path = Path(annotations_tsv).parent
    train_tsv, val_tsv = data_path/"train.tsv", data_path/"val.tsv"
    if train_tsv.is_file() and val_tsv.is_file():
        print("Re-using train/validation data split")
    else:
        df = pd.read_csv(annotations_tsv, sep='\t')
        print("\nDataset stats -") 
        print(df.reset_index().groupby('dataset').nunique())
        # split based on identifier column to prevent overlap of annotations from same/similar wavfiles 
        unique_identifiers = df.pst_or_master_tape_identifier.drop_duplicates()
        train, val = train_test_split(unique_identifiers, test_size=split)
        train_df, val_df = df.merge(train), df.merge(val)
        print("\nSplit stats - ")
        print(train_df.reset_index().groupby('dataset').nunique())
        print(val_df.reset_index().groupby('dataset').nunique())
        train_df.to_csv(train_tsv, sep='\t', index=False)
        val_df.to_csv(val_tsv, sep='\t', index=False)
    return str(train_tsv), str(val_tsv)

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-runRootPath', default=None, type=str, required=True)
    parser.add_argument('-dataPath', default=None, type=str, required=True)
    parser.add_argument('-model', default='AudioSet_fc_all', type=str, required=True)
    # select model, lr, lr plateau params
    parser.add_argument('-lr', default=0.0005, type=float, required=False)
    parser.add_argument('-lrPlateauSchedule', default="2,0.05,0.5", type=str, required=False)
    parser.add_argument('-batchSize', default=32, type=int, required=False)
    parser.add_argument('-minWindowS', default=params.WINDOW_S, type=float, required=False)
    parser.add_argument('-maxWindowS', default=params.WINDOW_S, type=float, required=False)
    parser.add_argument('--preTrainedModelPath', default=None, type=str, required=False)

    parser.add_argument('-printFreq', default=10, type=int, required=False)
    parser.add_argument('-numEpochs', default=30, type=int, required=False)
    parser.add_argument('-dataloadWorkers', default=1, type=int, required=False)
    args = parser.parse_args()

    # Create / check all directories
    num_classes, model_name = 2, args.model 
    runPath = Path(args.runRootPath) / ("{}_lr{}_run1".format(model_name,args.lr))
    if runPath.exists:
        runid = int(runPath.name.split("run")[-1]) + 1
        runPath = Path(args.runRootPath) / ("{}_lr{}_run{}".format(model_name,args.lr,runid))
    os.makedirs(runPath, exist_ok=True)

    ## initialize dataloaders

    data_path = Path(args.dataPath)
    wav_dir_path, tsv_path = data_path/"wav", data_path/"annotations.tsv"
    #  split into train/validation 
    train_tsv, val_tsv = split_annotations_train_val(tsv_path, 0.1)
    mean, invstd = data_path/params.MEAN_FILE, data_path/params.INVSTD_FILE

    specaug = SpecAug(2,12,2,6)
    print("Doing augmentation with specaug..")
    train_dataset = AudioFileDataset(
        wav_dir_path, train_tsv, mean=mean, invstd=invstd, transform=specaug
        )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchSize, 
                shuffle=True, drop_last=True, num_workers=args.dataloadWorkers,
                pin_memory=True)
    val_dataset = AudioFileDataset(
        wav_dir_path, val_tsv, mean=mean, invstd=invstd 
        )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    ## initialize model 
    num_classes, model_name = 2, args.model 
    if "ResNet" in model_name:
        model, curr_epoch = get_model_or_checkpoint(model_name,runPath) 
    elif "AudioSet" in model_name:
        model, curr_epoch = get_finetune_model(model_name,runPath,args.preTrainedModelPath) 
    model.train()

    ## initialize optimizers 
    ## loop epochs, train and checkpoint
    # optimizer = optim.SGD(
    #     filter(lambda p: p.requires_grad,model.parameters()),
    #     lr=args.lr, momentum=0.9
    # )
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad,model.parameters()),
        lr=args.lr
    )

    scheduler = lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-8, max_lr=1e-7, step_size_up=len(train_dataloader),
        cycle_momentum=False, mode='exp_range', gamma=1.05
    )

    # lr_plateau_schedule = [ float(p) for p in args.lrPlateauSchedule.split(',') ]
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=lr_plateau_schedule[0], threshold=lr_plateau_schedule[1],factor=lr_plateau_schedule[2]
    # )

    ## initialize logger
    writer = SummaryWriter(log_dir=runPath)
    print_freq = args.printFreq
    records = [PredScorer(), AverageMeter()]
    predscorer, epoch_loss =  records[0], records[1]

    # training
    iteration, logger = curr_epoch*len(train_dataloader), set_logger(runPath)
    for epoch in range(curr_epoch,args.numEpochs):
        iteration = train(iteration, train_dataloader, model,
                            optimizer, records, print_freq, epoch, args.batchSize, logger, writer, scheduler=scheduler)
        message = "\n### Epoch {}, Avg training loss: {} ###\n".format(epoch,epoch_loss.avg)
        logger.info(message)
        epoch_loss.reset()

        validate(val_dataloader, model, iteration, epoch, writer, records, logger)
        # scheduler.step(epoch_loss.avg)
        epoch_loss.reset()
        predscorer.reset()

        if epoch % 2 == 0:
            torch.save(model.state_dict(), runPath / (model_name + '_Iter_' + str(epoch)) )

    torch.save(model.state_dict(), runPath / (model_name + '_Iter_' + str(args.numEpochs)) )

    writer.close()
