from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


##############################
###### SET-UP ###############
###############################
def setReportingUp_trainingSimclr(args):
    """

    """
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y_%H_%M") 

    if args.dataset == "CIFAR10":
        data_dir = "cifar10"
    elif args.dataset == "IMAGENET":
        data_dir = "imagenet"
    else:
        data_dir = "satellite"

    dirpath = str(dt_string)
    if args.reload:
        if args.model_dir == "": raise NotImplementedError
        mkdirpath = "saved_models/selfsupervised/"+args.model_dir
        writer_train = SummaryWriter(log_dir = "runs/selfsupervised/"+args.model_dir+"/"+"training_positivesim")
        writer_val = SummaryWriter(log_dir = "runs/selfsupervised/"+args.model_dir+"/"+"validation_negativesim")
    else:
        mkdirpath = "runs/selfsupervised/"+data_dir+"/"+dirpath
        os.mkdir(mkdirpath)
        writer_train = SummaryWriter(log_dir=mkdirpath+"/"+"training_positivesim")
        writer_val = SummaryWriter(log_dir=mkdirpath+"/"+"validation_negativesim")
        mkdirpath = "saved_models/selfsupervised/"+data_dir+"/"+dirpath
        os.mkdir(mkdirpath) 

    return writer_train, writer_val, mkdirpath

def setReportingUp_trainingSupervised(args):
    """

    """
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y_%H_%M") 

    if args.dataset == "CIFAR10":
        data_dir = "cifar10"
    elif args.dataset == "IMAGENET":
        data_dir = "imagenet"
    else:
        data_dir = "satellite"

    dirpath = str(dt_string)
    if args.reload:
        if args.model_dir == "": raise NotImplementedError
        mkdirpath = "saved_models/supervised/"+args.model_dir
        writer_train = SummaryWriter(log_dir = "runs/supervised/"+args.model_dir+"/"+"training_positivesim")
        writer_val = SummaryWriter(log_dir = "runs/supervised/"+args.model_dir+"/"+"validation_negativesim")
    else:
        mkdirpath = "runs/supervised/"+data_dir+"/"+dirpath
        os.mkdir(mkdirpath)
        writer_train = SummaryWriter(log_dir=mkdirpath+"/"+"training_positivesim")
        writer_val = SummaryWriter(log_dir=mkdirpath+"/"+"validation_negativesim")
        mkdirpath = "saved_models/supervised/"+data_dir+"/"+dirpath
        os.mkdir(mkdirpath) 

    return writer_train, writer_val, mkdirpath


##############################
###### Update ###############
###############################
def updateWriter_trainingSimclr(writer_train, writer_val, train_loss_eval, val_loss_eval, train_acc, val_acc, pos_sim, neg_sim, train_loss, lr, epoch):
    """

    """
    writer_train.add_scalar("Loss_Eval/train_val", train_loss_eval, epoch)
    writer_val.add_scalar("Loss_Eval/train_val", val_loss_eval, epoch)
    writer_train.add_scalar("Accuracy/train_val", train_acc, epoch)
    writer_val.add_scalar("Accuracy/train_val", val_acc, epoch)
    writer_train.add_scalar("Similarity/train", pos_sim, epoch)
    writer_val.add_scalar("Similarity/train", neg_sim, epoch)
    writer_train.add_scalar("LR_Contrastive/train", lr, epoch)
    writer_train.add_scalar("Loss_Contrastive/train", train_loss, epoch)
    writer_train.flush()
    writer_val.flush()

    return writer_train, writer_val

def updateWriter_trainingSupervised(writer_train, writer_val, train_loss, val_loss, train_acc, val_acc, epoch):
    """

    """
    writer_train.add_scalar("Loss/train_val", train_loss, epoch)
    writer_train.add_scalar("Accuracy/train_val", train_acc, epoch)
    writer_val.add_scalar("Loss/train_val", val_loss, epoch)
    writer_val.add_scalar("Accuracy/train_val", val_acc, epoch)
    writer_train.flush()
    writer_val.flush()

    return writer_train, writer_val


##############################
###### Extended ###############
###############################
def updateWriter_trainingSimclr_extended(writer_train, train_loss, train_acc, val_acc, train_acc_long, val_acc_long, args):
    """

    """
    writer_train.add_hparams({"Arch/Data/Opt/Cos": args.arch+"_"+args.dataset+"_"+args.optimizer+"_"+str(args.cos),
                        "Epochs": args.epochs, 
                        "LR/WD/Dim/T": str(args.lr)+"_"+str(args.wd)+"_"+str(args.dim)+"_"+str(args.t), 
                        'Bsize': args.batch_size, 
                        "#Views": args.numberviews}, 
                        {'Cont.Loss': train_loss, 'Train_Acc': train_acc, 'Val_Acc': val_acc, 'Train_Acc_Ext': train_acc_long, 'Val_Acc_Ext': val_acc_long})
    writer_train.flush()

    return writer_train

def updateWriter_trainingSupervised_extended(writer_train, train_loss, train_acc, val_acc, args):
    """

    """
    writer_train.add_hparams({"architecture": args.arch, "dataset": args.dataset ,"epochs": args.epochs,"optimizer": args.optimizer, "Cosine-Schedule": args.cos,
                             "weigt-decay": args.wd, 'lr': args.lr, 'bsize': args.batch_size}, 
                             {'hparam/train_accuracy': train_acc, 'hparam/val_accuracy': val_acc,'hparam/loss': train_loss})
    writer_train.flush()

    return writer_train