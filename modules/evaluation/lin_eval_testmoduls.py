import torch 
import numpy as np 
from tqdm import tqdm
import math
import torch.nn as nn
from statistics import mean 

######################################
# Helper - Functions ############
######################################
def inference(loader, encoder_f):
    """
    Input:
        loader: A pytorch dataloader containing the data
        encoder_f: The encoder to create features from

        Creating features from encoder
    """
    encoder_f.eval()
    feature_vector = []
    labels_vector = []
    print("Computing features...")
    for step, (x, y) in enumerate(loader):
        with torch.no_grad():
            h = encoder_f(x.cuda(non_blocking=True))

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        #if step % 20 == 0:
         #   print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def get_features(encoder_f, train_loader, test_loader):
    """
    encoder_f: encoder to create features from
    train_loader: pytorch dataloader containing data
    test_loader: can be validation or testloader

        Creates training and test/validation data using inference function
    """
    train_X, train_y = inference(train_loader, encoder_f)
    test_X, test_y = inference(test_loader, encoder_f)
    return train_X, train_y, test_X, test_y

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    """
    Creates dataloaders from created features
    """
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


######################################
# Model ############
######################################
class LogisticRegression(nn.Module):
    """LOGISTIC REGRESSION MODEL"""
    def __init__(self, n_features, n_classes):
        """
        Input:
            n_featues: Type int - Dimension of encoder in simclr model (eg. resnet18)
            n_classes: Type int - Number of classes
        """
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        """Run model"""
        return self.model(x)


######################################
# Training / Validation ############
######################################
def train_lr(loader, model, criterion, optimizer, epoch, maxepochs, args):
    """
    Input:
        loader: Pytorch dataloader containing the data
        model: The model you want to train
        criterion: Loss function
        optimizer: The optimizer used during training
        epoch: Type int - the epoch the model is currently trained
        maxepochs: The maximal epochs you want to train
        args: Argument list

        Training the evaluation model
    """
    model.train()
    epoch_loss = []
    epoch_correct, epoch_total = 0, 0
    train_bar =  tqdm(loader)

    for x, y in train_bar:
        optimizer.zero_grad()
        
        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        epoch_total += y.size(0)
        epoch_correct += (predicted == y).sum().item()

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}, Acc_Train: {:.4f}'.format(epoch, maxepochs, optimizer.param_groups[0]['lr'], 
                                                                                                        mean(epoch_loss), float(epoch_correct)/epoch_total))

    return mean(epoch_loss), float(epoch_correct)/epoch_total

def test_lr(loader, model, criterion, epoch, maxepochs, args):
    """
    Input:
        loader: Pytorch dataloader containing the data
        model: The model you want to train
        criterion: Loss function
        optimizer: The optimizer used during training
        epoch: Type int - the epoch the model is currently trained
        maxepochs: The maximal epochs you want to train
        args: Argument list

        Testing/Validating the evaluation model
    """
    model.eval()
    epoch_loss = []
    epoch_correct, epoch_total = 0, 0
    test_bar = tqdm(loader)
    with torch.no_grad():
        for x, y in test_bar:
            x = x.to(args.device)
            y = y.to(args.device)

            output = model(x)
            loss = criterion(output, y)
            
            predicted = output.argmax(1)
            epoch_total += y.size(0)
            epoch_correct += (predicted == y).sum().item()

            epoch_loss.append(loss.item())
            
            test_bar.set_description('Val Epoch: [{}/{}], Loss: {:.4f}, Acc_Val: {:.4f}'.format(epoch, maxepochs, mean(epoch_loss), float(epoch_correct)/epoch_total))

    return mean(epoch_loss), float(epoch_correct)/epoch_total

##########################################################
# Feature Eval - Combines Training/Validation ############
##########################################################
def featureEval(model, trainloader, valloader, epoch, args, extended=False ,maxepochs_nonextended=30):
    """
    Input:
        model: the simclr-model you trained
        trainloader: Pytorch dataloader (training data)
        valloader: Pytorch dataloader (validation data)
        epoch: type int - epoch you are currently in
        args: Args list
        extended: Set to true for final evaluation on validation data (higher number of epochs), if false (lower number of epochs)
        maxepochs_nonextended: type int - For extended, the epochs_lineval is used from the args-list, if not, it uses maxepochs_nonextended as the number of epochs to train
                                            logistic regression
    """
    if args.dataparallel: encoder = model.module.encoder_f  
    else: encoder = model.encoder_f

    # Logistig Regression 
    linear_model = LogisticRegression(args.featureDim, args.numberClasses).to(args.device)
    optimizer_eval = torch.optim.Adam(linear_model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion_eval = torch.nn.CrossEntropyLoss()

    # Extract Features from encoder
    (train_X, train_y, test_X, test_y) = get_features(encoder, trainloader, valloader)
    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(train_X, train_y, test_X, test_y, args.batch_size)

    # Check Feature Quality
    all_valaccs = []
    all_vallosses = []
    all_trainaccs = []
    all_trainlosses = []

    if extended: 
        for epoch in range(1, args.epochs_lineval+1):
            train_loss, train_acc = train_lr(arr_train_loader, linear_model, criterion_eval, optimizer_eval, epoch, args.epochs_lineval, args)
            val_loss, val_or_test_acc = test_lr(arr_test_loader, linear_model, criterion_eval, epoch, args.epochs_lineval, args)  
            all_valaccs.append(val_or_test_acc)
            all_vallosses.append(val_loss)
            all_trainaccs.append(train_acc)
            all_trainlosses.append(train_loss)


    else:
        for epoch in range(1,maxepochs_nonextended+1):
            train_loss, train_acc = train_lr(arr_train_loader, linear_model, criterion_eval, optimizer_eval, epoch, maxepochs_nonextended, args)
            val_loss, val_or_test_acc = test_lr(arr_test_loader, linear_model, criterion_eval, epoch, maxepochs_nonextended, args)
            all_valaccs.append(val_or_test_acc)
            all_vallosses.append(val_loss)
            all_trainaccs.append(train_acc)
            all_trainlosses.append(train_loss)

    max_value = max(all_valaccs)
    max_index = all_valaccs.index(max_value)

    reported_valacc = max_value
    reported_valloss = all_vallosses[max_index]

    max_value = max(all_trainaccs)
    max_index = all_trainaccs.index(max_value)

    reported_trainacc = max_value
    reported_trainloss = all_vallosses[max_index]


    return (reported_trainloss, reported_valloss, reported_trainacc, reported_valacc)