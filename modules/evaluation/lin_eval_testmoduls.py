import torch 
import numpy as np 
from tqdm import tqdm
import math
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)

def inference(loader, encoder_f):
    encoder_f.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        # get encoding
        with torch.no_grad():
            h = encoder_f(x.cuda(non_blocking=True))

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def get_features(encoder_f, train_loader, test_loader):
    train_X, train_y = inference(train_loader, encoder_f)
    test_X, test_y = inference(test_loader, encoder_f)
    return train_X, train_y, test_X, test_y

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
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

def train_lr(loader, model, criterion, optimizer, epoch, args):
    model.train()
    total_loss, total_num, accuracy_epoch, train_bar = 0.0, 0, 0, tqdm(loader)
    for x, y in train_bar:
        optimizer.zero_grad()
        
        x = x.to("cuda")
        y = y.to("cuda")

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        total_num += loader.batch_size
        total_loss += loss.item() * loader.batch_size
        
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs_lineval, optimizer.param_groups[0]['lr'], total_loss / total_num))


    return total_loss, accuracy_epoch

def test_lr(loader, model, criterion, optimizer):
    model.eval()
    total_loss, total_num, accuracy_epoch, test_bar = 0.0, 0, 0, tqdm(loader)
    for x, y in test_bar:
        optimizer.zero_grad()
        
        x = x.to("cuda")
        y = y.to("cuda")

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        total_num += loader.batch_size
        total_loss += loss.item() * loader.batch_size
        
        test_bar.set_description('lr: {:.6f}, Loss: {:.4f}'.format(optimizer.param_groups[0]['lr'], total_loss / total_num))


    return total_loss, accuracy_epoch