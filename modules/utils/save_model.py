import os
import torch 

def save_model(model, optimizer, path, epoch):
    """
    Input: 
        model: The model you trained
        optimizer: The optimizer used to train model
        path:  type String - The path the model has to be saved to
        epoch: type int - The epoch when the model is saved

        Saving your trained model.
    """
    out = os.path.join(path+"/", "checkpoint_{}.tar".format(epoch))
    if isinstance(model, torch.nn.DataParallel):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, out)
    else:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, out)