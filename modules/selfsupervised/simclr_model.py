import torch.nn as nn
import torch.nn.functional as F
import torch 

class modelSIMCLR(nn.Module):
    """ SIMCLR MODEL"""

    def __init__(self, encoder, n_features, loss_func, dim=128, T=0.1):
        """ 
        Input:
            encoder: Encoder you want to use to get feature representations (eg. resnet18)
            n_features: Type int - The dimension of the encoder output, your feature dimension
            loss_func: The loss function you want to use (from loss_functions.py)
            eval_model: If you have labels, the model you want to evaluate your feature representations on
            eval_optimizer: The optimizer for your eval_model
            dim: Type int - The dimension the projection head outputs
            T: Type int - The temperature parameter in the contrastive loss function

            Creates a SIMCLR model
        """
        # Inherit from nn module (standard)
        super(modelSIMCLR, self).__init__()
        self.T = T
        self.n_features = n_features
        self.loss_func = loss_func
        # create the encoder
        self.encoder_f = encoder
        self.encoder_f.fc = torch.nn.Identity()
        #self.encoder_f.fc = torch.nn.Linear(512, 512)
        # Create Projection Head
        self.mlp = nn.Sequential (
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.BatchNorm1d(num_features=self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, dim, bias=False),
            nn.BatchNorm1d(num_features=dim)
        )

    def forward(self, im1, im2, args):
        """
        Input:
            im1: a batch of query images
            im2: differently augmented query images

            Runs the encoder, runs the projection head, normalizes the output of the projection head 
            and returns the contrastive loss given the loss_func from loss_functions.py 
        """
        # Get Representations
        h_i = self.encoder_f(im1)
        h_j = self.encoder_f(im2)
        # Get Feature Embeddings in lower Space through MLP
        z_i = self.mlp(h_i)
        z_j = self.mlp(h_j)
        # Normalize Feature Embeddings
        z_i = nn.functional.normalize(z_i, p=2, dim=1)
        z_j = nn.functional.normalize(z_j, p=2, dim=1) 
        
        return self.loss_func(z_i, z_j, self.T)
