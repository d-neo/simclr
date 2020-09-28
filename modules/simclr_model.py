

import torch.nn as nn
import torch.nn.functional as F
import torch 

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class modelSIMCLR(nn.Module):
    def __init__(self, encoder, n_features, dim=128, T=0.1):
        # Inherit from nn module (standard)
        super(modelSIMCLR, self).__init__()
        # Instance Variables
        self.trainloss = []
        self.T = T
        self.n_features = n_features
        # create the encoder
        self.encoder_f = encoder
        # Replace the fc layer with an Identity function
        self.encoder_f.fc = Identity()
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) 
        # where σ is a ReLU non-linearity.
        self.mlp = nn.Sequential (
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, dim, bias=False),
        )

    def contrastive_loss(self, out_1, out_2, args):
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.T)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.T)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        return loss

    def forward(self, im1, im2, args):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # Get Representations
        h_i = self.encoder_f(im1)
        h_j = self.encoder_f(im2)
        # Get Feature Embeddings in lower Space through MLP
        z_i = self.mlp(h_i)
        z_j = self.mlp(h_j)
        # Normalize Feature Embeddings
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1) 
        
        return self.contrastive_loss(z_i, z_j, args)