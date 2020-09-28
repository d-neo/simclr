import torch.nn as nn 

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class modelSupervised(nn.Module):
    def __init__(self, encoder, n_features):
        # Inherit from nn module (standard)
        super(modelSupervised, self).__init__()
        # Instance Variables
        self.trainloss = []
        self.n_features = n_features
        # create the encoder
        self.encoder_f = encoder
        # Replace the fc layer with an Identity function
        self.encoder_f.fc = Identity()
        # Add a Layer for classification
        self.classi = nn.Linear(self.n_features, 10, bias=False)

    def supervised_loss(self, out_1, out_2):
        pass
        
    def forward(self, im1):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
           Output
        """
        return self.classi(self.encoder_f(im1))