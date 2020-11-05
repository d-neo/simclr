import torch
from statistics import mean
import numpy as np 
import torch.nn as nn

# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu).cuda()
        v = torch.zeros_like(nu).cuda()
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8).cuda() - torch.logsumexp(self.M(C, u, v).cuda(), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8).cuda() - torch.logsumexp(self.M(C, u, v).transpose(-2, -1).cuda(), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

def contrastive_loss_cosine(out_1, out_2, T):
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / T)
    mask = (torch.ones_like(sim_matrix) - torch.eye(n_samples, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(n_samples, -1)
    # compute loss
    pos_sim = torch.exp(torch.sum(out_1*out_2, dim=-1) / T)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss

def contrastive_loss_cosine_extra(out_1, out_2, T):
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)
    # [2*B, 2*B]
    sim = torch.mm(out, out.t().contiguous())
    sim_matrix = torch.exp(sim / T)
    mask = (torch.ones_like(sim_matrix) - torch.eye(n_samples, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(n_samples, -1)
    # compute loss
    psim = torch.sum(out_1*out_2, dim=-1)
    pos_sim = torch.exp(psim / T)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return (loss, sim.mean(dim=-1).mean(), psim.mean())


def contrastive_loss_gaussian_euclidean(t1, t2, T):
    sigma = 1.0
    out = torch.cat([t1, t2], dim=0)
    n_samples = len(out)

    sim_matrix = torch.exp( (torch.exp( -torch.cdist(out, out, p=2) / (2 * sigma * sigma)))  / T )
    mask = (torch.ones_like(sim_matrix) - torch.eye(n_samples, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(n_samples, -1)

    pos_sim = torch.exp( (torch.exp( -torch.cdist(t1,t2, p=2) / (2*sigma*sigma)))  / T )
    mask = (torch.ones_like(pos_sim) - torch.eye(256, device=pos_sim.device)).bool()
    pos_sim = pos_sim.masked_select(~mask).view(256, -1)
    pos_sim = torch.reshape(pos_sim, (-1,))
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss

def contrastive_loss_sne_euclidean(t1, t2, T):
    #sigma = 1.0
    out = torch.cat([t1, t2], dim=0)
    n_samples = len(out)

    sim_matrix = torch.exp( (torch.exp( -torch.cdist(out, out, p=2)**2 ))  / T )
    mask = (torch.ones_like(sim_matrix) - torch.eye(n_samples, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(n_samples, -1)

    pos_sim = torch.exp( (torch.exp( -torch.cdist(t1,t2, p=2)**2))  / T )
    mask = (torch.ones_like(pos_sim) - torch.eye(256, device=pos_sim.device)).bool()
    pos_sim = pos_sim.masked_select(~mask).view(256, -1)
    pos_sim = torch.reshape(pos_sim, (-1,))
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss

def contrastive_loss_tsne_euclidean(t1, t2, T):
    #sigma = 1.0
    out = torch.cat([t1, t2], dim=0)
    n_samples = len(out)

    sim_matrix = torch.exp( (1/(1+torch.cdist(out, out, p=2)**2))  / T )
    mask = (torch.ones_like(sim_matrix) - torch.eye(n_samples, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(n_samples, -1)

    pos_sim = torch.exp( (1/(1+torch.cdist(t1,t2, p=2)**2))  / T )
    mask = (torch.ones_like(pos_sim) - torch.eye(256, device=pos_sim.device)).bool()
    pos_sim = pos_sim.masked_select(~mask).view(256, -1)
    pos_sim = torch.reshape(pos_sim, (-1,))
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss

def contrastive_loss_wasserstein_tsne(t1,t2,T):
    out = torch.cat([t1, t2], dim=0)
    n_samples = len(out)
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100).cuda()
    dist, P, sim_matrix = sinkhorn(out, out)
    sim_matrix = torch.exp( (1/(1+sim_matrix**2)) / T)
    mask = (torch.ones_like(sim_matrix) - torch.eye(n_samples, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(n_samples, -1)
    #print(sim_matrix)
    # compute loss
    dist, P, pos_sim = sinkhorn(t1,t2)
    pos_sim = torch.exp( (1/(1+pos_sim**2)) / T)
    mask = (torch.ones_like(pos_sim) - torch.eye(256, device=pos_sim.device)).bool()
    pos_sim = pos_sim.masked_select(~mask).view(256, -1)
    pos_sim = torch.reshape(pos_sim, (-1,))
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss