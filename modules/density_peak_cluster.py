
from calendar import c
import numpy as np
import torch
import torch.nn.functional as F
import heapq





def get_elu_dis(feats):
    distances = (feats ** 2).sum(dim=1, keepdim=True) - 2 * feats @ feats.t() + (feats.t() ** 2).sum(dim=0, keepdim=True)
    return distances

def cal_k_arr(dists, K):
    sorted, indices = torch.sort(dists, descending=False)
    return sorted[:, 1:K+1]

def localDensity(k_arr):
    k_arr = torch.mean(k_arr, dim=-1)
    print("k_arr: ", k_arr)
    rho = torch.exp(-k_arr)
    print("rho: ", rho)
    exit()
    return rho

def relativeDistance(dist, rho):
    max_rho, max_idx = torch.max(rho, dim=0)
    rel_rho = rho.unsqueeze(1) < rho.unsqueeze(0)
    beta = dist.masked_fill(~rel_rho, float("inf"))
    beta, _ = torch.min(beta, dim=-1)
    beta[max_idx] = torch.max(dist[max_idx])
    return beta


def knn_dpc(feats, K):
    dists = get_elu_dis(feats)
    print("dists: ", dists)
    k_arr = cal_k_arr(dists, K)
    print("k_arr: ", k_arr)
    rho = localDensity(k_arr)
    print("rho: ", rho)
    print("sorted rho idx: ", rho.sort(descending=True)[1])
    beta = relativeDistance(dists, rho)
    print("beta: ", beta)
    criterion = rho * beta
    sorted, indices = torch.sort(criterion, descending=True)
    return indices[:K].sort()[0]


if __name__ == "__main__":
    feats = torch.randn(100, 256).cuda()
    criterion = knn_dpc(feats, 8)
    #print(criterion)




