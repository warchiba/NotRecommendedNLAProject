import torch


def hit_ratio(users, items, target, k):
    scores = users @ items.T
    
    positions = torch.topk(scores, k, dim=1)[1]
    
    return torch.sum((positions == target.unsqueeze(1)).float()) / len(users)


def nDCG(users, items, target, k):
    device = users.device
    
    scores = users @ items.T
    
    positions = torch.topk(scores, k, dim=1)[1]
    
    return torch.sum((positions == target.unsqueeze(1)).float() / torch.log2(1 + torch.arange(1, k + 1).to(device))) / len(users)
    
    
def MRR(users, items, target, k):
    
    scores = users @ items.T
    
    positions = torch.topk(scores, k, dim=1)[1]
    
    ranks = torch.where(positions == target.unsqueeze(1))
    
    return torch.sum(1 / (1 + ranks[1])) / len(users)

