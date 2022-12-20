import numpy as np

import torch
from torch_scatter import scatter_sum
from tqdm.notebook import tqdm


class ALS:
    def __init__(self, factors, iterations=100, regularization=0.1, device='cpu', callback=None):
        
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.device = device
        self.callback = callback
        
    def fit(self, R):
        
        m, n = R.shape
        
        P = torch.randn(m, self.factors, device=self.device) / np.sqrt(self.factors)
        Q = torch.randn(n, self.factors, device=self.device) / np.sqrt(self.factors)
        I = torch.eye(self.factors, device=self.device)
        
        rows, cols = R.indices()
        
        rows = rows.to(self.device)
        cols = cols.to(self.device)
        
        logs = []
        for _ in tqdm(range(self.iterations)):
                
            A = torch.linalg.inv(Q.T @ Q + self.regularization * I) @ Q.T
            P = scatter_sum(A[:, cols], rows).T

            A = torch.linalg.inv(P.T @ P + self.regularization * I) @ P.T
            Q = scatter_sum(A[:, rows], cols).T
            
            if self.callback is not None:
                log = self.callback(P, Q)
                logs.append(log)
                
        self.P = P
        self.Q = Q
        
        return np.array(logs)
    
    def predict(self, id_user, k=None):
        
        if k is None:
            
            k = len(self.Q)
            
        p = self.P[id_user]
        
        scores = self.Q @ p
        
        top_k = torch.argsort(scores, descending=True)[:k]
        
        return top_k


class eALS:
    def __init__(self, factors, iterations, w=1, c=1, regularization=0.1, device='cpu', callback=None):
        self.w = w
        self.c = c
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.device = device
        self.callback = callback
        
    def fit(self, X):
        K = self.factors
        M, N = X.shape
        P = torch.randn(M, K, device=self.device) / np.sqrt(K)
        Q = torch.randn(N, K, device=self.device) / np.sqrt(K)
        
        rows, cols = X.indices()
        
        rows = rows.to(self.device)
        cols = cols.to(self.device)
        
        c = self.c
        if isinstance(c, int):
            c = torch.ones(N, device=self.device) * c
        else:
            c = torch.tensor(c, device=self.device)
        
        R_hat = (P[rows] * Q[cols]).sum(axis=1)
        
        logs = []
        
        for _ in tqdm(range(self.iterations)):
            S_q = (c.unsqueeze(1) * Q).T @ Q

            for f in range(K):
                r_hat = R_hat - P[rows, f] * Q[cols, f]

                nominator = scatter_sum((self.w - (self.w - c[cols]) * r_hat) * Q[cols, f], rows)
                nominator -= P @ S_q[:, f] - P[:, f] * S_q[f, f]

                denominator = scatter_sum((self.w - c[cols]) * Q[cols, f] ** 2, rows) + S_q[f, f]
                P[:, f] = nominator / (denominator + self.regularization)

                R_hat = r_hat + P[rows, f] * Q[cols, f]

            S_p = P.T @ P
            for f in range(K):
                r_hat = R_hat - P[rows, f] * Q[cols, f] 

                nominator = scatter_sum((self.w - (self.w - c[cols]) * r_hat) * P[rows, f], cols)
                nominator -= c * (Q @ S_p[:, f] - Q[:, f] * S_p[f, f])

                denominator = scatter_sum((self.w - c[cols]) * P[rows, f] ** 2, cols) + c * S_p[f, f] 
                Q[:, f] = nominator / (denominator + self.regularization)

                R_hat = r_hat + P[rows, f] * Q[cols, f]
                
            if self.callback is not None:
                log = self.callback(P, Q)
                logs.append(log)
            
        return np.array(logs)
    
    def predict(self, id_user, k=None):
        
        if k is None:
            
            k = len(self.Q)
            
        p = self.P[id_user]
        
        scores = self.Q @ p
        
        top_k = torch.argsort(scores, descending=True)[:k]
        
        return top_k

