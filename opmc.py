from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler
from time import time
from opmc import OPMC
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import _supervised
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import pickle


class OPMC:
    def __init__(self, max_iter=1000, tolerance=1e-5, device='cpu', update_beta=False):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.device = device
        self.update_beta = update_beta

    def fit(self, X, k):
        V = len(X)
        n = X[0].shape[0]
        c = k  # Set c equal to k as in MATLAB code
        # Initialize variables
        Y = torch.randint(1, k + 1, (n,), dtype=torch.int64, device=self.device)  # Random cluster assignments
        C = [torch.rand(k, c, device=self.device) for _ in range(V)]  # Randomly initialized C
        W = [torch.rand(c, X[v].shape[1], device=self.device) for v in range(V)]  # Randomly initialized W
        beta = torch.ones(V, device=self.device) / V  # Equal weights initially

        obj = []
        t = 0
        with torch.no_grad():
            while True:
                # Calculate objective
                obj.append(self._cal_obj(X, Y, C, W, beta, self.device))

                # Update W
                W = self._update_W(X, Y, C, self.device)

                # Update C
                C = self._update_C(X, Y, W, k, self.device)

                # Update Y
                Y = self._update_Y(X, C, W, beta, self.device)

                # Optionally update beta (if required in your application)
                if self.update_beta:
                    beta = self._update_beta_vals(X, Y, C, W)

                # Check convergence
                if t > 1 and abs(obj[-1] - obj[-2]) / obj[-1] < self.tolerance:
                    break
                t += 1
                if t >= self.max_iter:  # Exit if maximum iterations are reached
                    break

        return Y, C, W, beta, obj

    def _update_C(self, X, Y, W, k):
        V = len(X)
        n = X[0].shape[0]

        C = []
        for v in range(V):
            # Compute YTYvec (count of samples in each cluster)
            YTYvec = torch.zeros(k, device=self.device)
            for i in range(n):
                YTYvec[Y[i] - 1] += 1  # Subtract 1 since Python uses 0-based indexing

            # Compute YTYplusIinvvec (inverse with small epsilon for numerical stability)
            YTYplusIinvvec = 1.0 / (YTYvec + torch.finfo(torch.float32).eps)

            # Compute YTXv
            YTXv = torch.zeros((k, X[v].shape[1]), device=self.device)
            for j in range(1, k + 1):  # Loop over clusters (1-based in MATLAB)
                YTXv[j - 1, :] = X[v][Y == j].sum(dim=0)

            # Compute C[v]
            tmp = YTXv * YTYplusIinvvec.unsqueeze(1)  # Element-wise multiplication
            C_v = tmp @ W[v].T  # Matrix multiplication
            C.append(C_v)

        return C

    def _update_W(self, X, Y, C):
        V = len(X)
        k, c = C[0].shape  # Assuming all C[v] have the same dimensions
        W = []
        for v in range(V):
            # Compute YTXv
            YTXv = torch.zeros((k, X[v].shape[1]), device=self.device)
            for j in range(1, k + 1):  # Loop over clusters (1-based indexing in MATLAB)
                YTXv[j - 1, :] = X[v][Y == j].sum(dim=0) + torch.finfo(torch.float32).eps
            # Singular Value Decomposition (SVD)
            U, _, Vh = torch.linalg.svd(C[v].T @ YTXv, full_matrices=False)
            W_v = U @ Vh  # Compute W[v]
            W.append(W_v)
        return W

    def _update_Y(self, X, C, W, beta):
        V = len(X)
        n = X[0].shape[0]
        k = C[0].shape[0]

        # Precompute CW for each view
        CW = [C[v] @ W[v] for v in range(V)]

        # Compute loss for each cluster
        loss = torch.zeros((n, k), device=self.device)
        for v in range(V):
            loss += beta[v] * torch.cdist(X[v].unsqueeze(0), CW[v].unsqueeze(0).to(self.device)).squeeze(0)

        # Find the cluster assignment minimizing the loss
        _, Y = torch.min(loss, dim=1)
        return Y + 1  # Convert to 1-based indexing to match MATLAB

    def _update_beta_vals(self, X, Y, C, W):
        V = len(X)
        loss = torch.zeros(V, device=self.device)

        # Compute loss for each view
        for v in range(V):
            CWY = C[v][Y - 1, :] @ W[v]  # Use Y - 1 for 0-based indexing
            loss[v] = torch.sum((X[v] - CWY) ** 2)

        # Update beta
        tmp = 1.0 / loss
        beta = (tmp / torch.sum(tmp)) ** 2

        return beta

    def _cal_obj(self, X, Y, C, W, beta):
        V = len(X)
        loss = torch.zeros(V, device=self.device)

        # Compute loss for each view
        for v in range(V):
            CWY = C[v][Y - 1, :] @ W[v]  # Use Y - 1 for 0-based indexing
            loss[v] = torch.sum((X[v] - CWY) ** 2)

        # Compute total loss
        loss_sum = torch.dot(beta, loss)

        return loss_sum


class MultiviewDataset:
    def __init__(self, dataset_path):
        super().__init__()

        # Loading the dictionary back from the pickle file
        with open(dataset_path, "rb") as file:
            dataset_dict = pickle.load(file)

        self.views = dataset_dict["X"]
        self.labels = dataset_dict["Y"]
        self.dataset_name = dataset_dict["dataset_name"]

        if dataset_dict["sub_sample"][0] == True:
            self.num_of_sub_samples = dataset_dict["sub_sample"][1]
            self.sub_sample()
        print("Number of views:", len(self.views))
        print("Dimensions:", [v.shape[1] for v in self.views.values()])
        print("Unique labels:", np.unique(list(self.labels.values())[0]))

    def export(self):
        X = [v for n, v in self.views.items()]
        Y = list(self.labels.items())[0][1].flatten()
        return X, Y


class Trainer:
    def __init__(self, cfg):
        self.opmc = OPMC(device=cfg.device)
        self.device = cfg.device
        self.iterations = cfg.iterations
        self.samples, self.labels = MultiviewDataset(cfg.dataset).export()

    @staticmethod
    def clustering_accuracy(labels_true, labels_pred):
        labels_true, labels_pred = _supervised.check_clusterings(labels_true, labels_pred)
        value = _supervised.contingency_matrix(labels_true, labels_pred)
        [r, c] = linear_sum_assignment(-value)
        return value[r, c].sum() / len(labels_true)

    def get_evaluation_results(self, y_pred):
        ACC = self.clustering_accuracy(self.labels, y_pred)
        NMI = normalized_mutual_info_score(self.labels, y_pred)
        ARI = adjusted_rand_score(self.labels, y_pred)
        res = np.array([ACC, NMI, ARI])
        return res

    def train_eval(self):
        k = len(np.unique(self.labels))
        V = len(self.samples)

        # Normalize data
        X_normalized = []
        for v in range(V):
            scaler = StandardScaler()
            X_normalized.append(torch.tensor(scaler.fit_transform(self.samples[v].T).T, device=self.device).float())

        # Iterate and save results
        results = []
        for _ in tqdm(range(self.iterations)):
            start_time = time()
            Y, C, W, beta, obj = self.opmc.fit(X_normalized, k)  # Implement this function in Python
            elapsed_time = time() - start_time
            val = self.get_evaluation_results(Y.cpu().numpy())  # Implement this function in Python
            loss = obj[-1]
            # Save results
            res = {
                # 'data_name': data_name,
                # 'Y': Y,
                # 'C': C,
                # 'W': W,
                # 'beta': beta,
                'val': val,
                'obj': obj,
                'ts': elapsed_time,
                'loss': loss
            }
            results.append(res)

        # Get results corresponding to the minimal loss
        vals = [res['val'] for res in results]
        losses = [res['loss'] for res in results]
        min_loss_idx = np.argmin(losses)
        return vals[min_loss_idx]