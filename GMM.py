import matplotlib.pyplot as plt

import torch

class GMM:
    def __init__(self, n_components=1, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None

    def fit(self, X):
        # Enhanced Initialization
        global_mean = torch.mean(X, axis=0)
        global_cov = torch.eye(X.shape[1])
        self.means_ = global_mean + torch.randn((self.n_components, X.shape[1])) * 0.5
        self.covariances_ = torch.stack([global_cov for _ in range(self.n_components)])
        self.weights_ = torch.ones(self.n_components) / self.n_components

        prev_log_likelihood = float('-inf')
        for n_iter in range(self.max_iter):
            log_responsibilities = self._e_step(X)
            self._m_step(X, log_responsibilities)

            log_likelihood = (torch.exp(log_responsibilities) * log_responsibilities).sum()
            
            if n_iter % 100 == 0:
                print(f"Iteration {n_iter}, Log Likelihood: {log_likelihood.item()}")
                print(self.means_)
            
            if torch.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

    def _e_step(self, X):
        log_prob = torch.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            log_prob[:, k] = self._estimate_log_gaussian_prob(X, self.means_[k], self.covariances_[k])
        log_weights = torch.log(self.weights_.clamp(min=1e-10))
        log_prob += log_weights

        log_responsibilities = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)

        return log_responsibilities

    def _m_step(self, X, log_responsibilities):
        responsibilities = torch.exp(log_responsibilities)
        weights = responsibilities.sum(0)

        self.weights_ = (weights / X.shape[0])

        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = responsibilities[:, k].unsqueeze(1) * diff
            self.means_[k] = torch.sum(weighted_diff, axis=0) / (weights[k] + self.tol)
            
            cov_k = torch.mm(weighted_diff.T, diff) / weights[k]
            self.covariances_[k] = cov_k + torch.eye(X.shape[1]) * self.tol  # Ensure positive definite

    @staticmethod
    def _estimate_log_gaussian_prob(X, mean, covariance):
        diff = X - mean
        precision = torch.inverse(covariance)
        log_det = torch.logdet(covariance)
        mahalanobis_distance = (diff @ (precision @ diff.T)).diagonal()
        return -0.5 * (X.shape[1] * torch.log(torch.tensor(2.0 * torch.pi)) + log_det + mahalanobis_distance)

def main():
    # Generate synthetic data
    # torch.manual_seed(42)
    X = torch.cat((torch.randn(500, 2) + torch.tensor([-100, -100]), torch.randn(500, 2) + torch.tensor([100, 100]), torch.randn(500, 2) + torch.tensor([0, -100])))

    # Fit GMM
    gmm = GMM(n_components=3, max_iter=100000)
    # gmm = GMM(n_components=6, max_iter=10000)
    gmm.fit(X)
    print(gmm.means_)

    # Plot
    plt.scatter(X[:, 0], X[:, 1], c='blue', s=5)
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=10)
    plt.savefig(f"compute_saliency/projected/gmm.png")
    plt.show()

if __name__ == "__main__":
    main()
