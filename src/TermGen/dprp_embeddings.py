import numpy as np
from scipy.linalg import svd
import math

class DPRPEmbeddings:
    """
    Differentially Private Data Release via Random Projections (DPRP)
    Adapted for embedding vectors
    """
    
    def __init__(self, epsilon=1.0, delta=0.0001, budget_allocation=0.8):
        """
        Initialize DPRP for embeddings
        
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter
            budget_allocation: Fraction of budget for random projection (epsilon_1)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.budget_allocation = budget_allocation
        
        # Split privacy budget
        self.epsilon_1 = epsilon * budget_allocation
        self.epsilon_2 = epsilon * (1 - budget_allocation)
        self.delta_1 = delta * budget_allocation
        self.delta_2 = delta * (1 - budget_allocation)
        
    def compute_sigma_1(self, k1, Z=1.0):
        """
        Compute noise scale for random projection using Equation (6)
        
        Args:
            k1: Dimension of random projection
            Z: Sensitivity bound (set to 1 for normalized embeddings)
        """
        sigma_p = 1.0 / np.sqrt(k1)
        
        # From Equation (6)
        sqrt_part1 = np.sqrt(k1 + 2 * np.sqrt(k1 * np.log(2 / self.delta_1)) + 2 * np.log(2 / self.delta_1))
        sqrt_part2 = np.sqrt(2 * (np.log(1 / (2 * self.delta_1)) + self.epsilon_1)) / self.epsilon_1
        
        return Z * sigma_p * sqrt_part1 * sqrt_part2
    
    def compute_sigma_2(self, Z=1.0):
        """
        Compute noise scale for covariance matrix using Equation (8)
        
        Args:
            Z: Sensitivity bound (set to 1 for normalized embeddings)
        """
        return Z**2 * np.sqrt(2 * np.log(1.25/self.delta_2)) / self.epsilon_2
    
    def release_embeddings_modified(self, X, k1=None, k2_ratio=0.6):
        """
        Modified DPRP for embeddings (skipping steps 1-2, using X directly as projection)
        
        Args:
            X: Input embeddings (n x d)
            k1: Dimension for noise calculation (use d if None)
            k2_ratio: Fraction of singular values to keep
        
        Returns:
            X_prime: Differentially private embeddings
        """
        n, d = X.shape
        
        if k1 is None:
            k1 = d
            
        k2 = int(k2_ratio * d)
        
        # Step 3: Add noise directly to embeddings (treating X as P)
        sigma_1 = self.compute_sigma_1(k1, Z=1.0)
        M1 = np.random.normal(0, sigma_1, (n, d))
        P_prime = X + M1 # (n, d)
        
        # Step 4: Compute covariance matrix
        X_C = X.T @ X # (d, d)
        
        # Step 5: Add noise to covariance matrix and perform SVD
        sigma_2 = self.compute_sigma_2(Z=1.0)
        
        # Create symmetric noise matrix for covariance
        M2_upper = np.random.normal(0, sigma_2, (d, d))
        M2 = np.triu(M2_upper) + np.triu(M2_upper, 1).T  # Make symmetric
        
        X_C_noisy = X_C + M2 # (d, d)
        
        # SVD decomposition
        U_hat, S_hat, V_hat_T = svd(X_C_noisy) # U_hat: (d, d), S_hat: (d,), V_hat_T: (d, d)
        V_hat_prime = V_hat_T.T  # Convert to column format
        
        # Step 6: Take first k2 columns
        V_prime_k2 = V_hat_prime[:, :k2] # (d, k2)
        
        # Step 7: Reconstruction (modified for direct embedding case)
        # Since we're using X directly as P, we need to adapt the reconstruction
        # The original formula: X' = P'(V_k2^T R)^+ V_k2^T
        # For our case: X' = P'(V_k2^T)^+ V_k2^T
        
        V_k2_T = V_prime_k2.T # (k2, d)
        
        # Moore-Penrose pseudoinverse of V_k2^T
        V_k2_T_pinv = np.linalg.pinv(V_k2_T) # (d, k2)
        
        # Reconstruction
        X_prime = P_prime @ V_k2_T_pinv @ V_k2_T # (n, d) x (d, k2) x (k2, d) -> (n, d)
        
        return X_prime, {
            'sigma_1': sigma_1,
            'sigma_2': sigma_2,
            'k1': k1,
            'k2': k2,
            'V_prime_shape': V_hat_prime.shape,
            'V_k2_shape': V_prime_k2.shape
        }
    
    def release_embeddings_original(self, X, k1=None, k2_ratio=0.6):
        """
        Original DPRP algorithm adapted for embeddings
        
        Args:
            X: Input embeddings (n x d)
            k1: Dimension for random projection
            k2_ratio: Fraction of singular values to keep
        
        Returns:
            X_prime: Differentially private embeddings
        """
        n, d = X.shape
        
        if k1 is None:
            k1 = max(d, int(1.2 * d))  # Slightly higher than d for better utility
            
        k2 = int(k2_ratio * d)
        
        # Step 1-2: Create random projection
        R = np.random.normal(0, 1/math.sqrt(k1), (d, k1))
        P = X @ R
        
        # Step 3: Add noise to projection
        sigma_1 = self.compute_sigma_1(k1)
        M1 = np.random.normal(0, sigma_1, (n, k1))
        P_prime = P + M1
        
        # Step 4: Compute covariance matrix
        X_C = X.T @ X
        
        # Step 5: Add noise to covariance matrix and perform SVD
        sigma_2 = self.compute_sigma_2()
        
        # Create symmetric noise matrix
        M2_upper = np.random.normal(0, sigma_2, (d, d))
        M2 = np.triu(M2_upper) + np.triu(M2_upper, 1).T
        
        X_C_noisy = X_C + M2
        
        # SVD decomposition
        U_hat, S_hat, V_hat_T = svd(X_C_noisy)
        V_hat_prime = V_hat_T.T
        
        # Step 6: Take first k2 columns
        V_prime_k2 = V_hat_prime[:, :k2]
        
        # Step 7: Reconstruction
        V_k2_T = V_prime_k2.T
        
        # Moore-Penrose pseudoinverse of (V_k2^T @ R)
        V_k2_T_R = V_k2_T @ R
        V_k2_T_R_pinv = np.linalg.pinv(V_k2_T_R)
        
        # Final reconstruction
        X_prime = P_prime @ V_k2_T_R_pinv @ V_k2_T
        
        return X_prime, {
            'sigma_1': sigma_1,
            'sigma_2': sigma_2,
            'k1': k1,
            'k2': k2,
            'R_shape': R.shape,
            'P_shape': P.shape,
            'V_prime_shape': V_hat_prime.shape,
            'V_k2_shape': V_prime_k2.shape
        }

# Example usage
if __name__ == "__main__":
    # Generate sample embedding data
    np.random.seed(42)
    n, d = 100, 768  # 100 samples, 768-dim embeddings (e.g., BERT)
    
    # Simulate normalized embeddings
    X = np.random.randn(n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # L2 normalize
    
    # Initialize DPRP
    dprp = DPRPEmbeddings(epsilon=1.0, delta=0.0001, budget_allocation=0.8)
    
    # Test both approaches
    print("Testing modified approach (skip steps 1-2):")
    X_prime_modified, info_modified = dprp.release_embeddings_modified(X)
    print(f"Original shape: {X.shape}")
    print(f"Output shape: {X_prime_modified.shape}")
    print(f"Info: {info_modified}")
    
    print("\nTesting original approach:")
    X_prime_original, info_original = dprp.release_embeddings_original(X)
    print(f"Original shape: {X.shape}")
    print(f"Output shape: {X_prime_original.shape}")
    print(f"Info: {info_original}")
    
    # Compute utility metrics
    def compute_utility(X_orig, X_priv):
        """Compute utility metrics"""
        mse = np.mean((X_orig - X_priv)**2)
        cosine_sim = np.mean([
            np.dot(X_orig[i], X_priv[i]) / (np.linalg.norm(X_orig[i]) * np.linalg.norm(X_priv[i]))
            for i in range(X_orig.shape[0])
        ])
        return mse, cosine_sim
    
    mse_mod, cos_mod = compute_utility(X, X_prime_modified)
    mse_orig, cos_orig = compute_utility(X, X_prime_original)
    
    print(f"\nUtility Comparison:")
    print(f"Modified approach - MSE: {mse_mod:.4f}, Cosine similarity: {cos_mod:.4f}")
    print(f"Original approach - MSE: {mse_orig:.4f}, Cosine similarity: {cos_orig:.4f}")
