import numpy as np

def nmf(X, k, max_iter=5000, tol=1e-5, seed=0, normalize_signatures=True):

    np.random.seed(seed)
    eps = 1e-10
    
    n_tumors, n_mut_types = X.shape
    W = np.random.rand(n_tumors, k)
    H = np.random.rand(k, n_mut_types)
    
    errors = []
    prev_error = np.inf
    
    for it in range(max_iter):
        
        numerator_H = W.T @ X
        denominator_H = (W.T @ W @ H) + eps
        H *= numerator_H / denominator_H
        
        numerator_W = X @ H.T
        denominator_W = (W @ (H @ H.T)) + eps
        W *= numerator_W / denominator_W
        
        if normalize_signatures:
            row_sums = H.sum(axis=1, keepdims=True) + eps
            H /= row_sums
            W *= row_sums.T 
        
        recon = W @ H
        error = np.linalg.norm(X - recon, 'fro')
        errors.append(error)
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    
    return W, H, errors


X = np.array([
    [0.10, 0.30, 0.60],  
    [0.20, 0.40, 0.40],
    [0.25, 0.35, 0.40],
    [0.05, 0.05, 0.90],  
    [0.30, 0.40, 0.30]   
])

W, H, errors = nmf(X, k=4, max_iter=1000, tol=1e-8, seed=42)

np.set_printoptions(precision=3, suppress=True)

print("Final reconstruction error:", errors[-1])
print("\nExposures (W):\n", W)
print("\nSignatures (H):\n", H)
print("\nReconstruction (W @ H):\n", W @ H)
print("\nOriginal X:\n", X)
