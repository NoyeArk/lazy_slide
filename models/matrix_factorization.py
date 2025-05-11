import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def matrix_factorization(
    R: np.ndarray,
    K: int,
    alpha: float = 0.0002,
    beta: float = 0.02,
    steps: int = 5000,
    error_threshold: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Perform matrix factorization using gradient descent.

    Args:
        R: User-item rating matrix (m x n)
        K: Number of latent factors
        alpha: Learning rate (default: 0.0002)
        beta: Regularization parameter (default: 0.02)
        steps: Maximum number of iterations (default: 5000)
        error_threshold: Convergence threshold for loss (default: 0.001)

    Returns:
        Tuple containing:
        - user_matrix: User latent factor matrix (m x k)
        - item_matrix: Item latent factor matrix (n x k)
        - loss_history: List of loss values per iteration
    """
    try:
        # Input validation
        if not isinstance(R, np.ndarray) or R.ndim != 2:
            raise ValueError("R must be a 2D numpy array")
        if K <= 0 or not isinstance(K, int):
            raise ValueError("K must be a positive integer")
        if alpha <= 0 or beta < 0:
            raise ValueError("alpha must be positive and beta must be non-negative")

        # Initialize matrices
        m, n = R.shape
        user_matrix = np.random.rand(m, K)
        item_matrix = np.random.rand(n, K)
        loss_history = []

        print("Starting matrix factorization...")
        
        # Gradient descent
        for step in range(steps):
            # Update user_matrix and item_matrix
            for i in range(m):
                for j in range(n):
                    if R[i, j] > 0:
                        eij = R[i, j] - np.dot(user_matrix[i, :], item_matrix[j, :])
                        for k in range(K):
                            user_matrix[i, k] += alpha * (2 * eij * item_matrix[j, k] - beta * user_matrix[i, k])
                            item_matrix[j, k] += alpha * (2 * eij * user_matrix[i, k] - beta * item_matrix[j, k])

            # Calculate loss
            loss = 0.0
            for i in range(m):
                for j in range(n):
                    if R[i, j] > 0:
                        error = R[i, j] - np.dot(user_matrix[i, :], item_matrix[j, :])
                        loss += error ** 2
                        for k in range(K):
                            loss += (beta / 2) * (user_matrix[i, k] ** 2 + item_matrix[j, k] ** 2)
            
            loss_history.append(loss)
            
            # Check for convergence
            if loss < error_threshold:
                print(f"Converged at step {step + 1} with loss {loss:.6f}")
                break
            
            if (step + 1) % 500 == 0:
                print(f"Step {step + 1}, Loss: {loss:.6f}")

        print("Matrix factorization completed.")
        return user_matrix, item_matrix, loss_history

    except Exception as e:
        print(f"Error during matrix factorization: {str(e)}")
        raise

def plot_loss(loss_history: List[float]) -> None:
    """
    Plot the loss history over iterations.

    Args:
        loss_history: List of loss values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_history)), loss_history, 'b-')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs. Iteration")
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to demonstrate matrix factorization.
    """
    # Define sample rating matrix
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4]
    ])

    # Parameters
    K = 3
    alpha = 0.0002
    beta = 0.02
    steps = 5000

    # Run matrix factorization
    user_matrix, item_matrix, loss_history = matrix_factorization(R, K, alpha, beta, steps)

    # Calculate reconstructed matrix
    R_pred = np.dot(user_matrix, item_matrix.T)

    # Print results
    print("\nOriginal Matrix:")
    print(R)
    print("\nPredicted Matrix:")
    print(R_pred)
    print(f"\nFinal Loss: {loss_history[-1]:.6f}")

    # Plot loss history
    plot_loss(loss_history)

if __name__ == "__main__":
    main()
