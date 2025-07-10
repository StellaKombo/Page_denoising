import numpy as np

def page_construct(matrix: np.ndarray, L: int) -> np.ndarray:
    """
    Constructs a Page matrix from a 2D input matrix (e.g., weight matrix).
    Each column of the Page matrix is a vertical slice of L rows.
    """
    if matrix.ndim != 2:
        raise ValueError("Input to page_construct must be a 2D matrix.")

    N, M = matrix.shape
    countmax = N // L + int(N % L != 0)
    
    # Pad with zeros if necessary to make N divisible by L
    pad_len = countmax * L - N
    if pad_len > 0:
        matrix = np.pad(matrix, ((0, pad_len), (0, 0)), mode='constant')
    
    # Reshape into blocks of size L × M, stacked horizontally
    page_matrix = matrix.reshape(countmax, L, M).transpose(1, 0, 2).reshape(L, -1)
    return page_matrix

