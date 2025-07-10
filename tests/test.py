import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spectrum.optimal_SVHT_coef import optimal_SVHT_coef_sigma, optimal_SVHT_coef_sigma_known, optimal_SVHT_coef_sigma_unknown
from spectrum.page_construct import page_construct
import numpy as np
import unittest
import torch
import numpy as np
from spectrum.spectrum import ModelModifier
import matplotlib.pyplot as plt
from scipy.stats import norm, stats, kstest

def test_optimal_SVHT_known():
    beta = 0.5
    
    coef = optimal_SVHT_coef_sigma(beta, sigma_known=1)
    print(f"[KNOWN σ] beta = {beta}, SVHT Coef = {coef:.4f}")

def test_optimal_SVHT_unknown():
    beta = 0.5
    coef = optimal_SVHT_coef_sigma(beta, sigma_known=0)
    print(f"[UNKNOWN σ] beta = {beta}, SVHT Coef = {coef:.4f}")


def test_compare_known_vs_unknown():
    betas = np.linspace(0.1, 1.0, 10)
    for beta in betas:
        coef_known = optimal_SVHT_coef_sigma_known(beta)
        coef_unknown = optimal_SVHT_coef_sigma_unknown(beta)
        print(f"β = {beta:.2f} | known: {coef_known:.4f}, unknown: {coef_unknown:.4f}")

def get_optimal_L(weight_matrix: np.ndarray) -> int:
    d = weight_matrix.shape[0]
    L = int(0.25 * d)
    L = 2 ** int(np.round(np.log2(L)))
    return min(max(64, L), d)

def test_page_construct():

    # Create low-rank matrix with noise
    np.random.seed(42)
    U = np.random.randn(768, 10)
    V = np.random.randn(10, 768)
    noise = 0.01 * np.random.randn(768, 768)
    W = U @ V + noise

    # Construct Page matrix
    L = get_optimal_L(W)
    page = page_construct(W, L)

    # Plot singular values
    s = np.linalg.svd(page, compute_uv=False)

    beta = min(page.shape) / max(page.shape)
    coeff = optimal_SVHT_coef_sigma(beta, sigma_known=0)
    print(f"Optimal SVHT Coefficient: {coeff:.4f}")
    threshold = coeff * np.median(s)
    print(f"L: {L}, Page shape: {page.shape}, Rank selected: {threshold}")

    print(f"Threshold for SNR: {threshold:.4f}")  
    print(f"[DEBUG] S[:20] = {s[:20]}") 
    print(f"[DEBUG] signal = {np.sum(s[s > threshold])}, noise = {np.sum(s[s <= threshold])}")

    signal = np.sum(s[s > threshold])
    noise = np.sum(s[s <= threshold])
    noise_svs = s[s <= threshold]
    print(f"Noise singular values (count = {len(noise_svs)}): {noise_svs[:10]}")
    print(f"Mean noise energy: {noise_svs.mean():.4f}")
    print(f"Std dev of noise: {noise_svs.std():.4f}")
    snr = signal / (noise + 1e-8)
    print(f"SNR: {snr:.2f}")
    assert signal >= 9, f"Expected ~10 signal modes, got {signal}"
 
    plt.figure(figsize=(8, 4))
    plt.plot(s, label='Singular values', marker='o', markersize=3, linestyle='-', color='blue')
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.1f}')
    plt.title('Scree Plot (Page matrix)')
    plt.xlabel('Index')
    plt.ylabel('Singular Value (log)')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    mean = np.mean(noise_svs)
    std = np.std(noise_svs)
    stat, pval = kstest((noise_svs - mean) / std, 'norm')
    print(f"K-S test stat = {stat:.4f}, p-value = {pval:.4f}")
    counts, bins, patches = plt.hist(noise_svs, bins=20, color='gray', edgecolor='black', alpha=0.6, label='Histogram')

    x = np.linspace(bins[0], bins[-1], 200)
    bin_width = bins[1] - bins[0]
    gaussian_scaled = norm.pdf(x, mean, std) * len(noise_svs) * bin_width
    plt.plot(x, gaussian_scaled, 'r--', linewidth=2, label=f'Gaussian Fit\nμ={mean:.2f}, σ={std:.2f}')
    plt.hist(noise_svs, bins=20, color='gray', edgecolor='black')
    plt.title('Histogram of Noise Singular Values')
    plt.xlabel('Singular Value')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()

class DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(128, 128))

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = DummyLayer()
        self.linear2 = DummyLayer()

    def named_modules(self):
        return [('linear1.weight', self.linear1), ('linear2.weight', self.linear2)]

    def parameters(self):
        return list(self.linear1.parameters()) + list(self.linear2.parameters())

class TestModelModifier(unittest.TestCase):
    def test_page_svht_snr_calculation(self):
        model = DummyModel()
        modifier = ModelModifier()
        modifier.model = model
        modifier.use_page_svht = True

        modifier.calculate_snr_for_layer('weight')
        self.assertIn('linear1.weight', modifier.layer_snr)
        self.assertIn('linear2.weight', modifier.layer_snr)
        self.assertTrue(isinstance(modifier.layer_snr['linear1.weight']['snr'], float))

    def test_mp_snr_calculation(self):
        model = DummyModel()
        modifier = ModelModifier()
        modifier.model = model
        modifier.use_page_svht = True

        modifier.calculate_snr_for_layer('weight')
        self.assertIn('linear1.weight', modifier.layer_snr)
        self.assertIn('linear2.weight', modifier.layer_snr)
        self.assertTrue(isinstance(modifier.layer_snr['linear2.weight']['snr'], float))


if __name__ == "__main__":
    # test_optimal_SVHT_known()
    # test_optimal_SVHT_unknown()
    # test_compare_known_vs_unknown()
    test_page_construct()
    #unittest.main()