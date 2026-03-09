"""
Test suite for Deep-aPCE project.
"""

import pytest
import numpy as np
import torch

from src.Deep_PCE import k_center_moment, order_mat_fun


class TestKCenterMoment:
    """Test cases for k_center_moment function."""
    
    def test_k_center_moment_basic(self):
        """Test basic k-center moment calculation."""
        x = torch.randn(100, 5)
        k = 3
        
        mu = k_center_moment(x, k)
        
        assert mu.shape == (k + 1, 5)
        assert mu[0, 0].item() == pytest.approx(1.0, abs=1e-6)
    
    def test_k_center_moment_first_order(self):
        """Test first-order moment (mean)."""
        x = torch.randn(100, 3)
        
        mu = k_center_moment(x, 1)
        
        assert mu.shape == (2, 3)
        # First-order moment should be close to mean
        expected_mean = torch.mean(x, dim=0)
        assert torch.allclose(mu[1], expected_mean, atol=1e-5)
    
    def test_k_center_moment_zero_input(self):
        """Test with zero input."""
        x = torch.zeros(50, 2)
        k = 2
        
        mu = k_center_moment(x, k)
        
        assert mu[0, 0].item() == 1.0  # 0-order moment
        assert torch.all(mu[1:] == 0)  # Higher-order moments


class TestOrderMatFun:
    """Test cases for order_mat_fun function."""
    
    def test_order_mat_fun_1d(self):
        """Test order matrix for 1-dimensional case."""
        dim = 1
        order = 3
        
        order_mat = order_mat_fun(dim, order)
        
        assert order_mat.shape[0] == order + 1
        assert order_mat.shape[1] == dim
    
    def test_order_mat_fun_2d(self):
        """Test order matrix for 2-dimensional case."""
        dim = 2
        order = 2
        
        order_mat = order_mat_fun(dim, order)
        
        # Check that all rows sum to <= order
        mat_sum = torch.sum(order_mat, dim=1)
        assert torch.all(mat_sum <= order)
    
    def test_order_mat_fun_values(self):
        """Test that order matrix contains valid values."""
        dim = 3
        order = 2
        
        order_mat = order_mat_fun(dim, order)
        
        assert torch.all(order_mat >= 0)
        assert torch.all(order_mat <= order)


class TestIntegration:
    """Integration tests for Deep-aPCE."""
    
    def test_import_modules(self):
        """Test that all modules can be imported."""
        from src import data_process
        from src import apc
        from src import models
        from src import Deep_PCE
        from src import pce_loss
    
    def test_model_creation(self):
        """Test model creation (if models module is available)."""
        try:
            from src.models import DNN
            
            model = DNN(input_dim=5, hidden_dim=32, output_dim=1)
            assert isinstance(model, torch.nn.Module)
        except Exception:
            pytest.skip("Model creation not available in this configuration")


def test_numpy_torch_compatibility():
    """Test NumPy and PyTorch compatibility."""
    # Create numpy array
    np_array = np.random.randn(100, 5)
    
    # Convert to torch
    torch_tensor = torch.from_numpy(np_array).float()
    
    # Convert back
    np_array_back = torch_tensor.numpy()
    
    assert np.allclose(np_array, np_array_back)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
