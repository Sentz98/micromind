"""
Unit tests for micromind/networks/_utils

Run with: pytest test_phinet.py -v
"""
import pytest
from micromind.networks._utils import _make_divisible, correct_pad

class TestHelperFunctions:
    """Test suite for helper functions."""
    
    def test_make_divisible_basic(self):
        """Test _make_divisible with basic inputs."""
        assert _make_divisible(10, divisor=8) == 16
        assert _make_divisible(17, divisor=8) == 16
        assert _make_divisible(20, divisor=8) == 24
    
    def test_make_divisible_min_value(self):
        """Test _make_divisible with min_value parameter."""
        assert _make_divisible(5, divisor=8, min_value=16) == 16
        assert _make_divisible(1, divisor=8, min_value=8) == 8
    
    def test_make_divisible_ten_percent_rule(self):
        """Test that _make_divisible doesn't reduce by more than 10%."""
        result = _make_divisible(10, divisor=8)
        assert result >= 10 * 0.9
    
    def test_correct_pad_with_int_kernel(self):
        """Test correct_pad with integer kernel size."""
        padding = correct_pad((224, 224), 3)
        assert len(padding) == 4
        assert all(isinstance(p, int) for p in padding)
    
    def test_correct_pad_with_tuple_kernel(self):
        """Test correct_pad with tuple kernel size."""
        padding = correct_pad((224, 224), (3, 3))
        assert len(padding) == 4
    
    def test_correct_pad_with_none_input(self):
        """Test correct_pad with None in input shape."""
        padding = correct_pad((None, 224), 3)
        assert len(padding) == 4
    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])