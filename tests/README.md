# MicroMind Unit Test Structure

## Test Folder Structure

```
micromind/
├── micromind/
│   ├── __init__.py
│   ├── core.py
│   ├── enum.py
│   ├── callbacks.py
│   └── utils/
│       ├── helpers.py
│       └── checkpointer.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures and configuration
│   ├── test_core.py                   # Tests for core.py
│   ├── test_enum.py                   # Tests for enum.py  
│   ├── test_callbacks.py              # Tests for callbacks.py
│   ├── test_integration.py            # Integration tests
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── test_helpers.py            # Tests for utils/helpers.py
│   │   └── test_checkpointer.py       # Tests for utils/checkpointer.py
│   └── fixtures/
│       ├── __init__.py
│       ├── dummy_data.py              # Test data generators
│       └── mock_models.py             # Mock model implementations
├── pytest.ini                        # Pytest configuration
├── setup.py
└── requirements-test.txt              # Test dependencies
```

## Key Testing Principles for MicroMind

1. **Mock External Dependencies**: Mock PyTorch operations, accelerate, and file I/O
2. **Test Abstract Methods**: Use concrete implementations to test abstract base classes
3. **Callback System**: Ensure callbacks are called in the correct order with proper state
4. **Device Management**: Mock CUDA operations for consistent testing
5. **Integration Tests**: Test the complete training pipeline with minimal epochs

## Configuration Files

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=micromind
    --cov-report=html
    --cov-report=term-missing
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests that take more than a minute
    gpu: Tests that require GPU
```

### requirements-test.txt
```txt
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pytest-xdist>=3.2.0
torch>=1.12.0
accelerate>=0.20.0
tqdm>=4.64.0
torchinfo>=1.8.0
```

## Testing Strategy

### 1. Unit Tests
- Test individual methods and classes in isolation
- Mock external dependencies (PyTorch, accelerate, file I/O)
- Focus on business logic and edge cases
- Fast execution (< 1 second per test)

### 2. Integration Tests
- Test complete workflows (training, validation, export)
- Use minimal datasets and epochs
- Test callback interactions
- Verify end-to-end functionality

### 3. Fixtures and Mocks
- Reusable test data and mock objects
- Consistent test environment setup
- Mock expensive operations (GPU, I/O)

### 4. Parameterized Tests
- Test multiple scenarios with different parameters
- Reduce code duplication
- Comprehensive coverage of edge cases

## Running the Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=micromind --cov-report=html

# Run only unit tests
pytest -m unit

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v

# Run tests and stop on first failure
pytest -x
```

## Best Practices

1. **AAA Pattern**: Arrange, Act, Assert structure
2. **Descriptive Names**: Test names should describe what is being tested
3. **One Assertion**: Focus on testing one thing per test
4. **Isolation**: Tests should not depend on each other
5. **Mocking**: Mock external dependencies for reliability and speed
6. **Edge Cases**: Test boundary conditions and error scenarios
7. **Documentation**: Include docstrings for complex test scenarios