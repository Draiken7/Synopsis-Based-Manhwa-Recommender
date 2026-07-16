import pytest
from models.datasets import TextDataset # Adjust import path if needed

# --- Fixtures ---

@pytest.fixture
def valid_texts():
    return ["First synopsis.", "Second synopsis.", "Third synopsis."]

@pytest.fixture
def mixed_type_texts():
    # Simulating a scenario where pandas accidentally passed non-strings
    return ["Valid string", 404, 3.14159, None]

# --- Test Cases ---

def test_dataset_initialization(valid_texts):
    """Test 1: Does the dataset initialize and store the data correctly?"""
    dataset = TextDataset(valid_texts)
    assert dataset.texts == valid_texts

def test_dataset_length(valid_texts):
    """Test 2: Does __len__ return the exact number of records?"""
    dataset = TextDataset(valid_texts)
    
    # PyTorch's DataLoader relies heavily on len() being perfectly accurate
    assert len(dataset) == 3

def test_dataset_getitem_retrieval(valid_texts):
    """Test 3: Does __getitem__ return the correct text at the correct index?"""
    dataset = TextDataset(valid_texts)
    
    assert dataset[0] == "First synopsis."
    assert dataset[1] == "Second synopsis."
    assert dataset[2] == "Third synopsis."

def test_dataset_forces_string_conversion(mixed_type_texts):
    """Test 4 (Critical): Does it properly cast rogue data types to strings?"""
    # BERT tokenizer will crash if fed integers or NoneTypes.
    # Our __getitem__ uses str() to prevent this. Let's prove it works.
    dataset = TextDataset(mixed_type_texts)
    
    assert dataset[0] == "Valid string"
    assert dataset[1] == "404"       # Integer safely converted to string
    assert dataset[2] == "3.14159"   # Float safely converted to string
    assert dataset[3] == "None"      # NoneType safely converted to string

def test_index_out_of_bounds(valid_texts):
    """Test 5 (Negative): Does it throw a standard Python IndexError for bad indices?"""
    dataset = TextDataset(valid_texts)
    
    with pytest.raises(IndexError):
        # We only have 3 items (indices 0, 1, 2). Index 5 should crash.
        _ = dataset[5]