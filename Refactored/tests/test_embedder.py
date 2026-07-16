import pytest
import numpy as np
import torch
from models.embedder import BERTEmbedder 

# --- Fixtures ---

@pytest.fixture(scope="module")
def embedder():
    """
    Initializes the embedder once for the whole test module to save time.
    We use 'bert-tiny' so the test runs instantly and doesn't crash CI pipelines.
    BERT-Tiny has a hidden dimension of 128 instead of 768.
    """
    return BERTEmbedder(model_name="prajjwal1/bert-tiny", batch_size=2)

@pytest.fixture
def sample_texts():
    return [
        "A weak hunter levels up to become a god.",
        "A boy climbs a tower to find his friend.",
        "A martial artist reincarnates in a modern world.",
        "Someone gets stuck in a virtual reality game." # 4 texts total
    ]

# --- Test Cases ---

def test_embedder_initialization(embedder):
    """Test 1: Does it detect hardware and lock the model for inference?"""
    # Ensure model is strictly in evaluation mode, not training mode
    assert not embedder.model.training
    
    # Ensure device is one of the accepted PyTorch devices
    assert embedder.device.type in ['cpu', 'cuda', 'mps']

def test_generate_embeddings_shape_and_type(embedder, sample_texts):
    """Test 2: Does it output a NumPy matrix of the correct dimensions?"""
    embeddings = embedder.generate_embeddings(sample_texts)
    
    # 1. Must be a numpy array for scikit-learn clustering
    assert isinstance(embeddings, np.ndarray)
    
    # 2. Shape must be (Number of Texts, Hidden Dimension)
    # bert-tiny hidden dim is 128 (bert-base would be 768)
    assert embeddings.shape == (4, 128)

def test_batching_logic(embedder, sample_texts):
    """Test 3: Does it process data cleanly when the data size > batch size?"""
    # Our fixture set batch_size=2. We pass 4 texts. 
    # This forces the DataLoader to split into 2 separate batches and recombine them.
    embeddings = embedder.generate_embeddings(sample_texts)
    
    # If the stacking logic fails, this will not equal 4
    assert embeddings.shape[0] == len(sample_texts)

def test_padding_and_masking_consistency(embedder):
    """Test 4 (Architectural): Does the attention mask properly ignore padding?"""
    # Text A is long. Text B is short.
    # In a batch, Text B will be heavily padded with zeros.
    batch = [
        "This is a very long text to force the tokenizer to add a lot of padding to the next sentence.",
        "Short text."
    ]
    
    # Generate batched embeddings (uses padding and your custom mean pooling mask)
    batched_embeddings = embedder.generate_embeddings(batch)
    
    # Generate the embedding for the short text isolated (no padding added)
    isolated_embedding = embedder.generate_embeddings(["Short text."])
    
    # If your attention mask logic in embedder.py is correct, the embedding for 
    # "Short text." should be nearly identical whether it was padded in a batch or not.
    # We use np.allclose to account for microscopic floating-point rounding differences.
    assert np.allclose(batched_embeddings[1], isolated_embedding[0], atol=1e-5)

def test_empty_input(embedder):
    """Test 5 (Negative): How does it handle being fed nothing?"""
    # Depending on how strict you want to be, this should either return an empty array 
    # or raise a specific error. PyTorch DataLoaders usually crash on empty datasets.
    with pytest.raises(Exception):
        embedder.generate_embeddings([])