import pytest
import pandas as pd
from pathlib import Path
from data.loader import DataLoader 


@pytest.fixture
def valid_csv_path(tmp_path: Path) -> Path:
    """Dynamically generates a temporary valid CSV file for testing."""
    # tmp_path is a built-in pytest fixture that provides a temporary directory unique to the test invocation
    file_path = tmp_path / "valid_data.csv"
    
    df = pd.DataFrame({
        'title': ['Solo Leveling', 'Tower of God'],
        'synopsis': ['Hunter gets strong.', 'Boy climbs tower.'],
        'genres': ['Action, Fantasy', 'Adventure, Fantasy'],
        'score': [9.9, 9.5]
    })
    df.to_csv(file_path, index=False)
    
    return file_path

@pytest.fixture
def empty_csv_path(tmp_path: Path) -> Path:
    """Dynamically generates a temporary file with absolutely zero bytes."""
    file_path = tmp_path / "empty_data.csv"
    file_path.touch()
    return file_path


# --- Test Cases ---

def test_load_valid_csv_success(valid_csv_path):
    """Test 1 (Positive): Ensures normal files load successfully."""
    df = DataLoader.load_from_csv(valid_csv_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert 'title' in df.columns

def test_file_not_found():
    """Test 2 (Negative): Ensures the pipeline halts if the path is bad."""
    # fake_path = "s3://fake-bucket/this_does_not_exist.csv"
    fake_path = "./this_does_not_exist_at_all.csv"
    
    with pytest.raises(FileNotFoundError):
        DataLoader.load_from_csv(fake_path)

def test_empty_file(empty_csv_path):
    """Test 3 (Negative): Ensures the pipeline halts if the file has 0 bytes."""
    # If an empty file gets through, it crashes downstream operations.
    with pytest.raises(pd.errors.EmptyDataError):
        DataLoader.load_from_csv(empty_csv_path)

def test_accept_pathlib_objects(valid_csv_path):
    """Test 4 (Positive): Ensures it works with modern pathlib.Path objects, not just strings."""
    # valid_csv_path is already a Path object from the fixture
    df = DataLoader.load_from_csv(valid_csv_path)
    assert not df.empty