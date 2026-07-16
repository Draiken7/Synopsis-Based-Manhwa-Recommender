import pytest
import pandas as pd
import numpy as np
from data.dataprocessor import DataProcessor, DataSchemaError



# --- Fixtures (Mock Data Setup) ---

@pytest.fixture
def processor():
    """Returns a fresh instance of DataProcessor for each test."""
    return DataProcessor()


@pytest.fixture
def valid_data():
    """Returns a perfectly formatted, clean dataframe."""
    return pd.DataFrame({
        'title': ['Solo Leveling', 'Tower of God'],
        'synopsis': ['Hunter goes from weak to strong.', 'Boy climbs a tower for a girl.'],
        'genres': ['Action, Fantasy', 'Action, Adventure, Mystery'],
        'score': [9.9, 9.5]
    })
   
 
@pytest.mark.parametrize("column_to_drop", [
    'title', 
    'synopsis', 
    'genres', 
    'score'
    ])

def test_missing_columns(processor, valid_data, column_to_drop):
    """Ensures pipeline fails if ANY required column is missing."""
    bad_data = valid_data.drop(columns=[column_to_drop])
    
    with pytest.raises(DataSchemaError, match="Missing required columns"):
        processor.process(bad_data)


        
@pytest.mark.parametrize("col_to_corrupt, bad_value", [
    ('title', 12345),                 # Expected string, gets integer
    ('synopsis', True),               # Expected string, gets boolean
    ('genres', 45.67),                # Expected string, gets float
    ('score', 'high')      # Expected numeric, gets string
])

def test_type_mismatches(processor, valid_data, col_to_corrupt, bad_value):
    """Ensures pipeline fails if ANY column has the wrong data type."""
    bad_data = valid_data.copy()
    # Inject the bad data type into the entire column
    bad_data[col_to_corrupt] = [bad_value] * len(bad_data)
    
    with pytest.raises(DataSchemaError, match="Data type mismatch"):
        processor.process(bad_data)
        

def test_duplicate_removal(processor, valid_data):
    """Ensures exact duplicates are dropped cleanly."""
    # Create a duplicate of the first row and append it
    duplicate_row = valid_data.iloc[[0]]
    data_with_dupes = pd.concat([valid_data, duplicate_row], ignore_index=True)
    
    assert len(data_with_dupes) == 3 # Pre-check
    clean_df = processor.process(data_with_dupes)
    
    # The duplicate should be gone, back to 2 rows
    assert len(clean_df) == 2
    
    
def test_null_and_whitespace_handling_in_synopsis(processor, valid_data):
    """Ensures rows with missing or empty synopses are dropped entirely."""
    bad_data = valid_data.copy()
    
    # Inject various forms of "empty" text
    bad_data.loc[2] = ['Comic 3', '', 'Action', 5.0]          # Empty string
    bad_data.loc[3] = ['Comic 4', '   ', 'Action', 5.0]       # Whitespace only
    bad_data.loc[4] = ['Comic 5', np.nan, 'Action', 5.0]      # Real NaN
    bad_data.loc[5] = ['Comic 6', None, 'Action', 5.0]        # Python None
    
    clean_df = processor.process(bad_data)
    
    # Only the original 2 valid rows should survive
    assert len(clean_df) == 2
    assert 'Comic 3' not in clean_df['title'].values
    
    
def test_numeric_nan_imputation(processor, valid_data):
    """Ensures numeric columns get filled with 0 instead of dropping the row."""
    bad_data = valid_data.copy()
    bad_data.loc[0, 'score'] = np.nan
    
    clean_df = processor.process(bad_data)
    
    # Row should NOT be dropped, but the score should be 0.0
    assert len(clean_df) == 2
    assert clean_df.loc[0, 'score'] == 0.0

    
    
@pytest.mark.parametrize("invalid_genre_string", [
    '{"genre": "Action"}',     # JSON blob
    'Action; Fantasy',         # Semicolon instead of comma
    'Action | Fantasy',        # Pipe instead of comma
    '[Action, Fantasy]'        # List brackets
])

def test_strict_comma_separated_formatting(processor, valid_data, invalid_genre_string):
    """Ensures regex catches bad formatting in the tags/genres column."""
    bad_data = valid_data.copy()
    bad_data.loc[0, 'genres'] = invalid_genre_string
    
    with pytest.raises(DataSchemaError, match="not valid comma-separated strings"):
        processor.process(bad_data)