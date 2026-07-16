import pandas as pd
import logging
import re
from typing import Dict, Union, List

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSchemaError(Exception):
    """Custom exception for when the uploaded data does not match the expected schema."""
    pass

class DataProcessor:
    """
    Class to handle data validation and cleaning for the loaded dataset.
    This ensures that the data conforms to the expected schema and is free of nulls and duplicates before being used for model training or inference.
    """
    def __init__(self):
        # Define the strict contract: Column Name -> Expected Pandas Dtype
        self.schema: Dict[str, tuple] = {
            'title': ('object', 'string'),      # String
            'synopsis': ('object', 'string'),  # String
            'genres': ('object', 'string'),     # String (could be a list or comma-separated)
            'score': ('float64', 'int64')     
        }
        # 2. Define columns where NaNs should be filled with 0
        self.fill_zero_cols: List[str] = ['score']
        
        # 3. Define columns that must follow a comma-separated format
        self.comma_separated_cols: List[str] = ['genres']
        
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts all column names to lowercase to make validation case-agnostic."""
        df.columns = df.columns.str.lower().str.strip()
        return df

    def validate_and_clean_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates schema, handles missing values, and checks formatting."""
        
        # --- Requirement 1: Case Agnostic Schema & Type Check ---
        missing_cols = [col for col in self.schema.keys() if col not in df.columns]
        if missing_cols:
            raise DataSchemaError(f"Missing required columns (case-insensitive): {missing_cols}")
        
        #--- Requirement 2: Check for type errors ---
        type_errors = []
        for col, expected_types in self.schema.items():
            actual_type = str(df[col].dtype)
            if actual_type not in expected_types:
                type_errors.append(f"'{col}': (Expected {expected_types}, got {actual_type})")

        if type_errors:
             raise DataSchemaError(f"Data type mismatch: {', '.join(type_errors)}")
         
         # --- Requirement 2: Fill NaNs with 0 in specific columns ---
        for col in self.fill_zero_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
                logger.info(f"Filled NaNs with 0 in column: {col}")
                
        # --- Requirement 3: Check for comma-separated format ---
        # Pattern allows words, spaces, and commas (e.g., "Action, Fantasy, Sci-Fi")
        # It also allows single words without commas (e.g., "Action")
        comma_regex = re.compile(r'^[\w\s-]+(?:,\s*[\w\s-]+)*$')
        
        for col in self.comma_separated_cols:
            if col in df.columns:
                # Force to string, handle any accidental pandas NaNs
                df[col] = df[col].fillna('unknown').astype(str)
                
                # Check for invalid formats
                invalid_mask = ~df[col].str.match(comma_regex)
                if invalid_mask.any():
                    invalid_examples = df.loc[invalid_mask, col].head(3).tolist()
                    raise DataSchemaError(
                        f"Column '{col}' contains values that are not valid comma-separated strings. "
                        f"Examples of invalid data: {invalid_examples}"
                    )

        logger.info("Schema validation, imputation, and format checking passed.")
        return df
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main pipeline execution."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame!")
        else:
            df = df.copy()

        # Execute standardizations and validations
        df = self._standardize_columns(df)
        df = self.validate_and_clean_schema(df)
        
        # Standard final cleaning (dropping rows where the core text is missing)
        df = df.dropna(subset=['synopsis'])
        
        df['synopsis'] = df['synopsis'].astype(str)
        df = df[df['synopsis'].str.strip() != '']
        df = df[df['synopsis'].str.lower() != 'nan']
        
        df = df.drop_duplicates(subset=['synopsis']).reset_index(drop=True)
        df = df.drop_duplicates(subset=['title']).reset_index(drop=True)
        
        logger.info(f"Processing complete. Final dataset shape: {df.shape}")
        return df