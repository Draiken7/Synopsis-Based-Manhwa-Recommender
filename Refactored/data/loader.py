import pandas as pd
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles all data extraction from various storage systems."""
    def __init__(self):
        pass  # No instance variables needed for now
    
    @staticmethod
    def load_from_csv(file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads data from a local CSV file."""
        logger.info(f"Ingesting CSV data from {file_path}")
        try:
            # Using engine='c' for faster parsing
            df = pd.read_csv(file_path, engine='c') 
            logger.info(f"Successfully loaded {df.shape[0]} rows.")
            return df
        except FileNotFoundError:
            logger.error(f"CRITICAL: The file at {file_path} does not exist.")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"CRITICAL: The file at {file_path} is empty.")
            raise

    # --- Future AWS S3 Implementation Placeholder ---
    # @staticmethod
    # def load_from_s3(bucket_name: str, object_key: str) -> pd.DataFrame:
    #     """Downloads a CSV from AWS S3 and loads it into a DataFrame."""
    #     import boto3
    #     import io
    #     logger.info(f"Fetching s3://{bucket_name}/{object_key}")
    #     s3 = boto3.client('s3')
    #     obj = s3.get_object(Bucket=bucket_name, Key=object_key)
    #     return pd.read_csv(io.BytesIO(obj['Body'].read()))