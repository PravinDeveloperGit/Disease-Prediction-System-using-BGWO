import logging
import re
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Cleans raw symptom text by lowercasing, removing non-alphabetical special characters,
    and trimming extraneous whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters except letters, spaces, and commas
    text = re.sub(r'[^a-z\s,]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs full dataset cleaning:
    - Removes duplicate rows
    - Removes rows with missing target or symptom features
    - Cleans symptom text
    - Resets DataFrame index
    """
    logger.info("Cleaning Dataset: Removing duplicates and handling missing values...")
    try:
        initial_count = len(df)
        df = df.drop_duplicates()
        
        target_col = "disease" if "disease" in df.columns else ("diseases" if "diseases" in df.columns else None)
        if not target_col:
            raise ValueError("Dataset missing 'disease' target column.")
            
        df = df.dropna(subset=[target_col, "symptoms"])
        df["symptoms"] = df["symptoms"].apply(clean_text)
        df = df[df["symptoms"].str.strip() != ""]
        df = df.reset_index(drop=True)
        
        logger.info(f"Dataset cleaning complete: Reduced from {initial_count} to {len(df)} records.")
        return df
    except Exception as e:
        logger.error(f"Error during dataset cleaning: {str(e)}")
        raise e