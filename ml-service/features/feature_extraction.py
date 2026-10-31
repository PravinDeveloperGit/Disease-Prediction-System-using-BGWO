import pandas as pd
from preprocessing.text_vectorizer import SymptomVectorizer

def extract_features(df: pd.DataFrame, vectorizer: SymptomVectorizer = None, is_training: bool = True):
    """
    Unified feature extraction module.
    Converts raw text symptom column into TF-IDF numerical features.
    
    Parameters:
    - df: DataFrame containing 'symptoms' column
    - vectorizer: SymptomVectorizer. If is_training=False, this must be a pre-fitted vectorizer.
    - is_training: If True, fits and returns the vectorizer along with features.
    
    Returns:
    - If is_training=True: Tuple (X_tfidf_df, fitted_vectorizer)
    - If is_training=False: X_tfidf_df
    """
    print("[Features] Extracting Features (TF-IDF Vectorization)...")
    
    if is_training:
        if vectorizer is None:
            vectorizer = SymptomVectorizer()
        X_tfidf = vectorizer.fit_transform(df["symptoms"])
        print(f"[Features] Feature extraction completed: {X_tfidf.shape[1]} features extracted.")
        return X_tfidf, vectorizer
    else:
        if vectorizer is None:
            raise ValueError("[Error] Vectorizer must be provided for transforming in non-training mode.")
        X_tfidf = vectorizer.transform(df["symptoms"])
        print("[Features] Feature extraction (inference) completed.")
        return X_tfidf
