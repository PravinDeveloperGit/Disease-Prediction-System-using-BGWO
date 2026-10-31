import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class SymptomVectorizer:
    def __init__(self, max_features=None):
        """
        Custom vectorizer using TF-IDF.
        """
        # Split by comma or whitespace, and use standard english stop words.
        # token_pattern matches words of at least 1 character to keep abbreviations/short symptoms.
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            token_pattern=r'(?u)\b\w+\b',
            max_features=max_features
        )
        
    def fit(self, texts: pd.Series):
        """
        Fits the TF-IDF vectorizer on the symptoms text.
        """
        self.vectorizer.fit(texts)
        return self
        
    def transform(self, texts: pd.Series) -> pd.DataFrame:
        """
        Transforms the symptoms text to TF-IDF DataFrame with feature names as columns.
        """
        tfidf_matrix = self.vectorizer.transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=texts.index)
        
    def fit_transform(self, texts: pd.Series) -> pd.DataFrame:
        """
        Fits and transforms.
        """
        self.fit(texts)
        return self.transform(texts)
        
    def save(self, filepath: str):
        """
        Saves the fitted vectorizer object.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filepath: str):
        """
        Loads a saved SymptomVectorizer.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)
