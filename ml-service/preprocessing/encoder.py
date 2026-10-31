import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DiseaseEncoder:
    def __init__(self):
        """
        Wrapper around LabelEncoder to encode disease targets.
        """
        self.encoder = LabelEncoder()
        
    def fit(self, labels: pd.Series):
        self.encoder.fit(labels)
        return self
        
    def transform(self, labels: pd.Series):
        return self.encoder.transform(labels)
        
    def fit_transform(self, labels: pd.Series):
        return self.encoder.fit_transform(labels)
        
    def inverse_transform(self, encoded_labels):
        return self.encoder.inverse_transform(encoded_labels)
        
    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filepath: str):
        with open(filepath, "rb") as f:
            return pickle.load(f)
