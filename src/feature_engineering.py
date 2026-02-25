# Intrusion Detection Model/src/feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureEngineer:
    def __init__(self):
        self.selected_features = None
        self.selector = None
    
    def extract_statistical_features(self, X_sequence):
        """
        Extract statistical features from sequences
        """
        print("Extracting statistical features...")
        
        n_samples, seq_len, n_features = X_sequence.shape
        
        # Statistical features for each sequence
        statistical_features = np.zeros((n_samples, n_features * 4))
        
        for i in range(n_samples):
            seq_data = X_sequence[i]
            statistical_features[i, :n_features] = np.mean(seq_data, axis=0)
            statistical_features[i, n_features:2*n_features] = np.std(seq_data, axis=0)
            statistical_features[i, 2*n_features:3*n_features] = np.max(seq_data, axis=0)
            statistical_features[i, 3*n_features:4*n_features] = np.min(seq_data, axis=0)
        
        print(f"Statistical features extracted with shape {statistical_features.shape}")
        return statistical_features
    
    def select_best_features(self, X, y, k=50):
        """
        Select best features
        """
        print(f"Selecting top {k} features...")
        
        self.selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = self.selector.fit_transform(X, y)
        
        # Save indices of selected features
        self.selected_features = self.selector.get_support(indices=True)
        
        print(f"Features reduced to {X_selected.shape[1]}")
        return X_selected
    
    def create_flow_features(self, df):
        """
        Create flow-related features
        """
        print("Creating flow features...")
        
        # Calculate aggregate features for each flow
        if 'Flow Duration' in df.columns:
            flow_duration_min = df['Flow Duration'].min()
            flow_duration_max = df['Flow Duration'].max()
            if flow_duration_max > flow_duration_min:
                df['Flow_Duration_norm'] = (df['Flow Duration'] - flow_duration_min) / \
                                        (flow_duration_max - flow_duration_min)
        
        # Ratio of forward to backward packets
        if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
            df['Fwd_to_Bwd_Ratio'] = df['Total Fwd Packets'] / (df['Total Backward Packets'] + 1)
        
        # Packet rate
        if 'Flow Duration' in df.columns and 'Total Fwd Packets' in df.columns:
            df['Packet_Rate'] = df['Total Fwd Packets'] / (df['Flow Duration'] + 1)
        
        return df