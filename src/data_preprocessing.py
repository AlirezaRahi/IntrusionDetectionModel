# 2. src/data_preprocessing.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import glob

class DataPreprocessor:
    def __init__(self, data_path):
        """
        Data preprocessing class
        Args:
            data_path: Path to dataset folder
        """
        self.data_path = Path(data_path)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.attack_types = ['Arp_Spoofing', 'BotNet_DDOS', 'HTTP_Flood', 'ICMP_Flood', 
                            'MQTT_Flood', 'Port_Scanning', 'TCP_Flood', 'UDP_Flood', 'Normal']
        
    def load_and_combine_datasets(self):
        """
        Load and combine all CSV files from different folders
        """
        print("Starting dataset loading...")
        all_data = []
        
        # Path to Datasets folder
        datasets_path = self.data_path / 'Datasets'
        
        if not datasets_path.exists():
            datasets_path = self.data_path  # If Datasets folder doesn't exist
        
        # Search for all CSV files in all subfolders
        csv_files = glob.glob(str(datasets_path / '**' / '*_combined.csv'), recursive=True)
        csv_files.extend(glob.glob(str(datasets_path / '**' / '*.csv'), recursive=True))
        
        # Remove duplicate files (based on filename)
        unique_files = {}
        for file in csv_files:
            file_name = os.path.basename(file)
            if file_name not in unique_files:
                unique_files[file_name] = file
        
        csv_files = list(unique_files.values())
        print(f"Found {len(csv_files)} unique CSV files")
        
        for file in csv_files:
            try:
                df = pd.read_csv(file, nrows=100000)  # Limit records to prevent memory issues
                
                # Detect attack type from filename or folder
                attack_type = self._detect_attack_type(file)
                if attack_type and attack_type != 'Unknown':
                    df['Attack_Type'] = attack_type
                    all_data.append(df)
                    print(f"File {os.path.basename(file)} loaded with {len(df)} records - Type: {attack_type}")
                elif 'Farm-Flow' in str(file) or 'Farm_Flow' in str(file):
                    # Main Farm-Flow files might have Attack_Type column
                    if 'Attack_Type' in df.columns:
                        all_data.append(df)
                        print(f"File {os.path.basename(file)} loaded with {len(df)} records")
            except Exception as e:
                print(f"Error loading file {file}: {e}")
        
        if not all_data:
            raise Exception("No CSV files found for loading!")
        
        # Combine all dataframes
        print("Combining datasets...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicate records
        combined_df = combined_df.drop_duplicates()
        
        print(f"Total records after combination: {len(combined_df)}")
        print(f"Available columns: {combined_df.columns.tolist()}")
        
        # Display class distribution
        if 'Attack_Type' in combined_df.columns:
            print("\nClass distribution:")
            print(combined_df['Attack_Type'].value_counts())
        
        return combined_df
    
    def _detect_attack_type(self, file_path):
        """
        Detect attack type from file path
        """
        file_path_str = str(file_path)
        file_name = os.path.basename(file_path_str)
        
        for attack in self.attack_types:
            if attack in file_path_str or attack in file_name:
                return attack
        return 'Unknown'
    
    def preprocess_data(self, df):
        """
        Preprocess data
        """
        print("Starting data preprocessing...")
        
        # If Attack_Type column doesn't exist, use other columns
        if 'Attack_Type' not in df.columns:
            if 'Label' in df.columns:
                df['Attack_Type'] = df['Label']
            elif 'is_attack' in df.columns:
                df['Attack_Type'] = df['is_attack'].map({0: 'Normal', 1: 'Attack'})
            else:
                # If no label column exists, consider all as Normal
                df['Attack_Type'] = 'Normal'
        
        # Remove records with invalid Attack_Type
        df = df[df['Attack_Type'].notna()]
        df = df[df['Attack_Type'] != 'Unknown']
        
        print(f"After removing invalid classes: {len(df)} records")
        
        # Select numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Columns to drop
        columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label', 
                          'Fwd Header Length.1', 'id.orig_h', 'id.resp_h', 'proto', 
                          'service', 'conn_state', 'history', 'tunnel_parents', 'traffic']
        
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        # Separate features and labels
        X = df.drop(columns=existing_columns_to_drop + ['Attack_Type'], errors='ignore')
        y = df['Attack_Type']
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Replace inf values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Remove columns where all values are NaN
        X = X.dropna(axis=1, how='all')
        
        # Fill NaN values with column mean
        X = X.fillna(X.mean())
        
        # If still NaN exists, fill with 0
        X = X.fillna(0)
        
        print(f"Number of features after preprocessing: {X.shape[1]}")
        print(f"Number of classes: {len(y.unique())}")
        print(f"Available classes: {y.unique()}")
        
        return X, y
    
    def split_and_scale(self, X, y, test_size=0.2, val_size=0.1):
        """
        Split data into train, validation, test and scale
        """
        print("Starting data splitting and scaling...")
        
        # Check data existence
        if len(X) == 0:
            raise Exception("No data available for splitting!")
        
        # encode labels
        try:
            y_encoded = self.label_encoder.fit_transform(y)
            print(f"Identified classes: {self.label_encoder.classes_}")
        except Exception as e:
            print(f"Error in label encoding: {e}")
            # If problematic, consider all as one class
            y_encoded = np.zeros(len(y))
            self.label_encoder.classes_ = np.array(['Unknown'])
        
        # split into train+val and test
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, 
                stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
            )
        except Exception as e:
            print(f"Error in data splitting with stratify: {e}")
            # Try again without stratify
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42
            )
        
        # split train into train and validation
        val_relative_size = val_size / (1 - test_size)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_relative_size, random_state=42,
                stratify=y_temp if len(np.unique(y_temp)) > 1 else None
            )
        except Exception as e:
            print(f"Error in training data splitting: {e}")
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_relative_size, random_state=42
            )
        
        # scale features
        try:
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
        except Exception as e:
            print(f"Error in scaling: {e}")
            # If scaling fails, use original data
            X_train_scaled = X_train.values
            X_val_scaled = X_val.values
            X_test_scaled = X_test.values
        
        print(f"Number of training samples: {len(X_train)}")
        print(f"Number of validation samples: {len(X_val)}")
        print(f"Number of test samples: {len(X_test)}")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled), (y_train, y_val, y_test)
    
    def save_artifacts(self, save_dir):
        """
        Save scaler and label encoder
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, save_path / 'scaler.pkl')
        joblib.dump(self.label_encoder, save_path / 'label_encoder.pkl')
        print(f"Preprocessors saved in {save_path}")