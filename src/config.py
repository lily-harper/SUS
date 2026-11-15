from pathlib import Path

from sklearn.model_selection import PredefinedSplit

# seed 
SEED = 67

# paths 
CLEAN_DATA = "data/clean_runs.csv" 
TRAIN_DATA = "data/train.csv"

PARAMS_PATH = Path(__file__).resolve().parent / "params" 

from src.prep import temporal_split, make_xy
import pandas as pd 

def data_stuff(path):
    df = pd.read_csv(path)
    train, val = temporal_split(df, 2022)
    
    X_train, y_train = make_xy(train)
    X_val, y_val = make_xy(val)

    return X_train, y_train, X_val, y_val

def tune_set(X_train, y_train, X_val, y_val):
    X_tuning = pd.concat([X_train, X_val], ignore_index=True)
    y_tuning = pd.concat([y_train, y_val], ignore_index=True)
    validation_fold = [-1] * len(X_train) + [0] * len(X_val)
    splitter = PredefinedSplit(test_fold=validation_fold)

    return X_tuning, y_tuning, validation_fold, splitter
