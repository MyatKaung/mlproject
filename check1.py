import pandas as pd
from src.logger import logging
train_df = pd.read_csv("artifacts/train.csv")
test_df = pd.read_csv("artifacts/test.csv")

logging.info(f"Columns in train_df: {train_df.columns.tolist()}")
logging.info(f"Columns in test_df: {test_df.columns.tolist()}")