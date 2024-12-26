import kagglehub
import os
import pandas as pd

def download_data():
    # Download dataset dari Kaggle
    path = kagglehub.dataset_download("valakhorasani/electric-vehicle-charging-patterns")
    csv_file = os.path.join(path, 'ev_charging_patterns.csv')
    data = pd.read_csv(csv_file)
    print("Dataset berhasil diunduh di:", path)
    return data

if __name__ == "__main__":
    data = download_data()