import pandas as pd
import os

def load_and_preprocess():
    file_path = os.path.join(os.getcwd(), "ev_charging_patterns.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError("File ev_charging_patterns.csv tidak ditemukan. Pastikan file berada di direktori proyek.")

    # Load dataset
    data = pd.read_csv(file_path)

    # Preprocessing data (misalnya menangani missing values)
    data.dropna(inplace=True)
    
    return data

if __name__ == "__main__":
    try:
        data = load_and_preprocess()
        print(data.head())
    except FileNotFoundError as e:
        print(e)