import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import logging
import os

# Konfigurasi Matplotlib dan Logging
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-interaktif
logging.basicConfig(level=logging.INFO)

# Pastikan folder untuk gambar tersedia
if not os.path.exists('static/images'):
    os.makedirs('static/images')


def descriptive_stats(data):
    stats = data[['Energy Consumed (kWh)', 'Charging Duration (hours)', 'Charging Rate (kW)']].describe()
    return stats


def clustering_analysis(data):
    logging.info("Memulai PCA untuk clustering")
    data = data.copy()  # Buat salinan untuk menghindari modifikasi langsung
    pca = PCA(n_components=2)
    features = data[['Energy Consumed (kWh)', 'Charging Rate (kW)', 'Charging Cost (USD)']]
    pca_result = pca.fit_transform(features)

    logging.info("Menjalankan KMeans")
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(pca_result)
    data['Cluster'] = clusters

    silhouette_avg = silhouette_score(pca_result, clusters)
    logging.info(f"Silhouette Score: {silhouette_avg}")

    # Visualisasi clustering
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
    plt.title('Segmentasi Model Kendaraan (Clustering)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.savefig('static/images/clustering.png')
    plt.close()

    return silhouette_avg


def energy_usage_analysis(data):
    logging.info("Memulai analisis konsumsi energi")
    data = data.copy()  # Buat salinan
    model = LinearRegression()
    X = data[['Vehicle Age (years)']]
    y = data['Energy Consumed (kWh)']
    model.fit(X, y)
    score = model.score(X, y)
    logging.info(f"Skor regresi linear: {score}")

    # Plot hubungan
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Vehicle Age (years)'], data['Energy Consumed (kWh)'], alpha=0.5, label='Data')
    plt.plot(data['Vehicle Age (years)'], model.predict(X), color='red', label='Regresi Linear')
    plt.title('Hubungan Usia Kendaraan dan Konsumsi Energi')
    plt.xlabel('Vehicle Age (years)')
    plt.ylabel('Energy Consumed (kWh)')
    plt.legend()
    plt.savefig('static/images/energy_usage.png')
    plt.close()

    return score


def leaderboard_pca(data):
    logging.info("Menghitung leaderboard PCA")
    data = data.copy()  # Buat salinan
    numeric_columns = ['Energy Consumed (kWh)', 'Charging Duration (hours)', 'Charging Rate (kW)',
                       'Distance Driven (since last charge) (km)', 'Temperature (°C)']
    data.dropna(subset=numeric_columns, inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[numeric_columns])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    data['PCA1'] = pca_result[:, 0]
    data['PCA2'] = pca_result[:, 1]
    data['Composite Score'] = 0.7 * data['PCA1'] + 0.3 * data['PCA2']

    leaderboard = data.groupby('Vehicle Model')['Composite Score'].mean().reset_index()
    leaderboard['Rank'] = leaderboard['Composite Score'].rank(ascending=False)
    leaderboard = leaderboard.sort_values(by='Rank')

    # Visualisasi leaderboard
    plt.figure(figsize=(10, 6))
    plt.bar(leaderboard['Vehicle Model'], leaderboard['Composite Score'], color='blue', alpha=0.7)
    plt.xlabel('Vehicle Model')
    plt.ylabel('Composite Score')
    plt.title('EV Charging Efficiency Leaderboard (PCA)')
    plt.xticks(rotation=90)
    plt.savefig('static/images/leaderboard_pca.png')
    plt.close()

    return leaderboard


def leaderboard_lda(data):
    logging.info("Menghitung leaderboard LDA")
    data = data.copy()  # Buat salinan
    numeric_columns = ['Energy Consumed (kWh)', 'Charging Duration (hours)', 'Charging Rate (kW)',
                       'Distance Driven (since last charge) (km)', 'Temperature (°C)']
    target_column = 'Vehicle Model'
    data.dropna(subset=numeric_columns + [target_column], inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[numeric_columns])

    lda = LDA(n_components=1)
    lda_result = lda.fit_transform(X_scaled, data[target_column])
    data['LDA_Score'] = lda_result.flatten()

    leaderboard = data.groupby('Vehicle Model')['LDA_Score'].mean().reset_index()
    leaderboard['Rank'] = leaderboard['LDA_Score'].rank(ascending=False)
    leaderboard = leaderboard.sort_values(by='Rank')

    # Visualisasi leaderboard
    plt.figure(figsize=(10, 6))
    plt.bar(leaderboard['Vehicle Model'], leaderboard['LDA_Score'], color='green', alpha=0.7)
    plt.xlabel('Vehicle Model')
    plt.ylabel('LDA Score')
    plt.title('EV Charging Efficiency Leaderboard (LDA)')
    plt.xticks(rotation=90)
    plt.savefig('static/images/leaderboard_lda.png')
    plt.close()

    return leaderboard
