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

def ranking_analysis_adjustments(data):
    data['Charging Efficiency'] = (data['Energy Consumed (kWh)'] / data['Charging Duration (hours)']) / data['Battery Capacity (kWh)']
    data['Range per Charge'] = data['Distance Driven (since last charge) (km)'] / data['Charging Duration (hours)']
    data['Cost per kWh'] = data['Charging Cost (USD)'] / data['Energy Consumed (kWh)']
    data['Charging Speed'] = data['Charging Rate (kW)'] / data['Charging Duration (hours)']
    data['Battery Capacity per Hour'] = data['Battery Capacity (kWh)'] / data['Charging Duration (hours)']

    # Penyesuaian berdasarkan umur kendaraan
    data['Adjusted Charging Efficiency'] = data['Charging Efficiency'] * (1 - (0.02 * data['Vehicle Age (years)']))
    data['Adjusted Cost per kWh'] = data['Cost per kWh'] * (1 + (0.01 * data['Vehicle Age (years)']))
    data['Adjusted Range per Charge'] = data['Range per Charge'] * (1 - (0.01 * data['Vehicle Age (years)']))

    # Memberikan ranking untuk setiap kategori
    data['Efficiency Ranking'] = data['Adjusted Charging Efficiency'].rank(ascending=False)
    data['Range Ranking'] = data['Adjusted Range per Charge'].rank(ascending=False)
    data['Cost Ranking'] = data['Adjusted Cost per kWh'].rank(ascending=True)
    data['Speed Ranking'] = data['Charging Speed'].rank(ascending=False)
    data['Capacity Ranking'] = data['Battery Capacity per Hour'].rank(ascending=False)

    # Mengelompokkan berdasarkan model kendaraan dan mengambil rata-rata per kategori
    df_grouped = data.groupby('Vehicle Model').agg({
        'Efficiency Ranking': 'mean',
        'Range Ranking': 'mean',
        'Cost Ranking': 'mean',
        'Speed Ranking': 'mean',
        'Capacity Ranking': 'mean'
    }).reset_index()

    # Menambahkan kolom Overall
    df_grouped['Overall Ranking'] = df_grouped[['Efficiency Ranking', 'Range Ranking', 'Cost Ranking', 'Speed Ranking', 'Capacity Ranking']].sum(axis=1)
    df_grouped = df_grouped.sort_values('Overall Ranking', ascending=True)

    return df_grouped

def comparative_ranking_visualization(df_grouped):
    categories = ['Efficiency Ranking', 'Range Ranking', 'Cost Ranking', 'Speed Ranking', 'Capacity Ranking', 'Overall Ranking']

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.1
    index = range(len(df_grouped))

    for i, category in enumerate(categories):
        ax.barh([p + bar_width * i for p in index], df_grouped[category], bar_width, label=category)

    ax.set_yticks([p + bar_width * (len(categories) // 2) for p in index])
    ax.set_yticklabels(df_grouped['Vehicle Model'])
    ax.set_xlabel('Ranking Value')
    ax.set_title('Vehicle Model Rankings Comparison')

    ax.legend(title='Ranking Categories', loc='upper right')
    plt.tight_layout()
    plt.savefig('static/images/comparative_ranking.png')
    plt.close()
    
def pca_user_type_analysis(data):
    numerical_cols = [
        'Battery Capacity (kWh)', 'Energy Consumed (kWh)', 'Charging Duration (hours)',
        'Charging Rate (kW)', 'Charging Cost (USD)', 'State of Charge (Start %)',
        'State of Charge (End %)', 'Distance Driven (since last charge) (km)',
        'Temperature (°C)', 'Vehicle Age (years)'
    ]

    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numerical_cols])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    pca_df['User Type'] = data['User Type']

    plt.figure(figsize=(8, 6))
    for user_type in pca_df['User Type'].unique():
        subset = pca_df[pca_df['User Type'] == user_type]
        plt.scatter(subset['PCA1'], subset['PCA2'], label=user_type)

    plt.title('PCA Result for User Type')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/images/pca_user_type.png')
    plt.close()

    explained_variance = pca.explained_variance_ratio_
    return explained_variance

def pca_distribution_analysis(data):
    numerical_cols = [
        'Battery Capacity (kWh)', 'Energy Consumed (kWh)', 'Charging Duration (hours)',
        'Charging Rate (kW)', 'Charging Cost (USD)', 'State of Charge (Start %)',
        'State of Charge (End %)', 'Distance Driven (since last charge) (km)',
        'Temperature (°C)', 'Vehicle Age (years)'
    ]

    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numerical_cols])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    pca_df['Vehicle Model'] = data['Vehicle Model']
    pca_df['User Type'] = data['User Type']

    pca_grouped = pca_df.groupby(['Vehicle Model', 'User Type']).size().unstack(fill_value=0)

    percentage_pca = pca_grouped.div(pca_grouped.sum(axis=1), axis=0) * 100
    ax = percentage_pca.plot(kind='bar', stacked=True, figsize=(10, 6))

    plt.title('PCA Distribution by Vehicle Model and User Type')
    plt.xlabel('Vehicle Model')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.legend(title='User Type')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('static/images/pca_distribution.png')
    plt.close()

    return pca_grouped

def pca_df_func(data):
    numerical_cols = [
        'Battery Capacity (kWh)', 'Energy Consumed (kWh)', 'Charging Duration (hours)',
        'Charging Rate (kW)', 'Charging Cost (USD)', 'State of Charge (Start %)',
        'State of Charge (End %)', 'Distance Driven (since last charge) (km)',
        'Temperature (°C)', 'Vehicle Age (years)'
    ]

    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numerical_cols])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    pca_df['Vehicle Model'] = data['Vehicle Model']
    pca_df['User Type'] = data['User Type']

    return pca_df

def pca_summary_vehicle_model(pca_df):
    pca_summary = pca_df.groupby('Vehicle Model')[['PCA1', 'PCA2']].agg(['mean', 'std'])
    pca_summary.columns = ['PCA1 Mean', 'PCA1 Std', 'PCA2 Mean', 'PCA2 Std']
    return pca_summary