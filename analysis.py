import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
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
    
def lda_clustering_analysis(data):
    # Prepare data
    data = data.copy()  # Membuat salinan data asli untuk menjaga integritas
    numeric_columns = [
        'Energy Consumed (kWh)', 'Charging Duration (hours)', 'Charging Rate (kW)',
        'Distance Driven (since last charge) (km)', 'Temperature (°C)'
    ]
    target_column = 'Vehicle Model'  # Target kolom untuk LDA

    # Drop rows with missing values in required columns
    data.dropna(subset=numeric_columns + [target_column], inplace=True)  # Hapus baris dengan NaN

    # Standardize the data
    scaler = StandardScaler()  # Standarisasi data untuk memastikan mean = 0 dan std dev = 1
    X_scaled = scaler.fit_transform(data[numeric_columns])

    # Apply LDA
    lda = LDA(n_components=2)  # Mengurangi dimensi data menjadi 2 komponen utama
    X_lda = lda.fit_transform(X_scaled, data[target_column])  # LDA berdasarkan Vehicle Model

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)  # Melakukan clustering dengan 3 cluster
    data['Cluster'] = kmeans.fit_predict(X_lda)  # Menambahkan hasil prediksi cluster ke data

    # Map clusters to meaningful labels
    cluster_labels = {
        0: 'High Efficiency, Small Size',
        1: 'Medium Efficiency, Large Size',
        2: 'Low Efficiency, Medium Size'
    }
    data['Cluster Label'] = data['Cluster'].map(cluster_labels)  # Menambahkan label deskriptif

    # Visualize Clusters
    plt.figure(figsize=(10, 6))
    for cluster in range(kmeans.n_clusters):  # Iterasi setiap cluster
        cluster_points = X_lda[data['Cluster'] == cluster]  # Pilih titik data untuk cluster tersebut
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

    # Menambahkan elemen visualisasi
    plt.title('Vehicle Model Clustering with LDA')  # Judul plot
    plt.xlabel('LDA1')  # Label sumbu X
    plt.ylabel('LDA2')  # Label sumbu Y
    plt.legend()  # Menambahkan legenda
    plt.savefig('static/images/lda_clustering.png')  # Menyimpan grafik scatter
    plt.close()
    
    # Aggregate original data
    aggregation_rules = {
        "Energy Consumed (kWh)": "mean",
        "Charging Duration (hours)": "mean",
        "Charging Rate (kW)": "mean",
        "Distance Driven (since last charge) (km)": "mean",
        "Temperature (°C)": "mean"
    }

    # Mengelompokkan data berdasarkan Vehicle Model dan menghitung rata-rata
    aggregated_data = data.groupby("Vehicle Model").agg(aggregation_rules).reset_index()

    # Merge aggregated data with clustering labels
    cluster_info = data[["Vehicle Model", "Cluster Label"]].drop_duplicates()  # Ambil informasi cluster unik
    lda_data = pd.merge(aggregated_data, cluster_info, on="Vehicle Model", how="left")
    # Menghapus duplikasi jika ada
    lda_data = lda_data.drop_duplicates(subset="Vehicle Model")

    return lda_data  # Mengembalikan data hasil analisis


def pca_clustering_analysis(data):
    # Step 1: Standardize the Data
    numeric_columns = [
        'Energy Consumed (kWh)', 'Charging Duration (hours)', 'Charging Rate (kW)',
        'Distance Driven (since last charge) (km)', 'Temperature (°C)'
    ]

    # Menghapus baris yang memiliki nilai NaN pada kolom numerik
    data.dropna(subset=numeric_columns, inplace=True)

    # Standarisasi data numerik
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[numeric_columns])

    # Step 2: Apply PCA
    # Mengurangi dimensi data menjadi 2 komponen utama
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Step 3: K-Means Clustering
    # Melakukan clustering menggunakan algoritma K-Means (dengan 3 cluster)
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_pca)

    # Mapping cluster ke label yang lebih bermakna (opsional)
    cluster_labels = {
        0: 'High Efficiency, Small Size',
        1: 'Medium Efficiency, Large Size',
        2: 'Low Efficiency, Medium Size'
    }
    data['Cluster Label'] = data['Cluster'].map(cluster_labels)

    # Step 5: Visualize PCA Clusters
    plt.figure(figsize=(10, 6))
    for cluster in range(kmeans.n_clusters):
        # Memilih titik data untuk setiap cluster
        cluster_points = X_pca[data['Cluster'] == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

    # Menambahkan elemen visualisasi
    plt.title('Vehicle Model Clustering with PCA')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()
    plt.savefig('static/images/pca_clustering.png')  # Menyimpan grafik scatter
    plt.close()

    # Step 6: Aggregate Data (Optional Step for Analysis)
    # Menentukan aturan agregasi untuk setiap kolom numerik
    aggregation_rules = {
        "Energy Consumed (kWh)": "mean",
        "Charging Duration (hours)": "mean",
        "Charging Rate (kW)": "mean",
        "Distance Driven (since last charge) (km)": "mean",
        "Temperature (°C)": "mean"
    }

    # Mengelompokkan data berdasarkan Vehicle Model dan menghitung rata-rata
    aggregated_data = data.groupby("Vehicle Model").agg(aggregation_rules).reset_index()

    # Menggabungkan data agregat dengan informasi cluster
    cluster_info = data[["Vehicle Model", "Cluster Label"]].drop_duplicates()
    pca_data = pd.merge(aggregated_data, cluster_info, on="Vehicle Model", how="left")

    # Menghapus duplikasi jika ada
    pca_data = pca_data.drop_duplicates(subset="Vehicle Model")

    # Menghapus duplikasi berdasarkan Vehicle Model (jika ada)
    pca_data = pca_data.drop_duplicates(subset="Vehicle Model")

    return pca_data  # Mengembalikan data hasil analisis



# Fungsi untuk training LDA untuk fitur prediksi mobil
#LDA adalah model klasifikasi linier yang mencari kombinasi fitur untuk memaksimalkan perbedaan antar kelas
def train_lda_model(data): #parameter yang harus di isii [kapasitas baterai dalam kwh, lama durasi pengecharge an dengan jam, kecepatan pengisian dalam kilowatt]
    features = ["Battery Capacity (kWh)", "Charging Duration (hours)", "Charging Rate (kW)"] 
    target = "Vehicle Model" #target utama yang diprediksi di bagian vehicle model

    X = data[features] #fitur input
    y = data[target] #target prediksi

    #handle untuk nilai kosong (dari kolom" nya yang masih kosong, di isi menggunakan rata" )
    imputer = SimpleImputer(strategy='mean') #menggunakan library sklearn yaitu simpleImputer
    X = imputer.fit_transform(X)

    #data di standarisasi ulang agar semua model LDA bisa bekerja dengan optimal
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #membagi data jadi train 80% dan pengujian 20% pakai fungsi train_test_split
    #parameter random_state dibuat 42 agar hasil pembagian data konsisten
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    #melatih model LDA  
    lda = LDA()
    lda.fit(X_train, y_train)

    #akurasi model diukur menggunakan data pengujian (X_test, y_test) dengan metode score, yang menghitung proporsi prediksi yang benar
    accuracy = lda.score(X_test, y_test)
    
    #fungsi return Model LDA yang sudah di train, Scaler untuk preprocessing data baru, dan akurasi model
    return lda, scaler, accuracy

#prediksi model kendaraan berdasarkan input 
#param pakai LDA dari model yang sudah di train, Scaler = object StandardScaler yang telah dilatih , ...
def predict_vehicle_model(lda, scaler, battery_capacity, charging_duration, charging_rate):
    #data input disusun dalam bentuk array 2D diperlukan oleh metode scaler.transform() dan model LDA untuk melakukan prediksi
    input_data = scaler.transform([[battery_capacity, charging_duration, charging_rate]]) 
    #fungsi scaler.transform menormalisasi data input menggunakan scaler yang dilatih sebelumnya, jadi nilai-nilainya dalam scale yang sesuai
    
    #melakukan prediksi dengan model LDA digunakan untuk memprediksi kategori atau model kendaraan (lda.predict)
    prediction = lda.predict(input_data)
    #menghitung probabilitas kelas, lda.predict_proba menghasilkan probabilitas untuk setiap mobil
    #probabilitas menunjukkan seberapa keyakinan model terhadap setiap kemungkinan kelas 
    probabilities = lda.predict_proba(input_data)
    return prediction[0], probabilities #return prediction[0] untuk setiap Prediksi model kendaraan dengan probabilitas tertinggi
    #probabilities = Probabilitas untuk semua model kendaraan

#membuat visualisasi probabilitas prediksi model kendaraan dalam bentuk diagram batang (bar chart) dan menyimpannya sebagai file gambar
#parameternya probabilities array yang berisi probabilitas prediksi untuk setiap model kendaraan, vehicle_models  array berisi nama" model kendaraan yang sesuai dengan urutan probabilitas
def visualize_prediction_probabilities(probabilities, vehicle_models):
    plt.figure(figsize=(8, 6)) #dibuat ukuran 8 x 6 inch biar tidak terlalu besar
    plt.bar(vehicle_models, probabilities, alpha=0.7) #buat diagram batang, model kendaraan jadi sumbu x, probabilities jadi tinggi/sumbu y, alpha=0.7 Transparansi warna batang
    plt.xlabel("Vehicle Models") #label dan judul
    plt.ylabel("Prediction Probability")
    plt.title("Prediction Probabilities for Vehicle Models")
    plt.xticks(rotation=45) #putar label sumbu x agak miring untuk antisipasi nama panjang
    plt.tight_layout() #mengatur layout agar tidak terlalu terlalu sempit
    plt.savefig('static/images/prediction_probabilities.png') #save gambar untuk dikirim ke frontend
    plt.close() #close plt untuk save memory

# Function to classify efficiency into clusters (Low, Medium, High) with Battery Capacity and visualization
def kmeans_efficiency_classification(data, vehicle_model, charging_duration, start_soc, end_soc, vehicle_age, battery_capacity):
    # Filter dataset untuk model kendaraan yang sama
    filtered_data = data[data['Vehicle Model'] == vehicle_model]

    # Variabel input dan target
    features = ['Charging Duration (hours)', 'State of Charge (Start %)', 'State of Charge (End %)', 'Vehicle Age (years)', 'Battery Capacity (kWh)']

    # Hitung efisiensi dataset dengan rumus: (SOC End - SOC Start) / (Charging Duration * Battery Capacity)
    filtered_data['Efficiency'] = (
        (filtered_data['State of Charge (End %)'] - filtered_data['State of Charge (Start %)']) /
        (filtered_data['Charging Duration (hours)'] * (1 + 0.05 * filtered_data['Vehicle Age (years)']) * filtered_data['Battery Capacity (kWh)'])
    ) * 100  # Skalakan ke 100% (raw efficiency)

    # Data preprocessing: Ambil fitur input
    X = filtered_data[features]

    # Standarisasi data agar fitur memiliki skala yang sama
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Gunakan model K-Means untuk mengelompokkan data berdasarkan efisiensi
    kmeans = KMeans(n_clusters=3, random_state=42)
    filtered_data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Beri label kategori ke setiap cluster
    cluster_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    cluster_efficiency = filtered_data.groupby('Cluster')['Efficiency'].mean().sort_values().index
    for i, cluster in enumerate(cluster_efficiency):
        cluster_labels[cluster] = ['Low', 'Medium', 'High'][i]
    filtered_data['Efficiency Category'] = filtered_data['Cluster'].map(cluster_labels)

    # Buat input user berdasarkan parameter yang dimasukkan
    user_input = pd.DataFrame([[charging_duration, start_soc, end_soc, vehicle_age, battery_capacity]], columns=features)

    # Standarisasi input user
    user_scaled = scaler.transform(user_input)

    # Prediksi cluster input user menggunakan model K-Means
    user_cluster = kmeans.predict(user_scaled)[0]

    # Tentukan kategori efisiensi untuk input user
    user_efficiency_category = cluster_labels[user_cluster]

    # Visualisasi clustering
    plt.figure(figsize=(10, 6))
    for cluster in range(3):
        cluster_points = filtered_data[filtered_data['Cluster'] == cluster]
        plt.scatter(
            cluster_points['Charging Duration (hours)'],
            cluster_points['Efficiency'],
            label=f"Cluster {cluster} ({cluster_labels[cluster]})"
        )
    plt.scatter(
        charging_duration, 
        (end_soc - start_soc) / (charging_duration * (1 + 0.05 * vehicle_age) * battery_capacity) * 100,
        color='red', label='User Input', edgecolors='black', linewidths=2, s=100
    )
    plt.title("Clustering Visualization with Battery Capacity")
    plt.xlabel("Charging Duration (hours)")
    plt.ylabel("Efficiency (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/clustering_visualization.png')
    plt.close()

    # Output perbandingan
    comparison = {
        'vehicle_model': vehicle_model,
        'efficiency_category': user_efficiency_category,
        'visualization_path': 'static/images/clustering_visualization.png'
    }

    return comparison