from flask import Flask, render_template, request
import pandas as pd
from data_preprocessing import load_and_preprocess
from analysis import visualize_prediction_probabilities, knn_efficiency_comparison,train_lda_model, predict_vehicle_model,descriptive_stats, clustering_analysis, leaderboard_pca, leaderboard_lda, ranking_analysis_adjustments, comparative_ranking_visualization, lda_clustering_analysis,pca_clustering_analysis
import os

# Menangani masalah Matplotlib cache directory
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'matplotlib_cache')
if not os.path.exists(os.environ['MPLCONFIGDIR']):
    os.makedirs(os.environ['MPLCONFIGDIR'])

app = Flask(__name__)

# Load data
try:
    DATA = load_and_preprocess()
except FileNotFoundError as e:
    print(e)
    DATA = None

@app.route('/')
def home():
    return render_template('index.html', img_path='static/images/dataset-cover.jpg')

@app.route('/ranking')
def ranking():
    if DATA is None:
        return "Dataset tidak tersedia. Pastikan file CSV sudah diunduh dan diproses."

    # Statistik deskriptif
    stats = descriptive_stats(DATA)
    
    # Ranking berdasarkan kategori
    tabel = ranking_analysis_adjustments(DATA)
    if isinstance(tabel, pd.DataFrame):
        rankings = tabel.to_dict('records')  # Konversi DataFrame ke list of dictionaries
    else:
        return "Data ranking tidak dalam format yang diharapkan."

    # Visualisasi perbandingan ranking
    comparative_ranking_visualization(tabel)

    # Leaderboard PCA
    leaderboardPCA = leaderboard_pca(DATA)
    if isinstance(leaderboardPCA, pd.DataFrame):
        pca_items = []
        for _, row in leaderboardPCA.iterrows():
            pca_items.append({
                "rank": row["Rank"],
                "model": row["Vehicle Model"],
                "score": row["Composite Score"]
            })
    else:
        return "Leaderboard PCA tidak dalam format yang diharapkan."

    # Leaderboard LDA
    leaderboardLDA = leaderboard_lda(DATA)
    if isinstance(leaderboardLDA, pd.DataFrame):
        lda_items = []
        for _, row in leaderboardLDA.iterrows():
            lda_items.append({
                "rank": row["Rank"],
                "model": row["Vehicle Model"],
                "score": row["LDA_Score"]
            })
    else:
        return "Leaderboard LDA tidak dalam format yang diharapkan."
    
    return render_template(
        'ranking.html',
        stats=stats,
        rankings=rankings,  # Data ranking untuk tabel
        leaderboardPCA=pca_items,
        img_pathPCA='static/images/leaderboard_pca.png',
        leaderboardLDA=lda_items,
        img_pathLDA='static/images/leaderboard_lda.png',
        img_pathVisual='static/images/comparative_ranking.png'
    )


@app.route('/clustering')
def clustering():
    if DATA is None:
        return "Dataset tidak tersedia. Pastikan file CSV sudah diunduh dan diproses."

    lda_data = lda_clustering_analysis(DATA)
    pca_data = pca_clustering_analysis(DATA)
    return render_template('clustering.html', lda_data=lda_data.to_dict(orient='records'),pca_data=pca_data.to_dict(orient='records'), img_path='static/images/lda_clustering.png', img_path2='static/images/pca_clustering.png')
    
lda, scaler, accuracy = train_lda_model(DATA)
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        battery_capacity = float(request.form['battery_capacity'])
        charging_duration = float(request.form['charging_duration'])
        charging_rate = float(request.form['charging_rate'])

        predicted_model, class_probabilities = predict_vehicle_model(
            lda, scaler, battery_capacity, charging_duration, charging_rate
        )

        vehicle_models = lda.classes_
        probabilities = class_probabilities[0].tolist()
        visualize_prediction_probabilities(probabilities, vehicle_models)
        return render_template(
            'prediction.html',
            predicted_model=predicted_model,
            vehicle_models=vehicle_models,
            probabilities=probabilities,
            image = "static/images/prediction_probabilities.png",
            zip=zip  # Tambahkan zip ke konteks template
        )

    return render_template('prediction.html', predicted_model=None)

VEHICLE_MODELS = DATA['Vehicle Model'].unique().tolist()

@app.route('/check_efficiency', methods=['GET', 'POST'])
def check_efficiency():
    if request.method == 'POST':
        # Ambil input dari pengguna
        vehicle_model = request.form['vehicle_model']
        charging_duration = float(request.form['charging_duration'])
        start_soc = float(request.form['start_soc'])
        end_soc = float(request.form['end_soc'])
        vehicle_age = float(request.form['vehicle_age'])

        # Bandingkan dengan dataset menggunakan KNN
        efficiency_result = knn_efficiency_comparison(
            DATA, vehicle_model, charging_duration, start_soc, end_soc, vehicle_age
        )

        return render_template(
            'check_efficiency.html',
            efficiency_result=efficiency_result,
            input_data={
                'vehicle_model': vehicle_model,
                'charging_duration': charging_duration,
                'start_soc': start_soc,
                'end_soc': end_soc,
                'vehicle_age': vehicle_age
            },
            vehicle_models=VEHICLE_MODELS
        )

    return render_template('check_efficiency.html', efficiency_result=None, vehicle_models=VEHICLE_MODELS)

if __name__ == "__main__":
    app.run(debug=True)