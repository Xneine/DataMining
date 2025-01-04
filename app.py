from flask import Flask, render_template
import pandas as pd
from data_preprocessing import load_and_preprocess
from analysis import pca_df_func, pca_user_type_analysis, pca_distribution_analysis, pca_summary_vehicle_model, energy_usage_analysis,descriptive_stats, clustering_analysis, leaderboard_pca, leaderboard_lda, ranking_analysis_adjustments, comparative_ranking_visualization, lda_clustering_analysis,pca_clustering_analysis
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

@app.route('/insights')
def insights():
    if DATA is None:
        return "Dataset tidak tersedia. Pastikan file CSV sudah diunduh dan diproses."

    energy_score = energy_usage_analysis(DATA)
     
    pca_user_type = pca_user_type_analysis(DATA)
    pca_distribution = pca_distribution_analysis(DATA)
    pca_summary_vehicle = pca_summary_vehicle_model(pca_df_func(DATA))
    
    return render_template(
        'insights.html', 
        energy_score=energy_score, 
        img_path='static/images/energy_usage.png',
        pca_user_type_analysis = pca_user_type,
        pca_distribution_analysis = pca_distribution,
        pca_summary_vehicle_model = pca_summary_vehicle,
        img_path2 = 'static/images/pca_user_type.png',
        img_path3 = 'static/images/pca_distribution.png'
        )

if __name__ == "__main__":
    app.run(debug=True)