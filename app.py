from flask import Flask, render_template
import pandas as pd
from data_preprocessing import load_and_preprocess
from analysis import energy_usage_analysis,descriptive_stats, clustering_analysis, leaderboard_pca, leaderboard_lda
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
    return render_template('index.html')

@app.route('/ranking')
def ranking():
    if DATA is None:
        return "Dataset tidak tersedia. Pastikan file CSV sudah diunduh dan diproses."

    stats = descriptive_stats(DATA)
    
    # Process PCA leaderboard
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

    # Process LDA leaderboard
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
        leaderboardPCA=pca_items, 
        img_pathPCA='static/images/leaderboard_pca.png',
        leaderboardLDA=lda_items, 
        img_pathLDA='static/images/leaderboard_lda.png'
    )

@app.route('/clustering')
def clustering():
    if DATA is None:
        return "Dataset tidak tersedia. Pastikan file CSV sudah diunduh dan diproses."

    silhouette_score = clustering_analysis(DATA)
    return render_template('clustering.html', silhouette_score=silhouette_score, img_path='static/images/clustering.png')

@app.route('/insights')
def insights():
    if DATA is None:
        return "Dataset tidak tersedia. Pastikan file CSV sudah diunduh dan diproses."

    energy_score = energy_usage_analysis(DATA)
    return render_template('insights.html', energy_score=energy_score, img_path='static/images/energy_usage.png')

if __name__ == "__main__":
    app.run(debug=True)