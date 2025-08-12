from sklearn.metrics import average_precision_score

from onad.model.unsupervised.distance.knn import KNN
from onad.stream.streamer import ParquetStreamer, Dataset
from onad.transform.scale import MinMaxScaler
from onad.transform.pca import IncrementalPCA
from onad.utils.similarity.faiss_engine import FaissSimilaritySearchEngine

# Original pipeline: scaler | model (no PCA)
scaler = MinMaxScaler()
engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
model = KNN(k=55, similarity_engine=engine)
pipeline1 = scaler | model

# PCA pipeline: scaler | pca | model (with dimensionality reduction)
scaler2 = MinMaxScaler()
pca = IncrementalPCA(n_components=3, n0=100)  # Reduce to 3 components, warmup with 100 samples
engine2 = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
model2 = KNN(k=55, similarity_engine=engine2)
pipeline2 = scaler2 | pca | model2

# Collect results for both pipelines
labels, scores1, scores2 = [], [], []

with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
    for i, (x, y) in enumerate(streamer):
        # Learning phase: train both pipelines on normal samples
        if y == 0 and i < 2_000:
            pipeline1.learn_one(x)
            pipeline2.learn_one(x)
            continue
        
        pipeline1.learn_one(x)
        pipeline2.learn_one(x)
        
        score1 = pipeline1.score_one(x)  # Simple pipeline
        score2 = pipeline2.score_one(x)  # PCA pipeline
        
        if score1 is not None and score2 is not None:
            labels.append(y)
            scores1.append(score1)
            scores2.append(score2)

# Performance comparison
if len(labels) > 0:
    pr_auc1 = average_precision_score(labels, scores1)
    pr_auc2 = average_precision_score(labels, scores2)
    
    print("=== Performance Comparison ===")
    print(f"Original Pipeline (Scaler + KNN):     PR-AUC = {round(pr_auc1, 3)}")
    print(f"PCA Pipeline (Scaler + PCA + KNN):    PR-AUC = {round(pr_auc2, 3)}")
    print(f"Difference (PCA - Original):          {round(pr_auc2 - pr_auc1, 3)}")
    print(f"Relative improvement:                 {round((pr_auc2 - pr_auc1) / pr_auc1 * 100, 1)}%")
    print(f"Samples evaluated: {len(labels)}")
else:
    print("No samples were scored by both pipelines.")
