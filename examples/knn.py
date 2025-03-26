from sklearn.metrics import average_precision_score

from onad.model.unsupervised.dist.knn import KNN
from onad.stream.streamer import ParquetStreamer, Dataset
from onad.transform.scale import MinMaxScaler
from onad.utils.similarity.faiss_engine import FaissSimilaritySearchEngine

scaler = MinMaxScaler()

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
model = KNN(k=55, similarity_engine=engine)

pipeline = scaler | model

labels, scores = [], []
with ParquetStreamer(dataset=Dataset.FRAUD) as streamer:
    for i, (x, y) in enumerate(streamer):
        if y == 0 and i < 2_000:
            model.learn_one(x)
            continue
        model.learn_one(x)
        score = model.score_one(x)

        labels.append(y)
        scores.append(score)

print(f"PR_AUC: {round(average_precision_score(labels, scores), 3)}")  # 0.386
