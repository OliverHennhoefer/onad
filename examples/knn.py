from onad.metric.pr_auc import PRAUC
from onad.model.knn import KNN
from onad.transformer.scaler.normalize import MinMaxScaler
from onad.utils.similarity.faiss_engine import FaissSimilaritySearchEngine
from onad.utils.streamer.streamer import NPZStreamer

scaler = MinMaxScaler()

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
model = KNN(k=55, similarity_engine=engine)

pipeline = scaler | model

metric = PRAUC(n_thresholds=10)

with NPZStreamer("./data/shuttle.npz") as streamer:
    for x, y in streamer:
        pipeline.learn_one(x)
        score = pipeline.score_one(x)
        metric.update(y, score)

print(metric.get())
