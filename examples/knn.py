from onad.metric.pr_auc import PRAUC
from onad.model.knn import KNN
from onad.transformer.scaler.min_max import MinMaxScaler
from onad.utils.similarity.faiss_engine import FaissSimilaritySearchEngine
from onad.utils.streamer.datasets import Dataset
from onad.utils.streamer.streamer import NPZStreamer

scaler = MinMaxScaler()

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
knn = KNN(k=55, similarity_engine=engine)

pipeline = scaler | knn

metric = PRAUC(n_thresholds=10)

with NPZStreamer(Dataset.SHUTTLE) as streamer:
    for x, y in streamer:
        pipeline.learn_one(x)
        score = pipeline.score_one(x)
        metric.update(y, score)

print(metric.get())
