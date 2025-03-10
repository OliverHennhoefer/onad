from onad.metric.pr_auc import PRAUC
from onad.model.unsupervised.knn import KNN
from onad.stream.streamer import DataStreamer, Dataset
from onad.transform.scale import MinMaxScaler
from onad.utils.similarity.faiss_engine import FaissSimilaritySearchEngine

scaler = MinMaxScaler()

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
model = KNN(k=55, similarity_engine=engine)

pipeline = scaler | model

metric = PRAUC(n_thresholds=10)

with DataStreamer(dataset=Dataset.FRAUD) as streamer:
    for x, y in streamer:
        pipeline.learn_one(x)
        score = pipeline.score_one(x)
        metric.update(y, score)

print(metric.get())
