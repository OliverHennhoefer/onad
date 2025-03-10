from onad.metric.pr_auc import PRAUC
from onad.model.unsupervised.autoencoder import Autoencoder
from onad.stream.streamer import DataStreamer, Dataset
from onad.transform.scale import MinMaxScaler

scaler = MinMaxScaler()

model = Autoencoder(hidden_size=8, latent_size=4, learning_rate=0.005, seed=1)

pipeline = scaler | model

metric = PRAUC(n_thresholds=10)

with DataStreamer(dataset=Dataset.FRAUD) as streamer:
    for i, (x, y) in enumerate(streamer):
        if y == 0 and i < 50_000:
            pipeline.learn_one(x)
            continue
        pipeline.learn_one(x)
        score = pipeline.score_one(x)
        metric.update(y, score)

print(metric.get())
