from onad.metric.pr_auc import PRAUC
from onad.model.autoencoder import Autoencoder
from onad.transformer.scaler.normalize import MinMaxScaler
from onad.utils.streamer.datasets import Dataset
from onad.utils.streamer.streamer import NPZStreamer

scaler = MinMaxScaler()

model = Autoencoder(hidden_size=16, latent_size=4, learning_rate=0.05, seed=1)
pipeline = scaler | model

metric = PRAUC(n_thresholds=10)

with NPZStreamer(Dataset.FRAUD) as streamer:
    for x, y in streamer:
        pipeline.learn_one(x)
        score = pipeline.score_one(x)
        metric.update(y, score)

print(metric.get())
