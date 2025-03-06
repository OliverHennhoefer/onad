from onad.metric.pr_auc import PRAUC
from onad.model.asd_iforest import ASDIsolationForest
from onad.model.autoencoder import Autoencoder
from onad.transformer.scaler.normalize import MinMaxScaler
from onad.utils.streamer.streamer import ZippedCSVStreamer

scaler = MinMaxScaler()

model = Autoencoder(
    hidden_size=8, latent_size=4, learning_rate=0.005, seed=1
)

pipeline = scaler | model

pipeline = ASDIsolationForest(n_estimators=750, max_samples=2750, seed=1)

metric = PRAUC(n_thresholds=10)

with ZippedCSVStreamer("./data/194369_entity_1.zip") as streamer:
    for i, (x, y) in enumerate(streamer):
        if y == 0 and i < 194369:
            pipeline.learn_one(x)
            continue
        pipeline.learn_one(x)
        score = pipeline.score_one(x)
        metric.update(y, score)

print(metric.get())
