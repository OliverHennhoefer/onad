from sklearn.metrics import average_precision_score

from onad.model.unsupervised.autoencoder import Autoencoder
from onad.stream.streamer import ParquetStreamer, Dataset
from onad.transform.scale import MinMaxScaler
from onad.utils.architecture.autoencoder import VanillaAutoencoder

autoencoder = Autoencoder(
    model=VanillaAutoencoder(input_size=29), learning_rate=0.0001,
)

scaler = MinMaxScaler()

model = scaler | autoencoder

labels, scores = [], []
with ParquetStreamer(dataset=Dataset.FRAUD) as streamer:
    for i, (x, y) in enumerate(streamer):
        if i < 100_000:
            model.learn_one(x)
            continue
        model.learn_one(x)
        score = model.score_one(x)

        labels.append(y)
        scores.append(score)

print(f"PR_AUC: {round(average_precision_score(labels, scores), 3)}")  # 0.236
