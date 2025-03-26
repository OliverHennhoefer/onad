from sklearn.metrics import average_precision_score
from torch import optim, nn

from onad.model.unsupervised.deep.autoencoder import Autoencoder
from onad.stream.streamer import ParquetStreamer, Dataset
from onad.transform.scale import MinMaxScaler
from onad.utils.architecture.autoencoder import VanillaAutoencoder

model = VanillaAutoencoder(input_size=9, seed=1)

autoencoder = Autoencoder(
    model=model,
    optimizer=optim.Adam(model.parameters(), lr=0.001, weight_decay=0),
    criterion=nn.MSELoss(),
)

scaler = MinMaxScaler()

model = scaler | autoencoder

labels, scores = [], []
with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
    for i, (x, y) in enumerate(streamer):
        if i < 10_000:
            model.learn_one(x)
            continue
        model.learn_one(x)
        score = model.score_one(x)

        labels.append(y)
        scores.append(score)

print(f"PR_AUC: {round(average_precision_score(labels, scores), 3)}")  # 0.235
