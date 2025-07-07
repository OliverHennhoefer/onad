from sklearn.metrics import average_precision_score

from onad.model.unsupervised.forest.online_iforest import BoundedRandomProjectionOnlineIForest
from onad.stream.streamer import ParquetStreamer, Dataset

model = BoundedRandomProjectionOnlineIForest(
    num_trees=20,
    window_size=256,
    branching_factor=2,
    max_leaf_samples=2,
    type='fixed',
    subsample=1.0,
    n_jobs=1
)

labels, scores = [], []

with ParquetStreamer(dataset=Dataset.SHUTTLE) as streamer:
    for i, (x, y) in enumerate(streamer): 
        if y == 0 and i < 10_000:
            model.learn_one(x)
            continue
        
        model.learn_one(x)
        score = model.score_one(x)
        
        labels.append(y)
        scores.append(score)

print(f"PR_AUC: {round(average_precision_score(labels, scores), 3)}")
