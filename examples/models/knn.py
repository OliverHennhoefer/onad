from sklearn.metrics import average_precision_score

from onad.dataset import Dataset, load
from onad.model.unsupervised.distance.knn import KNN
from onad.transform.scale import MinMaxScaler
from onad.utils.similarity.faiss_engine import FaissSimilaritySearchEngine

scaler = MinMaxScaler()

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
model = KNN(k=45, similarity_engine=engine)

pipeline = scaler | model
labels, scores = [], []

# Load dataset using new API
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset.stream()):
    if i < 2_000:
        if y == 0:
            model.learn_one(x)
        continue
    model.learn_one(x)
    score = model.score_one(x)

    labels.append(y)
    scores.append(score)

print(f"PR_AUC: {round(average_precision_score(labels, scores), 3)}")  # 0.848
