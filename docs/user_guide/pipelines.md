# Pipelines

Pipelines chain transformers and models with `|`.

## Example: scaler + KNN

```python
from aberrant.model.distance import KNN
from aberrant.transform.preprocessing import MinMaxScaler
from aberrant.utils.similar.faiss_engine import FaissSimilaritySearchEngine

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
pipeline = MinMaxScaler() | KNN(k=45, similarity_engine=engine)
```

## Example: scaler + PCA + KNN

```python
from aberrant.model.distance import KNN
from aberrant.transform.preprocessing import StandardScaler
from aberrant.transform.projection import IncrementalPCA
from aberrant.utils.similar.faiss_engine import FaissSimilaritySearchEngine

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
pipeline = StandardScaler() | IncrementalPCA(n_components=3, n0=100) | KNN(
    k=45,
    similarity_engine=engine,
)
```

## Operational tip

Keep thresholding outside the model pipeline when you need fast runtime policy
changes (for example, changing alert sensitivity without retraining).
