# Model Evaluation

Evaluating anomaly detection models requires specialized metrics and techniques due to the unique challenges of imbalanced datasets and streaming scenarios. This guide covers evaluation strategies, metrics, and best practices for ONAD models.

## Evaluation Challenges in Anomaly Detection

### Imbalanced Data
- Anomalies are typically rare (1-5% of data)
- Traditional accuracy metrics are misleading
- Need metrics that focus on minority class performance

### Streaming Context
- No fixed test set available upfront
- Performance may drift over time
- Need online evaluation methods

### Label Scarcity
- True anomaly labels are often unavailable
- Semi-supervised evaluation approaches needed
- Proxy metrics for unsupervised scenarios

## Evaluation Metrics

### Classification Metrics

For labeled anomaly data, use these key metrics:

**Precision and Recall**
```python
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_detection(y_true, y_pred):
    """Evaluate binary anomaly detection"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'precision': precision,  # TP / (TP + FP)
        'recall': recall,       # TP / (TP + FN) 
        'f1_score': f1          # Harmonic mean of precision and recall
    }
```

**Area Under ROC Curve (AUC-ROC)**
```python
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

def evaluate_ranking(y_true, y_scores):
    """Evaluate anomaly score ranking"""
    auc_roc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find optimal threshold using Youden's index
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'auc_roc': auc_roc,
        'optimal_threshold': optimal_threshold,
        'fpr': fpr,
        'tpr': tpr
    }
```

**Average Precision (AP)**
```python
from sklearn.metrics import average_precision_score, precision_recall_curve

def evaluate_precision_recall(y_true, y_scores):
    """Evaluate precision-recall performance"""
    ap = average_precision_score(y_true, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Find threshold for desired precision
    desired_precision = 0.9
    valid_indices = precision >= desired_precision
    if np.any(valid_indices):
        best_recall = np.max(recall[valid_indices])
        threshold_idx = np.where((precision >= desired_precision) & (recall == best_recall))[0][0]
        threshold_for_precision = thresholds[threshold_idx]
    else:
        threshold_for_precision = None
    
    return {
        'average_precision': ap,
        'precision': precision,
        'recall': recall,
        'threshold_for_90_precision': threshold_for_precision
    }
```

### Streaming Evaluation

**Prequential Evaluation (Test-Then-Train)**
```python
from typing import List
import numpy as np

class PrequentialEvaluator:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.predictions = []
        self.true_labels = []
        self.scores = []
        
    def evaluate_point(self, model, x, y_true):
        """Evaluate model on one point, then train"""
        # Test: Get prediction before training
        score = model.score_one(x)
        self.scores.append(score)
        self.true_labels.append(y_true)
        
        # Train: Update model with new data
        model.learn_one(x)
        
        # Maintain sliding window
        if len(self.scores) > self.window_size:
            self.scores.pop(0)
            self.true_labels.pop(0)
    
    def get_current_performance(self, threshold=0.5):
        """Get current window performance"""
        if len(self.scores) < 10:  # Need minimum samples
            return None
        
        y_true = np.array(self.true_labels)
        y_scores = np.array(self.scores)
        y_pred = (y_scores > threshold).astype(int)
        
        if len(np.unique(y_true)) < 2:  # Need both classes
            return None
        
        return {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) == 2 else None
        }

# Usage
evaluator = PrequentialEvaluator(window_size=1000)

for data_point, label in labeled_stream:
    # Evaluate then train
    evaluator.evaluate_point(model, data_point, label)
    
    # Get performance every 100 points
    if len(evaluator.scores) % 100 == 0:
        perf = evaluator.get_current_performance()
        if perf:
            print(f"Current F1: {perf['f1']:.3f}, AUC: {perf['auc_roc']:.3f}")
```

**Delayed Labeling Evaluation**
```python
from collections import deque

class DelayedLabelEvaluator:
    def __init__(self, delay_window=100):
        self.delay_window = delay_window
        self.prediction_queue = deque()
        self.performance_history = []
    
    def add_prediction(self, x, score, timestamp):
        """Add prediction to delay queue"""
        self.prediction_queue.append({
            'data': x,
            'score': score,
            'timestamp': timestamp,
            'predicted': score > 0.5  # threshold
        })
    
    def add_label(self, timestamp, true_label):
        """Add delayed label and evaluate"""
        # Find matching prediction
        for i, pred in enumerate(self.prediction_queue):
            if pred['timestamp'] == timestamp:
                # Evaluate this prediction
                is_correct = (pred['predicted'] and true_label) or (not pred['predicted'] and not true_label)
                
                self.performance_history.append({
                    'timestamp': timestamp,
                    'correct': is_correct,
                    'true_positive': pred['predicted'] and true_label,
                    'false_positive': pred['predicted'] and not true_label,
                    'true_negative': not pred['predicted'] and not true_label,
                    'false_negative': not pred['predicted'] and true_label
                })
                
                # Remove from queue
                del self.prediction_queue[i]
                break
    
    def get_delayed_performance(self, window=1000):
        """Get performance from delayed labels"""
        if len(self.performance_history) < window:
            recent = self.performance_history
        else:
            recent = self.performance_history[-window:]
        
        if not recent:
            return None
        
        tp = sum(p['true_positive'] for p in recent)
        fp = sum(p['false_positive'] for p in recent)
        tn = sum(p['true_negative'] for p in recent)
        fn = sum(p['false_negative'] for p in recent)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': (tp + tn) / (tp + fp + tn + fn)
        }
```

### Unsupervised Evaluation

When labels are unavailable, use these proxy metrics:

**Reconstruction Error (for reconstruction-based models)**
```python
def reconstruction_evaluation(model, test_data, percentile=95):
    """Evaluate using reconstruction error"""
    errors = []
    
    for data_point in test_data:
        if hasattr(model, 'reconstruct'):
            reconstructed = model.reconstruct(data_point)
            error = np.linalg.norm(np.array(list(data_point.values())) - 
                                 np.array(list(reconstructed.values())))
            errors.append(error)
    
    threshold = np.percentile(errors, percentile)
    anomalies = np.sum(np.array(errors) > threshold)
    
    return {
        'mean_error': np.mean(errors),
        'error_std': np.std(errors),
        'threshold': threshold,
        'anomaly_count': anomalies,
        'anomaly_rate': anomalies / len(errors)
    }
```

**Silhouette Analysis**
```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

def silhouette_evaluation(data_points, anomaly_scores, n_clusters=2):
    """Evaluate using silhouette analysis"""
    # Convert to numpy array
    X = np.array([list(dp.values()) for dp in data_points])
    
    # Cluster based on anomaly scores
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    
    return {
        'silhouette_score': silhouette_avg,
        'cluster_labels': cluster_labels
    }
```

## Evaluation Frameworks

### Comprehensive Evaluation Pipeline

```python
import logging
from typing import Dict, Any
import time

class AnomalyDetectionEvaluator:
    def __init__(self, model, evaluation_config):
        self.model = model
        self.config = evaluation_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize evaluators
        self.prequential = PrequentialEvaluator(
            window_size=self.config.get('window_size', 1000)
        )
        self.delayed_label = DelayedLabelEvaluator(
            delay_window=self.config.get('delay_window', 100)
        )
        
        # Performance tracking
        self.performance_history = []
        self.start_time = time.time()
        
    def evaluate_stream(self, data_stream, label_stream=None):
        """Comprehensive stream evaluation"""
        point_count = 0
        
        for i, data_point in enumerate(data_stream):
            try:
                # Get model prediction
                score = self.model.score_one(data_point)
                
                # Prequential evaluation (if labels available)
                if label_stream and len(label_stream) > i:
                    label = label_stream[i]
                    self.prequential.evaluate_point(self.model, data_point, label)
                
                # Delayed label evaluation
                self.delayed_label.add_prediction(data_point, score, i)
                
                # Train model
                self.model.learn_one(data_point)
                point_count += 1
                
                # Periodic evaluation reporting
                if point_count % self.config.get('report_interval', 1000) == 0:
                    self._report_performance(point_count)
                    
            except Exception as e:
                self.logger.error(f"Evaluation error at point {i}: {e}")
                continue
        
        return self._final_evaluation_report(point_count)
    
    def _report_performance(self, point_count):
        """Report current performance"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Prequential performance
        prequential_perf = self.prequential.get_current_performance()
        
        # Delayed label performance
        delayed_perf = self.delayed_label.get_delayed_performance()
        
        report = {
            'timestamp': current_time,
            'points_processed': point_count,
            'processing_rate': point_count / elapsed,
            'prequential': prequential_perf,
            'delayed_labels': delayed_perf
        }
        
        self.performance_history.append(report)
        
        if prequential_perf:
            self.logger.info(
                f"Point {point_count}: F1={prequential_perf['f1']:.3f}, "
                f"Rate={point_count/elapsed:.1f}/sec"
            )
    
    def _final_evaluation_report(self, total_points):
        """Generate final evaluation report"""
        total_time = time.time() - self.start_time
        
        final_prequential = self.prequential.get_current_performance()
        final_delayed = self.delayed_label.get_delayed_performance()
        
        return {
            'summary': {
                'total_points': total_points,
                'total_time': total_time,
                'avg_processing_rate': total_points / total_time,
            },
            'final_performance': {
                'prequential': final_prequential,
                'delayed_labels': final_delayed
            },
            'performance_history': self.performance_history
        }

# Usage
config = {
    'window_size': 1000,
    'delay_window': 50,
    'report_interval': 500
}

evaluator = AnomalyDetectionEvaluator(model, config)
results = evaluator.evaluate_stream(data_stream, label_stream)
```

### Cross-Validation for Streaming

```python
from typing import List, Tuple
import numpy as np


class StreamingCrossValidator:
    def __init__(self, n_folds=5, window_size=1000):
        self.n_folds = n_folds
        self.window_size = window_size

    def validate_model(self, model_class, data_stream, labels, model_params=None):
        """Perform time-series cross-validation"""
        if model_params is None:
            model_params = {}

        # Convert stream to list for indexing
        data_list = list(data_stream)
        label_list = list(labels)

        fold_results = []
        fold_size = len(data_list) // self.n_folds

        for fold in range(self.n_folds):
            # Time-series split: use past data for training, future for testing
            train_end = (fold + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, len(data_list))

            if test_end <= test_start:
                continue

            # Initialize fresh model
            model = model_class(**model_params)

            # Train on past data
            for i in range(min(train_end, self.window_size)):
                model.learn_one(data_list[i])

            # Test on future data
            test_scores = []
            test_labels = []

            for i in range(test_start, test_end):
                score = model.score_one(data_list[i])
                test_scores.append(score)
                test_labels.append(label_list[i])

                # Continue learning (prequential)
                model.learn_one(data_list[i])

            # Evaluate fold
            if len(set(test_labels)) > 1:  # Need both classes
                fold_auc = roc_auc_score(test_labels, test_scores)
                threshold = np.percentile(test_scores, 95)
                fold_predictions = (np.array(test_scores) > threshold).astype(int)
                fold_f1 = f1_score(test_labels, fold_predictions)

                fold_results.append({
                    'fold': fold,
                    'auc_roc': fold_auc,
                    'f1_score': fold_f1,
                    'test_size': len(test_labels)
                })

        # Aggregate results
        if fold_results:
            return {
                'mean_auc': np.mean([r['auc_roc'] for r in fold_results]),
                'std_auc': np.std([r['auc_roc'] for r in fold_results]),
                'mean_f1': np.mean([r['f1_score'] for r in fold_results]),
                'std_f1': np.std([r['f1_score'] for r in fold_results]),
                'fold_results': fold_results
            }
        else:
            return None


# Usage
from onad.model.forest import OnlineIsolationForest

cv = StreamingCrossValidator(n_folds=5)
results = cv.validate_model(
    OnlineIsolationForest,
    data_stream,
    labels,
    model_params={'num_trees': 100, 'window_size': 1000}
)

if results:
    print(f"Cross-validation AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
    print(f"Cross-validation F1: {results['mean_f1']:.3f} ± {results['std_f1']:.3f}")
```

## Baseline Comparison

### Statistical Baselines

```python
class SimpleBaselines:
    @staticmethod
    def z_score_baseline(data_stream, threshold=3.0):
        """Simple z-score based anomaly detection"""
        mean_tracker = 0
        var_tracker = 0
        count = 0
        
        for data_point in data_stream:
            # Convert to single value (e.g., magnitude)
            value = np.linalg.norm(list(data_point.values()))
            
            # Update running statistics
            count += 1
            delta = value - mean_tracker
            mean_tracker += delta / count
            delta2 = value - mean_tracker
            var_tracker += delta * delta2
            
            # Calculate z-score
            if count > 1:
                std = np.sqrt(var_tracker / (count - 1))
                z_score = abs(value - mean_tracker) / std if std > 0 else 0
                yield z_score > threshold
            else:
                yield False
    
    @staticmethod
    def percentile_baseline(data_stream, window_size=1000, percentile=95):
        """Simple percentile-based anomaly detection"""
        window = deque(maxlen=window_size)
        
        for data_point in data_stream:
            value = np.linalg.norm(list(data_point.values()))
            window.append(value)
            
            if len(window) >= 10:  # Minimum window size
                threshold = np.percentile(window, percentile)
                yield value > threshold
            else:
                yield False

# Comparative evaluation
def compare_with_baselines(model, data_stream, labels):
    """Compare model with simple baselines"""
    # Convert stream to list for reuse
    data_list = list(data_stream)
    
    # Test model
    model_scores = []
    for data_point in data_list:
        model.learn_one(data_point)
        model_scores.append(model.score_one(data_point))
    
    # Test baselines
    z_scores = list(SimpleBaselines.z_score_baseline(data_list))
    percentile_scores = list(SimpleBaselines.percentile_baseline(data_list))
    
    # Evaluate all methods
    results = {}
    
    if len(set(labels)) > 1:
        # Model performance
        model_threshold = np.percentile(model_scores, 95)
        model_preds = (np.array(model_scores) > model_threshold).astype(int)
        results['model'] = {
            'auc_roc': roc_auc_score(labels, model_scores),
            'f1_score': f1_score(labels, model_preds)
        }
        
        # Baseline performance
        results['z_score'] = {
            'f1_score': f1_score(labels, z_scores)
        }
        
        results['percentile'] = {
            'f1_score': f1_score(labels, percentile_scores)
        }
    
    return results
```

---

!!! tip "Evaluation Best Practices"
    1. **Use Multiple Metrics**: Don't rely on a single metric; use precision, recall, F1, and AUC
    2. **Consider Class Imbalance**: Focus on metrics that handle imbalanced data well
    3. **Streaming Evaluation**: Use prequential evaluation for realistic performance assessment
    4. **Baseline Comparison**: Always compare against simple statistical baselines
    5. **Domain-Specific Metrics**: Consider business metrics relevant to your use case

!!! warning "Common Evaluation Pitfalls"
    - **Data Leakage**: Don't use future information for current predictions
    - **Threshold Selection**: Avoid optimizing threshold on test data
    - **Static Evaluation**: Don't assume performance remains constant over time
    - **Label Quality**: Be aware that anomaly labels may be noisy or incomplete