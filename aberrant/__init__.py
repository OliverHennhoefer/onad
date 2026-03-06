"""ABERRANT: Online Anomaly Detection library for streaming data.

A Python library implementing the online learning paradigm for anomaly
detection. All models process data one point at a time, updating their
state incrementally without storing historical data.

Modules:
    base: Core abstract classes (BaseModel, BaseTransformer, Pipeline)
    drift: Concept drift detection algorithms (ADWIN, KSWIN, PageHinkley)
    model: Anomaly detection models
    stream: Streaming data utilities
    transform: Data transformers (scalers, projections)
"""

__version__ = "0.3.1"
