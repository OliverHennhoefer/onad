"""
Threshold model example for simple boundary-based anomaly detection.

This example demonstrates using ThresholdModel with various configurations:
- One-sided thresholds (ceiling or floor only)
- Two-sided corridor (both ceiling and floor)
- Per-feature thresholds using dict parameters
"""

import numpy as np

from onad.model.threshold import ThresholdModel

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Generate synthetic temperature and pressure data
# Normal operating conditions: temp in [20, 80], pressure in [10, 50]
n_samples = 100
temperatures = rng.uniform(10, 90, n_samples)
pressures = rng.uniform(5, 60, n_samples)

data_stream = [
    {"temperature": float(t), "pressure": float(p)}
    for t, p in zip(temperatures, pressures)
]

print("=" * 70)
print("Threshold Model Anomaly Detection Demo")
print("=" * 70)

# Example 1: Ceiling only (one-sided upper bound)
print("\n1. CEILING ONLY (temperature > 80)")
print("-" * 70)
model1 = ThresholdModel(ceiling={"temperature": 80.0})
anomalies1 = []
for i, x in enumerate(data_stream):
    score = model1.score_one(x)
    if score > 0:
        anomalies1.append(i)
        print(
            f"  Sample {i}: temp={x['temperature']:.1f}, "
            f"pressure={x['pressure']:.1f} -> ANOMALY"
        )

print(f"Detected {len(anomalies1)} anomalies (high temperature)")

# Example 2: Floor only (one-sided lower bound)
print("\n2. FLOOR ONLY (pressure < 10)")
print("-" * 70)
model2 = ThresholdModel(floor={"pressure": 10.0})
anomalies2 = []
for i, x in enumerate(data_stream):
    score = model2.score_one(x)
    if score > 0:
        anomalies2.append(i)
        print(
            f"  Sample {i}: temp={x['temperature']:.1f}, "
            f"pressure={x['pressure']:.1f} -> ANOMALY"
        )

print(f"Detected {len(anomalies2)} anomalies (low pressure)")

# Example 3: Corridor (two-sided bounds)
print("\n3. CORRIDOR (temp in [20, 80] AND pressure in [10, 50])")
print("-" * 70)
model3 = ThresholdModel(
    ceiling={"temperature": 80.0, "pressure": 50.0},
    floor={"temperature": 20.0, "pressure": 10.0},
)
anomalies3 = []
for i, x in enumerate(data_stream):
    score = model3.score_one(x)
    if score > 0:
        anomalies3.append(i)
        reason = []
        if x["temperature"] > 80:
            reason.append("temp too high")
        elif x["temperature"] < 20:
            reason.append("temp too low")
        if x["pressure"] > 50:
            reason.append("pressure too high")
        elif x["pressure"] < 10:
            reason.append("pressure too low")

        print(
            f"  Sample {i}: temp={x['temperature']:.1f}, "
            f"pressure={x['pressure']:.1f} -> ANOMALY ({', '.join(reason)})"
        )

print(f"Detected {len(anomalies3)} anomalies (outside safe corridor)")

# Example 4: Scalar thresholds (applies to all features)
print("\n4. SCALAR THRESHOLD (any feature > 85)")
print("-" * 70)
model4 = ThresholdModel(ceiling=85.0)
anomalies4 = []
for i, x in enumerate(data_stream):
    score = model4.score_one(x)
    if score > 0:
        anomalies4.append(i)
        violations = [k for k, v in x.items() if v > 85.0]
        print(
            f"  Sample {i}: temp={x['temperature']:.1f}, "
            f"pressure={x['pressure']:.1f} -> ANOMALY ({', '.join(violations)} > 85)"
        )

print(f"Detected {len(anomalies4)} anomalies (any feature exceeds 85)")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total samples processed: {n_samples}")
print(f"Ceiling only (temp > 80):         {len(anomalies1)} anomalies")
print(f"Floor only (pressure < 10):       {len(anomalies2)} anomalies")
print(f"Corridor (safe operating range):  {len(anomalies3)} anomalies")
print(f"Scalar threshold (any > 85):      {len(anomalies4)} anomalies")
print("\nNote: ThresholdModel never learns, it uses fixed boundaries.")
