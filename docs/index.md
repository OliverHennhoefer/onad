# ONAD: Online Anomaly Detection Toolkit

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](https://github.com/OliverHennhoefer/onad/blob/main/LICENSE)
[![GitHub](https://img.shields.io/github/stars/OliverHennhoefer/onad?style=social)](https://github.com/OliverHennhoefer/onad)

**ONAD** is a comprehensive Python toolkit for **Online Anomaly Detection**, designed for real-time streaming data analysis. It provides state-of-the-art algorithms and utilities for detecting anomalies in continuous data streams with minimal latency and memory footprint.

## 🚀 Key Features

### **Streaming-First Design**
- **Low Memory Footprint**: Designed for constant memory usage regardless of stream length
- **Real-time Processing**: Process data points as they arrive with minimal latency
- **Incremental Learning**: Models adapt and learn from new data without full retraining

### **Comprehensive Algorithm Library**
- **Forest-based Models**: Online Isolation Forest, Mondrian Forest
- **SVM-based Models**: Adaptive SVM, GADGET SVM with graph structures
- **Statistical Models**: Moving averages, covariance analysis, Mahalanobis distance
- **Distance-based Models**: K-Nearest Neighbors, similarity search engines

### **Flexible Pipeline System**
- **Modular Architecture**: Compose complex detection pipelines from simple components
- **Data Transformers**: Built-in scaling, PCA, and preprocessing utilities
- **Stream Processing**: Efficient data streaming with configurable batch processing

### **Production-Ready**
- **Memory Management**: Configurable memory limits and automatic cleanup
- **Robust Error Handling**: Comprehensive validation and graceful failure recovery
- **Extensive Logging**: Detailed logging for monitoring and debugging
- **Type Safety**: Full type hints for better IDE support and code reliability

## 🎯 Use Cases

- **IoT Sensor Monitoring**: Detect anomalies in sensor readings from industrial equipment
- **Network Security**: Identify unusual network traffic patterns and potential threats
- **Financial Fraud Detection**: Monitor transactions for fraudulent activities
- **System Monitoring**: Detect performance anomalies in server metrics
- **Quality Control**: Identify defective products in manufacturing processes

## 📦 Quick Installation

```bash
# Basic installation
pip install onad

# With evaluation tools
pip install onad[eval]

# With deep learning support
pip install onad[dl]

# Full installation with all features
pip install onad[all]
```

## 🏃‍♂️ Quick Start

Here's a simple example to get you started:

```python
from onad.model.unsupervised.forest import OnlineIsolationForest
from onad.stream import ParquetStreamer, Dataset

# Initialize the model
model = OnlineIsolationForest(
    num_trees=100,
    window_size=1000,
    max_leaf_samples=32
)

# Stream data and detect anomalies
with ParquetStreamer(Dataset.FRAUD) as streamer:
    for features, label in streamer:
        # Learn from the data point
        model.learn_one(features)
        
        # Get anomaly score
        score = model.score_one(features)
        
        # Check if it's an anomaly
        if score > 0.7:  # threshold
            print(f"Anomaly detected! Score: {score:.3f}")
```

## 🏗️ Architecture Overview

ONAD follows a modular architecture with clear separation of concerns:

```
onad/
├── base/           # Abstract base classes and interfaces
├── model/          # Anomaly detection algorithms
│   ├── unsupervised/   # Unsupervised learning models
│   ├── supervised/     # Supervised learning models
│   └── statistics/     # Statistical methods
├── transform/      # Data preprocessing and transformation
├── stream/         # Data streaming and I/O utilities
└── utils/          # Helper functions and utilities
```

## 📚 What's Next?

- **[Installation Guide](installation.md)**: Detailed setup instructions
- **[Quick Start Tutorial](quickstart.md)**: Step-by-step introduction
- **[User Guide](user_guide/index.md)**: Comprehensive usage documentation
- **[Examples](examples/index.md)**: Real-world use cases and code samples
- **[API Reference](api/index.md)**: Complete API documentation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details on how to:

- Report bugs and request features
- Submit code contributions
- Improve documentation
- Share examples and use cases

## 📄 License

This project is licensed under the BSD License - see the [LICENSE](https://github.com/OliverHennhoefer/onad/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

- **Authors**: Oliver Hennhoefer, Pascal Heinzelmann, Marius Höll, Marco Catoir
- **Maintainer**: Oliver Hennhoefer
- Built with ❤️ for the anomaly detection community

---

!!! tip "Getting Help"
    - 📖 Check the [documentation](user_guide/index.md) for detailed guides
    - 💬 Ask questions in [GitHub Discussions](https://github.com/OliverHennhoefer/onad/discussions)  
    - 🐛 Report bugs in [GitHub Issues](https://github.com/OliverHennhoefer/onad/issues)