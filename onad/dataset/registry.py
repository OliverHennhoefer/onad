"""Dataset registry system for anomaly detection benchmarks.

This module provides a centralized registry of available datasets with rich metadata,
following the design patterns established in the anomaly detection research community.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass
class DatasetInfo:
    """Metadata for an anomaly detection dataset.

    Attributes:
        name: Human-readable name of the dataset
        description: Brief description of the dataset and its characteristics
        filename: Name of the file in the GitHub release (without extension)
        n_samples: Total number of samples in the dataset
        n_features: Number of features/dimensions
        anomaly_rate: Proportion of anomalous samples (0.0 to 1.0)
        source: Original source or reference for the dataset
        category: Dataset category (e.g., 'medical', 'security', 'benchmark')
    """

    name: str
    description: str
    filename: str
    n_samples: int
    n_features: int
    anomaly_rate: float
    source: str
    category: str


class Dataset(Enum):
    """Available datasets for anomaly detection experiments.

    This enumeration provides all built-in datasets that can be loaded
    using the load() function. Each dataset is preprocessed for anomaly
    detection tasks with normal and anomalous samples.

    Usage:
        from onad.dataset import load, Dataset
        dataset = load(Dataset.FRAUD)
        for features, label in dataset.stream():
            # process data
    """

    # Existing datasets (migrated from stream module)
    FRAUD = "fraud"
    SHUTTLE = "shuttle"
    SMD = "smd"

    # Medical/Health datasets
    ANNTHYROID = "annthyroid"
    BREAST = "breast"
    HEPATITIS = "hepatitis"
    LYMPHOGRAPHY = "lymphography"
    MAMMOGRAPHY = "mammography"
    THYROID = "thyroid"

    # Security/Network datasets
    BACKDOOR = "backdoor"
    HTTP = "http"
    SMTP = "smtp"
    IONOSPHERE = "ionosphere"

    # Machine Learning benchmark datasets
    CARDIO = "cardio"
    COVER = "cover"
    GLASS = "glass"
    LETTER = "letter"
    MAGIC_GAMMA = "magic_gamma"
    MNIST = "mnist"
    MUSK = "musk"
    OPTDIGITS = "optdigits"
    PAGEBLOCKS = "pageblocks"
    PENDIGITS = "pendigits"
    SATIMAGE2 = "satimage2"
    STAMPS = "stamps"
    DONORS = "donors"


# Central registry of dataset metadata
DATASET_REGISTRY: dict[Dataset, DatasetInfo] = {
    # Existing datasets (current ONAD datasets)
    Dataset.FRAUD: DatasetInfo(
        name="Credit Card Fraud Detection",
        description="European credit card fraud dataset with anonymized features",
        filename="fraud",
        n_samples=284807,
        n_features=30,
        anomaly_rate=0.00173,  # 0.173% fraud cases
        source="Kaggle Credit Card Fraud Dataset",
        category="financial",
    ),
    Dataset.SHUTTLE: DatasetInfo(
        name="Space Shuttle",
        description="NASA space shuttle dataset for anomaly detection",
        filename="shuttle",
        n_samples=58000,
        n_features=9,
        anomaly_rate=0.07,
        source="UCI Machine Learning Repository",
        category="engineering",
    ),
    Dataset.SMD: DatasetInfo(
        name="Server Machine Dataset",
        description="Server monitoring data for anomaly detection",
        filename="smd",
        n_samples=708405,
        n_features=38,
        anomaly_rate=0.042,  # 4.2% anomalies
        source="Server Machine Dataset",
        category="monitoring",
    ),
    # Medical/Health datasets
    Dataset.ANNTHYROID: DatasetInfo(
        name="Thyroid Disease (ANN)",
        description="Thyroid disease detection dataset for neural networks",
        filename="annthyroid",
        n_samples=7200,
        n_features=21,
        anomaly_rate=0.075,
        source="UCI Machine Learning Repository",
        category="medical",
    ),
    Dataset.BREAST: DatasetInfo(
        name="Breast Cancer Wisconsin",
        description="Breast cancer diagnostic dataset with cell characteristics",
        filename="breast",
        n_samples=569,
        n_features=30,
        anomaly_rate=0.373,  # Malignant cases
        source="UCI Machine Learning Repository",
        category="medical",
    ),
    Dataset.HEPATITIS: DatasetInfo(
        name="Hepatitis",
        description="Hepatitis patient data for outcome prediction",
        filename="hepatitis",
        n_samples=155,
        n_features=19,
        anomaly_rate=0.206,
        source="UCI Machine Learning Repository",
        category="medical",
    ),
    Dataset.LYMPHOGRAPHY: DatasetInfo(
        name="Lymphography",
        description="Lymphatic system diagnosis dataset",
        filename="lymphography",
        n_samples=148,
        n_features=18,
        anomaly_rate=0.041,
        source="UCI Machine Learning Repository",
        category="medical",
    ),
    Dataset.MAMMOGRAPHY: DatasetInfo(
        name="Mammography",
        description="Mammographic screening for breast cancer detection",
        filename="mammography",
        n_samples=11183,
        n_features=6,
        anomaly_rate=0.023,
        source="UCI Machine Learning Repository",
        category="medical",
    ),
    Dataset.THYROID: DatasetInfo(
        name="Thyroid Disease",
        description="Thyroid function diagnosis dataset",
        filename="thyroid",
        n_samples=3772,
        n_features=6,
        anomaly_rate=0.025,
        source="UCI Machine Learning Repository",
        category="medical",
    ),
    # Security/Network datasets
    Dataset.BACKDOOR: DatasetInfo(
        name="Network Backdoor",
        description="Network intrusion detection - backdoor attacks",
        filename="backdoor",
        n_samples=2329,
        n_features=196,
        anomaly_rate=0.028,
        source="KDD Cup 1999 Network Intrusion",
        category="security",
    ),
    Dataset.HTTP: DatasetInfo(
        name="HTTP Network Traffic",
        description="HTTP-based network anomaly detection",
        filename="http",
        n_samples=567498,
        n_features=3,
        anomaly_rate=0.004,
        source="KDD Cup 1999 Network Intrusion",
        category="security",
    ),
    Dataset.SMTP: DatasetInfo(
        name="SMTP Email Traffic",
        description="SMTP protocol anomaly detection",
        filename="smtp",
        n_samples=95156,
        n_features=3,
        anomaly_rate=0.0003,
        source="KDD Cup 1999 Network Intrusion",
        category="security",
    ),
    Dataset.IONOSPHERE: DatasetInfo(
        name="Ionosphere Radar",
        description="Ionosphere radar signal classification",
        filename="ionosphere",
        n_samples=351,
        n_features=33,
        anomaly_rate=0.359,
        source="UCI Machine Learning Repository",
        category="physics",
    ),
    # ML Benchmark datasets
    Dataset.CARDIO: DatasetInfo(
        name="Cardiovascular Disease",
        description="Cardiovascular disease dataset with patient vitals",
        filename="cardio",
        n_samples=1831,
        n_features=21,
        anomaly_rate=0.096,
        source="Cardiovascular Disease Dataset",
        category="medical",
    ),
    Dataset.COVER: DatasetInfo(
        name="Forest Cover Type",
        description="Forest cover type classification (class 4 as anomaly)",
        filename="cover",
        n_samples=286048,
        n_features=10,
        anomaly_rate=0.009,
        source="UCI Machine Learning Repository",
        category="environmental",
    ),
    Dataset.GLASS: DatasetInfo(
        name="Glass Identification",
        description="Glass type identification (float processed as anomaly)",
        filename="glass",
        n_samples=214,
        n_features=9,
        anomaly_rate=0.042,
        source="UCI Machine Learning Repository",
        category="materials",
    ),
    Dataset.LETTER: DatasetInfo(
        name="Letter Recognition",
        description="Letter recognition (vowels vs consonants)",
        filename="letter",
        n_samples=1600,
        n_features=32,
        anomaly_rate=0.025,
        source="UCI Machine Learning Repository",
        category="vision",
    ),
    Dataset.MAGIC_GAMMA: DatasetInfo(
        name="MAGIC Gamma Telescope",
        description="Gamma ray detection in telescope data",
        filename="magic_gamma",
        n_samples=19020,
        n_features=10,
        anomaly_rate=0.352,
        source="UCI Machine Learning Repository",
        category="astronomy",
    ),
    Dataset.MNIST: DatasetInfo(
        name="MNIST Handwritten Digits",
        description="MNIST digit recognition (digit 0 as normal, others anomalous)",
        filename="mnist",
        n_samples=7603,
        n_features=100,
        anomaly_rate=0.092,
        source="MNIST Database",
        category="vision",
    ),
    Dataset.MUSK: DatasetInfo(
        name="Musk Molecules",
        description="Musk vs non-musk molecule classification",
        filename="musk",
        n_samples=3062,
        n_features=166,
        anomaly_rate=0.032,
        source="UCI Machine Learning Repository",
        category="chemistry",
    ),
    Dataset.OPTDIGITS: DatasetInfo(
        name="Optical Digit Recognition",
        description="Optical digit recognition dataset",
        filename="optdigits",
        n_samples=5216,
        n_features=64,
        anomaly_rate=0.029,
        source="UCI Machine Learning Repository",
        category="vision",
    ),
    Dataset.PAGEBLOCKS: DatasetInfo(
        name="Page Layout Analysis",
        description="Document page layout block classification",
        filename="pageblocks",
        n_samples=5473,
        n_features=10,
        anomaly_rate=0.107,
        source="UCI Machine Learning Repository",
        category="document",
    ),
    Dataset.PENDIGITS: DatasetInfo(
        name="Pen-Based Recognition",
        description="Pen-based handwritten digit recognition",
        filename="pendigits",
        n_samples=6870,
        n_features=16,
        anomaly_rate=0.023,
        source="UCI Machine Learning Repository",
        category="vision",
    ),
    Dataset.SATIMAGE2: DatasetInfo(
        name="Satellite Image (Landsat)",
        description="Satellite image classification (red soil as anomaly)",
        filename="satimage2",
        n_samples=5803,
        n_features=36,
        anomaly_rate=0.013,
        source="UCI Machine Learning Repository",
        category="remote_sensing",
    ),
    Dataset.STAMPS: DatasetInfo(
        name="Stamp Verification",
        description="Postage stamp verification dataset",
        filename="stamps",
        n_samples=340,
        n_features=9,
        anomaly_rate=0.091,
        source="Postage Stamp Dataset",
        category="verification",
    ),
    Dataset.DONORS: DatasetInfo(
        name="Blood Donors",
        description="Blood donation prediction dataset",
        filename="donors",
        n_samples=748,
        n_features=4,
        anomaly_rate=0.239,
        source="UCI Machine Learning Repository",
        category="medical",
    ),
}


def get_dataset_info(dataset: Dataset) -> DatasetInfo:
    """Get metadata information for a specific dataset.

    Args:
        dataset: Dataset enum value

    Returns:
        DatasetInfo object with metadata

    Raises:
        KeyError: If dataset is not found in registry
    """
    if dataset not in DATASET_REGISTRY:
        raise KeyError(f"Dataset {dataset} not found in registry")
    return DATASET_REGISTRY[dataset]


def list_available() -> dict[str, DatasetInfo]:
    """List all available datasets with their metadata.

    Returns:
        Dictionary mapping dataset names to DatasetInfo objects
    """
    return {dataset.value: info for dataset, info in DATASET_REGISTRY.items()}


def list_by_category(category: str) -> dict[str, DatasetInfo]:
    """List datasets by category.

    Args:
        category: Category to filter by (e.g., 'medical', 'security', 'benchmark')

    Returns:
        Dictionary of datasets in the specified category
    """
    return {
        dataset.value: info
        for dataset, info in DATASET_REGISTRY.items()
        if info.category == category
    }


def get_categories() -> list[str]:
    """Get all available dataset categories.

    Returns:
        List of unique category names
    """
    return sorted({info.category for info in DATASET_REGISTRY.values()})
