"""Loss function types for deep learning models."""

from torch import nn

# Type alias for all PyTorch loss functions
LossFunction = (
    nn.L1Loss
    | nn.MSELoss
    | nn.SmoothL1Loss
    | nn.HuberLoss
    | nn.CrossEntropyLoss
    | nn.BCELoss
    | nn.BCEWithLogitsLoss
    | nn.NLLLoss
    | nn.KLDivLoss
    | nn.CTCLoss
    | nn.HingeEmbeddingLoss
    | nn.MarginRankingLoss
    | nn.MultiLabelMarginLoss
    | nn.MultiLabelSoftMarginLoss
    | nn.MultiMarginLoss
    | nn.CosineEmbeddingLoss
    | nn.TripletMarginLoss
    | nn.TripletMarginWithDistanceLoss
)

# More specific type alias for autoencoder reconstruction losses
AutoencoderLoss = nn.MSELoss | nn.L1Loss | nn.SmoothL1Loss | nn.HuberLoss
