from torch import nn

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
