"""Custom exceptions for the onad library."""


class OnadError(Exception):
    """Base exception class for all onad-specific errors."""

    pass


class ModelNotFittedError(OnadError):
    """Raised when a model method is called before the model has been fitted."""

    def __init__(self, message: str = "Model has not been fitted yet.") -> None:
        super().__init__(message)


class TransformationError(OnadError):
    """Raised when a transformation operation fails."""

    pass


class PipelineError(OnadError):
    """Raised when a pipeline operation fails."""

    pass


class ValidationError(OnadError):
    """Raised when input validation fails."""

    pass


class UnsupportedFeatureError(OnadError):
    """Raised when an unsupported feature is encountered."""

    def __init__(self, feature_name: str) -> None:
        super().__init__(f"Feature '{feature_name}' has not been seen during training.")
        self.feature_name = feature_name


class IncompatibleComponentError(PipelineError):
    """Raised when pipeline components are incompatible."""

    def __init__(self, component_name: str, expected_type: str) -> None:
        super().__init__(
            f"Component '{component_name}' is not compatible. Expected: {expected_type}"
        )
        self.component_name = component_name
        self.expected_type = expected_type
