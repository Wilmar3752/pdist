"""Custom exceptions for the pdist package."""


class PdistException(Exception):
    """Base exception for all pdist errors."""
    pass


class DataValidationError(PdistException):
    """Raised when input data is invalid."""
    pass


class FittingError(PdistException):
    """Raised when distribution fitting fails."""
    pass


class InsufficientDataError(PdistException):
    """Raised when there's not enough data to fit a distribution."""
    pass


class InvalidDistributionError(PdistException):
    """Raised when an invalid distribution is specified."""
    pass


class ConvergenceError(FittingError):
    """Raised when fitting algorithm fails to converge."""
    pass

