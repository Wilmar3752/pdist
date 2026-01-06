"""Custom exceptions for the bestdist package."""


class BestdistException(Exception):
    """Base exception for all bestdist errors."""
    pass


class DataValidationError(BestdistException):
    """Raised when input data is invalid."""
    pass


class FittingError(BestdistException):
    """Raised when distribution fitting fails."""
    pass


class InsufficientDataError(BestdistException):
    """Raised when there's not enough data to fit a distribution."""
    pass


class InvalidDistributionError(BestdistException):
    """Raised when an invalid distribution is specified."""
    pass


class ConvergenceError(FittingError):
    """Raised when fitting algorithm fails to converge."""
    pass

