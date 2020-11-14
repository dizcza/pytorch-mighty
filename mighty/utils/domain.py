from enum import Enum, unique


__all__ = [
    "MonitorLevel"
]


@unique
class MonitorLevel(Enum):
    """
    Defines the degree of exhaustive parameters visual exploration during
    training.
    """

    DISABLED = 0
    SIGNAL_TO_NOISE = 1  # signal-to-noise ratio
    FULL = 2  # sign flips, initial point difference
