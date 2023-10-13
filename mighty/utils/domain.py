__all__ = [
    "MonitorLevel"
]


class MonitorLevel:
    """
    Defines the degree of exhaustive parameters visual exploration during
    training.
    """

    DISABLED = 0
    SIGNAL_TO_NOISE           = 0b00001  # signal-to-noise ratio
    SIGN_FLIPS                = 0b00010  # weight sign flips after batch update
    WEIGHT_INITIAL_DIFFERENCE = 0b00100  # dist(w, w_initial)
    WEIGHT_HISTOGRAM          = 0b01000  # weight histograms
    WEIGHT_SNR_TRACE          = 0b10000  # how strong the weight signal is
    FULL                      = 0b11111  # everything

    DEFAULT = SIGNAL_TO_NOISE | SIGN_FLIPS | WEIGHT_INITIAL_DIFFERENCE
