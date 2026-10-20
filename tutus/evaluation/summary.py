"""Scalar summaries for new runs; inputs use the same [0, 1] scale."""

def harmonic_mean(*values):
    if not values:
        raise ValueError("At least one value is required")
    if any(value < 0 for value in values):
        raise ValueError("Harmonic mean requires nonnegative inputs")
    if any(value == 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)
