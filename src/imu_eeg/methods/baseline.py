"""Band-pass-only baseline. Preprocessing already applies this filter, so this
method is a passthrough -- it exists so every method has the same call signature."""
def clean(eeg, imu=None):
    return eeg.copy()