# NOTE: Settings in this file have global effect on the code. All parts of the pipeline have to run with the same settings.
# If you want to change a setting, you have to rerun all parts of the pipeline with the new setting. Otherwise, things will break.

# Root directory for the data
ROOT_DIR = "braintreebank" # "" usually
SAMPLING_RATE = 2048

# Disable file locking for HDF5 files. This is helpful for parallel processing.
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"