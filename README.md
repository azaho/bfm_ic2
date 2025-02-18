# bfm_ic

First, create a virtual environment and install the packages:
```
python -m venv .venv
source .venv/bin/activate
pip install beautifulsoup4 requests torch torchvision h5py pandas scipy numpy matplotlib seaborn wandb scikit-learn psutil ml_dtypes
```
<!-- then install xformers:
```
pip install xformers
``` -->

1. Download the braintreebank dataset (link: https://braintreebank.dev/)
First, run `braintreebank_download_extract.py` to download the zip files into `braintreebank_zip` directory and extract the zip files into `braintreebank` directory.
At this point, the folder `braintreebank_zip` can be deleted to free up space: `rm -rf braintreebank_zip`

2. Run the script `btbench_process_subject_trial_df.py` to process the subject trial data for benchmarking and save it to `btbench_subject_metadata` directory.

3. Now, you can run the script `train_model.py` to train the model. (use `run_train_model.sh` to run it on the cluster.)