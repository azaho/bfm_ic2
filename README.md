# bfm_ic

To run the script, you need to install the following packages:
```
pip install beautifulsoup4 requests torch torchvision h5py pandas scipy numpy matplotlib seaborn wandb scikit-learn
```
then install xformers:
```
pip install xformers
```

1. Download the braintreebank dataset (link: https://braintreebank.dev/)
First, run `braintreebank_download_extract.py` to download the zip files into `braintreebank_zip` directory and extract the zip files into `braintreebank` directory.
At this point, the folder `braintreebank_zip` can be deleted to free up space: `rm -rf braintreebank_zip`
