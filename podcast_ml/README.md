# Pod.Cast ML
- Pod.Cast ML code was written by Akash Mahajan (akash7190@gmail.com)
- Tools scripts were written by Prakruti Gogia (prakruti.gogia@gmail.com)

# Best model performance

# Reproduce test results
## Setup your virtual environment

Create a [virtual environment](https://docs.python.org/3.6/library/venv.html) to isolate and install package dependencies 
1. In your working directory, run `python -m venv podcast-venv`. This creates a directory `podcast-venv` with relevant files/scripts. 
2. On Mac, activate this environment with `source podcast-venv/bin/activate` and when you're done, `deactivate`. On Windows, activate with `.\podcast-venv\Scripts\activate.bat` and `.\podcast-venv\Scripts\deactivate.bat` when done
3. In an active environment, cd to `/data_ml` and run `python -m pip install --upgrade pip && pip install -r requirements.txt` 
## Run test
Download the test data by using the script

```bash
python data_ml/tools/download_and_combine_dataset.py --test_data_download_location <LOCATION>
```

Download the model from [model path](https://drive.google.com/drive/folders/1TrHCDrt8Plr27elsbgdJyfsDUwA7mzEp?usp=sharing) to a location on your machine ```<model-download-location>```

Regenerate test results
```shell
python data_ml/test.py --test_path <test-data-download-location> --model_path <model-download-location>
```