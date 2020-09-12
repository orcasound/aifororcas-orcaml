# Orcasound Baseline Detector 🎱 🐋

The implementation here focuses on binary detection of *orca calls* (that are in the audible range, hence fun to listen to and annotate :) ) 
We change the audio-preprocessing front-end to better match this task & fine-tune the fully-connected layers and classification head of the [AudioSet model](https://github.com/tensorflow/models/tree/master/research/audioset), specifically a [PyTorch port of the model/weights](https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch). The model is generating *local predictions* on a fixed window size of ~2.45s. Sampling and aggregation strategies for more *global detection* at minute/hourly/day-wise time scale would be a welcome contribution (helpful for a real-time detection pipeline, or processing 2-3 months of historical data from different hydrophone nodes).

> 1. The model was bootstrapped with scraped open data from WHOI Marine Mammal Database (see `src.scraper` and `notebooks/DataPreparation` for details)  
> 2. Labelled data in live conditions from Orcasound hydrophones has subsequently been added using the [Pod.Cast tool](https://github.com/orcasound/orcalabel-podcast). (see [DataArchives](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive) for details)
> 3. The mel spectrogram generation is changed to better suit this task (for details on choice of filterbank see `notebooks/DataPreparation`. Implementation is in `data_ml/src.params` and `data_ml/src.dataloader`)
> 4. Given limited domain data, and need for robustness to different acoustic conditions (hydrophone nodes, SNR, noise/disturbances) in live conditions, the baseline uses transfer learning.  
> 5. Data augmentation in the style of [SpecAug](https://arxiv.org/pdf/1904.08779.pdf) is also implemented, that acts as a helpful form of regularization 
> 6. The Pod.Cast website aims to generate labelled data in live conditions, with candidates for annotation created in an [active-learning-like](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) fashion. The goal is to generate more relevant labelled data through multiple rounds of the above feedback loop bootstrapped by the classifier 

# Directory structure:

```
- data_ml 						(current directory)
    train.py
    test.py
	- src						(module library)
	- notebooks					(for evaluation, data preparation)
    - tools
- models
- runs
- live_inference				(deploy trained model)
```

See documentation at [DataArchives](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive) for details on how to access and read datasets in a standard form. 


# Examples 

## Download datasets

This is a convenience script to download & uncompress latest combined training & test datasets. 
```
python data_ml/tools/download_datasets.py <LOCATION> (--only_train/--only_test)
```

## Training 

Pardon the brevity here, this is just a rough starting point, that will evolve significantly! Some of the code is still pretty rough, however `src.model` and `src.dataloader` are useful places to start. 

Training converges quite fast (~5 minutes on a GPU). Train/validation tensorboard logs & model checkpoints are saved to a directory in `runRootPath`. 

```
python train.py -dataPath ../train_data -runRootPath ../runs/test --preTrainedModelPath ../models/pytorch_vggish.pth -model AudioSet_fc_all -lr 0.0005
```

## Notebook for validation and testing 

See notebook `Evaluation.ipynb` (might be pretty rough, but should give a general idea)

## Reproduce testing results

1. Download the test data by using the script
2. Download a trained model from [model path](https://drive.google.com/drive/folders/1TrHCDrt8Plr27elsbgdJyfsDUwA7mzEp?usp=sharing) to a location on your machine ```<model-download-location>```
3. Regenerate test results

```shell
python tools/prepare_test_and_model_data.py --test_path <test-data-download-dir> --model_path <model-download-dir>

python data_ml/test.py --test_path <test-data-download-location> --model_path <model-download-location>
```

# Setup instructions 

1. [Windows] Get [pyenv-win](https://github.com/pyenv-win/pyenv-win) to manage python versions:
    1. `git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%/.pyenv` 
    2. Add the following to your shell PATH `%USERPROFILE%\.pyenv\pyenv-win\bin`, `%USERPROFILE%\.pyenv\pyenv-win\shims` 

2. [Mac] Get [pyenv](https://github.com/pyenv/pyenv) to manage python versions:
	1. Use homebrew and run `brew update && brew install pyenv`
	2. Follow from step 3 onwards [here](https://github.com/pyenv/pyenv#basic-github-checkout). This essentially adds the `pyenv init` command to your shell on startup 
	3. FYI this is a [commands reference](https://github.com/pyenv/pyenv/blob/master/COMMANDS.md)

3. [Common] Install and maintain the right Python version (3.6.8) 
    1. Run `pyenv --version` to check installation 
    2. Run `pyenv rehash` from your home directory, install python 3.6.8 with `pyenv install -l 3.6.8` (use 3.6.8-amd64 on Windows if relevant) and run `pyenv rehash` again 
    3. Cd to `/PodCast` and set a local python version `pyenv local 3.6.8` (or 3.6.8-amd64). This saves a `.python-version` file that tells pyenv what to use in this dir 
    4. Type `python --version` and check you're using the right one

(feel free to skip 1, 2, 3 if you prefer to use your own Python setup and are familiar with many of this)

4. Create a [virtual environment](https://docs.python.org/3.6/library/venv.html) to isolate and install package dependencies 
    1. In your working directory, run `python -m venv podcast-venv`. This creates a directory `podcast-venv` with relevant files/scripts. 
	2. On Mac, activate this environment with `source podcast-venv/bin/activate` and when you're done, `deactivate`
	   On Windows, activate with `.\podcast-venv\Scripts\activate.bat` and `.\podcast-venv\Scripts\deactivate.bat` when done
    3. In an active environment, cd to `/data_ml` and run `python -m pip install --upgrade pip && pip install -r requirements.txt` 

