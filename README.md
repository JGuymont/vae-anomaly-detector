# VAE Anomaly Detector

Code accompanying this [project](https://github.com/JGuymont/vae-anomaly-detector/blob/master/docs/report/report.pdf).

## Installation

In order to use the system, you will need to install the following dependencies:

- Pytorch
- Numpy

To install, build the source code from the repository by running the following command in your terminal:

```shell
git clone https://github.com/JGuymont/vae-anomaly-detector.git
cd vae-anomaly-detector
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## SMS Spam Collection Dataset

You can download the dataset on [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset/version/1). The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam. The file `spam.csv` contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.

To reproduce the experiment without changing the configuration, you should save the file `spam.csv` in the `data/` directory.

## Splitting the data

Start by splitting the data into a training set and a test set.

```shell
python split_data.py --train_size 0.5
```

Running this command in the terminal will create `train.csv` and a `test.csv` and save them in the directory `data/`. Note that you only need to run this command omce.

## Training

Train the model by running the flollowing command

```shell
python main.py --model boc
```
