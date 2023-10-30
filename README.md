# Feature-selector
Feature selector for ecg 12-lead classification

## Dataset
The dataset used in this project is the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) dataset. It contains 21837 10-second 12-lead ECGs from 18885 patients. The dataset is divided into 5490 training, 1825 validation and 10950 test samples. The dataset is annotated with 71 different diagnostic statements, which are divided into 19 diagnostic classes. The dataset is annotated by a single cardiologist. The dataset is available in the [PhysioNet](https://physionet.org/content/ptb-xl/1.0.1/) database.

- Put the dataset in the `dataset/ptb-xl/` folder.

## Model

```bash
git lfs install
git clone https://huggingface.co/MorphBai/Feature-Selector
```

## Test

```bash
python run.py --do_eval --eval_model ./checkpoint/selector/best_model.pth --model selector
```

## Train

```bash
python run.py --model_type demo --model resnet
```