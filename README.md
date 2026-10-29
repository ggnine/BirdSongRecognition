# BirdSongRecognition

Identify bird species from birdsong recordings with a ResNeSt50 CNN on mel-spectrograms. This was my entry work for Kaggle's [Cornell Birdcall Identification](https://www.kaggle.com/c/birdsong-recognition) competition (264 North American species).

> **Status:** 2020–21 learning project, not maintained. Most of the model and inference code comes from public sources (see Credits). My part is the training and data pipeline built around it.

## What's here

| File | What it is |
|---|---|
| `fork-of-ensemble-resnest50-gj-fortower.ipynb` | A fork of a public Kaggle inference notebook (ResNeSt50 baseline), with my own cells added for caching spectrograms and training. |
| `main.py` | The notebook turned into a script for a local machine. Contains my dataset caching (`save_dataset_to_file`, `FullDataset_new_from_file`) and the `train` / `validate` / `check_guess` / `do_sub` loop. |
| `res_util.py` | ResNeSt model code (third-party, see Credits). |
| `bird_co.py` | The competition's species-code to class-index mapping. |
| `test.py` | Scratch script that builds `fake_test.csv` (a local test set) from the training labels. |
| `fake_test.csv`, `lz.tsv` | A local test set and a prediction table in the competition's row format. |

## Running it (as far as is still true)

It needs the Kaggle competition data (`birdsong-recognition`, available on Kaggle after accepting the competition rules), PyTorch, librosa, OpenCV, pandas and scikit-learn. Paths are hard-coded for my machine at the time (`M:/birdsong-recognition/...`, `../input/...`), so edit them first. Pretrained baseline weights (`best_model.pth`) came from the public Kaggle training notebook and aren't included.

```
pip install torch librosa opencv-python pandas scikit-learn joblib audioread
python main.py
```

## Credits

- Inference notebook, forked from the public Kaggle kernel **"[Inference] Birdsong Baseline: ResNeSt50-fast"** and its training kernel (`training-birdsong-baseline-resnest50-fast`), birdsong-recognition competition, 2020. The original authors keep all rights to that code.
- ResNeSt (`res_util.py`): Hang Zhang et al., <https://github.com/zhanghang1989/ResNeSt>, Apache License 2.0.
- `mono_to_color`: daisukelab, <https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data>.
- Competition data and labels: Cornell Lab of Ornithology and Kaggle. Recordings come from [xeno-canto](https://xeno-canto.org/) under their recordists' licences. None of the data is redistributed here.

My own work: the dataset caching, training loop and validation code in `main.py`, `test.py`, and the added training cells in the notebook.
