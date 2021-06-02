import os
import sys

import gc
import time
import math
import shutil
import random
import warnings
import typing as tp
from pathlib import Path
from contextlib import contextmanager

# import yaml
from joblib import delayed, Parallel

# import cv2
import librosa
import audioread
import soundfile as sf

import numpy as np
import pandas as pd

# from fastprogress import progress_bar
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
import torch.utils.data as data
import glob
import re
# import os.path
from os import path
print("done")

# f = open("M:/birdsong-recognition/train_audio/test.txt")
# print(f.read())

# aa = pd.read_csv('lz.tsv', sep='\t', header=0)
# print(aa.head())
test_df = pd.read_csv('M:/birdsong-recognition/train.csv')      #('M:/birdsong-recognition/test.csv')
print(test_df[["ebird_code","filename","duration"]].sample(frac=1).head())
empt = pd.DataFrame() 
empt['birds'] = test_df["ebird_code"]
empt['audio_id'] = test_df["filename"]
empt['seconds'] = test_df["duration"]
empt['sites'] = [1 for x in range(len(test_df))]
print(empt.sample(frac=1).head())
empt.sample(frac=1).to_csv("fake_test.csv", index=False)
# df = pd.read_csv("M:/birdsong-recognition/mock_example_test_audio_summary.csv")
# df["site"] = [1 for x in range(len(df))]
# print(df.head())
# # df.to_csv("M:/birdsong-recognition/mock_example_test_audio_summary_1.csv", index=False)
# test_df_for_audio_id = df.loc[df["audio_id"] == "BLKFR-10-CPL"]
# print(test_df_for_audio_id)


# for i in range(21374):
#     if i <12325:
#         continue
#     # print(f"index: {i}")
#     # print ("File exists:"+str(path.exists('M:/birdsong-recognition/np/'+str(i)+'_*.npy')))
#     if i%1000==0:
#         print(i)
#     try:
#         filez = glob.glob('M:/birdsong-recognition/np/'+str(i)+'_*.npy')
#         bird_name = re.search(r"(?<=_).*(?=\.)", filez[0]).group(0)
#     except:
#         print(f"error index: {i}  bird_name: {bird_name}")
# print("completed searchfor bad")
# filez = glob.glob('M:/birdsong-recognition/np/*.npy')
# for f in filez:
#     print(f)
    # bird_num = re.search(r"(?<=_).*(?=\.)", filez[0]).group(0)
    # bird_name = re.search(r"(?<=_).*(?=\.)", filez[0]).group(0)


# clip, _ = librosa.load("./XC110084.mp3",
#                                sr=32000,
#                                mono=True)