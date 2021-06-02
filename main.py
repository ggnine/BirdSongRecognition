from res_util import *
from bird_co import *
# from other_util import *


# conda install pandas
# conda install librosa
# conda install pandas
# conda install pandas
# conda install pandas

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

import cv2
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

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500

TARGET_SR = 32000

model_config = {
    "base_model_name": "RESNET",
    "pretrained": True, #False,
    "num_classes": 264,#264,
    "trained_weights": "trained_model_4153237.pth"
}

melspectrogram_parameters = {
    "n_mels": 128,
    "fmin": 20,
    "fmax": 16000
}

# ROOT = "M:/birdsong-recognition"
INPUT_ROOT = Path("M:/")
RAW_DATA = INPUT_ROOT / "birdsong-recognition"
TRAIN_AUDIO_DIR = RAW_DATA / "train_audio"
TEST_AUDIO_DIR = RAW_DATA / "test_audio"

train_csv = pd.read_csv(RAW_DATA / "train.csv")


# TEST AUDIO STUFF _ NOT NEEDED NOW
# if not TEST_AUDIO_DIR.exists():
#     TEST_AUDIO_DIR = INPUT_ROOT / "birdcall-check" / "test_audio"
#     test = pd.read_csv(INPUT_ROOT / "birdcall-check" / "test.csv")
# else:
#     test = pd.read_csv(RAW_DATA / "test.csv")

# sub = pd.read_csv("../input/birdsong-recognition/sample_submission.csv")
# print(sub)
# print("_____sub")
# sub.to_csv("submission.csv", index=False)  # this will be overwritten if everything goes well
# print(sub)

def set_seed(seed: int = 42):
    # random.seed(seed)
    # np.random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore

set_seed(1213)


def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    """
    Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
    """
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
#     print(f"std {std}")
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


def get_model(args: tp.Dict):
    # # get resnest50_fast_1s1x64d
    model = ResNet(     
        Bottleneck, [3, 4, 6, 3],
        radix=1, groups=1, bottleneck_width=64,
        deep_stem=True, stem_width=32, avg_down=True,
        avd=True, avd_first=True)
    
    del model.fc
    # # use the same head as the baseline notebook.
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, args["num_classes"]))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    state_dict = torch.load(args["trained_weights"], map_location=device)
    model.load_state_dict(state_dict)
    
    return model


# SAVE ALL BIRDS WAV IMAGE TO FILE

# a.tofile('test2.dat')
# c = np.fromfile('test2.dat', dtype=int)
import os

def pad(arr,lenz):
    if arr.shape[0]>=lenz:
        return arr
    else:
        zz = np.zeros(lenz)
        zz[:len(arr)] = arr
        return zz


def save_dataset_to_file(df,sta,leng,img_size,melspectrogram_parameters,bird_num_to_start):  
    if not os.path.exists('M:/birdsong-recognition/np/'):
        os.makedirs('M:/birdsong-recognition/np/')
    
    for index,row in df.iterrows():
        # TO START AT A CERTAIN NUMBER
        # if index < bird_num_to_start:
        #     continue

        try:
            print(str(row.ebird_code) + "/" + row.filename)
            print(f"index: {index}")
            SR = 32000        
            print(f"in datagen: {index}")
            # oo=TRAIN_AUDIO_DIR / Path(str(row.ebird_code))
            clip, _ = librosa.load(os.path.join("M:/birdsong-recognition/train_audio/"+str(row.ebird_code) + "/" +str(row.filename)),
                                sr=SR,
                                mono=True,
                                res_type="kaiser_fast")

            start_seconds = sta
            length_in_sec = leng
        #         end_index = SR * end_seconds
            start_index = start_seconds * SR
            end_index = start_index + (length_in_sec * SR)

            temp_y = clip[start_index:end_index].astype(np.float32)
        #         print(f"amoutn : {(end_index-(SR*start_index))}")
        #         print(f"{temp_y.shape}")
            y = pad(temp_y,(end_index-start_index))

            melspec = librosa.feature.melspectrogram(y, sr=SR, **melspectrogram_parameters)
            melspec = librosa.power_to_db(melspec).astype(np.float32)

            image = mono_to_color(melspec)
            height, width, _ = image.shape
            image = cv2.resize(image, (int(width * img_size / height), img_size))
            image = np.moveaxis(image, 2, 0)
            image = (image / 255.0).astype(np.float32)
    #         print(image.flatten().shape)
    #         print(image.shape)
    #         print(type(image))
    #         print(type(image[0][0][0]))
            
            
            np.save('M:/birdsong-recognition/np/'+str(index)+'_'+row.ebird_code, image)
    #         image.tofile('./np/'+str(index)+'_'+row.ebird_code+'.dat')
            # c = np.fromfile('test2.dat', dtype=int)
        except:
            print("failed for some reason, index below")
            print(index)


#         return image, BIRD_CODE.get(row.ebird_code)
        




# VERSION 3.0 - FROM FILE -  FullDataset
import glob
import re

def pad(arr,lenz):
    if arr.shape[0]>=lenz:
        return arr
    else:
        zz = np.zeros(lenz)
        zz[:len(arr)] = arr
        return zz

class FullDataset_new_from_file(data.Dataset):
    def __init__(self, df: pd.DataFrame, indy=-1):
        self.df = df
        self.indy = indy
        
    def __len__(self):
        if self.indy > -1:
            return 1
        else:
            return len(self.df)
    
    def __getitem__(self, index: int):
        SR = 32000        
#         print(f"ind in datagen: {index}")
        if self.indy > -1:
            index = self.indy
#         image.tofile('./saves/'+str(index)+'_'+row.ebird_code+'.dat')
        filez = glob.glob('M:/birdsong-recognition/np/'+str(index)+'_*.npy')
        bird_name = re.search(r"(?<=_).*(?=\.)", filez[0]).group(0)
        # image = np.fromfile(filez[0], dtype=float)
        image = np.load(filez[0]).reshape(3,224,-1)
        
#         print(f"img: {image.shape} bird: {bird_name}  bridcode: {BIRD_CODE.get(bird_name)}")
        return image, BIRD_CODE.get(bird_name)
        
        
        
        



import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


# torch.set_default_tensor_type('torch.cuda.FloatTensor')
dtype = torch.cuda.FloatTensor
# torch.backends.cudnn.benchmark = True


model_train = None
valid_sampler = None
def train(train_df: pd.DataFrame,
               test_audio: Path,
               model_config: dict,
               mel_params: dict,
               target_sr: int,
               threshold=0.5, #was .5
               how_far_to_go = 25,
               batches = 100,
               epochs = 100
              ):
    global model_train
    global valid_sampler
    
#     use_cuda = torch.cuda.is_available()
    device = torch.device("cuda")

    model_train = get_model(model_config)
    model_train = model_train.cuda()
    warnings.filterwarnings("ignore")
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = optim.SGD(model_train.parameters(), lr=0.01, momentum=0.5)
    
    
    #GENERATE dataset/images on demand
#     dataset = FullDataset_new(df=train_df[:150],
#                           img_size=224,
#                           melspectrogram_parameters=mel_params)


    #GET dataset from preloaded Files
    dataset = FullDataset_new_from_file(df=train_df[:how_far_to_go])


    validation_split = .05
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    
    # SHUFFLE DATASET
    np.random.seed(42) #change seed if yu want
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    

    training_generator = torch.utils.data.DataLoader(dataset, batch_size=batches, 
                                            sampler=train_sampler)
    # validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                                 sampler=valid_sampler)


    # training_generator = data.DataLoader(dataset, batch_size=batches, shuffle=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    max_epochs = epochs
    count = 0
    all_losses = []
    for epoch in range(max_epochs):
        print(f"epoch {epoch}")
        # Training
        batch_losses = []
        
        ##https://gist.github.com/conormm/5b26a08029b900520bcd6fcd1f5712a0
        for ix, (local_batch, local_labels) in enumerate(training_generator):
#             print("pre_cudaAvail")
            if torch.cuda.is_available():
#                 print("post_cudaAvail")

                _X, _y = local_batch.type(dtype).cuda(), local_labels.long().cuda()  #torch.cuda.ByteTensor
    #             _X, _y = torch.tensor(local_batch, dtype=torch.float, device=device),torch.tensor(local_labels, dtype=torch.uint8, device=device)


    #             x = x.type(torch.cuda.FloatTensor)
#                 _X = Variable(_X, requires_grad=True).cuda()

    #             _y = Variable(_y, requires_grad=True)

#                 print(_y)
#                 print(_y.shape)
#                 print(f"xx shape: {_X.shape}")
                #==========Forward pass===============

                preds = model_train(_X)
                loss = criterion(preds, _y) #F.cross_entropy(preds, _y)  #

                #==========backward pass==============

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
#                 print(f"loss data: {loss.data}")
#                 print(loss.item())
#                 print(type(loss.item()))
#                 print(type(int(loss.item())))

                batch_losses.append(loss.item())
                all_losses.append(loss.item())
                torch.cuda.empty_cache()

            mbl = np.mean(np.sqrt(batch_losses)).round(3)
            if ix % 100 == 0:
                print("index {}, Batch loss: {}".format(ix, mbl))

        # if epoch % 2 == 0:
        print("Epoch [{}/{}], Batch loss: {}".format(epoch, max_epochs, np.mean(all_losses).round(3)))
    print("returning model ")
    return model_train

        
        
        


# import re
# import glob


# filez = glob.glob('./np/'+str(2)+'_*.npy')
# print(filez)

# image = np.fromfile(filez[0], dtype=float)
# image = np.load(filez[0])
# image.reshape(3,224,-1).shape
# a 


def check_guess(numba):
    datasetz = FullDataset_new_from_file(df=train_csv,indy=numba)

    loader = data.DataLoader(datasetz, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_train.eval()

    # image = image.to(device)
    for ix, (img, lab) in enumerate(loader):
    #     print(lab, img)
        with torch.no_grad():
    #         _X, _y = local_batch.type(dtype).cuda(), local_labels.long().cuda() 
            prediction_1 = model_train(img.type(dtype).cuda())   #F.sigmoid(model_train(img.type(dtype).cuda()))
#             print(prediction_1)
            proba = (prediction_1.detach().cpu().numpy().reshape(-1))
            events = proba >= .8
            
    #         print(np.argsort(-proba))
            labels = np.argsort(-proba).tolist()  #np.argsort(-proba)[:events.sum()].tolist()
            # print("_________________________")

            # print(f"actual labels: {INV_BIRD_CODE[lab.item()]}")

            # print(f"best guess: {INV_BIRD_CODE[labels[0]]}")
            # print(labels)
            # print(INV_BIRD_CODE[lab.item()] == INV_BIRD_CODE[labels[0]])
            # print("_________________________")
            return (INV_BIRD_CODE[lab.item()] == INV_BIRD_CODE[labels[0]])


def validate():
    print(f"validatins indices: {list(valid_sampler)}")
    # for zz in valid_sampler:
    #   print(zz)

    correct_arr = []
    for i in range(20):
        wasCorrect = check_guess(random.randint(0,how_far_to_go))
        correct_arr.append(wasCorrect)
    print("{0:.0%} Correct - FROM RANDO TRAIN SET".format(np.asarray(correct_arr).mean()))

    correct_arr = []
    for i in list(valid_sampler):
        wasCorrect = check_guess(i)
        correct_arr.append(wasCorrect)
    print("{0:.0%} Correct - FROM VALID SAMPLER".format(np.asarray(correct_arr).mean()))



how_far_to_go = 21373 #21373
print(how_far_to_go)

# BIG STEP - long process
# save_dataset_to_file(train_csv,5,5,224,melspectrogram_parameters,12327)


### TRAIN AND GUESS
# train(train_df=train_csv[:],
#             test_audio=TRAIN_AUDIO_DIR,
#             model_config=model_config,
#             mel_params=melspectrogram_parameters,
#             target_sr=TARGET_SR,
#             threshold=0.5,
#             how_far_to_go = how_far_to_go,
#             batches = 22, #was 100
#             epochs = 18
#            )
# torch.save(model_train.state_dict(), './trained_model_'+str(random.randint(1,10000000))+'.pth')

##### CHECK TRAINING
# validate()





# TEST STUFF
# _________________________


TEST_FOLDER = 'M:/birdsong-recognition/example_test_audio/' #'M:/birdsong-recognition/test_audio/'
test_df = pd.read_csv('M:/birdsong-recognition/mock_example_test_audio_summary_2.csv')      #('M:/birdsong-recognition/test.csv')
# test_info.head()


def prediction(test_df: pd.DataFrame,
               test_audio: Path,
               model_config: dict,
               mel_params: dict,
               target_sr=32000,
               threshold=0.5):
    model_pred = get_model(model_config)
    warnings.filterwarnings("ignore")
    prediction_dfs = []
    leng = 5
    SR = 32000
    img_size = 224

    for index,row in test_df.iterrows():
        # TO START AT A CERTAIN NUMBER
        # if index < bird_num_to_start:
        #     continue

        # try:
        if row.site != "site_3":
            SR = 32000        
            print(f"in datagen: {index}")
            clip, _ = librosa.load(os.path.join(TEST_FOLDER + row.audio_id + '.mp3'),
                                sr=SR,
                                mono=True,
                                res_type="kaiser_fast")

            start_seconds = int(row.seconds)-5
            length_in_sec = leng
        #         end_index = SR * end_seconds
            start_index = start_seconds * SR
            end_index = start_index + (length_in_sec * SR)

            temp_y = clip[start_index:end_index].astype(np.float32)
        #         print(f"amoutn : {(end_index-(SR*start_index))}")
        #         print(f"{temp_y.shape}")
            y = pad(temp_y,(end_index-start_index))

            melspec = librosa.feature.melspectrogram(y, sr=SR, **melspectrogram_parameters)
            melspec = librosa.power_to_db(melspec).astype(np.float32)

            image = mono_to_color(melspec)
            height, width, _ = image.shape
            image = cv2.resize(image, (int(width * img_size / height), img_size))
            image = np.moveaxis(image, 2, 0)
            image = (image / 255.0).astype(np.float32)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_pred.eval()
            with torch.no_grad():
                    prediction_1 = model_pred(torch.from_numpy(np.expand_dims(image, axis=0)).type(dtype = torch.cuda.FloatTensor).cuda())
                    # print(prediction_1)  #np.expand_dims(image))
                    # prediction_1 = model_pred(image.type(dtype = torch.cuda.FloatTensor).cuda()) 
                    proba = -prediction_1.detach().cpu().numpy().reshape(-1)
                    my_cutoff = 3
                    proba_filter = [proba<my_cutoff]
                    print(np.sum(proba_filter))
      
            labels = np.argsort(proba).tolist()
            print(labels)
            bz = ""
            for boid in labels[:np.sum(proba_filter)]:
                bz = bz+ str(INV_BIRD_CODE[boid]) +" "        # +"+"+str(INV_BIRD_CODE[labels[1]])+"+"+str(INV_BIRD_CODE[labels[2]])
            print(f"these boidz: {bz}")
            if len(bz) < 1:
                bz = "nocall"

            prediction_df = pd.DataFrame({
                "row_id": [str(row.row_id)],
                "birds": [bz]
                # "actual_bird_remove": [row.birds]
            })
            prediction_dfs.append(prediction_df)
        else:
            print(row.site)
            SR = 32000        
            print(f"in datagen: {index}")
            clip, _ = librosa.load(os.path.join(TEST_FOLDER + row.audio_id + '.mp3'),
                                sr=SR,
                                mono=True,
                                res_type="kaiser_fast")
            leng_np = clip.shape[0]//32000
            print(leng_np)
            num_five_intervals = leng_np//5
            print(num_five_intervals)

            c = np.arange(num_five_intervals-2)
            np.random.shuffle(c)
            how_many_random_samples = 10
            endz = len(c) if len(c) < how_many_random_samples else how_many_random_samples
            bz = []
            for time_index in c[:endz]:
                print(time_index*5)
                
                start_seconds = time_index*5
                length_in_sec = leng
            #         end_index = SR * end_seconds
                start_index = start_seconds * SR
                end_index = start_index + (length_in_sec * SR)

                temp_y = clip[start_index:end_index].astype(np.float32)
            #         print(f"amoutn : {(end_index-(SR*start_index))}")
            #         print(f"{temp_y.shape}")
                y = pad(temp_y,(end_index-start_index))

                melspec = librosa.feature.melspectrogram(y, sr=SR, **melspectrogram_parameters)
                melspec = librosa.power_to_db(melspec).astype(np.float32)

                image = mono_to_color(melspec)
                height, width, _ = image.shape
                image = cv2.resize(image, (int(width * img_size / height), img_size))
                image = np.moveaxis(image, 2, 0)
                image = (image / 255.0).astype(np.float32)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_pred.eval()
                with torch.no_grad():
                    prediction_1 = model_pred(torch.from_numpy(np.expand_dims(image, axis=0)).type(dtype = torch.cuda.FloatTensor).cuda())
                    # print(prediction_1)  #np.expand_dims(image))
                    # prediction_1 = model_pred(image.type(dtype = torch.cuda.FloatTensor).cuda()) 
                    proba = -prediction_1.detach().cpu().numpy().reshape(-1)
                    my_cutoff = 3
                    proba_filter = [proba<my_cutoff]
                    print(np.sum(proba_filter))

                
                labels = np.argsort(proba).tolist()
                print(labels)
                # print([INV_BIRD_CODE[labels[ii]] for ii in range(len(labels))])
                # print(INV_BIRD_CODE[248])
                
                for boid in labels[:np.sum(proba_filter)]:
                    bz.append(boid)
                    # bz = bz+ str(INV_BIRD_CODE[boid]) +" "        # +"+"+str(INV_BIRD_CODE[labels[1]])+"+"+str(INV_BIRD_CODE[labels[2]])
                print(f"these boidz: {bz}")
            print(len(bz))
            b_string = ""
            if len(bz) == 0:
                b_string = "nocall"
            else:
                for boid in list(set(bz)):
                    b_string = b_string+ str(INV_BIRD_CODE[boid]) +" "
            print(f"total boidz: {b_string}")

            prediction_df = pd.DataFrame({
                "row_id": [str(row.row_id)],
                "birds": [b_string]
                # "actual_bird_remove": [row.birds]
            })
            prediction_dfs.append(prediction_df)
            






        # except Exception as e: # work on python 3.x
        #     print('Failed: '+ str(e))

        
    
    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)
    return prediction_df


def do_sub():
    submission = prediction(test_df=test_df,
                            test_audio=TEST_FOLDER,
                            model_config=model_config,
                            mel_params=melspectrogram_parameters,
                            target_sr=32000,
                            threshold=0.56
                            
                        )
    print(submission)
    submission.to_csv("submission.csv", index=False)


# RUN TEST FILE AND CREATE SUBMISSION
do_sub()


# TODO
# 1. change filename_seconds to rwo_id
# ----make sure prediction works with test.csv
# 2.make sure site stuff works