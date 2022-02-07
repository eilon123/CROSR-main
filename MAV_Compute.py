
import os, sys
import glob
import time
import scipy as sp
from scipy.io import loadmat, savemat
import pickle
import os.path as path
import torch
import numpy as np
import argparse


def compute_mean_vector(category_index,save_path,featurefilepath,):
    
    featurefile_list = os.listdir(os.path.join(featurefilepath,category_index))
    folder_name = category_index
    correct_features = []
    for featurefile in featurefile_list:
        
        feature = torch.from_numpy(np.load(os.path.join(featurefilepath,folder_name,featurefile)))

        predicted_category = torch.max(feature[0:10],dim=0)[1].item()
        # done in get model features now
        # if(predicted_category == int(category_index)):

        correct_features.append(feature)

        # correct_features.append(feature - np.mean(feature))
        # correct_features.append(feature[predicted_category])
    # a = torch.zeros((len(correct_features), correct_features[0].shape[0]))
    # for i in range(a.shape[0]):
    #     a[i, :] = correct_features[i]
    correct_features = torch.cat(correct_features,0)

    mav = torch.mean(correct_features)
    # mav = torch.mean(torch.Tensor(correct_features),dim=0)
    # mav = a.mean(dim=0)
    # mav = torch.tensor(np.mean(correct_features))
    os.makedirs(os.path.join(save_path), exist_ok=True)
    np.save(os.path.join(save_path,folder_name+".npy"),mav.data.numpy(),allow_pickle=False)

def get_args():
    parser = argparse.ArgumentParser(description='Get activation vectors')
    parser.add_argument('--save_path',default="./saved_MAVs/cifar10/",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--feature_dir',default="./saved_features/cifar10/train",type=str,help="Path to save the ensemble weights")
    parser.set_defaults(argument=True)

    return parser.parse_args()


def main():
    args = get_args()

    for class_no in os.listdir(args.feature_dir):
        compute_mean_vector(class_no,args.save_path,args.feature_dir)


if __name__ == "__main__":
    main()

