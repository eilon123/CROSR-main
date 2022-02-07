import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
import torch
from scipy.io import loadmat

from openmax_utils import *
from evt_fitting import weibull_tailfitting, query_weibull

import numpy as np
import libmr
import scipy.stats as st

from sklearn.metrics import roc_auc_score
import random
import torch.nn as nn
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.integrate as integrate
random.uniform(0, 1)

red_patch = mpatches.Patch(color='red', label='same class')
green_patch = mpatches.Patch(color='green', label='diff class')
blue_patch = mpatches.Patch(color='blue', label='diff class')


def get_args():
    parser = argparse.ArgumentParser(description='Get open max probability and compute AUROC')
    parser.add_argument('--MAV_path',default="./saved_MAVs/cifar10/",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--distance_scores_path',default="./saved_distance_scores/cifar10/",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--feature_dir',default="./saved_features/cifar10",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--alpha_rank',default=10,type=int,help="Alpha rank classes to consider")
    parser.add_argument('--weibull_tail_size',default=20,type=int,help="Tail size to fit")
    parser.add_argument('--overclass', '-overclass', default=0, action='store_true')

    parser.set_defaults(argument=True)

    return parser.parse_args()
args = get_args()

def recalibrate_scores(weibull_model, img_features,
                        alpharank = 5, distance_type = 'eucos'):
                        # alpharank = 5, distance_type = 'euclidean'):

    # img_features = img_features[0:10]
    NCLASSES = len(list(img_features[0:10]))
    ranked_list = img_features[0:10].argsort().ravel()[::-1]

    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
    # if args.overclass:
    #     alpha_weights = [x for pair in zip(alpha_weights, alpha_weights) for x in pair]
    ranked_alpha = np.zeros(NCLASSES)
    for i in range(len(alpha_weights)):

        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    openmax_layer = []
    openmax_unknown = []

    openmax_eilon = []
    open_eilon = 0


    for cls_indx in range(NCLASSES):

        if cls_indx ==5:
                # and args.overclass==0:
            break
        category_weibull = query_weibull(cls_indx, weibull_model, distance_type = distance_type)
        distance = compute_distance(img_features[cls_indx], category_weibull[0],
                                            distance_type = distance_type)

        wscore = category_weibull[2].w_score(distance)
        k = category_weibull[2].get_params()[1]
        lamda = category_weibull[2].get_params()[0]

        w = ((k / lamda) * (distance / lamda) ** (k - 1) * np.exp(-(distance / lamda) ** k))
        # modified_unit = img_features[cls_indx] * ( 1 - wscore*ranked_alpha[cls_indx] )
        modified_unit = img_features[cls_indx]*(1-ranked_alpha[cls_indx])
        openmax_layer += [modified_unit]
        openmax_unknown += [img_features[cls_indx] - modified_unit]
        #extra
        openmax_eilon += [img_features[cls_indx] * (wscore*ranked_alpha[cls_indx] )]
        open_eilon += img_features[cls_indx] *(1-wscore)


    openmax_eilon += [open_eilon]

    output = softmax(openmax_eilon)

    openmax_fc8 = np.asarray(openmax_layer)
    openmax_score_u = np.asarray(openmax_unknown)

    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u)
    """
    logits = [] 
    for indx in range(NCLASSES):
        logits += [sp.exp(img_features[indx])]
    den = sp.sum(sp.exp(img_features))
    softmax_probab = logits/den

    return np.asarray(openmax_probab), np.asarray(softmax_probab)
    """

    output = np.argmax(output)

    return openmax_probab,output
def get_scores(data_type,weibull_model,feature_path):

    results = []
    cnt=0
    decline = 0
    for cls_no in os.listdir(os.path.join(feature_path,data_type)):
        
        for filename in os.listdir(os.path.join(feature_path,data_type,cls_no)):

            img_features = np.load(os.path.join(feature_path,data_type,cls_no,filename))

            openmax ,pred=  recalibrate_scores(weibull_model, img_features)

            results.append(openmax)
            if  pred == int(cls_no):
                cnt += 1
            if pred == 5 and int(cls_no) >4:
                decline +=1
            # if openmax <  0.999999517479464:
            #     decline+=1

    print("correct is ",cnt)
    print("declined is ",decline)
    return np.array(results)

def getProb(c,d):
    k = c.get_params()[1]
    lamda = c.get_params()[0]



    p = integrate.quad(lambda dist: ((k / lamda) * (dist / lamda) ** (k - 1) * np.exp(-(dist / lamda) ** k)), 0,d)

    return p[0]
def getdistribution(data_type,w,feature_path):
    dist = np.linspace(0, 4, 10000)
    src = os.path.join(feature_path, 'val')
    if data_type == 'val':
        for cls_no in os.listdir(src):
            if  cls_no != '1':
                continue
            score = list()
            sameclasspoints = list()
            sameclassscore = list()
            otherclasspoints = list()
            otherclassscore = list()
            category_weibull = query_weibull(cls_no, w,distance_type = 'eucos')
            for d in dist:
                score.append(category_weibull[2].w_score(d))
            # for cls_num in (os.listdir(os.path.join(feature_path, data_type))):
            for filename in os.listdir(os.path.join(feature_path, data_type, cls_no)):
                img_features = np.load(os.path.join(feature_path, data_type, cls_no, filename))
                for cls_num in os.listdir(src):
                    img_features_cls = img_features[int(cls_num)]
                    # img_features_cls = img_features

                    if int(cls_no) != int(cls_num):
                        c = query_weibull(cls_num, w, distance_type='eucos')
                        distance = compute_distance(img_features, c[0], distance_type='eucos')
                        wscore = c[2].w_score(distance)
                        p = getProb(category_weibull[2], distance)
                        wscore = p
                        otherclassscore.append(wscore)
                        otherclasspoints.append(distance)

                    else:
                        distance = compute_distance(img_features, category_weibull[0], distance_type='eucos')
                        p = getProb(category_weibull[2],distance)
                        wscore = category_weibull[2].w_score(distance)
                        wscore = p
                        sameclasspoints.append(distance)
                        sameclassscore.append(wscore)
            if args.overclass:

                s = 'weibull/overclass/cdf/' + data_type + '/' + 'sameclass'
            else:
                s = 'weibull/regular/cdf/' + data_type + '/' + 'sameclass'
            os.makedirs(s, exist_ok=True)

            plt.plot(dist,score)
            plt.scatter(sameclasspoints,sameclassscore,color='red',marker='*')


            plt.legend(handles=[red_patch])
            s = "class num " + cls_no
            plt.title(s)
            if args.overclass:

                s = 'weibull/overclass/cdf/' + data_type + '/' + 'sameclass/' +  str(cls_no)
            else:
                s = 'weibull/regular/cdf/' + data_type + '/' + 'sameclass/' +  str(cls_no)

            plt.savefig(s)
            plt.clf()
            plt.plot(dist, score)
            plt.scatter(otherclasspoints, otherclassscore, color='green')
            s = "class num " + cls_no
            plt.title(s)
            plt.legend(handles=[green_patch])
            if args.overclass:

                s = 'weibull/overclass/cdf/' + data_type + '/' + 'diffclass'
            else:
                s = 'weibull/regular/cdf/' + data_type + '/' + 'diffclass'
            os.makedirs(s, exist_ok=True)
            s = s + '/' +  str(cls_no)

            plt.savefig(s)
            plt.clf()
    else:
        src2 = os.path.join(feature_path, 'open_set')
        for cls_open in os.listdir(src2):
            for cls_no in os.listdir(src):
                score = list()
                sameclasspoints = list()
                sameclassscore = list()
                otherclasspoints = list()
                otherclassscore = list()
                category_weibull = query_weibull(cls_no, w, distance_type='eucos')
                for d in dist:
                    score.append(category_weibull[2].w_score(d))
                # for cls_num in (os.listdir(os.path.join(feature_path, data_type))):
                for filename in os.listdir(os.path.join(src2, cls_open)):
                    img_features = np.load(os.path.join(feature_path, data_type, cls_open, filename))

                    img_features_cls = img_features[int(cls_no)]
                    # img_features_cls = img_features
                    distance = compute_distance(img_features, category_weibull[0], distance_type='eucos')
                    wscore = category_weibull[2].w_score(distance)

                    otherclassscore.append(wscore)
                    otherclasspoints.append(distance)

                if args.overclass:

                    s = 'weibull/overclass/cdf/' + data_type
                else:
                    s = 'weibull/regular/cdf/' + data_type
                os.makedirs(s, exist_ok=True)
                plt.plot(dist, score)
                plt.scatter(otherclasspoints, otherclassscore, color='blue')
                s = "class num " + cls_no
                plt.title(s)
                plt.legend(handles=[blue_patch])
                if args.overclass:
                    s = 'weibull/overclass/cdf/' + data_type + '/' + str(cls_no)
                else:
                    s = 'weibull/regular/cdf/' + data_type + '/' + str(cls_no)
                plt.savefig(s)
                plt.clf()
def getPDF(weibull_model,args,distance_type):
    if distance_type == 'eucos' or 'cosine':
        distance = np.linspace(0, 2, 10000)
    else:
        distance = np.linspace(0, 200, 10000)

    totlist = list()
    for cls_indx in range(5 * (1 + args.overclass)):
        wscore = list()
        distl = list()
        distl2 = list()
        # meantrain_vec = np.load(os.path.join(mean_path, str(cls_indx) + ".npy"))
        category_weibull = query_weibull(cls_indx, weibull_model, distance_type=distance_type)
        k = category_weibull[2].get_params()[1]
        lamda = category_weibull[2].get_params()[0]
        # tailtofit = sorted(distance_scores)[-20:]
        # k = st.exponweib.fit(tailtofit, floc=0, f0=1)[1]
        # k=24
        for dist in distance:
            wscore.append((k / lamda) * (dist / lamda) ** (k - 1) * np.exp(-(dist / lamda) ** k))
            # wscore.append(category_weibull[2].pdf(dist))

        fig, ax = plt.subplots()
        color = 'tab:red'
        ax.plot(distance, wscore, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        feature_path = args.feature_dir
        data_type = "val"

        onelist = list()
        for cls_no in os.listdir(os.path.join(feature_path, data_type)):
            if int(cls_no) != cls_indx:

                for filename in os.listdir(os.path.join(feature_path, data_type, cls_no)):
                    img_features = np.load(os.path.join(feature_path, data_type, cls_no, filename))
                    c = query_weibull(cls_no, weibull_model, distance_type=distance_type)
                    distl.append(compute_distance(img_features, c[0], distance_type=distance_type))

            else:

                for filename in os.listdir(os.path.join(feature_path, data_type, cls_no)):
                    if filename == '5977.npy':
                        dbg =3
                    c = query_weibull(cls_no, weibull_model, distance_type=distance_type)
                    img_features = np.load(os.path.join(feature_path, data_type, cls_no, filename))
                    distl2.append(compute_distance(img_features, c[0], distance_type=distance_type))


        totlist.append(onelist)
        # plt.show()
        color = 'tab:blue'
        s = 'weibull/'
        if args.overclass:
            s += 'overclass'
        else:
            s += 'regular'
        os.makedirs(s, exist_ok=True)
        s = s + '/' + str(cls_indx)
        ax2 = ax.twinx()
        # ax2.hist(distl, bins=1000, density=True, color=color)
        ax2.tick_params(axis='y', labelcolor=color)


        color = 'tab:red'
        ax3 = ax.twinx()
        ax3.hist(distl2, bins=1000, density=True, color=color)
        ax3.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()


        ax3.legend(handles=[red_patch,blue_patch])
        plt.savefig(s)
        plt.clf()
        dist2 = np.linspace(0, 5, 100000)


        meana = 0
        for dist in dist2:
            meana += dist ** (1 / k) * np.exp(-dist)
        meana *= lamda
        x = 3
    print("finish")

def main():
    args = get_args()
    distance_type = 'eucos'
    distance_path = args.distance_scores_path
    mean_path = args.MAV_path
    alpha_rank = args.alpha_rank
    weibull_tailsize = args.weibull_tail_size
    weibull_model = weibull_tailfitting(mean_path, distance_path,
                                        tailsize=weibull_tailsize,distance_type = distance_type)


    getPDF(weibull_model,args,distance_type)
    getdistribution("val",weibull_model,args.feature_dir)
    getdistribution("open_set", weibull_model, args.feature_dir)
    in_dist_scores = get_scores("val",weibull_model,args.feature_dir)
    open_set_scores = get_scores("open_set",weibull_model,args.feature_dir)

    print("The AUROC is ",calc_auroc(in_dist_scores, open_set_scores))


if __name__ == "__main__":


    main()
