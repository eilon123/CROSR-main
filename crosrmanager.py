import os
import shutil
import argparse
def parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--unrconst', '-unrconst', default=0, action='store_true')
    parser.add_argument('--overclass', '-overclass', default=0, action='store_true')
    parser.add_argument('--feat', '-feat', default=0, action='store_true')
    parser.add_argument('--full', '-full', default=0, action='store_true')
    parser.add_argument('--res', '-res', default=0, action='store_true')

    args = parser.parse_args()
    return args
args = parse()
if os.path.exists("saved_features"):
    shutil.rmtree("saved_features")
if os.path.exists("saved_MAVs"):

    shutil.rmtree("saved_MAVs")
if os.path.exists("saved_distance_scores"):

    shutil.rmtree("saved_distance_scores")

# python train_net.py

# 3) Compute the activation vectors for images
cmd = 'python get_model_features.py'
if args.overclass:
    cmd += ' -overclass'
if args.unrconst:
    cmd += ' -unrconst'
if args.feat:
    cmd += ' -feat'
if args.full:
    cmd += ' -full'
if args.res:
    cmd += ' -res'
os.system(cmd)


# 4) Compute the MAV (mean activation vector) for each class category
print('mav compute')
os.system('python MAV_Compute.py')


# 5) Compute the distance scores for activation features of training set
os.system('python compute_distances.py')


# 6) Fit Weibull distribution for each category and calculate openmax scores (Note that this code needs to be run in Python 2.7.)
if args.overclass:
    os.system('python compute_openmax.py -overclass')
else:
    os.system('python compute_openmax.py')

