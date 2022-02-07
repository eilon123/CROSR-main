import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import DHR_Net as models
import numpy as np
import pickle
import os
from PIL import Image
import argparse
from models import *
def get_args():
    parser = argparse.ArgumentParser(description='Get activation vectors')
    parser.add_argument('--dataset_dir', default="./data", type=str, help="Number of members in ensemble")
    parser.add_argument('--num_classes', default=6, type=int, help="Number of classes in dataset")
    parser.add_argument('--means', nargs='+', default=[0.4914, 0.4822, 0.4465], type=float,
                        help="channelwise means for normalization")
    parser.add_argument('--stds', nargs='+', default=[0.2023, 0.1994, 0.2010], type=float,
                        help="channelwise std for normalization")
    parser.add_argument('--save_path', default="./saved_features/cifar10", type=str,
                        help="Path to save the ensemble weights")
    parser.add_argument('--load_path', default="checkpoint/ckpt.pth", type=str,
                        help="Path to save the ensemble weights")
    parser.add_argument('--unrconst', '-unrconst', default=0, action='store_true')
    parser.add_argument('--overclass', '-overclass', default=0, action='store_true')
    parser.add_argument('--feat', '-feat', default=0, action='store_true')
    parser.add_argument('--full', '-full', default=0, action='store_true')
    parser.add_argument('--res', '-res', default=0, action='store_true')


    parser.add_argument('--extraclasses', default=1, type=int, help='Classes used in testing')

    parser.set_defaults(argument=True)

    return parser.parse_args()

args = get_args()

print(args.unrconst)
def epoch_openset(net, save_path, root):
    net.eval()
    if args.res:
        args.unrconst =1
    with torch.no_grad():
        for folder in os.listdir(os.path.join(root, "open_set")):
            count = 0

            for file_name in os.listdir(os.path.join(root, "open_set", str(folder))):

                # if (count >= 120):
                #     break
                count = count + 1

                image = Image.open(os.path.join(root, "open_set", str(folder), file_name)).convert("RGB")
                image = transform_test(image)
                image = torch.unsqueeze(image, 0)
                image = image.cuda(non_blocking=True)
                if args.overclass:
                    logits, _, _, _ = net(image)
                elif args.res:
                    logits, _, _, _ = net(image)
                else:

                    logits, inlayer, latent = net(image)
                squeezed_latent = []
                squeezed_latent.append(torch.squeeze(logits))
                if args.feat:
                    latent = inlayer
                if args.unrconst :
                    x=3
                else:
                    for layer in latent:

                        m = nn.AdaptiveAvgPool2d((1, 1))
                        new_layer = torch.squeeze(m(layer))
                        squeezed_latent.append(new_layer)

                feature = torch.cat(squeezed_latent, 0)

                save_name = file_name.split(".")[0]
                _,out = logits.max(1)

                folderw = folder
                # if args.overclass:
                #     folderw = str(out.item())
                os.makedirs(os.path.join(save_path, "open_set", str(folderw)),exist_ok=True)
                np.save(os.path.join(save_path, "open_set", str(folderw), save_name + ".npy"), feature.cpu().data.numpy(),
                        allow_pickle=False)


def epoch_train(net, save_path, root):
    net.eval()
    i=0
    arr = np.zeros(10)
    cnt =0
    if args.res:
        args.unrconst =1
    with torch.no_grad():
        for folder in os.listdir(os.path.join(root, "train")):
            print(folder)

            i +=1

            if i==6:
                continue
            j=0

            for file_name in os.listdir(os.path.join(root, "train", str(folder))):
                j+=1
                image = Image.open(os.path.join(root, "train", str(folder), file_name)).convert("RGB")
                image = transform_test(image)
                image = torch.unsqueeze(image, 0)
                image = image.cuda(non_blocking=True)
                if args.overclass:
                    logits, _ ,_,_= net(image)
                elif args.res:
                    logits, _, _, _ = net(image)
                else:

                    logits, inlayer, latent = net(image)

                squeezed_latent = []
                squeezed_latent.append(torch.squeeze(logits))
                if args.feat:
                    latent = inlayer
                if args.unrconst :

                    x=3
                else:
                    for layer in latent:


                        m = nn.AdaptiveAvgPool2d((1, 1))
                        new_layer = torch.squeeze(m(layer))
                        squeezed_latent.append(new_layer)

                feature = torch.cat(squeezed_latent, 0)

                save_name = file_name.split(".")[0]
                _,out = logits.max(1)


                if int(out/2) != int(folder) and args.overclass:
                    continue
                if (out) != int(folder) and args.overclass ==0:
                    continue
                folderw = folder
                cnt +=1
                arr[out.item()] +=1
                if args.overclass:
                    folderw = str(out.item())
                os.makedirs(os.path.join(save_path, "train", str(folderw)),exist_ok=True)
                np.save(os.path.join(save_path, "train", str(folderw), save_name + ".npy"), feature.cpu().data.numpy(),
                        allow_pickle=False)
            print("exit")

    print(arr)
def epoch_val(net, save_path, root):
    net.eval()
    if args.res:
        args.unrconst =1
    with torch.no_grad():
        for folder in os.listdir(os.path.join(root, "val")):

            for file_name in os.listdir(os.path.join(root, "val", str(folder))):

                image = Image.open(os.path.join(root, "val", str(folder), file_name)).convert("RGB")
                image = transform_test(image)
                image = torch.unsqueeze(image, 0)
                image = image.cuda(non_blocking=True)
                if args.overclass:
                    logits, _, _, _ = net(image)
                elif args.res:
                    logits, _, _, _ = net(image)
                else:

                    logits, inlayer, latent = net(image)
                squeezed_latent = []
                squeezed_latent.append(torch.squeeze(logits))
                if args.feat:
                    latent = inlayer
                if args.unrconst :
                    x = 3
                else:
                    for layer in latent:
                        if args.unrconst and not(args.feat):
                            break
                        m = nn.AdaptiveAvgPool2d((1, 1))
                        new_layer = torch.squeeze(m(layer))
                        squeezed_latent.append(new_layer)

                feature = torch.cat(squeezed_latent, 0)

                save_name = file_name.split(".")[0]
                _, out = logits.max(1)
                # if int(out / 2) != int(folder) and args.overclass:
                #     continue
                # if (out) != int(folder) and args.overclass == 0:
                #     continue

                folderw = folder
                # if args.overclass:
                #     folderw = str(out.item())
                os.makedirs(os.path.join(save_path, "val", str(folderw)), exist_ok=True)
                np.save(os.path.join(save_path, "val", str(folderw), save_name + ".npy"), feature.cpu().data.numpy(),
                        allow_pickle=False)



def main():
    global transform_test

    seed = 222
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    num_classes = 10
    # args.num_classes
    print("Num classes " + str(num_classes))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.means, args.stds),
    ])
    root = args.dataset_dir

    if args.overclass and args.extraclasses == 1:
        args.extraclasses = 2
    if args.overclass:
        net = ResNet18(num_classes=20)
    elif args.res:
        net = ResNet18(num_classes=10)
    else:
        net = models.DHRNet(num_classes, extraclasses=args.extraclasses)
    print(args.unrconst)
    if args.overclass:
        args.load_path = "resnet overclass/ckpt.pth"
        # args.load_path = "overclass/ckpt.pth"
    elif args.res and args.full:
        args.load_path = "resnetfull/ckpt.pth"
    elif args.unrconst:
        args.load_path = "unrconst/ckpt.pth"
    print(args.load_path)
    checkpoint = torch.load(args.load_path, map_location="cpu")
    if args.overclass or args.res:
        new_weights = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
    else:
        new_weights = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}

    net.load_state_dict(new_weights)
    net.cuda()

    epoch_train(net, args.save_path, root)
    epoch_val(net, args.save_path, root)
    epoch_openset(net, args.save_path, root)
    print("finish get feat")


if __name__ == "__main__":
    main()
