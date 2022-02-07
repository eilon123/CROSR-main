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
import argparse
import matplotlib.pyplot as plt
from entropyloss import *
from tsne import *
from models import *
hist = np.zeros(10)

def get_args():
    parser = argparse.ArgumentParser(description='Train DHR Net')
    parser.add_argument('--lr',default=0.05,type=float,help="learning rate")
    parser.add_argument('--epochs',default=500,type=int,help="Number of training epochs")
    parser.add_argument('--batch_size',default=500,type=int,help="Batch size")
    parser.add_argument('--dataset_dir',default="./data/cifar10",type=str,help="Number of members in ensemble")
    parser.add_argument('--num_classes',default=6,type=int,help="Number of classes in dataset")
    parser.add_argument('--means',nargs='+',default=[0.4914, 0.4822, 0.4465], type=float,help="channelwise means for normalization")
    parser.add_argument('--stds',nargs='+',default=[0.2023, 0.1994, 0.2010],type=float,help="channelwise std for normalization")
    parser.add_argument('--momentum',default=0.9,type=float,help="momentum")
    parser.add_argument('--weight_decay',default=0.0005,type=float,help="weight decay")
    parser.add_argument('--save_path',default="./save_models/cifar10",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--unrconst', '-unrconst', default=0, action='store_true')
    parser.add_argument('--trans', '-trans', default=0, action='store_true')
    parser.add_argument('--r', '-r', default=0, action='store_true')
    parser.add_argument('--train', '-t', default=0, action='store_true')
    parser.add_argument('--overclass', '-overclass', default=0, action='store_true')
    parser.add_argument('--imgsave', '-imgsave', default=0, action='store_true')
    parser.add_argument('--tsne', '-tsne', default=0, action='store_true')
    parser.add_argument('--full', '-full', default=0, action='store_true')
    parser.add_argument('--res', '-res', default=0, action='store_true')


    parser.add_argument('--extraclasses', default=1, type=int, help='Classes used in testing')

    parser.set_defaults(argument=True)

    return parser.parse_args()

def tensorToImg(in_ten):
    imgs = []
    for t in in_ten:
        img = np.transpose(t.numpy(),(1,2,0))
        imgs.append(img)
    return imgs

def saveIMAG(imgs,labels,address):
    address = "data/" + address + "/"
    for i,img in enumerate(imgs):
        os.makedirs(address + "{:d}".format(int(labels[i])), exist_ok=True)
        plt.imsave(address + "{:d}/{:d}.png".format(int(labels[i]), int(hist[int(labels[i])])), img)
        # os.makedirs("data/openset/{:d}".format(int(labels[i])),exist_ok=True)
        # plt.imsave("data/openset/{:d}/{:d}.png".format(int(labels[i]), int(hist[int(labels[i])])), img)
        hist[labels[i]] += 1
def epoch_train(epoch_no,net,trainloader,optimizer,ent=0):
        
    net.train() 
    correct=0
    total=0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconst_loss = 0.0
    iter=0
    cls_criterion = nn.CrossEntropyLoss()
    reconst_criterion = nn.MSELoss()

    args = get_args()

    if args.overclass:
        args.unrconst = 1
    for i,data in enumerate(trainloader):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if args.imgsave:
            imgs = tensorToImg(inputs)
            saveIMAG(imgs, labels, "train")
            continue
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits, reconstruct,_ = net(inputs)
        if args.overclass:
            uniquenessLoss, uniformLoss, newOutputs = ent.CalcEntropyLoss(logits,0)
            logits = newOutputs
            reconst_loss = torch.Tensor([0])

        cls_loss = cls_criterion(logits, labels)
        if args.unrconst == False:
            reconst_loss = reconst_criterion(reconstruct,inputs)

        # if(torch.isnan(cls_loss) or torch.isnan(reconst_loss)):
        #     print("Nan at iteration ",iter)
        #     cls_loss=0.0
        #     reconst_loss=0.0
        #     logits=0.0
        #     reconstruct = 0.0
        #     continue
        if args.unrconst:
            loss = cls_loss
        else:
            loss = cls_loss + reconst_loss
        if args.overclass:
            loss += uniformLoss + uniquenessLoss
        loss.backward()
        optimizer.step()  

        total_loss = total_loss + loss.item()
        total_cls_loss = total_cls_loss + cls_loss.item()
        if args.unrconst == 0:
            total_reconst_loss = total_reconst_loss + reconst_loss.item()

        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        iter = iter + 1
    if args.imgsave:
        return
    return [(100 * (correct / total)), (total_cls_loss/iter), (total_reconst_loss/iter), (total_loss/iter)]
    
def epoch_val(net,testloader,ent=0):

    net.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconst_loss = 0.0
    iter=0
    cls_criterion = nn.CrossEntropyLoss()
    reconst_criterion = nn.MSELoss()
    batch_idx = 0
    args = get_args()
    totalPredict = 0
    output = 0

    with torch.no_grad():
        for data in testloader:

            images, labels = data
            if args.imgsave:

                imgs = tensorToImg(images)
                saveIMAG(imgs, labels,"val")
                continue
            images=images.cuda(non_blocking=True)
            labels=labels.cuda(non_blocking=True)
            if args.res:
                logits ,_,_,_= net(images)
            else:
                logits, reconstruct,_ = net(images)
            if args.overclass:
                uniquenessLoss, uniformLoss, newOutputs = ent.CalcEntropyLoss(logits, 0)
                _,predictedext = logits.max(1)
                logits = newOutputs
                _,predicted = logits.max(1)
                totalPredict, output = getFeat(batch_idx, True, predicted, predictedext, totalPredict,
                                           logits, output)

            batch_idx = 1
            if args.res == 0:
                cls_loss = cls_criterion(logits, labels)
            if args.res==0:
                reconst_loss = reconst_criterion(reconstruct,images)
        
                loss = cls_loss + reconst_loss
                total_loss = total_loss + loss.item()

                total_cls_loss = total_cls_loss + cls_loss.item()

            # total_reconst_loss = total_reconst_loss + reconst_loss.item()
            if args.res:
                prob, predicted = logits.max(1)
            else:
                _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            iter = iter + 1
    if args.imgsave:
        exit()
    if (args.tsne or args.train ==0 )and args.overclass:
        showtsne(output, totalPredict, numClasses=20)
        plt.figure()
        plt.hist(totalPredict, bins=20)
        plt.show()
        # plt.savefig(self.address[0:13] + "predict_hist")

    return [(100 * (correct / total)), (total_cls_loss/iter), (total_reconst_loss/iter), (total_loss/iter)]
                 
def setTrans(net,optimizer):
    for idx, l in enumerate(next(net.children()).children()):
        for param in l.parameters():
            param.requires_grad = idx ==22

            if idx == 22:
                if param.dim() > 1:  # wieghts
                    torch.nn.init.kaiming_uniform_(param)
                            # torch.nn.init.xavier_uniform_(param)
                else:  # Bias
                    torch.nn.init.zeros_(param)
    optimizer.param_groups[0]["lr"] = 1e-2

def main():

    seed = 222
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    args = get_args()

    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    momentum= args.momentum
    weight_decay= args.weight_decay
    means = args.means
    stds = args.stds
    

    num_classes = args.num_classes
    num_classes = 10
    print("Num classes "+str(num_classes))

    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, hue=0.3),
        transforms.RandomAffine(degrees=30,translate =(0.2,0.2),scale=(0.75,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means,stds),
    ])




    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means,stds),
    ])
    if args.imgsave:
        transform_test = transforms.Compose([

            transforms.ToTensor(),

        ])
        transform_train = transforms.Compose([

            transforms.ToTensor(),

        ])
    if args.overclass:
        args.unrconst = True
    root = args.dataset_dir
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    indices = []
    if not(args.trans) and not(args.full):
        for ind in range(5):
            indices +=([i for i, l in enumerate(trainset.targets[:]) if l == ind])
        indices.sort()

        trainset.targets = [trainset.targets[i] for i in indices]
        trainset.data = [trainset.data[i] for i in indices]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True,pin_memory=True,drop_last=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    indices = []
    if not(args.trans) and not(args.full):
        for ind in range(5):
            indices += ([i for i, l in enumerate(testset.targets[:]) if l == ind])
        indices.sort()

        testset.targets = [testset.targets[i] for i in indices]
        testset.data = [testset.data[i] for i in indices]

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False,pin_memory=True,drop_last=True)
    if args.overclass and args.extraclasses == 1:
        args.extraclasses =2
    if args.res:
        net = ResNet18(
            num_classes=10)
    else:
        net = models.DHRNet(num_classes,args.extraclasses)
    ent = EntropyLoss(args,'cuda')
    best_acc = 0

    if args.r:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        if args.overclass:
            address = "overclass/ckpt.pth"
        elif args.res and args.full:
            address = "resnetfull/ckpt.pth"
        elif args.unrconst and not args.full:
            address = "unrconst/ckpt.pth"
        elif args.full:
            address = "full/ckpt.pth"
        else:
            address = "checkpoint/ckpt.pth"
        checkpoint = torch.load(address, map_location="cpu")
        if args.res:
            new_weights = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
        else:
            new_weights = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        net.load_state_dict(new_weights)



    net = torch.nn.DataParallel(net.cuda())

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    if args.trans:
        setTrans(net,optimizer)
    for epoch in range(epochs):  # loop over the dataset multiple times
        if args.train:
            train_acc = epoch_train(epoch,net,trainloader,optimizer,ent)
            print(
                "Train accuracy and cls, reconstruct and total loss for epoch " + str(epoch) + " is " + str(train_acc))

        test_acc = epoch_val(net,testloader,ent)
        scheduler.step()
        print("Test accuracy and cls, reconstruct and total loss for epoch "+str(epoch)+" is "+str(test_acc))
        if args.train==0:
            break
        if not(args.trans)  and test_acc[0] > best_acc:
            best_acc = test_acc[0]
            print("-----------------------saving")
            if args.overclass:
                address = "overclass/ckpt.pth"
                print(address)
                torch.save({'epoch': epoch,
                            'model_state_dict': net.module.state_dict(),
                            'train_acc': train_acc[0],
                            'train_loss': train_acc[3],
                            'val_acc': test_acc[0],
                            'val_loss': test_acc[3]},
                           address)

            elif args.unrconst and not args.full:

                address = "unrconst/ckpt.pth"
                print(address)
                torch.save({'epoch': epoch,
                            'model_state_dict': net.module.state_dict(),
                            'train_acc': train_acc[0],
                            'train_loss': train_acc[3],
                            'val_acc': test_acc[0],
                            'val_loss': test_acc[3]},
                           address)
            elif args.full:
                address = "full/ckpt.pth"
                print(address)
                torch.save({'epoch': epoch,
                            'model_state_dict': net.module.state_dict(),
                            'train_acc': train_acc[0],
                            'train_loss': train_acc[3],
                            'val_acc': test_acc[0],
                            'val_loss': test_acc[3]},
                           address)
            else:
                torch.save({'epoch':epoch,
                             'model_state_dict':net.module.state_dict(),
                             'train_acc':train_acc[0],
                             'train_loss':train_acc[3],
                              'val_acc':test_acc[0] ,
                              'val_loss':test_acc[3]},
                  "checkpoint/ckpt.pth")




if __name__ == "__main__":
    main()
    
