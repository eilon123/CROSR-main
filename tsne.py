import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import wandb

colors_per_class = {
    'plane' : [254, 202, 87],
    'car' : [255, 107, 107],
    'bird' : [10, 189, 227],
    'cat' : [255, 159, 243],
    'deer' : [16, 172, 132],
    'dog' : [128, 80, 128],
    'frog' : [87, 101, 116],
    'horse' : [52, 31, 151],
    'ship' : [0, 0, 0],
    'truck' : [100, 100, 255],
    'unknown': [159, 159, 159]

}

colors_per_class = {
    'one' : [254, 202, 87],
    'two' : [255, 107, 107],
    'three' : [10, 189, 227],
    'four' : [255, 159, 243],
    'five' : [16, 172, 132],
    'six' : [128, 80, 128],
    'seven' : [87, 101, 116],
    'eight' : [52, 31, 151],
    'nine' : [0, 0, 0],
    'zero' : [100, 100, 255],

}

colors_per_classEx = {
    'plane1' : [254, 202, 87],
    'plane2' : [254, 202, 0],
    'car1' : [255, 107, 107],
    'car2' : [255, 107, 0],
    'bird1' : [10, 189, 227],
    'bird2' : [10, 189, 0],
    'cat1' : [255, 159, 243],
    'cat2': [255, 159, 0],
    'deer1' : [16, 172, 132],
    'deer2' : [16, 172, 0],
    'dog1' : [128, 80, 128],
    'dog2' : [128, 80, 0],
    'frog1' : [87, 101, 116],
    'frog2' : [87, 101, 0],
    'horse1' : [52, 31, 151],
    'horse2' : [52, 31, 0],
    'ship1' : [0, 0, 0],
    'ship2' : [0, 0, 255],
    'truck1' : [100, 100, 255],
    'truck2' : [100, 100, 0],
    'unknown': [159, 159, 159],

}


def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))

def showtsne(features,targets ,address=0,mode=0,numClasses=10):
    if numClasses > 10:
        colors_per_classt = colors_per_classEx
    else:
        colors_per_classt = colors_per_class
    print("starting TSNE")
    tsne = TSNE(n_components=2).fit_transform(features)
    print("finish TSNE")
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    cm = plt.get_cmap('gist_rainbow')

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    # initialize a matplotlib plot
    NUM_COLORS = 20
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_color_cycle([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    # for every class, we'll add a scatter plot separately
    for ind ,label in enumerate(colors_per_classt):

        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(targets) if l == ind]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        color = np.array(colors_per_classt[label], dtype=np.float) / 255

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # wandb.log({"point_cloud": wandb.Object3D(ax)})
    # finally, show the plot
    if address==0:
        plt.show(        )
    else:
        plt.savefig(address+ 'tsne.png')

        wandb.log({mode: wandb.Image(address + 'tsne.png')})
    # plt.show()

# scale and move the coordinates so they fit [0; 1] range
def getFeat(idx,overclass,predicted,predictedext,totalPredict,current_outputs,output):
    if overclass:
        if idx == 0:
            output = current_outputs.cpu().numpy()
            totalPredict = predictedext.cpu()
        else:
            output = np.concatenate((output, current_outputs.cpu().numpy()))
            totalPredict = np.concatenate((totalPredict, predictedext.cpu()))

    else:

        if idx == 0:
            output = current_outputs.cpu().numpy()
            totalPredict = predicted.cpu()
        else:
            output = np.concatenate((output, current_outputs.cpu().numpy()))
            totalPredict = np.concatenate((totalPredict, predicted.cpu()))

    return totalPredict , output
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
