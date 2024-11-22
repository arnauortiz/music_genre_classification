import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_curve
import numpy as np
import wandb
import matplotlib.pyplot as plt


def init_weights(model, method = 'Kaiming'):
    """
    Initialization method for the weights and biases of layer of the model.

    Parameters:
    - model: Model that we want to initialize weights and biases.
    - method: Method of initialization. Kaiming (Kaiming Uniform) and Xavier (Xavier Uniform) supported. By default Kaiming is the selected method.
    """

    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            if method == 'Kaiming':
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif method == 'Xavier':
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)      

def calcular_parametres_del_model(model):
    """
    Function that returns the total number of parameters of the model.
    
    Parameters:
    - model: Model whose number of parameters we want to evaluate.

    Return:
    - pytorch_total_params: Number of parameters (int).
    """

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# trainable parameters: {:,}".format(pytorch_total_params))
    return pytorch_total_params



def plot_roc_curve(targets, probas, class_names):
    """
    Plots the roc curve and logs it to wandb.

    Parameters:
    - targets: List containing the ground truth of the genres.
    - probas: List containing the probabilities of predicting each genre for each spectrogram.
    - class_names: List containing the class names.
    """
    y_true = np.array(targets)
    y_probas = np.array(probas)
    fpr = dict()
    tpr = dict()

    for i in range(8):
        fpr[i], tpr[i], _ = roc_curve(y_true,y_probas[...,i], pos_label = i)

        plt.plot(fpr[i], tpr[i], lw=2, label=class_names[i])


    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    wandb.init(project="MusicGenreClassification")

    wandb.log({"chart": plt})


def showImage(img, ax):
    # convert the tensor to numpy
    out = img.numpy()
    # Bring to the 0-255 range
    out = out - out.min()
    out = out / out.max()
    out = out * 255
    out = out.astype('uint8')
    # Plot image
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(out, cmap='gray')

def showConvMap(conv_map):
    # Create a grid of images
    h = conv_map.shape[0] # = number of images in the batch
    w = conv_map.shape[1] # = number of activation maps per image
    fig, ax = plt.subplots(h, w, figsize=(10, 10))

    # Plot activation maps
    for i in range(conv_map.shape[0]):
        for j in range(conv_map.shape[1]):
            showImage(conv_map[i][j], ax[i, j])

    fig.tight_layout()
def hook_ShowOutput(module, input, output):
    print("Output shape:", output.shape)
    showConvMap(output.cpu())
