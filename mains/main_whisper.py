import os
import random
import wandb

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from train import *
from test import *
from utils.utils import *
from tqdm.auto import tqdm
from models.models import *
from models.models_utils import *
import whisper_loader
import download_data
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
#import whisper_loader


# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 30
batch_size = 25       # number of samples during training
test_batch_size = 25  # number of samples for test 
train_size = 0.8
test_size = 0.2

train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}


def main(model, epochs=30):
    """
    Overall, this script performs the following steps: data loading, model initialization, training loop, evaluation, logging, and model saving.
    """
    whisper_loader.main()
    """
    In order to disable WandB, comment both lines below, line 74, line 47 in test.py and lines 23, 64 in train.py
    """
    #wandb.login()
    #with wandb.init(project="MusicGenreClassificationDefinitive"):

    data_list, genres_list = LoadDataPipeline('whisper_tiny')
    
    train_dataloader,test_dataloader, targets = CreateTrainTestLoaders(data_list, genres_list, train_size,
                                                                train_kwargs, test_kwargs, False, sample_fraction=1)

    print("Initializing model")
    #model = WaveNet(input_shape=(80,2997),num_classes=8)
    model = model
    print("Paràmetes model RNN:",calcular_parametres_del_model(model))
    model.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #weight_decay = 1e-4

    #Scheduler that will modify the learning ratio dinamically according to the test loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    model.to(device)
    print("Beginning epochs")

    accmax = 0
    whisper_accuracies = []
    for epoch in range(1, epochs + 1):
        print("Epoch: ", epoch)
        loss_train_epoch = train(model, device, train_dataloader, optimizer, loss, epoch)
        loss_test_epoch, prediction, probas, acc = test(model, device, test_dataloader, loss)
        scheduler.step(loss_test_epoch)
        if acc > accmax:
            print('Max Accuracy Increased, Saving Predictions')
            accmax = acc
            y_pred = prediction
        whisper_accuracies.append(acc)
    return model, targets, probas, y_pred, accmax, train_dataloader, test_dataloader, whisper_accuracies


if __name__ == "__main__":
    model, targets, probas, y_pred, accmax, _ , _, _ = main(RNN('whisper'))

    class_names =['Electronic','Experimental','Folk','Hip-Hop',
        'Instrumental', 'International', 'Pop', 'Rock']

    plot_roc_curve(targets, probas, class_names)
    #wandb.log({"conf_mat" : wandb.plot.confusion_matrix(preds=prediction,y_true=targets,class_names=class_names)})
    PATH="./modelsguardats/" + model.name
    torch.save(model.state_dict(), PATH)
