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
import mains.main_spectrograms as main_spectrograms
import mains.main_mfccs as main_mfccs
import mains.main_chroma as main_chroma
import mains.main_whisper as main_whisper
import download_data
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import itertools


# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 30
batch_size = 50        # number of samples during training
test_batch_size = 50  # number of samples for test 
train_size = 0.8
test_size = 0.2

train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}

def majority_vote(predictions_list):
    """
    Perform majority voting on a list of predictions from different models.
    """
    # Transpose the list of lists so that we can iterate over each set of predictions for each sample
    predictions_list = list(zip(*predictions_list))
    
    final_predictions = []
    for predictions in predictions_list:
        most_common_prediction, count = Counter(predictions).most_common(1)[0]
        # If there's a tie or all predictions are different, default to the first prediction
        if count == 1:
            final_predictions.append(predictions[0])
        else:
            final_predictions.append(most_common_prediction)
    
    return final_predictions

def highest_probability(probas_spec, probas_mfccs, probas_chroma, probas_whisper, 
                          predictions_spec, predictions_mfccs, predictions_chroma, predictions_whisper):
    highest_probabilities = []
    for prob_spec, prob_mfcc, prob_chroma, prob_whisper, pred_spec, pred_mfcc, pred_chroma, pred_whisper in zip(
        probas_spec, probas_mfccs, probas_chroma, probas_whisper, 
        predictions_spec, predictions_mfccs, predictions_chroma, predictions_whisper):
        
        max_prob_spec = np.max(prob_spec)
        max_prob_mfcc = np.max(prob_mfcc)
        max_prob_chroma = np.max(prob_chroma)
        max_prob_whisper = np.max(prob_whisper)

        max_probs = [max_prob_spec, max_prob_mfcc, max_prob_chroma, max_prob_whisper]
        max_index = np.argmax(max_probs)

        if max_index == 0:
            highest_probabilities.append(pred_spec)
        elif max_index == 1:
            highest_probabilities.append(pred_mfcc)
        elif max_index == 2:
            highest_probabilities.append(pred_chroma)
        else:
            highest_probabilities.append(pred_whisper)

    return highest_probabilities

def highest_mean_probability(probas_spec, probas_mfccs, probas_chroma, probas_whisper, 
                          predictions_spec, predictions_mfccs, predictions_chroma, predictions_whisper):
    mean_probs = []
    for prob_spec, prob_mfcc, prob_chroma, prob_whisper in zip(
        probas_spec, probas_mfccs, probas_chroma, probas_whisper):

        mean_prob_spec = np.mean(prob_spec)
        mean_prob_mfcc = np.mean(prob_mfcc)
        mean_prob_chroma = np.mean(prob_chroma)
        mean_prob_whisper = np.mean(prob_whisper)

        mean_probs.append((mean_prob_spec, mean_prob_mfcc, mean_prob_chroma, mean_prob_whisper))

    highest_mean_probabilities = []
    for mean_prob, pred_spec, pred_mfcc, pred_chroma, pred_whisper in zip(
        mean_probs, predictions_spec, predictions_mfccs, predictions_chroma, predictions_whisper):

        max_mean_prob_index = np.argmax(mean_prob)
        if max_mean_prob_index == 0:
            highest_mean_probabilities.append(pred_spec)
        elif max_mean_prob_index == 1:
            highest_mean_probabilities.append(pred_mfcc)
        elif max_mean_prob_index == 2:
            highest_mean_probabilities.append(pred_chroma)
        else:
            highest_mean_probabilities.append(pred_whisper)

    return highest_mean_probabilities

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
fashion_mnist_classes = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental',
                         'International', 'Pop', 'Rock']

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(8):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(fashion_mnist_classes)


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 128))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

if __name__ == "__main__":
    
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB")

    model = EmbeddingRNN('spectrogram').to(device)
    n_classes = 8
    #model = ClassificationNet(embedding_net, n_classes=n_classes)
    model.to(device)
    model_spectrograms, targets, probas_spectrograms, predictions_spectrograms,accuracy_spectrograms, train_spectrograms_dataloader, test_spectograms_dataloader, spectrogram_accuracies = main_spectrograms.main(model,epochs)
    train_embeddings_spectrograms_baseline, train_labels_baseline = extract_embeddings(train_spectrograms_dataloader, model)
    val_embeddings_spectrograms_baseline, val_labels_baseline = extract_embeddings(test_spectograms_dataloader, model)
    #print(train_embeddings_spectrograms_baseline)

    
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB")

    model = EmbeddingRNN('mfccs').to(device)
    n_classes = 8
    #model = ClassificationNet(embedding_net, n_classes=n_classes)
    model.to(device)
    model_mfccs, targets, probas_mfccs, predictions_mfccs,accuracy_mfccs, train_mfccs_dataloader, test_mfccs_dataloader, mfcc_accuracies = main_mfccs.main(model,epochs)
    train_embeddings_mfccs_baseline, train_labels_baseline = extract_embeddings(train_mfccs_dataloader, model)
    val_embeddings_mfccs_baseline, val_labels_baseline = extract_embeddings(test_mfccs_dataloader, model)
    #print(train_embeddings_mfccs_baseline)
    
    torch.cuda.empty_cache()
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB")

    model = EmbeddingRNN('chroma').to(device)
    n_classes = 8
    #model = ClassificationNet(embedding_net, n_classes=n_classes)
    model.to(device)
    model_chroma, targets, probas_chroma, predictions_chroma, accuracy_chroma,train_chroma_dataloader, test_chroma_dataloader, chroma_accuracies = main_chroma.main(model,epochs)
    train_embeddings_chroma_baseline, train_labels_baseline = extract_embeddings(train_chroma_dataloader, model)
    val_embeddings_chroma_baseline, val_labels_baseline = extract_embeddings(test_chroma_dataloader, model)
    #print(train_embeddings_mfccs_baseline)

    torch.cuda.empty_cache()
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB")

    model = EmbeddingRNN('whisper').to(device)
    n_classes = 8
    #model = ClassificationNet(embedding_net, n_classes=n_classes)
    model.to(device)
    model_whisper, targets, probas_whisper, predictions_whisper, accuracy_whisper,train_whisper_dataloader, test_whisper_dataloader, whisper_accuracies = main_whisper.main(model,epochs)
    train_embeddings_whisper_baseline, train_labels_baseline = extract_embeddings(train_whisper_dataloader, model)
    val_embeddings_whisper_baseline, val_labels_baseline = extract_embeddings(test_whisper_dataloader, model)
    #print(train_embeddings_whisper_baseline)

    torch.cuda.empty_cache()
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB")

    train_embeddings_concatenated = np.concatenate(
    (train_embeddings_spectrograms_baseline,
        train_embeddings_mfccs_baseline,
        train_embeddings_chroma_baseline,
        train_embeddings_whisper_baseline), axis=1
    )
    print(train_embeddings_concatenated.shape)

    val_embeddings_concatenated = np.concatenate(
    (val_embeddings_spectrograms_baseline,
        val_embeddings_mfccs_baseline,
        val_embeddings_chroma_baseline,
        val_embeddings_whisper_baseline), axis=1
    )
    print(train_embeddings_concatenated.shape)

    print(f"final_predictions sp: {predictions_spectrograms[:10]}")
    print(len(predictions_spectrograms))
    print(f"final_predictions mf: {predictions_mfccs[:10]}")
    print(len(predictions_mfccs))
    print(f"final_predictions ch: {predictions_chroma[:10]}")
    print(len(predictions_chroma))
    print(f"final_predictions wh: {predictions_whisper[:10]}")
    print(len(predictions_chroma))

    all_predictions = [predictions_spectrograms, predictions_mfccs, predictions_chroma, predictions_whisper]
    final_predictions_mj = majority_vote(all_predictions)
    final_predictions_hp = highest_probability(probas_spectrograms, probas_mfccs, probas_chroma, probas_whisper, predictions_spectrograms, predictions_mfccs, predictions_chroma, predictions_whisper)
    final_predictions_hmp = highest_mean_probability(probas_spectrograms, probas_mfccs, probas_chroma, probas_whisper, predictions_spectrograms, predictions_mfccs, predictions_chroma, predictions_whisper)

    # Print to debug
    print(f"final_predictions: {final_predictions_mj[:10]}")
    print(f"targets: {targets[:10]}")
    print(len(final_predictions_mj),len(targets))

        # Print to debug
    print(f"final_predictions: {final_predictions_hp[:10]}")
    print(f"targets: {targets[:10]}")
    print(len(final_predictions_hp),len(targets))

        # Print to debug
    print(f"final_predictions: {final_predictions_hmp[:10]}")
    print(f"targets: {targets[:10]}")
    print(len(final_predictions_hmp),len(targets))

    print(f'Spectrograms Accuracy: {accuracy_spectrograms}')
    print(f'MFCCS Accuracy: {accuracy_mfccs}')
    print(f'Chroma Accuracy: {accuracy_chroma}')
    print(f'Whisper Accuracy: {accuracy_whisper}')


    # Evaluate the final predictions
    print(classification_report(targets, final_predictions_mj, target_names=[
        'Electronic','Experimental','Folk','Hip-Hop','Instrumental', 'International', 'Pop', 'Rock']))

    print(accuracy_score(targets, final_predictions_mj))

        # Evaluate the final predictions
    print(classification_report(targets, final_predictions_hp, target_names=[
        'Electronic','Experimental','Folk','Hip-Hop','Instrumental', 'International', 'Pop', 'Rock']))

    print(accuracy_score(targets, final_predictions_hp))

        # Evaluate the final predictions
    print(classification_report(targets, final_predictions_hp, target_names=[
        'Electronic','Experimental','Folk','Hip-Hop','Instrumental', 'International', 'Pop', 'Rock']))

    print(accuracy_score(targets, final_predictions_hp))

    all_labels = []

    for data in train_spectrograms_dataloader:
        inputs, labels = data
        all_labels.extend(labels.cpu().numpy())  # Assuming labels are on the same device as the DataLoader

    # Convert to a NumPy array
    all_labels = np.array(all_labels)

    X_train = train_embeddings_concatenated
    X_test = val_embeddings_concatenated
    y_train = all_labels
    y_test = targets

    
    Xtrain_tensor = torch.tensor(X_train, dtype=torch.float32)
    Xtest_tensor = torch.tensor(X_test, dtype=torch.float32)
    ytrain_tensor = torch.tensor(y_train, dtype=torch.long)  # Assuming ytrain contains integer labels
    ytest_tensor = torch.tensor(y_test, dtype=torch.long)    # Assuming ytest contains integer labels

    # Create TensorDataset for train and test sets
    train_dataset = TensorDataset(Xtrain_tensor, ytrain_tensor)
    test_dataset = TensorDataset(Xtest_tensor, ytest_tensor)

    model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    embedding_accuracies = []
    predictions_embeddings = []
    
    def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=10):
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")
            
            
            # Test the model
            test_model(model, criterion, test_loader)

    def test_model(model, criterion, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 0)
                total += 1
                correct += (predicted == labels).item()
                predictions_embeddings.append(predicted)
        accuracy = correct / total

        print(f"Test Accuracy: {accuracy:.4f}")
        #wandb.log({"test_accuracy": correct / total})
        embedding_accuracies.append(accuracy)

    train_model(model, criterion, optimizer, train_dataset, test_dataset, epochs)

    print("spectrogram accuracies: ", spectrogram_accuracies)
    print("mfcc accuracies: ",mfcc_accuracies)
    print("chroma accuracies: ",chroma_accuracies)
    print("whisper accuracies: ", whisper_accuracies)
    print("embeding accuracies:" ,embedding_accuracies)
    
    epochs = [i+1 for i in range(epochs)]
    plt.plot(epochs, spectrogram_accuracies, label='Spectrograms')
    plt.plot(epochs, mfcc_accuracies, label='MFCCs')
    plt.plot(epochs, chroma_accuracies, label='Chroma Frequencies')
    plt.plot(epochs, whisper_accuracies, label='Whisper')
    plt.plot(epochs, embedding_accuracies, label='Embeddings')
    
    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.legend()

    plt.savefig("./visualitzations/accuracies.png")
    # Display the plot
    plt.show()
    
    def compute_scores(y_true, y_pred):
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        accuracy = accuracy_score(y_true, y_pred)
        return precision, recall, f1, accuracy

    # Example usage:
    models = ["Spctrograms", "MFCCs", "Chroma Frequencies", "Whisper", "Embeddings", "Voting Classifier","Highest Probability","Highest mean probability"]
    y_true = targets  
    y_preds = [
        predictions_spectrograms,  
        predictions_mfccs,  
        predictions_chroma,  
        predictions_whisper,  
        predictions_embeddings[:len(targets)],
        final_predictions_mj,
        final_predictions_hp,
        final_predictions_hmp   
    ]

    # Print header
    print("Model\tPrecision\tRecall\tF1 Score\tAccuracy")

    # Loop through each model
    for i, model in enumerate(models):
        # Compute scores for the current model
        precision, recall, f1, accuracy = compute_scores(y_true, y_preds[i])
        
        # Print scores for each class
        
        print(f"{model}\t\t{precision:.4f}\t\t{recall:.4f}\t{f1:.4f}\t\t{accuracy:.4f}")
    
    # Map genre labels to indices and vice versa
    genre_dict = {'Electronic': 0, 'Experimental': 1, 'Folk': 2, 'Hip-Hop': 3,
              'Instrumental': 4, 'International': 5, 'Pop': 6, 'Rock': 7}
    reverse_genre_dict = {v: k for k, v in genre_dict.items()}
    targets_labels = [reverse_genre_dict[target] for target in targets]
    predictions_labels = [reverse_genre_dict[prediction] for prediction in predictions_spectrograms]

    # Compute confusion matrix
    cm = confusion_matrix(targets_labels, predictions_labels, labels=list(genre_dict.keys()))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Spectrograms')
    plt.colorbar()
    tick_marks = np.arange(len(genre_dict))
    plt.xticks(tick_marks, genre_dict.keys(), rotation=45)
    plt.yticks(tick_marks, genre_dict.keys())

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("./visualitzations/cm_spectrograms.png")

    predictions_labels = [reverse_genre_dict[prediction] for prediction in final_predictions_mj]

        # Compute confusion matrix
    cm = confusion_matrix(targets_labels, predictions_labels, labels=list(genre_dict.keys()))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Majority Voting')
    plt.colorbar()
    tick_marks = np.arange(len(genre_dict))
    plt.xticks(tick_marks, genre_dict.keys(), rotation=45)
    plt.yticks(tick_marks, genre_dict.keys())

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("./visualitzations/cm_mj.png")

    predictions_labels = [reverse_genre_dict[prediction] for prediction in final_predictions_hp]

        # Compute confusion matrix
    cm = confusion_matrix(targets_labels, predictions_labels, labels=list(genre_dict.keys()))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Highest Probability')
    plt.colorbar()
    tick_marks = np.arange(len(genre_dict))
    plt.xticks(tick_marks, genre_dict.keys(), rotation=45)
    plt.yticks(tick_marks, genre_dict.keys())

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("./visualitzations/cm_hp.png")

    predictions_labels = [reverse_genre_dict[prediction] for prediction in final_predictions_hmp]

      # Compute confusion matrix
    cm = confusion_matrix(targets_labels, predictions_labels, labels=list(genre_dict.keys()))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Highest Mean Probability')
    plt.colorbar()
    tick_marks = np.arange(len(genre_dict))
    plt.xticks(tick_marks, genre_dict.keys(), rotation=45)
    plt.yticks(tick_marks, genre_dict.keys())

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("./visualitzations/cm_hmp.png")