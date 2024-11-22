import tqdm
import numpy as np
import wandb
import torch



def test(model, device, test_loader, criterion):
    """
    Performs the evaluation of a trained model on a test dataset. 
    
    Parameters:
    - model: This is the trained model that will be evaluated.
    - device: The device (e.g., CPU or GPU) on which the evaluation will be performed.
    - test_loader: A data loader that provides the test dataset in batches.
    - criterion: The loss criterion used to calculate the loss between the model's predictions and the target labels.
    
    Returns:
    - np.mean(losses): The mean of all the losses calculated during the evaluation.
    - all_preds: A list that contains the predicted labels for all the test samples.
    """

    losses = []
    model.eval()
    all_preds = []
    probas = []

    t = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
    t.set_description('Test')
    with torch.no_grad():
        correct, total = 0, 0
        for batch_idx, (data, target) in t: #iterem sobre les dades
            data, target = data.to(device), target.to(device)
            output, probs, _ = model(data)
            #output = model(data) #for wavenet
            probas.extend(probs.detach().cpu().numpy())
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, target.long())
            losses.append(loss.item())
            t.set_postfix(loss=loss.item())
            all_preds.extend(predicted.detach().cpu().numpy())
            total += target.size(0)
            correct += (predicted == target).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        #wandb.log({"test_accuracy": correct / total})
        

    return np.mean(losses), all_preds, probas, correct/total