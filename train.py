import tqdm
import numpy as np
import wandb

def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    Performs the training of a model on a given dataset, 
    updates the model's parameters using backpropagation and the specified optimizer, 
    logs the training progress, and returns the mean loss value calculated during training.    
    
    Parameters:
    - model: The model to be trained.
    - device: The device (e.g., CPU or GPU) on which the training will be performed.
    - train_loader: A data loader that provides the training dataset in batches.
    - optimizer: The optimizer used to update the model's parameters.
    - criterion: The loss criterion used to calculate the loss between the model's predictions and the target labels.
    - epoch: The current epoch number.
    
    Returns:
    - np.mean(losses): The mean of all the losses calculated during training.
    
    """
    #wandb.watch(model, criterion, log="all", log_freq=10)

    losses = []
    model.train()
    example_ct = 0  

    t = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    t.set_description('Train')
    for batch_idx, (data, target) in t:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #backpropagation
        output, _, _ = model(data)
        #output = model(data) #for wavenet
        loss = criterion(output, target.long())
        loss.backward()

        optimizer.step()

        example_ct +=  len(data)

        losses.append(loss.item())
        t.set_postfix(loss=loss.item())

        if ((batch_idx + 1) % 25) == 0:
               train_log(loss, example_ct, epoch)

    return np.mean(losses)


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    """
    Used to log the loss value and print it during the training process, providing progress updates to the user.
    
    Parameters:
    - loss: The loss value to be logged.
    - example_ct: The total number of examples seen so far.
    - epoch: The current epoch number.
    
    Returns:
    - None
    """
    #wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
