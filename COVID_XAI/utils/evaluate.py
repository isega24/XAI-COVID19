import sys
import cv2
import torch
torch.set_num_threads(3)
import torch.nn.functional as F
import torch.optim as optim
import tqdm

def eval(model,
        dataloader,
        loss_fn,
        criterion,
        device,):
    model.eval()
    loss_sum = 0
    correct = 0
    # Actualizar el mensaje del tqdm para mostrar el loss y el acc progresivo
    processed = 0
    loader = tqdm.tqdm(dataloader)
    with torch.no_grad():
        for batch_index, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data = data.float()
            output = model(data)
            loss_ = loss_fn(output, target)
            loss_sum += loss_
            correct += criterion(output, target)
            processed += len(output)
            loader.set_description(desc=f'Loss={loss_sum/processed } Batch_index= {batch_index} Accuracy={100*correct/processed:0.2f}%')
        
    return loss_sum/len(dataloader), correct/len(dataloader)