import torch
import os
torch.set_num_threads(4)
import torch.nn.functional as F
import tqdm



def train(model, train_loader, optimizer,loss,criterion, device):
    # Usar tqdm para mostrar el progreso de la ejecución
    # y además mostrar el loss y el acc progresivo
    model.to(device)
    model.train()
    dataloader = tqdm.tqdm(train_loader)
    
    correct = 0
    processed = 0
    total_loss = 0
    total_0 = 0
    for batch_index, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        '''
        from matplotlib import pyplot as plt
        plt.imshow(data[0][0], cmap='gray')
        print(target[0])
        plt.show()
        exit()
        '''
        
        
        data = data.float()
        optimizer.zero_grad()
        # Forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # Compute the loss
        loss_ = loss(output, target)
        total_loss += loss_.item()
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss_.backward()
        # Update the weights
        optimizer.step()
        # Update dataloader-tqdm
        
        
        correct += criterion(output, target)
        processed += len(data)
        dataloader.set_description(desc=f'Loss={total_loss/processed} Criterion={100*correct/processed:0.2f}')
        
        
    return total_loss / len(train_loader)
def val(model, val_loader,loss,criterion, device):
    # Usar tqdm para mostrar el progreso de la ejecución
    # y además mostrar el loss y el acc progresivo
    model.to(device)
    model.eval()
    dataloader = tqdm.tqdm(val_loader)
    correct = 0
    processed = 0
    total_loss = 0
    with   torch.no_grad():
        for batch_index, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            data = data.float()
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # Compute the loss
            loss_ = loss(output, target)
            total_loss += loss_.item()
            # Update dataloader-tqdm
            
            correct += criterion(output, target)
            processed += len(data)
            dataloader.set_description(desc=f'Loss={total_loss/processed} Criterion={100*correct/processed:0.2f}')
    return total_loss / len(val_loader)
    
def train_epochs(name,model, train_loader, val_loader, optimizer,loss, epochs, device,criterion):
    BestLoss = float('inf')
    for epoch in range(1, epochs + 1):
        
        train_loss = train(model, train_loader, optimizer,loss,criterion, device)
        val_loss = val(model, val_loader,loss, criterion,device)
        print(f'Epoch: {epoch:02} Train_loss: {train_loss:.3f} Val_loss: {val_loss:.3f}')
        # Si el val_loss es menor que el mejor val_loss, guardar el modelo
        if val_loss < BestLoss:
            BestLoss = val_loss
            print(f'Epoch: {epoch:02} New best validation loss achieved. Saving model...')
            # Si el directorio ./saved_models no existe, crearlo
            if not os.path.isdir('./saved_models'):
                os.mkdir('./saved_models')
            torch.save(model.state_dict(), f'./saved_models/{name}.pt')
    return model