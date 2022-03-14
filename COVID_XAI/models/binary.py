import torch
torch.set_num_threads(3)
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


def BinaryModel(name,num_classes,ModelConstructor=EfficientNet):
    model = ModelConstructor.from_pretrained(name,num_classes=num_classes)
    # Freeze al parameters
    for param in model.parameters():
        param.requires_grad = False
    # Add new layers with a softmax activation
    in_features = model._fc.in_features
    model._fc = torch.nn.Sequential(
        torch.nn.Linear(in_features, in_features/3),       # Add a linear layer
        torch.nn.ReLU(),
        torch.nn.Linear(in_features/3, num_classes),
        torch.nn.Softmax(dim=1)               # Add a softmax activation
    )
    return model

def BinaryLoss(pred, target):
    # pred : (batch_size, 2)
    # target : (batch_size, 2)
    target = target[:,0]
    target = torch.argmax(target, dim=1)
    target[target > 1] = 1
    

    
    
    # target : (batch_size, )
    return F.cross_entropy(pred, target)
def BinaryCriterion(pred,target):
    # Accuracy metric
    target = target[:,0]
    target = torch.argmax(target, dim=1)
    target[target > 1] = 1
    pred = torch.argmax(pred, dim=1)
    correct = pred.eq(target).sum().item()
    return correct 