import torch
torch.set_num_threads(3)
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


def BinaryModel(name,num_classes,ModelConstructor=EfficientNet):
    model = ModelConstructor.from_pretrained(name,num_classes=num_classes)
    # Freeze al parameters
    #for param in model.parameters():
    #    param.requires_grad = False
    # Add new layers with a softmax activation
    in_features = model._fc.in_features
    fc = torch.nn.Sequential(
        torch.nn.Linear(in_features, int(in_features/3)),       # Add a linear layer
        torch.nn.ReLU(),
        torch.nn.Linear(int(in_features/3), num_classes)
    )
    model = torch.nn.ModuleList([model.extract_features, fc])
    return model

def BinaryLoss(pred, target):
    pred = torch.softmax(pred, dim=1)
    
    target = target[:,0]
    target = torch.argmax(target, dim=1)
    target[target > 1] = 1 
    
    return F.cross_entropy(pred, target)

def BinaryCriterion(pred,target):
    pred = torch.argmax(pred, dim=1)
    
    target = target[:,0]
    target = torch.argmax(target, dim=1)
    target[target > 1] = 1
    correct = pred.eq(target).sum().item()

    return correct 