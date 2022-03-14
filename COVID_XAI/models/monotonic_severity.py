import torch
torch.set_num_threads(3)
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

def MonotonicModel(name,num_classes,ModelConstructor=EfficientNet):
    model = ModelConstructor.from_pretrained(name,num_classes=num_classes)
    # Freeze al parameters
    for param in model.parameters():
        param.requires_grad = False
    # Add new layers with a softmax activation
    in_features = model._fc.in_features
    model._fc = torch.nn.Sequential(
        torch.nn.Linear(in_features, in_features//2),       # Add a linear layer
        torch.nn.ReLU(),
        torch.nn.Linear(in_features//2, num_classes),       # Add a linear layer
        torch.nn.Sigmoid()               # Add a softmax activation
    )
    return model

def MonotonicLoss(pred, target):
    
    target = torch.cumsum(target[:,0], dim=1)
    
    target = target.float()
    # target : (batch_size, 4)
    return F.binary_cross_entropy(pred, target)
def MonotonicCriterion(pred,target):
    # Accuracy metric
    target = target[:,0]
    
    target = torch.argmax(target, dim=1)
    pred = torch.argmax(pred, dim=1)
    
    correct0 = torch.sum(pred.eq(target) & (target.eq(0))).float()
    
    correct_rest = torch.sum(pred.not_equal(0) & (target.not_equal(0))).float()
    percentage = torch.sum(correct0+correct_rest)
    
    return percentage