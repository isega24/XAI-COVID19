import torch
torch.set_num_threads(3)
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class Monotonic(torch.nn.Module):
    def __init__(self,name,num_classes,ModelConstructor=EfficientNet):
        super(Monotonic, self).__init__()
        self.feature_extractor = ModelConstructor.from_pretrained(name,num_classes=num_classes)
        # Freeze al parameters
        self.feature_extractor.freeze()
        # Add new layers with a softmax activation
        in_features = self.feature_extractor._fc.in_features
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, int(in_features/3)),       # Add a linear layer
            torch.nn.ReLU(),
            torch.nn.Linear(int(in_features/3), num_classes)
        )
        
    def forward(self,x):
        x = self.feature_extractor.extract_features(x)
        x = self.max_pool(x)
        x = x.view(x.shape[0],-1)
        return self.fc(x)

def MonotonicModel(name,num_classes,ModelConstructor=EfficientNet):
    return Monotonic(name,num_classes,ModelConstructor)

def MonotonicLoss(pred, target):
    target = torch.cumsum(target[:,0], dim=1)
    target = target.float()
    pred = torch.sigmoid(pred)
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