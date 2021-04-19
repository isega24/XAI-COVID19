import sys
import cv2
import torch
torch.set_num_threads(3)
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from efficientnet_pytorch import EfficientNet
from .load_data import COVIDGR
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from GenerativeAdversarialExplanation.lime_with_autovectors.limeAV import LIMEImgAug
from skimage.segmentation import quickshift
import numpy as np

SEED = 314159265
#from  import LIMEImgAug

device = "cuda" if torch.cuda.is_available() else "cpu" 

def myquickshift(image,final_shape = (64,64)):
    initial_shape = image.shape
    image_ = image.numpy()
    image_ = cv2.resize(image_,final_shape)
    
    segments = quickshift(image_,ratio=1,kernel_size=5,max_dist=100,sigma=5,random_seed=SEED)
    segments = cv2.resize(segments,initial_shape[0:2],interpolation=cv2.INTER_NEAREST)
    return segments

transformObject = LIMEImgAug(segmentation_fn=myquickshift, n_hide=4,num_classes=4,final_size=(224,224))
trainIMG = transformObject.transform
trainTarget = transformObject.target_transform
testIMG = torchvision.transforms.Lambda(lambda x:transformObject.transform(x,train=False))
testTarget = transformObject.target_transform


data = COVIDGR(data_folder="./data/COVIDGR-09-04/Revised-croped/",severity=True)
train, test = random_split(data,lengths=[int(len(data)*0.8),len(data)-int(len(data)*0.8)],
    generator=torch.Generator().manual_seed(SEED))#train_test_split(data,stratify=True,test_size=0.2,random_state=SEED)

trainTransform = lambda x: (trainIMG(x[0]),trainTarget(x[1]))
testTransform = lambda x: (testIMG(x[0]),testTarget(x[1]))

def before(sample):
    imagen = torch.unsqueeze(torch.Tensor(sample[0][0]),dim=-1)
    imagen = imagen.repeat((1,1,3))
    imagen = imagen.double()
    target = torch.argmax(sample[1])
    return (imagen,target)

def after(sample):
    imagen = torch.Tensor(sample[0][0:1,:,:])
    return (imagen,sample[1])


train.dataset.addTransform(torchvision.transforms.Compose([before,
    trainTransform,
    after]) )
test.dataset.addTransform(torchvision.transforms.Compose([before,
    testTransform,
    after]))


batch_size = 16
TrainLoader = DataLoader(train,batch_size=batch_size,shuffle=True)
TestLoader = DataLoader(test,batch_size=batch_size,shuffle=False)

clasifier = EfficientNet.from_pretrained("efficientnet-b0",num_classes=4,in_channels=1)
clasifier.to(device)
def criterion_classification(ypred,y_label):

    ypred = torch.sigmoid(ypred)
    clase = torch.argmax(y_label,axis=-1)

    indexes = torch.arange(0,len(y_label[0]))
    indexes = indexes.unsqueeze(0).repeat((len(ypred),1))
    clase = clase.unsqueeze(-1).repeat((1,len(y_label[0])))
    
    condition = indexes.to(device) <= clase.to(device)
    y_label = torch.where(condition,1,0)

    y_label = y_label.type(torch.FloatTensor).to(device)
    

    #y_label = torch.where(clase >  ,0,1)
    
    return F.mse_loss(ypred,y_label)

if len(sys.argv) == 1:
    epochsC = 100
else:
    epochsC = int(sys.argv[1])

optimizer = optim.Adam(clasifier.parameters(), lr=0.0001,weight_decay=0.01,amsgrad=True)
epochs = range(epochsC)


best_loss = None
for epoch in epochs:
    batches = tqdm.tqdm(enumerate(TrainLoader),total = len(TrainLoader))
    running_loss = 0.0
    total, m = 0,0
    clasifier.train()
    print(f"Epoch {epoch}")
    for i, data in batches:
        # get the inputs; data is a list of [inputs, labels]
        data = data
        inputs, labels = data
        inputs = inputs.float().to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = clasifier(inputs)

        labelsAcc = torch.argmax(labels,axis=-1)
        outputsAcc = torch.argmax(outputs,axis=-1)
        result = (outputsAcc == labelsAcc).float()
        m += torch.sum(result)
        total+=len(result)

        loss = criterion_classification(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_reported = loss.item()
        running_loss += loss_reported
        batches.set_postfix({"Running loss": running_loss/total*batch_size,
            "Running Acc":(m/total).cpu().numpy()})
    
    accIterator = tqdm.tqdm(enumerate(TestLoader),total=len(TestLoader))
    total,m = 0, 0.0
    clasifier.eval()
    last_loss = 0.0
    for i, data in accIterator :
        
        data = data
        inputs, labels = data
        total +=len(inputs)
        inputs = inputs.float().to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        outputs = clasifier(inputs)
        last_loss += criterion_classification(outputs, labels).item()

        labels = torch.argmax(labels,axis=-1)
        outputs = torch.argmax(outputs,axis=-1)
        result = (outputs == labels).float()
        m += torch.sum(result)
        accIterator.set_postfix({"Running Val loss": last_loss/total*batch_size,
            "Running Val Acc":(m/total).cpu().numpy()})
    if best_loss is None or best_loss > last_loss:
        best_loss = last_loss
        torch.save(clasifier.state_dict(), "./models/clasifier_ordinal.pt")
        print(f"New best model: val_loss {best_loss}")
m = 0

accIterator = tqdm.tqdm(enumerate(TestLoader),total=len(TestLoader))
total = 0
clasifier.eval()
for i, data in accIterator :
    
    data = data
    inputs, labels = data
    total +=len(inputs)
    inputs = inputs.float().to(device)
    labels = labels.to(device)

    # forward + backward + optimize
    outputs = clasifier(inputs)

    labels = torch.argmax(labels,axis=-1)
    outputs = torch.argmax(outputs,axis=-1)
    result = (outputs == labels).float()
    m += torch.sum(result)
    accIterator.set_postfix({"Running Val acc":(m/total).cpu().numpy()})
print(m*1.0/total)
