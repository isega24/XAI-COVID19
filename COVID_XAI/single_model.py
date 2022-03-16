import torch
import torchvision

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from Damax.DAMAX.damax import DamaxAug

from COVID_XAI.utils.load_data import COVIDGR
from COVID_XAI.utils.training import train_epochs
from COVID_XAI.utils.evaluate import eval





torch.set_num_threads(3)
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu" 

parser = argparse.ArgumentParser(description='')

parser.add_argument('--epochs', metavar='E', type=int,default=50,
                    help='Number of epochs trained')

parser.add_argument('--seed', metavar='S', type=int,default=31415,
                    help='Seed for random number generator')
parser.add_argument('--model', metavar='M', type=str,default="efficientnet",
                    help='Specific model to use')
parser.add_argument('--name', metavar='N', type=str,default="efficientnet-b0",
                    help='Specific model to use')
parser.add_argument('--mode', metavar='m', type=str,default="binary",
                    help='Binary, severity or monotonic')
parser.add_argument('--lr', metavar='L', type=float,default=0.0001,
                    help='Binary, severity or monotonic')
args = parser.parse_args()

Dataset = COVIDGR()
num_classes = 2 if args.mode == "binary" else 5
transformObject = DamaxAug(n_hide=2,num_classes=num_classes,final_size=(224,224))

trainIMG = transformObject.transform
trainTarget = transformObject.target_transform
testIMG = torchvision.transforms.Lambda(lambda x:(transformObject.transform(x[0],train=False),x[1]))
testTarget = transformObject.target_transform
covidgr = COVIDGR(transform=testIMG)

Train,Test = random_split(covidgr, [5*int(len(covidgr)//6),len(covidgr)-5*int(len(covidgr)//6)],generator=torch.Generator().manual_seed(args.seed))
Train, Val = random_split(Train, [int(len(Train)*0.9), len(Train)-int(len(Train)*0.9)],generator=torch.Generator().manual_seed(args.seed))

batch_size = 8
TrainLoader = DataLoader(Train,batch_size=batch_size,shuffle=True)
ValLoader = DataLoader(Val,batch_size=batch_size,shuffle=False)
TestLoader = DataLoader(Test,batch_size=batch_size,shuffle=False)


if args.mode == "binary":
    from COVID_XAI.models.binary import BinaryModel as Model, BinaryLoss as Loss, BinaryCriterion as Criterion
elif args.mode == "severity":
    from COVID_XAI.models.severity import SeverityModel as Model, SeverityLoss as Loss, SeverityCriterion as Criterion
elif args.mode == "monotonic":
    from COVID_XAI.models.monotonic_severity import MonotonicModel as Model, MonotonicLoss as Loss, MonotonicCriterion as Criterion
else:
    print("No model described")
    exit()

if args.model == "efficientnet":
    from efficientnet_pytorch import EfficientNet as ModelConstructor
elif args.model == "resnet":
    from torchvision.models import ResNet as ModelConstructor
elif args.model == "vgg":
    from torchvision.models import VGG as ModelConstructor
else:
    print("No model specified")
    exit()
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name(args.name,num_classes=num_classes)
#model = Model(args.name,num_classes=num_classes,ModelConstructor=ModelConstructor)

name = args.mode+"_"+args.model+"_"+str(args.seed)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_epochs(name=name,model=model, train_loader=TrainLoader,val_loader= ValLoader, optimizer=optimizer,loss=Loss,epochs=args.epochs,device=device,criterion=Criterion)
eval(model=model,dataloader=TestLoader,loss_fn=Loss,criterion=Criterion,device=device)
    