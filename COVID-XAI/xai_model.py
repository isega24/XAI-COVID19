import torch
from GenerativeAdversarialExplanation.lime_with_autovectors.limeAV import LIMEImgAug,LIMEAVExplainer
from torch.nn.functional import softmax
from torch.utils.data.dataloader import DataLoader
from .load_data import COVIDGR
from torch.utils.data import random_split
torch.set_num_threads(3)
import torchvision
import tqdm
import numpy as np
from efficientnet_pytorch import EfficientNet
import cv2
from skimage.segmentation import quickshift, mark_boundaries

from lime import lime_image

SEED = 314159265


device = "cuda" if torch.cuda.is_available() else "cpu" 

data = COVIDGR(data_folder="./data/COVIDGR-09-04/Revised-croped/")
_, Test = random_split(data,lengths=[int(len(data)*0.8),len(data)-int(len(data)*0.8)],
    generator=torch.Generator().manual_seed(SEED))

def myquickshift(image : torch.Tensor,final_shape = (64,64)):
    initial_shape = image.shape
    image_ = np.array(image)
    if image_.shape[0] == 1:
        image_ = image_[0]
        initial_shape = image_.shape
        image_ = np.double(np.repeat(np.expand_dims(image_,-1),3,axis=-1))
    image_ = cv2.resize(image_,final_shape)
    segments = quickshift(image_,ratio=1,kernel_size=5,max_dist=100,sigma=5,random_seed=SEED)
    segments = cv2.resize(segments,initial_shape[0:2],interpolation=cv2.INTER_NEAREST)
    return segments

def neutral(image):
    image = np.array(image) 
    return np.zeros(image.shape) + image.mean()

transformObject = LIMEImgAug(segmentation_fn=myquickshift, fn_neutral_image=neutral,n_hide=4,num_classes=2,final_size=(224,224))
testIMG = torchvision.transforms.Lambda(lambda x:transformObject.transform(x,train=False))
testTarget = transformObject.target_transform
testTransform = lambda x: (testIMG(x[0]),testTarget(x[1]))

def before(sample):
    #import matplotlib.pyplot as plt
    #plt.imsave("Original.jpg",sample[0][0],cmap="gray")
    imagen = torch.unsqueeze(torch.Tensor(sample[0][0]),dim=-1)
    imagen = imagen.repeat((1,1,3))
    imagen = imagen.double()
    target = torch.argmax(sample[1])
    return (imagen,target)

def after(sample):
    imagen = torch.Tensor(sample[0][0:1,:,:])
    #import matplotlib.pyplot as plt
    #plt.imsave("Ocluded.jpg",imagen[0],cmap="gray")
    #exit()
    return (imagen,sample[1])

Test.dataset.addTransform(torchvision.transforms.Compose([before,
    testTransform,
    after]))

batch_size = 16
TestLoader = DataLoader(Test,batch_size=batch_size,shuffle=False)


classifiers = []
names = ["clasifier","clasifier_no_transf"]
for file in [f"./models/{name}.pt" for name in names]:
    
    classifier = EfficientNet.from_pretrained("efficientnet-b0",num_classes=2,in_channels=1)# Classification(num_classes=10)
    loaded_dict = torch.load(file)
    classifier.load_state_dict(loaded_dict)
    classifier.to(device)
    classifier.eval()
    classifiers.append(classifier)


for data in tqdm.tqdm(TestLoader,total=len(TestLoader)):
    inputs,labels = data
    for inp in inputs:

        for i,classifier in enumerate(classifiers):
            def classify(image):
                image = np.expand_dims(image,axis=0)
                image = torch.Tensor(image).to(device)
                #image = torch.unsqueeze(image,0)
                image.to(device)
                result = classifier(image)
                return result
            def classify1(image):
                image = np.expand_dims(image,axis=0)
                image = torch.Tensor(image).to(device)
                image = torch.unsqueeze(image,0)
                image.to(device)
                result = classifier(image)
                return result
            def classify_lime(images):
                images = images[:,:,:,0]
                result = []
                for image in images:
                    result.append(softmax(classify1(image),dim=-1).cpu().detach().numpy())
                result = np.array(result)
                
                result = result[:,0,:]
                return result

            limeAV = LIMEAVExplainer(classify,transformObject,top_k=2)
            segments = myquickshift(inp)
            
            expl_matrix,seg = limeAV.explain_instance_regression(inp,segments,name=names[i],random=False,max_examples=1000)
            
            limeExplainer = lime_image.LimeImageExplainer()
            limeExpl =limeExplainer.explain_instance(np.double(np.array(inp[0])) ,
                classify_lime,
                segmentation_fn=myquickshift,
                random_seed=SEED,
                num_samples=1000)
            
            image, mask = limeExpl.get_image_and_mask(0,positive_only=False,hide_rest=False)
            img_boundry = mark_boundaries(image/255.0, mask)
            
            import matplotlib.pyplot as plt
            plt.imsave(f"{names[i]}_lime.jpg",img_boundry)
        exit()