import os
import torch
from skimage import io
from torch.utils.data import Dataset
import numpy as np
import torchvision


class COVIDGR(Dataset):
    def __init__(self,data_folder="./data/COVIDGR-09-04/Revised/",
            metadata_file="./data/COVIDGR-09-04/Metadatos.csv",
            severity="False",
            transform = None):
        self.data_folder = data_folder
        self.metadata_file = metadata_file
        self.dataset = []
        self.transform = transform

        if self.transform is None:
            self.transform = self._default_transform
        with open(metadata_file,"r") as csv_file:
            for i, line in enumerate(csv_file):
                line = line.split(",")
                if i > 0:
                    filename = line[2]
                    if not os.path.exists(f"{data_folder}{filename}.jpg"):
                        print(f"Row: {i} ID: {line[0]} Date: {line[1]} File: {data_folder}{filename}.jpg")

                    elif line[7] == "PA":
                        if not severity:
                            clase = line[6]
                            if clase in ["VP","FN"]:
                                clase = [1,0]
                            elif clase in ["VN","FP"]:
                                clase = [0,1]
                            self.dataset.append((f"{filename}.jpg",clase))
                        else:
                            positivo = line[6] in ["VP","FN"]
                            if positivo:
                                clase = line[12]
                                if clase == "NORMAL":
                                    clase = [0,1,0,0,0]
                                elif clase == "LEVE":
                                    clase = [0,0,1,0,0]
                                elif clase == "MODERADO":
                                    clase = [0,0,0,1,0]
                                elif clase == "GRAVE":
                                    clase = [0,0,0,0,1]
                            else:
                                clase = [1,0,0,0,0]
                            self.dataset.append((f"{filename}.jpg",clase))

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.data_folder, self.dataset[idx][0])
        image = io.imread(img_name,as_gray=True)

        clase = np.array([self.dataset[idx][1]])
        sample = image,clase

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def _default_transform(self,sample):
        sample_ = [None,None]
        sample_[0] = torch.unsqueeze(torch.Tensor(sample[0]),dim=0)*255
        sample_[1] =  torch.Tensor(sample[1])[0]
        return sample_
    
    def addTransform(self,new_transform):
        self.transform = torchvision.transforms.Compose([self.transform,
        new_transform])

if __name__ == "__main__":
    data = COVIDGR(data_folder="./data/COVIDGR-09-04/Revised-croped/")
    
    for d in data:
        print(d[0].shape,d[1].shape)
