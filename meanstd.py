import torch
import torchvision
from torchvision import datasets, transforms

def get_meanstd(datadir):
    train_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.ToTensor()
            ])
    train_set = datasets.ImageFolder(datadir,transform=train_transforms)
    
    loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), num_workers=1)
    
    channels_sum,channels_squared_sum,num_batches=0,0,0
    
    for data,_ in loader:
        channels_sum+=torch.mean(data,[0,2,3])
        channels_squared_sum+=torch.mean(data**2,[0,2,3])
        num_batches+=1
        
    mean=channels_sum/num_batches
    std=(channels_squared_sum/num_batches-mean**2)**0.5
    return mean,std

#plt.hist(data[0].flatten())
#plt.axvline(data[0].mean())
