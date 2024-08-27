import os
import sys
sys.path.append('../')

import torch
import h5py

from escnn import gspaces
from escnn import nn

from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *

from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import RandomRotation
# from torchvision.transforms import Pad
# from torchvision.transforms import Resize
# from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
# from torchvision.transforms import InterpolationMode

import numpy as np
from scipy.ndimage import rotate
from PIL import Image

device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

class c8steerable3Dcnn(torch.nn.Module):
    def __init__(self):
        super(c8steerable3Dcnn,self).__init__()

        self.r3_act = gspaces.icoOnR3()
        
        in_type = nn.FieldType(self.r3_act, [self.r3_act.trivial_repr])

        self.input_type = in_type
        # Convolution 1
        out_type = nn.FieldType(self.r3_act, 24*[self.r3_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.MaskModule(in_type, 29, margin=1),
            nn.R3Conv(in_type,out_type,kernel_size=7,padding=1,bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # Convolution 2
        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r3_act, 48*[self.r3_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R3Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased3D(in_type, sigma=0.66, stride=2)
        )

        # Convolution 3
        in_type = self.block2.out_type
        out_type = nn.FieldType(self.r3_act, 48*[self.r3_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R3Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # Convolution 4
        in_type = self.block3.out_type
        out_type = nn.FieldType(self.r3_act, 64*[self.r3_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R3Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased3D(in_type, sigma=0.66, stride=2)
        )

        # Convolution 5
        in_type = self.block4.out_type
        out_type = nn.FieldType(self.r3_act, 64*[self.r3_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R3Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # Convolution 6
        in_type = self.block5.out_type
        out_type = nn.FieldType(self.r3_act, 32*[self.r3_act.regular_repr])
        self.block6 = nn.SequentialModule(
            nn.R3Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = nn.PointwiseAvgPoolAntialiased3D(in_type, sigma=0.66, stride=1, padding=0)
        self.gpool = nn.GroupPooling(out_type)

    def forward(self, input: torch.Tensor):
        x = nn.GeometricTensor(input, self.input_type)

        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        
        x = self.gpool(x)

        x = x.tensor
        # x = self.fully_net(x.reshape(x.shape[0], -1))
        return x

class h5dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform

        with h5py.File(file_path[0],'r') as file:
            self.data_shape = file['/data'].shape
        self.num_samples = len(file_path)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        with h5py.File(self.file_path[index],'r') as file:
            data = file['/data'][:]
        if self.transform:
            data = self.transform(data)

        return data
    
class NumpyTo3DTensor:
    def __call__(self, array):
        return torch.from_numpy(array).unsqueeze(0).float()


def list_of_h5_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]

def equivariance_test(model, dataloader):
    model.eval()
    results = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            orig_data = data.to(device)

            angle = 45
            transformed_data = rotate(orig_data.cpu().numpy(), angle, axes=(2,3), reshape=False)
            transformed_data = torch.from_numpy(transformed_data).to(device)

            original_output = model(orig_data)
            transformed_output = model(transformed_data)

            equivariant = torch.allclose(original_output, transformed_output, rtol=1e-5, atol=1e-5)
            results.append(equivariant)

            print(f'File {idx+1}: Equivariant = {equivariant}')

    return results

directory = './../../escnn_trial_data/67files/test_file/'
file_path = list_of_h5_files(directory)

transform = Compose([
    NumpyTo3DTensor(),
])

dataset = h5dataset(file_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

model = c8steerable3Dcnn().to(device)
results = equivariance_test(model, dataloader)

for idx, result in enumerate(results):
    print(f'File {file_path[idx]}: Equivariant = {result}')