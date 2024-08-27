
from typing import Tuple, Union
from collections import defaultdict

from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *

import time
import torch
from torch import nn, optim
import numpy as np
import numpy
from torch.utils.data import DataLoader, TensorDataset, Dataset

from scipy import stats

import pandas as pd
import os
import h5py
import math


class ResBlock(EquivariantModule):

    def __init__(self, in_type: FieldType, channels: int, out_type: FieldType = None, stride: int = 1, features: str = '2_96'):

        super(ResBlock, self).__init__()

        self.in_type = in_type
        if out_type is None:
            self.out_type = self.in_type
        else:
            self.out_type = out_type

        self.gspace = self.in_type.gspace

        if features == 'ico':
            L = 2
            grid = {'type': 'ico'}
        elif features == '2_96':
            L = 2
            grid = {'type': 'thomson_cube', 'N': 4}
        elif features == '2_72':
            L = 2
            grid = {'type': 'thomson_cube', 'N': 3}
        elif features == '3_144':
            L = 3
            grid = {'type': 'thomson_cube', 'N': 6}
        elif features == '3_192':
            L = 3
            grid = {'type': 'thomson_cube', 'N': 8}
        elif features == '3_160':
            L = 3
            grid = {'type': 'thomson', 'N': 160}
        else:
            raise ValueError()

        so3: SO3 = self.in_type.fibergroup
        #####

        # number of samples for the discrete Fourier Transform
        S = len(so3.grid(**grid))

        #  We try to keep the width of the model approximately constant
        _channels = channels / S
        _channels = int(round(_channels))

        # Build the non-linear layer
        # Internally, this module performs an Inverse FT sampling the `_channels` continuous input features on the `S`
        # samples, apply ELU pointwise and, finally, recover `_channels` output features with discrete FT.
        ftelu = FourierELU(self.gspace, _channels, irreps=so3.bl_irreps(L), inplace=True, **grid)
        res_type = ftelu.in_type

        print(f'ResBlock: {in_type.size} -> {res_type.size} -> {self.out_type.size} | {S*_channels}')

        self.res_block = SequentialModule(
            R3Conv(in_type, res_type, kernel_size=3, padding=1, bias=False, initialize=False),
            IIDBatchNorm3d(res_type, affine=True),
            ftelu,
            R3Conv(res_type, out_type, kernel_size=3, padding=1, stride=stride, bias=False, initialize=False),
        )

        # self.deconv = R3ConvTransposed(self.out_type, self.out_type, kernel_size=3, padding=1, stride=stride, output_padding=stride-1, bias=False, initialize=False)
        # self.reconv = R3Conv(self.out_type, self.out_type, kernel_size=3, padding=1, stride=1, bias=False, initialize=False)

        if stride > 1:
            self.downsample = PointwiseAvgPoolAntialiased3D(in_type, .33, 2, 1)
        else:
            self.downsample = lambda x: x

        if self.in_type != self.out_type:
            self.skip = R3Conv(self.in_type, self.out_type, kernel_size=1, padding=0, bias=False)
        else:
            self.skip = lambda x: x

    def forward(self, input: GeometricTensor):

        assert input.type == self.in_type

        # res = self.res_block(input)
        # res = self.deconv(res)
        # res = self.reconv(res)

        return self.skip(self.downsample(input)) + self.res_block(input)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape


class SE3CNN(nn.Module):

    def __init__(self, pool: str = "snub_cube", res_features: str = '2_96', init: str = 'delta'):

        super(SE3CNN, self).__init__()

        self.gs = rot3dOnR3()

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr])

        self._init = init

        layer_types = [
            (FieldType(self.gs, [self.build_representation(2)] * 3), 200),
            (FieldType(self.gs, [self.build_representation(3)] * 2), 480),
            (FieldType(self.gs, [self.build_representation(3)] * 6), 480),
            (FieldType(self.gs, [self.build_representation(3)] * 12), 960),
            (FieldType(self.gs, [self.build_representation(3)] * 8), None),
        ]

        blocks = [
            R3Conv(self.in_type, layer_types[0][0], kernel_size=5, padding=2, stride=1, bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                ResBlock(layer_types[i][0], layer_types[i][1], layer_types[i+1][0], stride=1, features=res_features)
            )

        # For pooling, we map the features to a spherical representation (bandlimited to freq 2)
        # Then, we apply pointwise ELU over a number of samples on the sphere and, finally, compute the average
        # # (i.e. recover only the frequency 0 component of the output features)
        if pool == "icosidodecahedron":
            # samples the 30 points of the icosidodecahedron
            # this is only perfectly equivarint to the 12 tethrahedron symmetries
            grid = self.gs.fibergroup.sphere_grid(type='ico')
        elif pool == "snub_cube":
            # samples the 24 points of the snub cube
            # this is perfectly equivariant to all 24 rotational symmetries of the cube
            grid = self.gs.fibergroup.sphere_grid(type='thomson_cube', N=1)
        else:
            raise ValueError(f"Pooling method {pool} not recognized")

        ftgpool = QuotientFourierELU(self.gs, (False, -1), 128, irreps=self.gs.fibergroup.bl_irreps(2), out_irreps=self.gs.fibergroup.bl_irreps(0), grid=grid)

        final_features = ftgpool.in_type
        blocks += [
            R3Conv(layer_types[-1][0], final_features, kernel_size=3, padding=1, stride=1, bias=False, initialize=False),
            ftgpool,
        ]
        C = ftgpool.out_type.size
        print(f'The size of C which we are taking as input for classifier is {C}')
        self.blocks = SequentialModule(*blocks)


        H = 256
        self.classifier = nn.Sequential(
            nn.Linear(128*3*3*3, H, bias=False),
            #nn.Linear(C, H, bias=False),

            #nn.BatchNorm1d(H, affine=True),
            nn.LayerNorm(H),
            nn.ELU(inplace=True),
            nn.Dropout(.1),
            nn.Linear(H, H // 2, bias=False),

            #nn.BatchNorm1d(H // 2, affine=True),
            nn.LayerNorm(H // 2),
            nn.ELU(inplace=True),
            nn.Dropout(.1),
            nn.Linear(H//2, 10, bias=True),
        )

    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, R3Conv):
                if self._init == 'he':
                    init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_representation(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs.trivial_repr]

        SO3 = self.gs.fibergroup

        polinomials = [self.gs.trivial_repr, SO3.irrep(1)]

        for k in range(2, K+1):
            polinomials.append(
                polinomials[-1].tensor(SO3.irrep(1))
            )

        return directsum(polinomials, name=f'polynomial_{K}')

    def forward(self, input: torch.Tensor):

        input = GeometricTensor(input, self.in_type)

        features = self.blocks(input)
        #print(f"Shape before reshape: {features.shape}")

        shape = features.shape
        # features = features.tensor.view(shape[0], -1)
        features = torch.mean(features.tensor, dim=1, keepdim=True)
        features = features.view(-1, 1, 67, 67, 67)
        #features = features.tensor.reshape(shape[0], shape[1])
        out = features
        #out = self.classifier(features)
        #print(f'Output shape is {out.shape}')
        # out = out.view(shape)

        return out
    
class PairedH5Dataset(Dataset):
    def __init__(self, input_data, output_data):
        assert input_data.shape == output_data.shape, "Input and output data must have the same shape"
        self.input_data = input_data.reshape((1, 1, *input_data.shape))
        self.output_data = output_data.reshape((1, 1, *output_data.shape))

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        x = self.input_data[idx].clone().detach().float()
        y = self.output_data[idx].clone().detach().float()
        # x = torch.tensor(self.input_data[idx], dtype=torch.float32)
        # y = torch.tensor(self.output_data[idx], dtype=torch.float32)
        return x, y

def load_h5_data(file_path, dataset_name='/data'):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
        tensor_data = torch.tensor(data, dtype=torch.float32)  # Convert numpy array to PyTorch tensor
    return tensor_data


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    # Paths to the training and testing directories
    train_directory = '/global/cfs/cdirs/m4231/Abhishek/escnn_data/train/'
    test_directory = '/global/cfs/cdirs/m4231/Abhishek/escnn_data/test/'

    # Build and initialize the SE(3) equivariant model
    model = SE3CNN(pool='snub_cube', res_features='2_96', init='he').to(device)
    model.init()
    

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20

    epoch_list = []
    loss_list = []
    unique_name_list = []

    train_files = [f for f in os.listdir(train_directory) if f.endswith('.h5')]
    test_files = [f for f in os.listdir(test_directory) if f.endswith('.h5')]

    model.train()
    print('Training Initiated ...')

    for unique_name in set(f[:-7] for f in train_files if f.endswith('_rho.h5')):
        print(f'\nFile: {unique_name}')
        tic = time.time()
        rho_filepath = os.path.join(train_directory, unique_name + '_rho.h5')     
        sad_filepath = os.path.join(train_directory, unique_name + '_sad.h5')

        rho_data = load_h5_data(rho_filepath)
        sad_data = load_h5_data(sad_filepath)

        train_dataset = PairedH5Dataset(sad_data, rho_data)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        for epoch in range(num_epochs):
            #model.train()
            toc = time.time()
            running_loss = 0.0
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                error = torch.abs(outputs - targets)
                print(f'Max train error: {error.max().item()} | Min train error: {error.min().item()}')
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()*inputs.size(0)
            epoch_loss = running_loss/len(train_dataloader.dataset)
            epoch_list.append(epoch+1)
            loss_list.append(epoch_loss)
            unique_name_list.append(unique_name)
            print(f'Epoch: {epoch+1} | Loss: {epoch_loss: .4f}')
            print(f'Time taken in epoch {epoch+1} is {time.time()-toc} seconds')
        print(f'Time taken {unique_name} is for all {num_epochs} epochs is {time.time() - tic} seconds')

    df = pd.DataFrame({
         'Epoch': epoch_list,
         'Loss': loss_list,
         'Name': unique_name_list
    })

    csv_file_path = 'loss_vals.csv'
    df.to_csv(csv_file_path, index=False)
    print('Training Complete !!!')
    
    model.eval()                  

    print('Equivariance Test Initiated ...')
    
    for test_unique_name in set(f[:-7] for f in test_files if f.endswith('_rho.h5')):

        test_rho_path = os.path.join(test_directory, test_unique_name + '_rho.h5')
        test_sad_path = os.path.join(test_directory, test_unique_name + '_sad.h5')

        test_rho_data = load_h5_data(test_rho_path)
        test_sad_data = load_h5_data(test_sad_path)

        test_dataset = PairedH5Dataset(test_sad_data, test_rho_data)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        correct = 0
        test_loss = 0.0

        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                inputs_rot90 = torch.rot90(inputs, k=1, dims=(2,3))
                outputs_rot90 = model(inputs_rot90)
                rotated_outputs = torch.rot90(outputs, k=1, dims=(2,3))

                print(f'Max input: {inputs.max().item()} | Max output: {outputs.max().item()} | Max target: {targets.max().item()}')
                error = torch.abs(outputs - targets)
                eq_error = torch.abs(outputs_rot90 - rotated_outputs)
                print(f'Max equivariance error: {eq_error.max().item()} | Min equivariance error: {eq_error.min().item()}')
                print(f'Max test error: {error.max().item()} | Min test error: {error.min().item()}')
                loss = criterion(outputs, targets)
                test_loss += loss.item()*inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()

                outputs_rot90_np = outputs_rot90.detach().cpu().numpy()
                rotated_outputs_np = rotated_outputs.detach().cpu().numpy()
                h5_file_path = 'outputs_rot90_data.h5'
                with h5py.File(h5_file_path,'w') as hf:
                    hf.create_dataset('op_rot',data=outputs_rot90_np)

                h5_file_path = 'rotated_outputs_data.h5'
                with h5py.File(h5_file_path,'w') as hf:
                    hf.create_dataset('rot_op',data=rotated_outputs_np)

        accuracy = correct/len(test_dataloader.dataset)
        test_loss /= len(test_dataloader.dataset)
        print(f'Test Loss: {test_loss: .4f}')

    print('Equivariance Test Complete !!!')


