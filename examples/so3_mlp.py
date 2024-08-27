import sys
sys.path.append('../')

import torch
import numpy as np

from escnn import gspaces
from escnn import nn
from escnn import group

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SO3MLP(nn.EquivariantModule):
    
    def __init__(self, n_classes=10):
        
        super(SO3MLP, self).__init__()
        
        # the model is equivariant to the group SO(3)
        self.G = group.so3_group()
        
        # since we are building an MLP, there is no base-space
        self.gspace = gspaces.no_base_space(self.G)
        
        # the input contains the coordinates of a point in the 3D space
        self.in_type = self.gspace.type(self.G.standard_representation())
        
        # Layer 1
        # We will use the representation of SO(3) acting on signals over a sphere, bandlimited to frequency 1
        # To apply a point-wise non-linearity (e.g. ELU), we need to sample the spherical signals over a finite number of points.
        # Note that this makes the equivariance only approximate.
        # The representation of SO(3) on spherical signals is technically a quotient representation,
        # identified by the subgroup of planar rotations, which has id=(False, -1) in our library
        
        # N.B.: the first this model is instantiated, the library computes numerically the spherical grids, which can take some time
        # These grids are then cached on disk, so future calls should be considerably faster.
        
        activation1 = nn.QuotientFourierELU(
            self.gspace,
            subgroup_id=(False, -1),
            channels=3, # specify the number of spherical signals in the output features
            irreps=self.G.bl_sphere_representation(L=1).irreps, # include all frequencies up to L=1
            grid=self.G.sphere_grid(type='thomson', N=16), # build a discretization of the sphere containing 16 equally distributed points            
            inplace=True
        )
        
        # map with an equivariant Linear layer to the input expected by the activation function, apply batchnorm and finally the activation
        self.block1 = nn.SequentialModule(
            nn.Linear(self.in_type, activation1.in_type),
            nn.IIDBatchNorm1d(activation1.in_type),
            activation1,
        )
        
        # Repeat a similar process for a few layers
        
        # 8 spherical signals, bandlimited up to frequency 3
        activation2 = nn.QuotientFourierELU(
            self.gspace,
            subgroup_id=(False, -1),
            channels=8, # specify the number of spherical signals in the output features
            irreps=self.G.bl_sphere_representation(L=3).irreps, # include all frequencies up to L=3
            grid=self.G.sphere_grid(type='thomson', N=40), # build a discretization of the sphere containing 40 equally distributed points            
            inplace=True
        )
        self.block2 = nn.SequentialModule(
            nn.Linear(self.block1.out_type, activation2.in_type),
            nn.IIDBatchNorm1d(activation2.in_type),
            activation2,
        )
        
        # 8 spherical signals, bandlimited up to frequency 3
        activation3 = nn.QuotientFourierELU(
            self.gspace,
            subgroup_id=(False, -1),
            channels=8, # specify the number of spherical signals in the output features
            irreps=self.G.bl_sphere_representation(L=3).irreps, # include all frequencies up to L=3
            grid=self.G.sphere_grid(type='thomson', N=40), # build a discretization of the sphere containing 40 equally distributed points            
            inplace=True
        )
        self.block3 = nn.SequentialModule(
            nn.Linear(self.block2.out_type, activation3.in_type),
            nn.IIDBatchNorm1d(activation3.in_type),
            activation3,
        )
        
        # 5 spherical signals, bandlimited up to frequency 2
        activation4 = nn.QuotientFourierELU(
            self.gspace,
            subgroup_id=(False, -1),
            channels=5, # specify the number of spherical signals in the output features
            irreps=self.G.bl_sphere_representation(L=2).irreps, # include all frequencies up to L=2
            grid=self.G.sphere_grid(type='thomson', N=25), # build a discretization of the sphere containing 25 equally distributed points            
            inplace=True
        )
        self.block4 = nn.SequentialModule(
            nn.Linear(self.block3.out_type, activation4.in_type),
            nn.IIDBatchNorm1d(activation4.in_type),
            activation4,
        )
        
        # Final linear layer mapping to the output features
        # the output is a 5-dimensional vector transforming according to the Wigner-D matrix of frequency 2
        self.out_type = self.gspace.type(self.G.irrep(2))
        self.block5 = nn.Linear(self.block4.out_type, self.out_type)
    
    def forward(self, x: nn.GeometricTensor):
        
        # check the input has the right type
        assert x.type == self.in_type
        
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
     
        return x
    
    def evaluate_output_shape(self, input_shape: tuple):
        shape = list(input_shape)
        assert len(shape) ==2, shape
        assert shape[1] == self.in_type.size, shape
        shape[1] = self.out_type.size
        return shape
    
model = SO3MLP().to(device)
np.set_printoptions(linewidth=10000, precision=4, suppress=True)

model.eval()

B = 10

# generates B random points in 3D and wrap them in a GeometricTensor of the right type
x = model.in_type(torch.randn(B, 3))
print(f'Input data is {x}')
print(f'Input data shape is {x.shape}')
print(f'Input data type is {x.type}')


print('##########################################################################################')
with torch.no_grad():
    y = model(x.to(device)).to('cpu')
    print("Outputs' magnitudes")
    print(torch.linalg.norm(y.tensor, dim=1).numpy().reshape(-1))
    print('##########################################################################################')
    print("Errors' magnitudes")
    for r in range(8):
        # sample a random rotation
        g = model.G.sample()
        
        x_transformed = g @ x
        x_transformed = x_transformed.to(device)

        y_transformed = model(x_transformed).to('cpu')
        
        # verify that f(g@x) = g@f(x)=g@y
        print(torch.linalg.norm(y_transformed.tensor - (g@y).tensor, dim=1).numpy().reshape(-1))        

print('##########################################################################################')
print()