from typing import Tuple, Union
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *

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

        S = len(so3.grid(**grid))

        _channels = channels / S
        _channels = int(round(_channels))

        ftelu = FourierELU(self.gspace, _channels, irreps=so3.bl_irreps(L), inplace=True, **grid)
        res_type = ftelu.in_type

        self.res_block = SequentialModule(
            R3Conv(in_type, res_type, kernel_size=3, padding=1, bias=False, initialize=False),
            IIDBatchNorm3d(res_type, affine=True),
            ftelu,
            R3Conv(res_type, self.out_type, kernel_size=3, padding=1, stride=stride, bias=False, initialize=False),
        )

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
            R3Conv(self.in_type, layer_types[0][0], kernel_size=5, padding=2, bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                ResBlock(layer_types[i][0], layer_types[i][1], layer_types[i+1][0], 2, features=res_features)
            )

        if pool == "icosidodecahedron":
            grid = self.gs.fibergroup.sphere_grid(type='ico')
        elif pool == "snub_cube":
            grid = self.gs.fibergroup.sphere_grid(type='thomson_cube', N=1)
        else:
            raise ValueError(f"Pooling method {pool} not recognized")

        ftgpool = QuotientFourierELU(self.gs, (False, -1), 128, irreps=self.gs.fibergroup.bl_irreps(2), out_irreps=self.gs.fibergroup.bl_irreps(0), grid=grid)

        final_features = ftgpool.in_type
        blocks += [
            R3Conv(layer_types[-1][0], final_features, kernel_size=3, padding=0, bias=False, initialize=False),
            ftgpool,
        ]
        self.C = ftgpool.out_type.size
        print(f'Size of C : {self.C}')
        self.blocks = SequentialModule(*blocks)
        self.C = self.C*3*3*3
        self.hidden_size = 256
        self.classifier = nn.Sequential(
            nn.Linear(self.C, self.hidden_size, bias=False),
            nn.BatchNorm1d(self.hidden_size, affine=True),
            nn.ELU(inplace=True),
            nn.Dropout(.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2, bias=False),
            nn.BatchNorm1d(self.hidden_size // 2, affine=True),
            nn.ELU(inplace=True),
            nn.Dropout(.1),
            nn.Linear(self.hidden_size // 2, 10, bias=True),
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

        shape = features.shape
        print(f"Shape before reshaping: {shape}")

        num_elements = torch.prod(torch.tensor(shape[1:])).item()
        features = features.tensor.view(shape[0], num_elements)
        print(f"Shape after reshaping: {features.shape}")

        out = self.classifier(features)

        return out

if __name__ == '__main__':
    m = SE3CNN(pool='snub_cube', res_features='2_96', init='he')
    m.init()

    device = 'cuda'
    m.eval()

    test_sizes = [(3,1,67,67,67)]

    for size in test_sizes:
        x = torch.randn(size)
        y = m(x)
        print(f"Input size: {size}, Output shape: {y.shape}")

        x_x90 = x.rot90(1, (2, 3))
        x_z90 = x.rot90(1, (3, 4))
        x_y90 = x.rot90(1, (2, 4))
        x_y180 = x.rot90(2, (2, 4))
        x_fy = x.flip(dims=[3])
        x_fx = x.flip(dims=[2])

        y_x90 = m(x_x90)
        y_z90 = m(x_z90)
        y_y90 = m(x_y90)
        y_y180 = m(x_y180)
        y_fy = m(x_fy)
        y_fx = m(x_fx)

        print('TESTING INVARIANCE:                     ')
        print('90 degrees ROTATIONS around X axis:  ' + ('YES' if torch.allclose(y, y_x90, atol=1e-5, rtol=1e-4) else 'NO'))
        print('90 degrees ROTATIONS around Y axis:  ' + ('YES' if torch.allclose(y, y_y90, atol=1e-5, rtol=1e-4) else 'NO'))
        print('90 degrees ROTATIONS around Z axis:  ' + ('YES' if torch.allclose(y, y_z90, atol=1e-5, rtol=1e-4) else 'NO'))
        print('180 degrees ROTATIONS around Y axis: ' + ('YES' if torch.allclose(y, y_y180, atol=1e-5, rtol=1e-4) else 'NO'))
        print('REFLECTIONS on the Y axis:           ' + ('YES' if torch.allclose(y, y_fx, atol=1e-5, rtol=1e-4) else 'NO'))
        print('REFLECTIONS on the Z axis:           ' + ('YES' if torch.allclose(y, y_fy, atol=1e-5, rtol=1e-4) else 'NO'))
