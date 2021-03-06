'Test the utility functions.'


# pylint: disable=C0413
# Not all the modules can be placed at the top of the files as we need
# first to change the PYTHONPATH before to import the modules.
import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import numpy as np
from scipy.special import logsumexp
import torch
import beer
from basetest import BaseTest


class TestUtilityFunctions(BaseTest):

    def setUp(self):
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)

    def test_logsumexp(self):
        val1 = beer.logsumexp(self.data, dim=0).numpy()
        val2 = logsumexp(self.data.numpy(), axis=0)
        self.assertArraysAlmostEqual(val1, val2)
        val1 = beer.logsumexp(self.data, dim=1).numpy()
        val2 = logsumexp(self.data.numpy(), axis=1)
        self.assertArraysAlmostEqual(val1, val2)

    def test_onehot(self):
        ref = torch.range(0, 2).long()
        labs1 = beer.onehot(ref, 3).long()
        labs2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertArraysAlmostEqual(labs1.numpy(), labs2)


__all__ = ['TestUtilityFunctions']
