import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import CenterCrop
import torchvision
import pytorch_ssim
import torch.optim as optim
import glob
import os
import pdb
import gc
from conv64 import VAE, Flatten, Reshape
from torchsummary import summary



def samps(model):
	N = 8

	m, v = model.gaussian_parameters(model.prior.squeeze(0), dim=-2)
	idx = torch.distributions.categorical.Categorical(torch.ones(model.k)/model.k).sample((N,))
	m, v = m[idx], v[idx]
	h = model.sample_gaussian(m, v)
	
	H = model.decoder.forward(h)
	L, R = H.size(-1)//2 - 128, H.size(-1)//2 + 128
	H = H[:,:,L:R,L:R].clamp(0,1)

	H = F.interpolate(H, scale_factor=3)
	torchvision.utils.save_image(H, 'sample.png', nrow=8, padding=0, pad_value=0)
	
if __name__=='__main__':
	#model = VAE()
	#print(model.prior.min(), model.prior.max())
	model = torch.load('Model_all_600.pth', map_location=torch.device('cpu'))
	model.eval()
	model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.device = torch.device('cpu')
	model.to(model.device)
	samps(model)
