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
#from sppnet.spp_layer import SPP2D

from torch.utils import data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, b):
        'Initialization'
        self.CD = np.load("Clinical.npz")
        self.patient_ids, self.CD = self.CD['arr_0'], self.CD['arr_1']
        self.b = b
        self.labels = self.CD[:,0]
        self.CD = self.CD[:,1:]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.patient_ids)

  def __getitem__(self, q):
        'Generates one sample of data'

        #Patient ID
        ID = self.patient_ids[q]

        # Load MRI Image
        I = np.load('MRIs/' + ID + '.npz')
        I, i = I['arr_0'], I['arr_1']

        # Load Clinical Data
        #print(ID)
        x = self.CD[q]

        # Load Labels
        y = self.labels[q]
        N = np.random.choice(len(os.listdir("Processed/"+ID+"/")), self.b)
        H = np.array([np.load("Processed/"+ID+"/"+str(n)+".npy") for n in N])

        I = torch.from_numpy(I.astype(float))
        i = torch.tensor([float(i[-2])])
        x = torch.from_numpy(x)
        S = torch.from_numpy(np.array(y))
        P = torch.from_numpy(np.array(y > 1).astype(float))
        H = torch.from_numpy(H)
        return (I, i, x), (S, P, H)





class Encoder(nn.Module):
    #MRNet Convolutional Architecture
    def __init__(self, filter_size = [5, 5, 4, 4], filters = [64, 64, 128, 256], fcs = [256, 256, 128, 2*64], i_dim = 1, x_dim = 3, h_dim = 128):
        super(Encoder, self).__init__()
        assert(len(filter_size) == len(filters))
        self.fcs = fcs
        self.fcs[-1] = 2*h_dim
        self.filters = filters
        self.filter_size = filter_size
        self.x_dim = x_dim
        self.h_dim = h_dim


        #Series of Convolutions
        self.convs = [nn.Conv3d(1, filters[0], kernel_size = (1, filter_size[0], filter_size[0]), stride = 1, padding = 0)]
        for k in range(1, len(filters)):
            self.convs.append(nn.Conv3d(filters[k-1], filters[k], kernel_size = (1, filter_size[k], filter_size[k]), stride = 1, padding =0))
        self.convs = nn.ModuleList(self.convs)

        #Series of Fully Connected
        size = filters[-1] + x_dim + i_dim
        self.FC = [nn.Linear(size, fcs[0])]
        for k in range(1,len(fcs)):
            self.FC.append(nn.Linear(fcs[k-1], fcs[k]))
        self.FC = nn.ModuleList(self.FC)

        #Pooling
        self.MP  = nn.MaxPool3d((1, 2, 2), stride=(1,2,2))              #For Use After Every Conv
        self.MP2 = nn.AdaptiveMaxPool2d((126,126))
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))                #For Use After All Convs
        self.AMP = nn.AdaptiveMaxPool1d((1,))   #For Use Between Images within a Batch/Series

        self.DO = nn.Dropout(0.5)

    def forward(self, I, i, x):
        #print(I.shape)
        #Series of Convolutions
        if mem:
            a = torch.cuda.memory_allocated(device=device)
            #print("enc PreConv",a/1e9)
        for k in range(len(self.convs)):
            I = self.convs[k](I)
            I = F.relu(I)

            if mem:
                a = torch.cuda.memory_allocated(device=device)
                #print("enc PostConv",k, a/1e9)

            if k == (len(self.convs) - 1):
                pass
            elif k > 0:
                #MaxPool after every conv except last
                I = self.MP(I)
            elif k == 0:
                s = I.size(2)
                f = I.size(1)
                I = I.reshape(-1, s*f, I.size(3), I.size(4))
                I = self.MP2(I)
                I = I.reshape(-1, f, s, 126, 126)
            I = self.DO(I)
            #print(I.shape)
        if mem:
            a = torch.cuda.memory_allocated(device=device)
            #print("enc Postconv, Pre Pool",a/1e9)


        #GlobalAveragePool after last conv
        I = I.mean(dim=-1)
        #print(I.shape)
        I = I.mean(dim=-1)
        #print(I.shape)
        #MaxPool over Batch
        I = self.AMP(I)
        #print(I.shape)
        if mem:
            a = torch.cuda.memory_allocated(device=device)
            print("enc Postpool, Pre Cat/Dense",a/1e9)

        #Flatten and Concatenate
        I = torch.flatten(I, start_dim=1)
        #print(I.shape)
        try:
            I = torch.cat((I, i, x), dim=-1)
        except:
            pdb.set_trace()
        #print(I.shape)
        #Series of Fully Connected
        for k in range(len(self.FC)):
            I = self.FC[k](I)
            if k < len(self.FC) -1:
              I = F.softplus(I)
              I = self.DO(I)
        if mem:
            a = torch.cuda.memory_allocated(device=device)
            print("enc Post Cat/Dense, Pre split MV",a/1e9)

        m, v = torch.split(I, I.size(-1) // 2, dim=-1)
        v = F.softplus(v) + 1e-8
        #print("mv shape", m.shape, v.shape)
        return m, v


class Decoder(nn.Module):
    #MRNet Architecture
    def __init__(self, filter_size = [5, 5, 5, 5], filters = [256, 128, 32, 6], fcs = [256, 256, 512, 1024, 4096], x_dim = 3, h_dim=512):
        super(Decoder, self).__init__()
        assert(len(filter_size) == len(filters))

        self.fcs = fcs
        self.filters = filters
        self.filter_size = filter_size
        self.x_dim = x_dim
        self.h_dim = fcs[0]

        #Series of Fully Connected
        self.FC = []
        for k in range(1,len(fcs)):
            self.FC.append(nn.Linear(fcs[k-1], fcs[k]))
        self.FC = nn.ModuleList(self.FC)

        #Series of Deconvolutions
        self.convs = [nn.ConvTranspose3d(fcs[-1]//64, filters[0], kernel_size=(1, filter_size[0], filter_size[0]), stride = 1, padding = 0)]
        for k in range(1, len(filters)):
            self.convs.append(nn.ConvTranspose3d(filters[k-1], filters[k], kernel_size=(1, filter_size[k], filter_size[k]), stride=1, padding=0))
        self.convs = nn.ModuleList(self.convs)
        #Pooling
        self.US1  = nn.Upsample(scale_factor=(1,4,4), mode='trilinear')#, align_corners=True)          #For Use After Every Deconv
        self.US2  = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')#, align_corners=True)          #For Use After Every Deconv
        self.pools = nn.ModuleList([self.US1, self.US2, self.US2, self.US2])
        #Crop
        #self.crop = CenterCrop((512,512))
        self.DO = nn.Dropout(0.5)

    def forward(self, H, b):

        #Series of Fully Connected
        for k in range(len(self.FC)):
            H = self.FC[k](H)
            H = F.softplus(H)
            H = self.DO(H)
            #print("FC ", k, H.shape)
        #pdb.set_trace()
        H = (H.reshape(H.size(0), b, self.fcs[-1]//64, 8, 8)).permute(0,2,1,3,4)
        #print("Post FC ", k, H.shape)

        #Series of Convolutions
        #print("num convs", len(self.convs))
        for k in range(len(self.convs)):
            H = self.pools[k](H)
            #print("Pool ", k, H.shape)
            H = self.convs[k](H)
            #print("Conv ", k, H.shape)

            if k < len(self.convs)-1:
                H = F.softplus(H)
                H = self.DO(H)

        left = H.size(4)//2-128
        right = H.size(4)//2+128
        H = H[:,:,:,left:right, left:right]
        #print("Post Crop", H.shape)
        m = H[:,:3]
        v = F.softplus(H[:,3:]) + 1e-8
        return m, v #Mean and Variance for each of RGB


class Extractor(nn.Module):
    #MRNet Convolutional Architecture
    def __init__(self, filter_size = [5, 5, 4, 4], filters = [64, 64, 128, 256], fcs = [256, 256, 256, 256], i_dim = 1, x_dim = 3):
        super(Extractor, self).__init__()
        assert(len(filter_size) == len(filters))
        self.fcs = fcs
        self.filters = filters
        self.filter_size = filter_size
        self.x_dim = x_dim


        #Series of Convolutions
        self.convs = [nn.Conv3d(1, filters[0], kernel_size = (1, filter_size[0], filter_size[0]), stride = 1, padding = 0)]
        for k in range(1, len(filters)):
            self.convs.append(nn.Conv3d(filters[k-1], filters[k], kernel_size = (1, filter_size[k], filter_size[k]), stride=1, padding=0))
        self.convs = nn.ModuleList(self.convs)


        #Series of Fully Connected
        size = filters[-1] + x_dim + i_dim
        self.FC = [nn.Linear(size, fcs[0])]
        for k in range(1,len(fcs)):
            self.FC.append(nn.Linear(fcs[k-1], fcs[k]))
        self.FC = nn.ModuleList(self.FC)

        #Pooling
        self.MP  = nn.MaxPool3d((1, 2, 2), stride=(1,2,2))              #For Use After Every Conv
        self.MP2 = nn.AdaptiveMaxPool2d((62,62))
        self.AMP = nn.AdaptiveMaxPool1d((1))   #For Use Between Images within a Batch/Series

        self.DO = nn.Dropout(0.5)

    def forward(self, I, i, x):
        I = I[0].unsqueeze(1)

        #Series of Convolutions
        for k in range(len(self.convs)):
            I = self.convs[k](I)
            I = F.softplus(I)

            if k == (len(self.convs) - 1):
                #GlobalAveragePool after last conv
                #I = self.GAP(I)
                I = self.MP(I)
                #pass
            elif k > 0:
                #MaxPool after every conv except last
                I = self.MP(I)
            elif k == 0:
                s = I.size(2)
                f = I.size(1)
                I = I.reshape(-1, s*f, I.size(3), I.size(4))
                I = self.MP2(I)
                I = I.reshape(-1, f, s, 62, 62)
            I = self.DO(I)

        #MaxPool over Batch
        I = I.mean(dim=-1)
        #print(I.shape)
        I = I.mean(dim=-1)
        #print(I.shape)
        I = self.AMP(I)
        #print(I.shape)

        #Flatten and Concatenate
        I = torch.flatten(I, start_dim=1)
        I = torch.cat((I, i, x), dim=-1)

        #Series of Fully Connected
        for fc in self.FC:
            I = fc(I)
            I = F.softplus(I)
            I = self.DO(I)
        return I



class Trunk(nn.Module):
    "Combined output of H"
    def __init__(self, fchs = [256, 256, 256], fcs = [256, 256, 256, 256], fc1 = [256, 128, 64, 1], fc2 = [256, 128, 64, 1], ext_dim = 256, h_dim = 64):
        super(Trunk, self).__init__()

        self.h_dim = h_dim
        self.ext_dim = ext_dim
        self.fchs = [self.h_dim] + fchs
        self.fcs = [self.fchs[-1] + ext_dim] + fcs
        self.fc1 = [self.fcs[-1]] + fc1
        self.fc2 = [self.fcs[-1]] + fc2

        #Fully Connected for H, pre joint
        self.FCHs = []
        for k in range(1,len(self.fchs)):
            self.FCHs.append(nn.Linear(self.fchs[k-1], self.fchs[k]))
        self.FCHs = nn.ModuleList(self.FCHs)


        #Fully Connected, post joint
        self.FCs = []
        for k in range(1,len(self.fcs)):
            self.FCs.append(nn.Linear(self.fcs[k-1], self.fcs[k]))
        self.FCs = nn.ModuleList(self.FCs)

        #Fully Connected, branch 1
        self.FC1 = []
        for k in range(1,len(self.fcs)):
            self.FC1.append(nn.Linear(self.fc1[k-1], self.fc1[k]))
        self.FC1 = nn.ModuleList(self.FC1)

        #Fully Connected, branch 2
        self.FC2 = []
        for k in range(1,len(self.fcs)):
            self.FC2.append(nn.Linear(self.fc2[k-1], self.fc2[k]))
        self.FC2 = nn.ModuleList(self.FC2)



        self.DO = nn.Dropout(0.5)

    def forward(self, E, H=0):

        '''
        E : output of extractor(I, x). has shape (batch, output_of_extractor)
        H : sampled h vectors.         has shape (batch, b, h_dim)
        '''
        #E = E.unsqeeze(1).expand(-1, H.size(1), -1)
        
        #PreProcess H's
        if type(H) != int:
            for fc in self.FCHs:
                #print(H.shape, fc)
                H = fc(H)
                H = F.softplus(H)
                H = self.DO(H)
            H = H.mean(dim = 1) #Average across b axis


            #Combine E, H
            x = torch.cat((E, H), dim=-1)
        else:
            x = E


        #Main Trunk
        for fc in self.FCs:
            x = fc(x)
            x = F.softplus(x)
            x = self.DO(x)

        #Two Branches
        x1 = 1.0*x
        x2 = 1.0*x
        for k in range(len(self.FC1)-1):
            x1 = self.FC1[k](x1)
            x1 = F.softplus(x1)# - F.softplus(-x1-0.5) #Zellu
            x1 = self.DO(x1)
        for k in range(len(self.FC2)-1):
            x2 = self.FC2[k](x2)
            x2 = F.softplus(x2)# - F.softplus(-x2-0.5) #Zellu
            x2 = self.DO(x2)

        x1 = F.softplus(self.FC1[-1](x1))
        x2 = F.sigmoid(self.FC2[-1](x2))

        return x1, x2
        
        
class LBN(object):
    """Lower Bound Net"""
    def __init__(self):
        super(LBN, self).__init__()

        self.h_dim = 0
        self.extractor = Extractor(filter_size = [5, 5, 4, 4], filters = [64, 64, 128, 256], fcs = [256, 256, 256, 256], x_dim = 3)
        self.trunk = Trunk(fchs = [], fcs = [256, 256, 256, 256], fc1 = [256, 128, 64, 1], fc2 = [256, 128, 64, 1], ext_dim = 256, h_dim = self.h_dim)
        
        self.SLoss = nn.MSELoss()
        self.PLoss = nn.BCELoss()

    def forward(self, I, x, hist=False):

        ext = self.extractor.forward(I, x)
        S, P = self.trunk.forward(ext)

        return S, P
        
class ImagiNet(nn.Module):
    def __init__(self, x_dim, h_dim = 128, width = 256, k = 500, b = 2, gpu_ids=[]):
        super(ImagiNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.h_dim = h_dim
        self.b = b #Number of generated hist
        
        self.encoder = Encoder(filter_size = [5, 5, 4, 4], filters = [32, 64, 128, 512], fcs = [2048, 2048, 2*self.h_dim], x_dim = 3, i_dim = 1, h_dim = self.h_dim)
        self.decoder = Decoder(filter_size = [5, 5, 5, 5], filters = [128, 64, 32, 6], fcs = [self.h_dim, 2048, 4096], x_dim = 3, h_dim = self.h_dim)
        self.extractor = Extractor(filter_size = [5, 5, 4, 4], filters = [32, 64, 128, 128], fcs = [128, 128, 128, 128], x_dim = 3, i_dim = 1)
        self.trunk = Trunk(fchs = [128, 128, 128], fcs = [128, 128, 128, 128], fc1 = [128, 128, 64, 1], fc2 = [128, 128, 64, 1], ext_dim = 128, h_dim = self.h_dim)

        self.HLoss = pytorch_ssim.DSSIM(window_size = 32, size_average = True)
        self.SLoss = nn.MSELoss()
        self.PLoss = nn.BCELoss()

    def forward(self, I, i, x, hist=True):
        if mem:
            a = torch.cuda.memory_allocated(device=device)
            print("Pre MV",a/1e9)
        m, v = self.encoder.forward(I, i, x)
        if ((m + v) != (m + v)).any():
            pdb.set_trace()
        if mem:
            a = torch.cuda.memory_allocated(device=device)
            print("Post MV",a/1e9)
        ext = self.extractor.forward(I, i, x)
        if mem:
            a = torch.cuda.memory_allocated(device=device)
            print("Post Extractor",a/1e9)

        h = self.sample_gaussian(m, v, self.b)

        if mem:
            a = torch.cuda.memory_allocated(device=device)
            print("Post Sample",a/1e9)

        self.KLoss = self.Kloss(m, v)
        #pdb.set_trace()
        S, P = self.trunk.forward(ext, h)
        if hist:
            #H = self.sample_gaussian_image(self.decoder.forward(h,self.b))
            #H = torch.sigmoid(0.01*H) # H in {0, 1, 2, ... 255}
            H = torch.sigmoid(0.1*self.decoder.forward(h,self.b)[0])
            return S, P, H
        else:
            return S, P

    def Kloss(self, qm, qv, pm=0, pv=1):
        try:
            element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        except:
            pm = torch.zeros(qm.shape).to(device)
            pv = torch.ones(qv.shape).to(device)
            element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.sum(-1)
        return kl

    def sample_gaussian(self, m, v, b = 0):
        if b > 0:
            m = m.unsqueeze(1).expand(-1, b, -1)
            v = v.unsqueeze(1).expand(-1, b, -1)
        return torch.sqrt(v)*torch.randn(v.shape).to(device) + m

    def sample_gaussian_image(self, mv):
        m, v = mv
        n = torch.randn(v.shape).to(device)
        n = torch.sqrt(v)*n + m
        #print("gauss", n.shape)
        return n


B = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImagiNet(x_dim=3, h_dim = 2048, b=B).to(device)
optimizer = optim.Adadelta(model.parameters())
trainset = Dataset(B)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False) #TODO TRAINSETs
mem = False
Enc = True
if Enc:
    model.extractor.requires_grad = False
    model.trunk.requires_grad = False
epochs = 100

for epoch in range(epochs):
    for q, Data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = Data
        I, i, x = inputs
        S, P, H = labels

        if mem:
            a = torch.cuda.memory_allocated(device=device)
            print("PreData on Device",a/1e9)
        if I.size(-1) == 512:
            DS = nn.AdaptiveMaxPool3d((I.size(-3), 256,256))
            I = DS(I)
        while I.size(-3) > 60:
            I = I[:,::2]
        I = I.unsqueeze(1).float().to(device)/256
        print(I.shape, i.shape, x.shape)
        i = i.float().to(device)
        x = x.float().to(device)
        S = S.float().to(device)
        P = P.float().to(device)
        H = H.permute(0,1,4,2,3)
        H = H[:,:,:3,:,:]
        H = H.reshape(-1, 3, 512, 512) #5D to 4D
        H = H.float().to(device)/256
        DS = nn.AdaptiveMaxPool2d((256,256))
        H = DS(H)
        #pdb.set_trace()
        if mem:
            a = torch.cuda.memory_allocated(device=device)
            print("Data on Device, PreForward",a/1e9)

        # zero the parameter gradients
        optimizer.zero_grad()


        # forward + backward + optimize
        s, p, h = model.forward(I, i, x)
        #pdb.set_trace()
        h = h.permute(0,2,1,3,4)
        h = h.reshape(-1, 3, H.size(-2), H.size(-1))
        if mem:
            a = torch.cuda.memory_allocated(device=device)
            print("Post Forward",a/1e9)
        #print(h.shape, H.shape)
        print(h.mean().detach().item(), H.mean().detach().item(), s.mean().detach().item(), S.mean().detach().item(), p.mean().detach().item(), P.mean().detach().item())
        #pdb.set_trace()
        hl, sl, pl, kl = model.HLoss(h, H), model.SLoss(s,S)*(1-Enc), model.PLoss(p, P)*(1-Enc), 4*model.KLoss
        #hl, sl, pl, kl = 256*model.SLoss(h, H), model.SLoss(s,S)*(1-Enc), model.PLoss(p, P)*(1-Enc), model.KLoss/10
        loss = (hl + sl + pl + kl).mean()
        if mem:
            a = torch.cuda.memory_allocated(device=device)
            print("Post backward",a/1e9)
        loss.to(device)
        loss.backward()
        optimizer.step()

        torchvision.utils.save_image(torch.cat((h,H),dim=0), 'gen.png', nrow=model.b, padding=2, pad_value=0)

        # print statistics
        print("Predict", s.item(), p.item())
        print("Label", S.item(), P.item())
        print("Epoch", epoch, "batch", q, "loss", loss.item(), "hl", 
hl.detach().item(), "sl", sl.detach().item(), "pl", pl.detach().item(), "kl", kl.item())
        torch.cuda.empty_cache()
        del inputs, labels, loss, hl, sl, pl
        del I, i, x, S, P, H, h, s, p, q, Data



torch.save(model.state_dict(), 'Model.pth')
