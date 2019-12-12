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
from torchsummary import summary
#from sppnet.spp_layer import SPP2D
#import samplesomethings.samps
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

        #for ID in glob.glob("Processed/*"):
        #  if os.path.basename(ID) not in self.patient_ids:
        #    os.system('mv '+ID+" Processed/unused")

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.patient_ids)*100

  def __getitem__(self, q):
        'Generates one sample of data'


        #Patient ID
        id = q//100
        ID = self.patient_ids[id]

        # Load MRI Image
        #I = np.load('MRIs/' + ID + '.npz')
        #I, i = I['arr_0'], I['arr_1']

        # Load Clinical Data
        #print(ID)
        #x = self.CD[q]

        # Load Labels
        #y = self.labels[q]
        #N = np.random.choice(len(os.listdir("Processed/"+ID+"/")), 1, replace=False)[0]
        #N = np.random.choice(self.b, self.b, replace=False)
        #N = [0]
        H = np.load(glob.glob("Processed/"+ID+"/*.npy")[q%100])

        H = torch.from_numpy(H)
        return H

class Z(nn.Module):
	def __init__(self, weights = 1):
		super().__init__()
		self.weights = weights
	def forward(self, input):
		a = 10
		s = 2
		return F.softplus(a*input)/a - F.softplus(-a*(input+s))/a


class Flatten(nn.Module):
    def forward(self, input):
        #print(input.shape)
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.size(0), *self.shape)

class ConvTransposeRenorm2d(nn.ConvTranspose2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)
        output_padding = (output_padding, output_padding)
        super(nn.ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode)

  def forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return F.conv_transpose2d(input, self.weight, torch.zeros_like(self.bias), self.stride, self.padding, output_padding, self.groups, self.dilation)/F.conv_transpose2d(torch.ones_like(input), torch.ones_like(self.weight), torch.zeros_like(self.bias), self.stride, self.padding, output_padding, self.groups, self.dilation) + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
  





class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.h_dim = 512
        self.k = 500
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = self.Encoder()
        self.decoder = self.Decoder()
        self.prior = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.h_dim)/ np.sqrt(self.k * self.h_dim), requires_grad=False)
        self.iter = 0

        self.HLoss1 = pytorch_ssim.DSSIM(window_size = 11, size_average = True)
        #self.HLoss2 = pytorch_ssim.DSSIM(window_size = 64, size_average = True)
        self.SLoss = nn.MSELoss()

    def Encoder(self):
        return nn.Sequential(nn.Conv2d(   3,  16, 5, 2, 0), nn.LeakyReLU(),

                             nn.Conv2d(  16,  32, 5, 1, 0), nn.LeakyReLU(),
                             nn.Conv2d(  32,  32, 5, 1, 0), nn.LeakyReLU(),
                             nn.Conv2d(  32,  64, 5, 1, 0), nn.LeakyReLU(),
                             nn.Conv2d(  64,  64, 5, 1, 0), nn.LeakyReLU(),

                             nn.Conv2d(  64, 128, 4, 1, 0), nn.LeakyReLU(),
                             nn.Conv2d( 128, 128, 3, 1, 0), nn.LeakyReLU(),
                             nn.Conv2d( 128, 256, 3, 1, 0), nn.LeakyReLU(),
                             nn.Conv2d( 256, 256, 3, 1, 0), nn.LeakyReLU(),
                             Flatten(),

                             nn.Linear(6400, 2048), nn.LeakyReLU(),
                             nn.Linear(2048, self.h_dim), nn.LeakyReLU(),
                             nn.Linear(self.h_dim, 2*self.h_dim))

    def Decoder(self):
        return nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.LeakyReLU(),
                             nn.Linear(self.h_dim, 2048), nn.LeakyReLU(),
                             nn.Linear(2048, 6400), nn.LeakyReLU(), Reshape((-1, 5, 5)),

                             nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                             nn.Conv2d(256, 128, 3, 1, 0), nn.LeakyReLU(),
                             nn.Conv2d(128, 128, 3, 1, 0), nn.LeakyReLU(),
                             nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                             nn.Conv2d(128, 128, 3, 1, 0), nn.LeakyReLU(),
                             nn.Conv2d(128, 64, 3, 1, 0), nn.LeakyReLU(),
                             nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                             nn.Conv2d(64, 64, 3, 1, 0), nn.LeakyReLU(),
                             nn.Conv2d(64, 64, 3, 1, 0), nn.LeakyReLU(),
                             nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                             nn.Conv2d( 64, 64, 3, 1, 0), nn.LeakyReLU(),
                             nn.Conv2d( 64, 32, 3, 1, 0), nn.LeakyReLU(),
                             nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                             nn.Conv2d(32, 16, 3, 1, 0), nn.LeakyReLU(),
                             nn.Conv2d(16, 16, 3, 1, 0), nn.LeakyReLU(),
                             nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                             nn.Conv2d(16, 8, 1, 1, 0), nn.LeakyReLU(),
                             nn.Conv2d(8, 3, 1, 1, 0))
                             #nn.Conv2d(16, 3, 1, 1, 0))

                             #ConvTransposeRenorm2d(64, 64, 2, 2, 0), nn.LeakyReLU(),
                             #nn.ConvTranspose2d(64, 32, 1, 1, 0), nn.LeakyReLU(),
                             #nn.ConvTranspose2d(32, 3, 1, 1, 0))
                             #nn.ConvTranspose2d(64*3, 3, 4, 4, 3, groups=3))
                             #nn.ConvTranspose2d(32*3, 3, 3, 2, 0, groups=3)'''
                             #nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                             #nn.Conv2d(256, 64*3, 3, 1, )
                             #)

    def HLoss(self, x, X):
        return self.HLoss1(x, X)
    def forward(self, I):
        m = self.encoder.forward(I)
        m, v = torch.split(m, m.size(-1) //2, dim = -1)
        v = F.softplus(v) + 1e-8
        #v = torch.exp(v)

        h = self.sample_gaussian(m, v)
        #self.KLoss = self.Kloss(m, v) #G
        self.KLoss = self.log_normal(h, m, v) - self.log_normal_mixture(h, *self.gaussian_parameters(self.prior, dim=1)) #GM

        H = self.decoder(h)
        #return H
        L, R = H.size(-1)//2 - 32, H.size(-1)//2 + 32

        #if self.iter < 1000:
        #  return(torch.sigmoid(H/100-0.5)[:,:,L:R,L:R])
        #  return(H[:,:,L:R,L:R])
        #else:
        #return(H[:,:,L:R,L:R].clamp(0,1))
        H = H[:,:,L:R,L:R]
        if self.iter < 1000:
            H = (torch.min(F.leaky_relu(H, 1/(self.iter/10+1)), torch.ones_like(H)) - torch.min(F.leaky_relu(1-H, 1/(self.iter/10+1)), torch.ones_like(H))+1)/2.0
            return(H)
        else:
            return(H.clamp(0,1))

    def sample_gaussian(self, m, v, b = 0):
        if b > 0:
            m = m.unsqueeze(1).expand(-1, b, -1)
            v = v.unsqueeze(1).expand(-1, b, -1)
        return torch.sqrt(v)*torch.randn(v.shape).to(self.device) + m

    def Kloss(self, qm, qv, pm=0, pv=1):
        try:
            element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        except:
            pm = torch.zeros(qm.shape).to(device)
            pv = torch.ones(qv.shape).to(device)
            element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.sum(-1)
        return kl

    def log_normal_mixture(self, z, m, v):
        # (batch , dim) -> (batch , 1, dim)
        z = z.unsqueeze(1)
        # (batch , 1, dim) -> (batch , mix , dim) -> (batch , mix)
        log_prob = self.log_normal(z, m, v)
        # (batch , mix) -> (batch ,)
        log_prob = self.log_mean_exp(log_prob , dim=1)
        return log_prob

    def log_normal(self, x, m, v):
        element_wise = -0.5 * (torch.log(v) + (x - m).pow(2) / v + np.log(2 * np.pi))
        log_prob = element_wise.sum(-1)
        return  log_prob

    def gaussian_parameters(self, h, dim=-1):
        """
        Converts generic real-valued representations into mean and variance
        parameters of a Gaussian distribution

        Args:
            h: tensor: (batch, ..., dim, ...): Arbitrary tensor
            dim: int: (): Dimension along which to split the tensor for mean and
                variance

        Returns:
            m: tensor: (batch, ..., dim / 2, ...): Mean
            v: tensor: (batch, ..., dim / 2, ...): Variance
        """
        m, h = torch.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        return m, v

    def log_mean_exp(self, x, dim):
        """
        Compute the log(mean(exp(x), dim)) in a numerically stable manner

        Args:
            x: tensor: (...): Arbitrary tensor
            dim: int: (): Dimension along which mean is computed

        Return:
             _: tensor: (...): log(mean(exp(x), dim))
        """
        return self.log_sum_exp(x, dim) - np.log(x.size(dim))

    def log_sum_exp(self, x, dim=0):
        """
        Compute the log(sum(exp(x), dim)) in a numerically stable manner

        Args:
            x: tensor: (...): Arbitrary tensor
            dim: int: (): Dimension along which sum is computed

        Return:
            _: tensor: (...): log(sum(exp(x), dim))
        """
        max_x = torch.max(x, dim)[0]
        new_x = x - max_x.unsqueeze(dim).expand_as(x)
        return max_x + (new_x.exp().sum(dim)).log()

if __name__=='__main__':

  B = 64
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #model = VAE().to(device)
  from LSUV import LSUVinit
  rf = 398 #resme from this epoch
  model = torch.load('Model_all_'+str(rf)+'.pth')
  model.prior = torch.load('Model_prior_'+str(rf)+'.pth')
  load = True
  init = False
  model.prior.requires_grad=True
  #model.iter = 0
  model.sz = 64
  summary(model, (3, model.sz, model.sz))

  #optimizer = optim.Adam(model.parameters(), lr = 3e-4)
  optimizer = optim.Adam(model.parameters(), lr = 1e-6)
  optimizer.load_state_dict(torch.load('Optim_'+str(rf)+'.pth'))
  for g in optimizer.param_groups:
    g['lr'] = 1.0e-5

  #optimizer = optim.SGD(model.parameters(), lr = 1e-4)
  #optimizer = optim.Adadelta(model.parameters())
  mem = False
  epochs = 800

  losses = []
  if load == True:
    losses = list(np.load('Loss.npy'))
  #mnist = torchvision.datasets.MNIST('/media/heinzlab/Data/', download=True, transform=torchvision.transforms.Compose([torchvision.transforms.Resize(64), torchvision.transforms.ToTensor()]))
  #trainloader = torch.utils.data.DataLoader(mnist,batch_size=16, shuffle=True,num_workers=2)
  trainset = Dataset(1)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=B, shuffle=True, num_workers=8, pin_memory=True) #TODO TRAINSETs
    

  subepochs = 1
  for epoch in range(rf, epochs*subepochs):
    for q, Data in enumerate(trainloader, 0):
      for iolimloop in range(1):
      #if q in [1,2]:
          # get the inputs; data is a list of [inputs, labels]
          #inputs, _= Data
          I = Data#inputs = Data
          if mem:
              a = torch.cuda.memory_allocated(device=device)
              print("PreData on Device",a/1e9)
          
          
          I = I.permute(0,3,1,2)
          I = I[:,:3] 
  
          if q == 0:
            subepoch = epoch%subepochs
            epoch = epoch//subepochs

          Or = subepoch%4
          

          #MNIST
          #H = torch.repeat_interleave(H, 3, dim=1)
          

          I = I.float()

          DS = nn.AdaptiveAvgPool2d((256,256))
          I = DS(I)/256
          #A, B, C, D, = I[:,:,:128,:128], I[:,:,:128,128:], I[:,:,128:,:128], I[:,:,128:,128:]
          I = torch.cat((I[:,:,:128,:128],I[:,:,:128,128:],I[:,:,128:,:128],I[:,:,128:,128:]), dim = 0)#.to(device)
          I = torch.cat((I[:,:,:64,:64],I[:,:,:64,64:],I[:,:,64:,:64],I[:,:,64:,64:]), dim = 0).to(device)
          if epoch == 0 and q == 0 and load == False and init == False:
            print("LSUVINIT")
            model = LSUVinit(model, I, needed_std = I.reshape(I.size(0), I.size(1), -1).std(dim=-1).mean().item(), std_tol= 1e-4, max_attempts = 50, cuda=True)
            init = True

          #I = torchvision.transforms.ToTensor()(torchvision.transforms.functional.rotate(torchvision.transforms.ToPILImage()(I), Or*90))
          #I = I.unsqueeze(0).to(device)
          #pdb.set_trace()
          if mem:
              a = torch.cuda.memory_allocated(device=device)
              print("Data on Device, PreForward",a/1e9)

          # zero the parameter gradients


          # forward + backward + optimize
          model.iter += 1
          pred = model.forward(I)
          #pdb.set_trace()
          if mem:
              a = torch.cuda.memory_allocated(device=device)
              print("Post Forward",a/1e9)
          mse = 256**2*((pred-I)**2).mean(-1).mean(-1).mean(-1) # +256/(epoch+1)**2*((pred.reshape(I.size(0), I.size(1), -1).std(dim=-1)-I.reshape(I.size(0), I.size(1), -1).std(dim=-1))**2).mean(-1)
          hl = 100*model.HLoss(pred,I)
          kl = torch.max(80*torch.ones_like(model.KLoss), model.KLoss)*(mse <= 400.0).float() + torch.max(500*torch.ones_like(model.KLoss), model.KLoss)*(mse > 400.).float()

          #if mse.mean().item() < 400:
          #  pass
          #elif mse.mean().item() < 1000:
          #  kl = torch.max(500*torch.ones_like(kl), kl)
          #else:
          #  kl = torch.max(2000*torch.ones_like(kl), kl)

          channels = 3
          grad_x_weights = torch.tensor([[1, -1]], dtype=torch.float32).to(model.device)
          grad_x_weights = grad_x_weights.expand(channels, 1, 1, 2)
          Gx = F.conv2d(I, grad_x_weights, groups=I.shape[1], padding=1)
          gx = F.conv2d(pred, grad_x_weights, groups=pred.shape[1], padding=1)
          grad_y_weights = torch.tensor([[ 1],[ -1]], dtype=torch.float32).to(model.device)
          grad_y_weights = grad_y_weights.expand(channels, 1, 2, 1)
          Gy = F.conv2d(I, grad_y_weights, groups=I.shape[1], padding=1)
          gy = F.conv2d(pred, grad_y_weights, groups=pred.shape[1], padding=1)


          adS = torch.zeros_like(mse)#1.0*(((Gx-gx)**2).sum(-1).sum(-1).sum(-1) + ((Gy-gy)**2).sum(-1).sum(-1).sum(-1))
          #bowl = 2*(F.relu(I-1) + F.relu(-I)).sum(-1).sum(-1).sum(-1)
          loss = (hl + mse + kl + adS).mean()#*(10 - 9*np.exp(-self.iter/1000))
          print("Epoch", epoch, "batch", q, "loss", np.round(loss.item(),4), "kl", np.round(kl.mean().item(),4), "mse", np.round(mse.mean().item(),4), "img means", "({}, {})".format(np.round(pred.mean().item(),3), np.round(I.mean().item(),3)))
          print("means", model.gaussian_parameters(model.prior)[0].min().item(),  model.gaussian_parameters(model.prior)[0].max().item(), "std", "({} {})".format(np.round(pred.reshape(pred.size(0),pred.size(1),-1).std(dim=-1).mean().item(), 3), np.round(I.reshape(I.size(0), I.size(1),-1).std(dim=-1).mean().item(), 3)), "DSSIM", hl.mean().item())
          if model.iter%100 == 0:
              losses.append([kl.mean().item(), mse.mean().item(), adS.mean().item(), hl.mean().item()])
          nbatches = 1
          loss = loss/nbatches
          loss.backward()
          #loss = loss*nbatches
          if q == 0:
              a = torch.cuda.memory_allocated(device=device)
              print("Post Backward",a/1e9)
          if (q+1)%nbatches == 0 or q == trainset.__len__()//B:
            print("step")
            optimizer.step()
            optimizer.zero_grad()


          if (q + epoch)%10 == 1:
              img = torch.cat((pred[:8],I[:8]),dim=0)
              nrow = I.size(0)
              nrow = 8
              torchvision.utils.save_image(img, 'gen.png', nrow=nrow, padding=1)

          #if (epoch+1)%2==0 and q == 0:
          if model.iter%2000 == 0:
              img = torch.cat((pred[:8],I[:8]),dim=0)
              nrow = I.size(0)
              nrow = 8
              torchvision.utils.save_image(img, 'gen_'+str(epoch+1)+"_"+str(model.iter)+'.png', nrow=nrow, padding=1)
              np.save('Loss.npy', np.array(losses))
          # print 

          if epoch in [0,1,2,3,4] or epoch%100==0:
            #samps(model)
            pass

          #if model.iter == 500:
          #    torch.save(optimizer.state_dict(), 'Optim_iter_'+str(model.iter)+'.pth')

    if ((epoch+1)%1)==0:
      torch.save(model.state_dict(), 'Model_'+str(epoch+1)+'.pth')
      torch.save(model, 'Model_all_'+str(epoch+1)+'.pth')
      torch.save(optimizer.state_dict(), 'Optim_'+str(epoch+1)+'.pth')
      torch.save(model.prior, 'Model_prior_'+str(epoch+1)+'.pth')
  torch.save(model.state_dict(), 'Model.pth')
  torch.save(model, 'Model_all.pth')

  np.save('Loss.npy', np.array(losses))
