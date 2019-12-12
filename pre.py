import openslide
#import matplotlib.pyplot as plt
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
import pdb
import os
import glob
import random

patient = np.genfromtxt("patient_gbm.txt", dtype=str, delimiter='\t', skip_header=3)
CIDs = patient[:,0]



# In[12]:


svspath = 'Histology/'
destination = 'Processed/'
imsize = 512
imnum = 300
svsdirectories = os.listdir(svspath)
SVSs = []
for d in svsdirectories:
    try:
        SVSs.append(glob.glob(svspath + d +'/*.svs')[0])
    except:
        pass
SVS_IDs = [os.path.basename(fname)[:12] for fname in SVSs]

common = []
for ID in SVS_IDs:
    if ID in CIDs:
        common.append(ID)
SVS_IDs = common

random.shuffle(SVSs)

for ID in SVS_IDs:
    if ID in CIDs:
        try:
            os.stat(destination + ID )
        except:
            os.mkdir(destination + ID )
for svs in SVSs:
    ID = os.path.basename(svs)[:12]
    if ID in CIDs:
        print(ID)
        path = destination + ID
        if len(os.listdir(path))< imnum:
            file = openslide.OpenSlide(svs)
        while len(os.listdir(path))< imnum:
            i = np.random.choice(file.level_dimensions[0][0]-imsize)
            j = np.random.choice(file.level_dimensions[0][1]-imsize)

            things = glob.glob(os.path.join(path, str(len(os.listdir(path)))) )
            things = [os.path.basename(thing) for thing in things]
            #pdb.set_trace()
            #print("Try")
            proceed = False
            N = len(things)
            iteration = 0
            if N == 0:
              proceed = True
            else:
              while proceed == False and iteration < N:
                iteration += 1
                x = int(thing[:6])
                y = int(thing[7:13])

                if ((i-x)**2 + (j-y)**2) > 128**2:
                  proceed = True

            if proceed:

              ROI = np.array(file.read_region((i,j), 0, (imsize,imsize)))
              print("ROI generated")
              S = entropy(ROI.mean(axis=2)/255, disk(10)).mean()

              x = f'{i:06}'
              y = f'{j:06}'
              if S > 4.0:# and S < 5.7 :
                  savename = os.path.join(path, x+"_"+y+".npy")
                  np.save(savename, ROI)
                  print("Saving Plot")

# iteration = 0
# xs = []
# entropies =[]
# imsize = 512
# while len(xs)< 40:
#     i = np.random.choice(file.level_dimensions[0][0]-imsize)
#     j = np.random.choice(file.level_dimensions[0][1]-imsize)
#     ROI = np.array(file.read_region((i,j), 0, (imsize,imsize)))
#     s = entropy(ROI.mean(axis=2)/255, disk(10)).mean()
#     if s > 5.5:
#         xs.append(ROI)
#         iteration +=1
#         print(iteration)
#         entropies.append(s)
# entropies = np.array(entropies)

# check = np.load('/Users/aidan/Documents/CS236/Project/processed/TCGA-26-A7UX/0.npy')
# plt.imshow(check)

# plt.figure(figsize= (6, 3*len(xs)))
# for i in range(len(xs)):
#     plt.subplot(len(xs),1,i+1)
#     plt.imshow(xs[i][:,:,:3])
# plt.show()

# plt.hist(entropies)
