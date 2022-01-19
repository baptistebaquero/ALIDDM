import GlobVar as GV
from utils import *
from ALIDDM_utils import *


from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np 
import torchvision.models as models
from pytorch3d.renderer import look_at_rotation
import torch
from monai.transforms import ToTensor
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import fly_by_features as fbf
import json
from collections import deque
import statistics
import matplotlib.pyplot as plt
import math
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import math
import torch.optim as optim
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from tqdm.std import tqdm
from statistics import mean




icosahedron = CreateIcosahedron(1, 1)
sphere_points = []
for pid in range(icosahedron.GetNumberOfPoints()):
    spoint = icosahedron.GetPoint(pid)
    sphere_points.append([point for point in spoint])

CAMERA_POSITION = np.array(sphere_points[:])
# CAMERA_POSITION = np.array(sphere_points)


class Agent:
    def __init__(
        self,
        renderer, 
        target,
        device, 
        # label,
        save_folder = "", 
        radius = 1,
        verbose = True,
        ):
        super(Agent, self).__init__()
        self.renderer = renderer
        self.device = device
        self.target = target
        self.writer = SummaryWriter(os.path.join(save_folder,"Run_"+target))
        # self.label = label
        self.camera_points = torch.tensor(CAMERA_POSITION).type(torch.float32).to(self.device)
        self.scale = 0
        self.radius = radius
        self.verbose = verbose


    def position_agent(self, text, vert, label, device):
        # print(text)
        # print(int(label))
        final_pos = torch.empty((0)).to(device)
        for mesh in range(len(text)):
            index_pos_land = (text[mesh]==int(label)).nonzero(as_tuple=True)[0]
            lst_pos = []
            for index in index_pos_land:
                lst_pos.append(vert[mesh][index])
            position_agent = sum(lst_pos)/len(lst_pos)
            final_pos = torch.cat((final_pos,position_agent.unsqueeze(0).to(device)),dim=0)
        # print(final_pos.shape)
        self.positions = final_pos
        return self.positions

    
    def GetView(self,meshes):
        spc = self.positions
        img_lst = torch.empty((0)).to(self.device)

        for sp in self.camera_points:
            sp_i = sp*self.radius
            # sp = sp.unsqueeze(0).repeat(self.batch_size,1)
            current_cam_pos = spc + sp_i
            R = look_at_rotation(current_cam_pos, at=spc, device=self.device)  # (1, 3, 3)
            # print( 'R shape :',R.shape)
            # print(R)
            T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:, :, None])[:, :, 0]  # (1, 3)

            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T.to(self.device))
            images = images.permute(0,3,1,2)
            images = images[:,:-1,:,:]
            
            # print(images.shape)
            pix_to_face, zbuf, bary_coords, dists = self.renderer.rasterizer(meshes)
            zbuf = zbuf.permute(0, 3, 1, 2)
            # print(dists.shape)
            y = torch.cat([images, zbuf], dim=1)
            # print(y)

            img_lst = torch.cat((img_lst,y.unsqueeze(0)),dim=0)
        img_batch =  img_lst.permute(1,0,2,3,4)
        
        return img_batch

    def SetScale(self,scale):
        self.scale = scale
        self.radius = self.radius_lst[scale]


def PlotAgentViews(view):
    for batch in view:
        if batch.shape[0] > 5:
            row = int(math.ceil(batch.shape[0]/5)) 
            f, axarr = plt.subplots(nrows=row,ncols=5)
            c,r = 0,0
            for image in batch:
                image = image.permute(1,2,0)[:,:,:-1]
                axarr[r,c].imshow(image)
                c += 1
                if c == 5:c,r = 0,r+1
        else:
            f, axarr = plt.subplots(nrows=1,ncols=batch.shape[0])
            for i,image in enumerate(batch):
                image = image.permute(1,2,0)[:,:,:-1]
                axarr[i].imshow(image)
        plt.show()

