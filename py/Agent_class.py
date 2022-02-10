from operator import itemgetter
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
from scipy import linalg



icosahedron = CreateIcosahedron(1, 1)
sphere_points=[]
sphere_points = ([0,0,1],
                 np.array([0.5,0.,1.0])/linalg.norm([0.5,0.5,1.0]),
                 np.array([-0.5,0.,1.0])/linalg.norm([-0.5,-0.5,1.0]),
                 np.array([0,0.5,1])/linalg.norm([1,0,1]),
                 np.array([0,-0.5,1])/linalg.norm([0,1,1])
                )
# for pid in range(icosahedron.GetNumberOfPoints()):
#     spoint = icosahedron.GetPoint(pid)
#     sphere_points.append([point for point in spoint])

# got = itemgetter(0,4)(sphere_points)
# CAMERA_POSITION = np.array(got)
# print(sphere_points)
CAMERA_POSITION = np.array(sphere_points)


class Agent:
    def __init__(
        self,
        renderer, 
        renderer2,
        device, 
        radius = 1,
        verbose = True,
        ):
        super(Agent, self).__init__()
        self.renderer = renderer
        self.renderer2=renderer2
        self.device = device
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

    
    def GetView(self,meshes,rend=False):
        spc = self.positions
        img_lst = torch.empty((0)).to(self.device)
        seuil = 0.5

        for sp in self.camera_points:
            sp_i = sp*self.radius
            # sp = sp.unsqueeze(0).repeat(self.batch_size,1)
            current_cam_pos = spc + sp_i
            R = look_at_rotation(current_cam_pos, at=spc, device=self.device)  # (1, 3, 3)
            # print( 'R shape :',R.shape)
            # print(R)
            T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:, :, None])[:, :, 0]  # (1, 3)

            if rend:
                renderer = self.renderer2
                images = renderer(meshes_world=meshes.clone(), R=R, T=T.to(self.device))
                y = images[:,:,:,:-1]

                # yd = torch.where(y[:,:,:,:]<=seuil,0.,0.)
                yr = torch.where(y[:,:,:,0]>seuil,1.,0.).unsqueeze(-1)
                yg = torch.where(y[:,:,:,1]>seuil,2.,0.).unsqueeze(-1)
                yb = torch.where(y[:,:,:,2]>seuil,3.,0.).unsqueeze(-1)

                y = ( yr + yg + yb).to(torch.float32)

                y = y.permute(0,3,1,2)
              
            else:
                renderer = self.renderer
                images = self.renderer(meshes_world=meshes.clone(), R=R, T=T.to(self.device))
                images = images.permute(0,3,1,2)
                images = images[:,:-1,:,:]

                pix_to_face, zbuf, bary_coords, dists = self.renderer.rasterizer(meshes.clone())
                zbuf = zbuf.permute(0, 3, 1, 2)
                y = torch.cat([images, zbuf], dim=1)

            img_lst = torch.cat((img_lst,y.unsqueeze(0)),dim=0)
        img_batch =  img_lst.permute(1,0,2,3,4)
        
        return img_batch
    
    def get_view_rasterize(self,meshes):
        spc = self.positions
        img_lst = torch.empty((0)).to(self.device)
        tens_pix_to_face = torch.empty((0)).to(self.device)

        for sp in self.camera_points:
            sp_i = sp*self.radius
            current_cam_pos = spc + sp_i
            R = look_at_rotation(current_cam_pos, at=spc, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:, :, None])[:, :, 0]  # (1, 3)
              
            renderer = self.renderer
            images = renderer(meshes_world=meshes.clone(), R=R, T=T.to(self.device))
            images = images.permute(0,3,1,2)
            images = images[:,:-1,:,:]

            pix_to_face, zbuf, bary_coords, dists = renderer.rasterizer(meshes.clone())
            zbuf = zbuf.permute(0, 3, 1, 2)
            y = torch.cat([images, zbuf], dim=1)

            img_lst = torch.cat((img_lst,y.unsqueeze(0)),dim=0)
            tens_pix_to_face = torch.cat((tens_pix_to_face,pix_to_face.unsqueeze(0)),dim=0)
        img_batch =  img_lst.permute(1,0,2,3,4)
    
        return img_batch , tens_pix_to_face      


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

