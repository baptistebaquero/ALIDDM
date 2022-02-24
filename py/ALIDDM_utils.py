import os
import glob
import json
import csv
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
import random
import vtk
import torch

import GlobVar as GV
from GlobVar import SELECTED_JAW
from utils import ReadSurf
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
from utils import(
    PolyDataToTensors
) 
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex,blending

from torch.utils.data import DataLoader

from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from monai.transforms import (
    ToTensor
)
import monai
import torchvision.transforms as transforms
from shader import *
from post_process import NeighborPoints

def GetLandmarkPosFromLP(lm_pos,target):
    lst_lm =  GV.LANDMARKS[GV.SELECTED_JAW]
    lm_coord = torch.empty((0)).cpu()
    for lst in lm_pos:
        lst = lst.cpu()
        # print(lst_lm)
        lm_coord = torch.cat((lm_coord,lst[lst_lm.index(target)].unsqueeze(0)),dim=0)

    return lm_coord

def GenPhongRenderer(image_size,blur_radius,faces_per_pixel,device):
    
    # cameras = FoVOrthographicCameras(znear=0.1,zfar = 10,device=device) # Initialize a ortho camera.

    cameras = FoVPerspectiveCameras(znear=0.01,zfar = 10, fov= 90, device=device) # Initialize a perspective camera.

    raster_settings = RasterizationSettings(        
        image_size=image_size, 
        blur_radius=blur_radius, 
        faces_per_pixel=faces_per_pixel, 
    )

    lights = PointLights(device=device) # light in front of the object. 

    rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
    
    b = blending.BlendParams(background_color=(0,0,0))
    phong_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights,blend_params=b)
    )
    mask_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=MaskRenderer(device=device, cameras=cameras, lights=lights,blend_params=b)
    )
    return phong_renderer,mask_renderer
    

def GenDataSet(df,dir_patients,flyBy,device,label):
    SELECTED_JAW = GV.SELECTED_JAW
    df_train = df.loc[df['for'] == "train"]
    df_train = df_train.loc[df_train['jaw'] == SELECTED_JAW]
    df_val = df.loc[df['for'] == "val"]
    df_val = df_val.loc[df_val['jaw'] == SELECTED_JAW]
    df_train = df_train.loc[df_train[label] == 1]
    df_val = df_val.loc[df_val[label] == 1]

    # print(df.loc[df['for'] == "test"])

    # print(df_train)

    train_data = flyBy(
        df = df_train,
        device=device,
        dataset_dir=dir_patients,
        rotate=False
        )

    val_data = flyBy(
        df = df_val,
        device=device,
        dataset_dir=dir_patients,
        rotate=False
        )

    return train_data,val_data

def generate_sphere_mesh(center,radius,device,color = [1,1,1]):
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(center[0],center[1],center[2])
    sphereSource.SetRadius(radius)

    # Make the surface smooth.
    sphereSource.SetPhiResolution(10)
    sphereSource.SetThetaResolution(10)
    sphereSource.Update()
    vtk_landmarks = vtk.vtkAppendPolyData()
    vtk_landmarks.AddInputData(sphereSource.GetOutput())
    vtk_landmarks.Update()

    verts_teeth,faces_teeth = PolyDataToTensors(vtk_landmarks.GetOutput())

    verts_rgb = torch.ones_like(verts_teeth)[None]  # (1, V, 3)
    # verts_rgb[:,0:] *= 0.1
    verts_rgb[:,:, 0] *= color[0]  # red
    verts_rgb[:,:, 1] *= color[1]  # green
    verts_rgb[:,:, 2] *= color[2]  # blue


    # verts_rgb[:,:2] *= 0


    # color_normals = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(fbf.GetColorArray(surf, "Normals"))/255.0)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh = Meshes(
        verts=[verts_teeth], 
        faces=[faces_teeth],
        textures=textures).to(device)
    
    return mesh,verts_teeth,faces_teeth,verts_rgb.squeeze(0)

def GenDataSplitCSV(dir_data,csv_path,val_p,test_p):
    patient_dic = {}
    normpath = os.path.normpath("/".join([dir_data, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(img_fn):
            basename = os.path.basename(img_fn).split('.')[0].split("_P")
            jow = basename[0][0]
            patient_ID = "P"+basename[1]
            if patient_ID not in patient_dic.keys():
                patient_dic[patient_ID] = {"L":{},"U":{}}
            
            if ".json" in img_fn:
                patient_dic[patient_ID][jow]["lm"] = img_fn
            elif ".vtk" in img_fn:
                patient_dic[patient_ID][jow]["surf"] = img_fn

    # print(patient_dic)
    test_p = test_p/100
    val_p = val_p/100
    val_p = val_p/(1-test_p)

    patient_dic = list(patient_dic.values())
    random.shuffle(patient_dic)

    # print(patient_dic)
    df_train, df_test = train_test_split(patient_dic, test_size=test_p)
    df_train, df_val = train_test_split(df_train, test_size=val_p)

    data_dic = {
        "train":df_train,
        "val":df_val,
        "test":df_test
        }

    fieldnames = ['for','jaw','surf', 'landmarks']
    for lab in range(2,32):
        fieldnames.append(str(lab))
    data_list = []
    for type,dic in data_dic.items():
        for patient in dic:
            for jaw,data in patient.items():
                if jaw == "L":
                    rows = {
                        'for':type,
                        'jaw':jaw,
                        'surf':data["surf"].replace(dir_data,"")[1:],
                        'landmarks':data["lm"].replace(dir_data,"")[1:],
                        }
                    # print(data["surf"])
                    read_surf = ReadSurf(data["surf"])
                    ids = ToTensor(dtype=torch.int64, device=GV.DEVICE)(vtk_to_numpy(read_surf.GetPointData().GetScalars("PredictedID")))
                    # print(ids)

                    for label in range(2,32):
                        
                        if label in ids:
                            present = 1
                        else:
                            present = 0
                        
                        rows[str(label)] = present

                    data_list.append(rows)
    
    with open(csv_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)
    # return outfile


def PlotDatasetWithLandmark(target,dataLoader):
    for batch, (V, F, CN, LP, MR, SF) in enumerate(dataLoader):
        radius = 0.02
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )

        lm_pos = torch.empty((0)).to(GV.DEVICE)
        for lst in LP:
            lm_pos = torch.cat((lm_pos,lst[GV.LM_SELECTED_LST.index(target)].unsqueeze(0)),dim=0)

        # print(lm_pos[0])
    PlotMeshAndSpheres(meshes,lm_pos,radius,[0.3,1,0.3])

def PlotMeshAndSpheres(meshes,sphere_pos,radius,col):
    dic = {
    "teeth_mesh": meshes,
    }
    for id,pos in enumerate(sphere_pos):
        # print(pos)
        mesh,verts_teeth,faces_teeth,textures = generate_sphere_mesh(pos,radius,GV.DEVICE,col)
        dic[str(id)] = mesh
    # print(dic)
    plot_fig(dic)


def plot_fig(dic):
    fig = plot_scene({"subplot1": dic},     
        xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
        yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
        zaxis={"backgroundcolor":"rgb(200, 230, 200)"}, 
        axis_args=AxisArgs(showgrid=True))
    fig.show()


def Generate_Mesh(verts,faces,text,lst_landmarks,device):
    verts_rgb = torch.ones_like(text)[None].squeeze(0)  # (1, V, 3)
    verts_rgb[:,:, 0] *= 1  # red
    verts_rgb[:,:, 1] *= 0  # green
    verts_rgb[:,:, 2] *= 0  # blue
    text = verts_rgb

    for landmark in lst_landmarks:
        batch_verts = torch.empty((0)).to(device)
        batch_faces = torch.empty((0)).to(device)
        batch_text = torch.empty((0)).to(device)
        # print(text)
        for position in landmark:
            mesh_l,verts_l,faces_l,text_l = generate_sphere_mesh(position,0.01,device,[0,1,0])
            batch_text = torch.cat((batch_text,text_l.unsqueeze(0).to(device)),dim=0)
            batch_verts = torch.cat((batch_verts,verts_l.unsqueeze(0).to(device)),dim=0)
            batch_faces = torch.cat((batch_faces,faces_l.unsqueeze(0).to(device)),dim=0)
       
        verts,faces,text = merge_meshes(verts,faces,text,batch_verts,batch_faces,batch_text)
    
    textures = TexturesVertex(verts_features=text)
        
    meshes =  Meshes(
        verts=verts,   
        faces=faces, 
        textures=textures
    )
    return meshes


def merge_meshes(verts_1,faces_1,text_1,verts_2,faces_2,text_2):

    verts = torch.cat([verts_2,verts_1], dim=1)
    faces = torch.cat([faces_2,faces_1+verts_2.shape[1]], dim=1)
    text = torch.cat([text_2,text_1], dim=1)

    return verts,faces,text

def Get_lst_landmarks(LP,lst_names_land):
    lst_landmarks=[]
    for landmarks in lst_names_land:
        lm_coords = GetLandmarkPosFromLP(LP,landmarks)
        lst_landmarks.append(lm_coords)
    return lst_landmarks

def Generate_land_Mesh(lst_landmarks,device):
    verts = torch.empty((0)).to(device)
    faces = torch.empty((0)).to(device)
    text = torch.empty((0)).to(device)
    
    for landmark in lst_landmarks:
        batch_verts = torch.empty((0)).to(device)
        batch_faces = torch.empty((0)).to(device)
        batch_text = torch.empty((0)).to(device)
    
        for position in landmark:
            mesh_l,verts_l,faces_l,text_l = generate_sphere_mesh(position,0.02,device)
            tensor_text = torch.ones_like(verts_l).to(device)*0
            batch_text = torch.cat((batch_text,tensor_text.unsqueeze(0).to(device)),dim=0)
            batch_verts = torch.cat((batch_verts,verts_l.unsqueeze(0).to(device)),dim=0)
            batch_faces = torch.cat((batch_faces,faces_l.unsqueeze(0).to(device)),dim=0)
       
        verts,faces,text = merge_meshes(verts,faces,text,batch_verts,batch_faces,batch_text)
    
    textures = TexturesVertex(verts_features=text)
        
    meshes =  Meshes(
        verts=verts,   
        faces=faces, 
        textures=textures
    )
    return meshes

def Convert_RGB_to_grey(lst_images):
    tens_images = torch.empty((0)).to(GV.DEVICE)
    for image in lst_images:
        # print(image.shape)
        # image = image.to(GV.DEVICE)
        image = image.cpu()
        image = image[:-1,:,:]
        image = transforms.ToPILImage()(image)
        image = transforms.Grayscale(num_output_channels=1)(image)
        image = transforms.ToTensor()(image)
        # print(image[0])
        for row in image[0]:
            for pix in row:
                new_pix = torch.nn.Threshold(pix>0.5, 1)
                pix = new_pix

        tens_images = torch.cat((tens_images,image.unsqueeze(0).to(GV.DEVICE)),dim=0)

    return tens_images

# def Gen_patch(lst_surf, V, RED, LP, label, step_neighbor):
#     lst_landmarks = Get_lst_landmarks(LP,GV.LABEL[label])
#     for surf_index,surf in enumerate(lst_surf):
#         locator = vtk.vtkOctreePointLocator()
#         locator.SetDataSet(surf)
#         locator.BuildLocator()
#         for color_index,landmark_coord in enumerate(lst_landmarks):
#             patch = []

#             landmark_coord =landmark_coord.to(GV.DEVICE)
#             pid = locator.FindClosestPoint(landmark_coord[surf_index].cpu().numpy())
#             all_neighbor_pid = NeighborPoints(surf,pid)
#             patch.append(all_neighbor_pid)
        
#             for step in range(step_neighbor):
#                 for ids in all_neighbor_pid:
#                     all_neighbor_pid_step = NeighborPoints(surf,ids)
#                     patch.append(all_neighbor_pid_step)           
#                     all_neighbor_pid = all_neighbor_pid_step

#             patch = np.concatenate(patch).tolist()
                        
#             patch = np.unique(patch)
#             color = [torch.tensor([1,0,0]),torch.tensor([0,1,0]),torch.tensor([0,0,1])]
#             for i,index in enumerate(patch):
#                 RED[surf_index][index] = color[color_index]                
          
        
#     return RED

def Gen_patch(V, RED, LP, label, radius):
    lst_landmarks = Get_lst_landmarks(LP,GV.LABEL[label])
    color_index=0
    for landmark_coord in lst_landmarks:
        landmark_coord =landmark_coord.unsqueeze(1).to(GV.DEVICE)
        distance = torch.cdist(landmark_coord, V, p=2)
        distance = distance.squeeze(1)
        index_pos_land = (distance<radius).nonzero(as_tuple=True)
        color = [torch.tensor([1,0,0]),torch.tensor([0,1,0]),torch.tensor([0,0,1])]
        for i,index in enumerate(index_pos_land[0]):
            RED[index][index_pos_land[1][i]] = color[color_index]
        color_index +=1 
    
    return RED

def Gen_one_patch(V, RED, radius, coord):
    landmark_coord = coord.unsqueeze(0).to(GV.DEVICE)
    distance = torch.cdist(landmark_coord, V, p=2)
    distance = distance.squeeze(1)
    # print(distance.shape)
    # index_pos_land = torch.where(distance<radius,distance,torch.ones(distance.shape)*0)
    index_pos_land = (distance<radius).nonzero(as_tuple=True)
    # print('index_pos_land',index_pos_land)
    # print(RED[index_pos_land])
    for i,index in enumerate(index_pos_land[0]):
        # print(RED[index][index_pos_land[1][i]])
        RED[index][index_pos_land[1][i]] = torch.tensor([0,1,0])
              
    return RED

def Gen_mesh_patch(surf,V,F,CN,LP,label):
    verts_rgb = torch.ones_like(CN)[None].squeeze(0)  # (1, V, 3)
    verts_rgb[:,:, 0] *= 0  # red
    verts_rgb[:,:, 1] *= 0  # green
    verts_rgb[:,:, 2] *= 0  # blue 
    # patch_region = Gen_patch(surf, V, verts_rgb, LP, label, 3)
    patch_region = Gen_patch(V, verts_rgb, LP, label, 0.02)    
    textures = TexturesVertex(verts_features=patch_region)
    meshes = Meshes(
        verts=V,   
        faces=F, 
        textures=textures
    ).to(GV.DEVICE) # batchsize
    
    return meshes

def Gen_mesh_one_patch(V,F,CN,coord):
    verts_rgb = torch.ones_like(CN)[None].squeeze(0)  # (1, V, 3)
    verts_rgb[:,:, 0] *= 1  # red
    verts_rgb[:,:, 1] *= 0  # green
    verts_rgb[:,:, 2] *= 0  # blue
    patch_region = Gen_one_patch(V, verts_rgb, 0.02,coord)
    textures = TexturesVertex(verts_features=patch_region)
    meshes = Meshes(
        verts=V,   
        faces=F, 
        textures=textures
    ).to(GV.DEVICE) # batchsize
    
    return meshes