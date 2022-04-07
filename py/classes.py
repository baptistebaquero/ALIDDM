from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import GlobVar as GV
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from utils import(
    ReadSurf,
    ScaleSurf,
    RandomRotation,
    ComputeNormals,
    GetColorArray,
    GetTransform
)
from monai.transforms import (
    ToTensor
)
import pandas as pd
import json
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence

from vtk import vtkMatrix4x4
from vtk import vtkMatrix3x3
import vtk

class FlyByDataset(Dataset):
    def __init__(self, df, device, dataset_dir='', rotate=False):
        self.df = df
        self.device = device
        self.dataset_dir = dataset_dir
        self.rotate = rotate
    def set_env_params(self, params):
        self.params = params

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        surf = ReadSurf(os.path.join(self.dataset_dir,self.df.iloc[idx]["surf"])) # list of dico like [{"model":... ,"landmarks":...},...]
        # print(surf)
        surf, mean_arr, scale_factor= ScaleSurf(surf) # resize my surface to center it to [0,0,0], return the distance between the [0,0,0] and the camera center and the rescale_factor
        if self.rotate:
            surf, angle, vector = RandomRotation(surf)
        else:
            angle = 0 
            vector = np.array([0, 0, 1])
        
        surf = ComputeNormals(surf) 
        landmark_pos = self.get_landmarks_position(idx, mean_arr, scale_factor, angle, vector)
        color_normals = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
        verts = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(surf.GetPoints().GetData()))
        faces = ToTensor(dtype=torch.int64, device=self.device)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
        region_id = ToTensor(dtype=torch.int64, device=self.device)(vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID")))
        region_id = torch.clamp(region_id, min=0)
        landmark_pos = torch.tensor(landmark_pos,dtype=torch.float32).to(self.device)
        mean_arr = torch.tensor(mean_arr,dtype=torch.float64).to(self.device)
        scale_factor = torch.tensor(scale_factor,dtype=torch.float64).to(self.device)

        return surf, verts, faces, region_id, color_normals, landmark_pos, mean_arr, scale_factor
   
    def get_landmarks_position(self,idx, mean_arr, scale_factor, angle, vector):
       
        data = json.load(open(os.path.join(self.dataset_dir,self.df.iloc[idx]["landmarks"])))
        markups = data['markups']
        landmarks_lst = markups[0]['controlPoints']

        lst_lm =  GV.LANDMARKS[GV.SELECTED_JAW]
        landmarks_position = np.zeros([len(lst_lm), 3])
        # resc_landmarks_position = np.zeros([number_of_landmarks, 3])
        for landmark in landmarks_lst:
            label = landmark["label"]
            if label in lst_lm:
                landmarks_position[lst_lm.index(label)] = Downscale(landmark["position"],mean_arr,scale_factor)

        landmarks_pos = np.array([np.append(pos,1) for pos in landmarks_position])
        if angle:
            transform = GetTransform(angle, vector)
            transform_matrix = arrayFromVTKMatrix(transform.GetMatrix())
            landmarks_pos = np.matmul(transform_matrix,landmarks_pos.T).T
        return landmarks_pos[:, 0:3]

        
def pad_verts_faces(batch):
    surf = [s for s, v, f, ri, cn, lp, sc, ma  in batch]
    verts = [v for s, v, f, ri, cn, lp, sc, ma  in batch]
    faces = [f for s, v, f, ri, cn, lp, sc, ma  in batch]
    region_id = [ri for s, v, f, ri, cn, lp, sc, ma  in batch]
    color_normals = [cn for s, v, f, ri, cn, lp, sc, ma, in batch]
    landmark_position = [lp for s, v, f, ri, cn, lp, sc, ma in batch]
    scale_factor = [sc for s, v, f, ri, cn, lp , sc, ma  in batch]
    mean_arr = [ma for s, v, f, ri, cn,lp, sc, ma   in batch]

    return surf, pad_sequence(verts, batch_first=True, padding_value=0.0), pad_sequence(faces, batch_first=True, padding_value=-1),region_id, pad_sequence(color_normals, batch_first=True, padding_value=0.), landmark_position, mean_arr, scale_factor




def arrayFromVTKMatrix(vmatrix):
  """Return vtkMatrix4x4 or vtkMatrix3x3 elements as numpy array.
  The returned array is just a copy and so any modification in the array will not affect the input matrix.
  To set VTK matrix from a numpy array, use :py:meth:`vtkMatrixFromArray` or
  :py:meth:`updateVTKMatrixFromArray`.
  """

  if isinstance(vmatrix, vtkMatrix4x4):
    matrixSize = 4
  elif isinstance(vmatrix, vtkMatrix3x3):
    matrixSize = 3
  else:
    raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
  narray = np.eye(matrixSize)
  vmatrix.DeepCopy(narray.ravel(), vmatrix)
  return narray

def Upscale(landmark_pos,mean_arr,scale_factor):
    new_pos_center = (landmark_pos/scale_factor) + mean_arr
    return new_pos_center

def Downscale(pos_center,mean_arr,scale_factor):
    landmarks_position = (pos_center - mean_arr) * scale_factor
    return landmarks_position