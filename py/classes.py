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

        return verts, faces, region_id, color_normals, landmark_pos, mean_arr, scale_factor
   
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

# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement. 
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#             path (str): Path for the checkpoint to be saved to.
#                             Default: 'checkpoint.pt'
#             trace_func (function): trace print function.
#                             Default: print            
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.path = path
#         self.trace_func = trace_func
#     def __call__(self, val_loss, agents):

#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, agents)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, agents)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, agents):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

#         for a in agents:
#             torch.save(a.state_dict(), os.path.join(self.path, "_aid_" + str(a.agent_id) + ".pt"))
#         self.val_loss_min = val_loss

        
def pad_verts_faces(batch):
    verts = [v for v, f, ri, cn, lp, sc, ma  in batch]
    faces = [f for v, f, ri, cn, lp, sc, ma  in batch]
    region_id = [ri for v, f, ri, cn, lp, sc, ma  in batch]
    color_normals = [cn for v, f, ri, cn, lp, sc, ma, in batch]
    landmark_position = [lp for v, f, ri, cn, lp, sc, ma in batch]
    scale_factor = [sc for v, f, ri, cn, lp , sc, ma  in batch]
    mean_arr = [ma for v, f, ri, cn,lp, sc, ma   in batch]

    return pad_sequence(verts, batch_first=True, padding_value=0.0), pad_sequence(faces, batch_first=True, padding_value=-1),region_id, pad_sequence(color_normals, batch_first=True, padding_value=0.), landmark_position, mean_arr, scale_factor




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