import argparse
import datetime
from os import lstat
from unittest.mock import patch

from torch import pixel_shuffle
from ALIDDM_utils import *
from classes import *
import pandas as pd
import GlobVar as GV
from Agent_class import *
from prediction_utils import *
from monai.networks.nets import UNet
import cv2 as cv
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


def main(args):
    writer_pred = SummaryWriter()

    phong_renderer,mask_renderer = GenPhongRenderer(args.image_size,args.blur_radius,args.faces_per_pixel,GV.DEVICE)

    agent = Agent(
        renderer=phong_renderer,
        renderer2=mask_renderer,
        device=GV.DEVICE,
        radius=args.sphere_radius,
    )

    # recover images/pix_to_face of the patient
    SURF = ReadSurf(args.model_teeth)    
    surf_unit, mean_arr, scale_factor= ScaleSurf(SURF)
    (V, F, CN, RI) = GetSurfProp(surf_unit, mean_arr, scale_factor)
    # print(LP)
    agent.position_agent(RI,V,args.label,GV.DEVICE)
    textures = TexturesVertex(verts_features=CN)
    meshe = Meshes(
                verts=V,   
                faces=F, 
                textures=textures
                ).to(GV.DEVICE)
    images_model , tens_pix_to_face_model=  agent.get_view_rasterize(meshe) #[batch,num_ima,channels,size,size] torch.Size([1, 2, 4, 224, 224])
    tens_pix_to_face_model = tens_pix_to_face_model.permute(1,0,4,2,3) #tens_pix_to_face : torch.Size([1, 2, 1, 224, 224])
    # print(tens_pix_to_face_model,tens_pix_to_face_model.shape)
    # pred (for now y_true)
  
    net = UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=4,
        channels=( 16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=4
    ).to(GV.DEVICE)
    
    inputs = torch.empty((0)).to(GV.DEVICE)
    for i,batch in enumerate(images_model):
        inputs = torch.cat((inputs,batch.to(GV.DEVICE)),dim=0) #[num_im*batch,channels,size,size]

    inputs = inputs.to(dtype=torch.float32)
    net.load_state_dict(torch.load(args.model))
    images_pred = net(inputs)

    print(images_pred.type(),inputs.type())
    print(images_pred.shape,inputs.shape)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, num_classes=4)


    val_pred_outputs_list = decollate_batch(images_pred)                
    val_pred_outputs_convert = [
        post_pred(val_pred_outputs_tensor) for val_pred_outputs_tensor in val_pred_outputs_list
    ]
    val_pred = torch.empty((0)).to(GV.DEVICE)
    for image in images_pred:
        val_pred = torch.cat((val_pred,post_pred(image).unsqueeze(0).to(GV.DEVICE)),dim=0)
    
    # print(val_pred.shape)
    # writer_pred.add_images("prediction",val_pred[:,1:,:,:])
    # writer_pred.add_images("prediction_input",inputs[:,:-1,:,:])

    # writer_pred.close()
    
    # print(inputs.shape)
    # print(images_pred.shape)
    
    
    pred_data = images_pred.detach().cpu().unsqueeze(0).type(torch.int16) #torch.Size([1, 2, 2, 224, 224])
    # print(images_pred.shape,pred_data.shape)
    pred_data = torch.argmax(pred_data, dim=2).unsqueeze(2)
    
    
    # print(pred_data.shape)
    # meshe_2 = Gen_mesh_one_patch(V,F,CN,LP)
    # images_pred =  agent.GetView(meshe_2,rend=True) #images : torch.Size([1, 12, 3, 20, 20])   tens_pix_to_face : torch.Size([12, 1, 20, 20, 1])

    # recover where there is the landmark in the image
    index_label_land_r = (pred_data==1.).nonzero(as_tuple=False) #torch.Size([6252, 5])
    index_label_land_g = (pred_data==2.).nonzero(as_tuple=False) #torch.Size([6252, 5])
    index_label_land_b = (pred_data==3.).nonzero(as_tuple=False) #torch.Size([6252, 5])
    
    # print(index_label_land)
    # print(tens_pix_to_face_model.shape)
    # recover the face in my mesh 
    num_faces_r = []
    num_faces_g = []
    num_faces_b = []

    # print(index_label_land)
    # for index in index_label_land:
    #     # print(index)
    #     num_faces.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]])
    [num_faces_r.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]]) for index in index_label_land_r]
    [num_faces_g.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]]) for index in index_label_land_g]
    [num_faces_b.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]]) for index in index_label_land_b]
      
    dico_rgb = {}
    dico_rgb[f'{GV.LABEL[args.label][0]}'] = num_faces_r
    dico_rgb[f'{GV.LABEL[args.label][1]}'] = num_faces_g
    dico_rgb[f'{GV.LABEL[args.label][2]}'] = num_faces_b
    # print(dico_rgb)
    
    # recover position 
    # print(num_faces)
    
    locator = vtk.vtkOctreePointLocator()
    locator.SetDataSet(surf_unit)
    locator.BuildLocator()
    
    groupe_data = {}

    for land_name,list_face_ids in dico_rgb.items():
        print(land_name)
        list_face_id=[]
        for faces in list_face_ids:
            faces_int = int(faces.item())
            # print(faces_int)
            juan = F[0][faces_int]
            # print(juan)
            list_face_id += [int(juan[0].item()) , int(juan[1].item()) , int(juan[2].item())]
        
        vert_coord = 0
        for vert in list_face_id:
            # print(V[0][vert])
            vert_coord += V[0][vert]

        landmark_pos = vert_coord/len(list_face_id)
        pid = locator.FindClosestPoint(landmark_pos.cpu().numpy())
        closest_landmark_pos = torch.tensor(surf_unit.GetPoint(pid))

        upscale_landmark_pos = Upscale(closest_landmark_pos,mean_arr,scale_factor)
        final_landmark_pos = upscale_landmark_pos
        # print(final_landmark_pos)
        
        coord_dic = {"x":final_landmark_pos[0],"y":final_landmark_pos[1],"z":final_landmark_pos[2]}
        groupe_data[f'{land_name}']=coord_dic
    # print(groupe_data)
    lm_lst = GenControlePoint(groupe_data)
    # print(lm_lst)
    WriteJson(lm_lst,os.path.join(args.out_path,f"Lower_P20_teeth_{args.label}_Pred.json"))



def GetSurfProp(surf_unit, surf_mean, surf_scale):     
    surf = ComputeNormals(surf_unit)
    # landmark_pos = get_landmarks_position(surf_mean, surf_scale)
    # landmark_pos = torch.tensor(landmark_pos,dtype=torch.float32).to(GV.DEVICE)
    color_normals = ToTensor(dtype=torch.float32, device=GV.DEVICE)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
    verts = ToTensor(dtype=torch.float32, device=GV.DEVICE)(vtk_to_numpy(surf.GetPoints().GetData()))
    faces = ToTensor(dtype=torch.int64, device=GV.DEVICE)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
    region_id = ToTensor(dtype=torch.int64, device=GV.DEVICE)(vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID")))
    region_id = torch.clamp(region_id, min=0)

    return verts.unsqueeze(0), faces.unsqueeze(0), color_normals.unsqueeze(0), region_id.unsqueeze(0)

# def get_landmarks_position(mean_arr, scale_factor):
    
#     data = json.load(open(args.jsonfile))
#     markups = data['markups']
#     landmarks_lst = markups[0]['controlPoints']

#     landmark_name=  GV.LABEL[args.label]

#     for landmark in landmarks_lst:
#         label = landmark["label"]
#         if label in landmark_name:
#             landmarks_position = Downscale(landmark["position"],mean_arr,scale_factor)

#     # landmarks_pos = np.array([np.append(pos,1) for pos in landmarks_position])
    
#     return landmarks_position #landmarks_pos[:, 0:3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic Landmark Identification on Digital Dental Model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--model_teeth', type=str, help='path of 3D model of the teeth of 1 patient', default='/home/jonas/Desktop/Baptiste_Baquero/data_ALIDDM/data/patients/P20/Lower/Lower_P20.vtk')
    # input_param.add_argument('--model_teeth', type=str, help='path of 3D model of the teeth of 1 patient', default='/Users/luciacev-admin/Desktop/data_ALIDDM/data/Patients /P3/Lower/Lower_P3.vtk')
    # input_param.add_argument('--jsonfile', type=str, help='path of jsonfile of the teeth of 1 patient', default='/home/jonas/Desktop/Baptiste_Baquero/data_ALIDDM/data/patients/P10/Lower/Lower_P10.json')

    # input_group.add_argument('--dir_cash', type=str, help='Output directory of the training',default=parser.parse_args().dir_data+'/Cash')
    input_param.add_argument('--model', type=str, help='loading of model', default='/home/jonas/Desktop/Baptiste_Baquero/data_ALIDDM/data/best_models/best_metric_model_24.pth')

    #Environment
    input_param.add_argument('-j','--jaw',type=str,help="Prepare the data for uper or lower landmark training (ex: L U)", default="L")
    input_param.add_argument('-sr', '--sphere_radius', type=float, help='Radius of the sphere with all the cameras', default=0.2)
    input_param.add_argument('--label', type=str, help='label of the teeth',default="24")


    #Prediction data
    input_param.add_argument('--image_size',type=int, help='size of the picture', default=224)
    input_param.add_argument('--blur_radius',type=int, help='blur raius', default=0)
    input_param.add_argument('--faces_per_pixel',type=int, help='faces per pixels', default=1)
 
    input_param.add_argument('--out_path',type=str, help='path where jsonfile is saved', default='/home/jonas/Desktop/Baptiste_Baquero/data_ALIDDM/data/out_json')

    
    args = parser.parse_args()
    main(args)




