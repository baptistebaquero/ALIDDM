import argparse
import os

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
import shutil
from scipy import linalg

def main(args):
    sphere_points_L = ([0,0,1],
                    np.array([0.5,0.,1.0])/linalg.norm([0.5,0.5,1.0]),
                    np.array([-0.5,0.,1.0])/linalg.norm([-0.5,-0.5,1.0]),
                    np.array([0,0.5,1])/linalg.norm([1,0,1]),
                    np.array([0,-0.5,1])/linalg.norm([0,1,1])
                    )
    sphere_points_U = ([0,0,-1],
                    np.array([0.5,0.,-1])/linalg.norm([0.5,0.5,-1]),
                    np.array([-0.5,0.,-1])/linalg.norm([-0.5,-0.5,-1]),
                    np.array([0,0.5,-1])/linalg.norm([1,0,-1]),
                    np.array([0,-0.5,-1])/linalg.norm([0,1,-1])
                    )
    GV.DEVICE = torch.device(f"cuda:{args.num_device}" if torch.cuda.is_available() else "cpu")
    GV.SELECTED_JAW = args.jaw
    
    if GV.SELECTED_JAW == "U":
        lst_label = args.label_U
        dir_model = args.model_U
        # csv_file = args.csv_file_U
        GV.CAMERA_POSITION = np.array(sphere_points_U)

    else :
        lst_label = args.label_L
        dir_model = args.model_L
        csv_file = args.csv_file_L
        GV.CAMERA_POSITION = np.array(sphere_points_L)

    print(GV.CAMERA_POSITION)
    print(csv_file)
    print(lst_label)
    print(dir_model)
    # vtk_normpath = os.path.normpath("/".join([args.vtk_dir,'**','']))
    lst_vtkfiles = []
    # for vtkfile in sorted(glob.iglob(vtk_normpath, recursive=True)):
    #     if os.path.isfile(vtkfile) and True in [ext in vtkfile for ext in [".vtk"]]:
    #         lst_vtkfiles.append(vtkfile)
    
    # print(lst_vtkfiles)


    df = pd.read_csv(csv_file)
    df_test = df.loc[df['for'] == "test"]
    # print(df_test['surf'])
    for vtkfile in df_test['surf']:
        full_vtkfile = args.patient_path + '/' + vtkfile
        # print(full_vtkfile)
        lst_vtkfiles.append(full_vtkfile)
    
    print(lst_vtkfiles)
    


    for path_vtk in lst_vtkfiles:
        num_patient = os.path.basename(path_vtk).split('.')[0].split('_')[1][1:]
        print(f"prediction for patient {num_patient} :", path_vtk )
        groupe_data = {}
        
        for label in lst_label:
            model = os.path.join(dir_model,f"best_metric_model_{label}.pth")
            print("Loading model :", model, "for patient :", num_patient, "label :", label)
            phong_renderer,mask_renderer = GenPhongRenderer(args.image_size,args.blur_radius,args.faces_per_pixel,GV.DEVICE)

            agent = Agent(
                renderer=phong_renderer,
                renderer2=mask_renderer,
                radius=args.sphere_radius,
            )

            SURF = ReadSurf(path_vtk)    
            surf_unit, mean_arr, scale_factor= ScaleSurf(SURF)
            (V, F, CN, RI) = GetSurfProp(surf_unit, mean_arr, scale_factor)
     
            if int(label) in RI.squeeze(0):
                agent.position_agent(RI,V,label)
                textures = TexturesVertex(verts_features=CN)
                meshe = Meshes(
                            verts=V,   
                            faces=F, 
                            textures=textures
                            ).to(GV.DEVICE)
                images_model , tens_pix_to_face_model=  agent.get_view_rasterize(meshe) #[batch,num_ima,channels,size,size] torch.Size([1, 2, 4, 224, 224])
                tens_pix_to_face_model = tens_pix_to_face_model.permute(1,0,4,2,3) #tens_pix_to_face : torch.Size([1, 2, 1, 224, 224])
                       
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
                net.load_state_dict(torch.load(model))
                images_pred = net(inputs)

                post_pred = AsDiscrete(argmax=True, to_onehot=True, num_classes=4)


                val_pred_outputs_list = decollate_batch(images_pred)                
                val_pred_outputs_convert = [
                    post_pred(val_pred_outputs_tensor) for val_pred_outputs_tensor in val_pred_outputs_list
                ]
                val_pred = torch.empty((0)).to(GV.DEVICE)
                for image in images_pred:
                    val_pred = torch.cat((val_pred,post_pred(image).unsqueeze(0).to(GV.DEVICE)),dim=0)
                         
                
                pred_data = images_pred.detach().cpu().unsqueeze(0).type(torch.int16) #torch.Size([1, 2, 2, 224, 224])
                pred_data = torch.argmax(pred_data, dim=2).unsqueeze(2)
                
                
      
                # recover where there is the landmark in the image
                index_label_land_r = (pred_data==1.).nonzero(as_tuple=False) #torch.Size([6252, 5])
                index_label_land_g = (pred_data==2.).nonzero(as_tuple=False) #torch.Size([6252, 5])
                index_label_land_b = (pred_data==3.).nonzero(as_tuple=False) #torch.Size([6252, 5])

                # recover the face in my mesh 
                num_faces_r = []
                num_faces_g = []
                num_faces_b = []
            
                for index in index_label_land_r:
                    num_faces_r.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]]) 
                for index in index_label_land_g:
                    num_faces_g.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]])
                for index in index_label_land_b:
                    num_faces_b.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]]) 
                
                
                last_num_faces_r = remove_extra_faces(F,num_faces_r,RI,int(label))
                last_num_faces_g = remove_extra_faces(F,num_faces_g,RI,int(label))
                last_num_faces_b = remove_extra_faces(F,num_faces_b,RI,int(label))       
               
                # print(F.shape,RI.shape)
                # print(F)
                # print(RI)
                # print("")
                # print(last_num_faces_r)
                # print('num_faces_r',len(num_faces_r))
                # print('last_num_faces_r',len(last_num_faces_r))

                dico_rgb = {}
                dico_rgb[f'{GV.LABEL[label][0]}'] = last_num_faces_r
                dico_rgb[f'{GV.LABEL[label][1]}'] = last_num_faces_g
                dico_rgb[f'{GV.LABEL[label][2]}'] = last_num_faces_b
                
                
                locator = vtk.vtkOctreePointLocator()
                locator.SetDataSet(surf_unit)
                locator.BuildLocator()
                
                for land_name,list_face_ids in dico_rgb.items():
                    print(land_name)
                    list_face_id=[]
                    for faces in list_face_ids:
                        faces_int = int(faces.item())
                        juan = F[0][faces_int]
                        list_face_id += [int(juan[0].item()) , int(juan[1].item()) , int(juan[2].item())]
                    
                    vert_coord = 0
                    for vert in list_face_id:
                        vert_coord += V[0][vert]
                    if len(list_face_id) != 0:
                        landmark_pos = vert_coord/len(list_face_id)
                        pid = locator.FindClosestPoint(landmark_pos.cpu().numpy())
                        closest_landmark_pos = torch.tensor(surf_unit.GetPoint(pid))

                        upscale_landmark_pos = Upscale(closest_landmark_pos,mean_arr,scale_factor)
                        final_landmark_pos = upscale_landmark_pos
                        
                        coord_dic = {"x":final_landmark_pos[0],"y":final_landmark_pos[1],"z":final_landmark_pos[2]}
                        groupe_data[f'{land_name}']=coord_dic

        lm_lst = GenControlePoint(groupe_data)
        out_path = os.path.join(args.out_path,f"P{num_patient}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_path_jaw = os.path.join(out_path,os.path.basename(path_vtk).split('.')[0].split('_')[0])
        if not os.path.exists(out_path_jaw):
            os.makedirs(out_path_jaw)
 
        copy_file = os.path.join(out_path_jaw,os.path.basename(path_vtk))
        final_out_path = shutil.copy(path_vtk,copy_file)
        
        landmark_path = os.path.join(os.path.dirname(path_vtk),f"Lower_P{num_patient}.json")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        copy_json_file =  os.path.join(out_path_jaw,os.path.basename(landmark_path))
        final_outpath_json = shutil.copy(landmark_path,copy_json_file)
        # final_out_path = shutil.copytree(path_vtk,out_path_L)

        WriteJson(lm_lst,os.path.join(out_path_jaw,f"Lower_P{num_patient}_Pred.json"))



def GetSurfProp(surf_unit, surf_mean, surf_scale):     
    surf = ComputeNormals(surf_unit)
    color_normals = ToTensor(dtype=torch.float32, device=GV.DEVICE)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
    verts = ToTensor(dtype=torch.float32, device=GV.DEVICE)(vtk_to_numpy(surf.GetPoints().GetData()))
    faces = ToTensor(dtype=torch.int64, device=GV.DEVICE)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
    region_id = ToTensor(dtype=torch.int64, device=GV.DEVICE)(vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID")))
    region_id = torch.clamp(region_id, min=0)

    return verts.unsqueeze(0), faces.unsqueeze(0), color_normals.unsqueeze(0), region_id.unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic Landmark Identification on Digital Dental Model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    # input_param.add_argument('--model_teeth', type=str, help='path of 3D model of the teeth of 1 patient', default='/home/jonas/Desktop/Baptiste_Baquero/data_ALIDDM/data/patients/P20/Lower/Lower_P20.vtk')
    # input_param.add_argument('--vtk_dir', type=str, help='path of 3D model of the teeth of 1 patient', default='/home/luciacev-admin/Desktop/Baptiste_Baquero/Project/ALIDDM/data/Upper_jaw_lab')
    input_param.add_argument('--csv_file_L', type=str, help='path of the csv', default='/home/luciacev-admin/Desktop/Baptiste_Baquero/Project/ALIDDM/data/data_split/Lower/data_splitfold2.csv')
    # input_param.add_argument('--csv_file_U', type=str, help='path of the csv', default='/home/luciacev-admin/Desktop/Baptiste_Baquero/Project/ALIDDM/data/data_split/Upper')
    input_param.add_argument('--patient_path', type=str, help='path of the patient folder', default='/home/luciacev-admin/Desktop/Baptiste_Baquero/Project/ALIDDM/data/patients')

    # input_param.add_argument('--model_teeth', type=str, help='path of 3D model of the teeth of 1 patient', default='/Users/luciacev-admin/Desktop/data_ALIDDM/data/Patients /P3/Lower/Lower_P3.vtk')
    # input_param.add_argument('--jsonfile', type=str, help='path of jsonfile of the teeth of 1 patient', default='/home/jonas/Desktop/Baptiste_Baquero/data_ALIDDM/data/patients/P10/Lower/Lower_P10.json')

    # Model directories
    input_param.add_argument('--model_U', type=str, help='loading of model', default='/home/luciacev-admin/Desktop/Baptiste_Baquero/Project/ALIDDM/data/models/Upper/models_csv/')
    input_param.add_argument('--model_L', type=str, help='loading of model', default='/home/luciacev-admin/Desktop/Baptiste_Baquero/Project/ALIDDM/data/models/Lower/models_csv0')

    # Environment
    input_param.add_argument('--jaw',type=str,help="Prepare the data for uper or lower landmark training (ex: L U)", default="L")
    input_param.add_argument('--sphere_radius', type=float, help='Radius of the sphere with all the cameras', default=0.2)
    input_param.add_argument('--label_L', type=list, help='label of the teeth',default=["18","19","20","21","22","23","24"])#,"25","26","27","28","29","30","31"])
    input_param.add_argument('--label_U', type=list, help='label of the teeth',default=(["2","3","4","5","6","7","8","9","10","11","12","13","14","15"]))

    # Prediction data
    input_param.add_argument('--num_device',type=str, help='cuda:0 or cuda:1', default='0')
    input_param.add_argument('--image_size',type=int, help='size of the picture', default=224)
    input_param.add_argument('--blur_radius',type=int, help='blur raius', default=0)
    input_param.add_argument('--faces_per_pixel',type=int, help='faces per pixels', default=1)
 
    input_param.add_argument('--out_path',type=str, help='path where jsonfile is saved', default='/home/luciacev-admin/Desktop/Baptiste_Baquero/Project/ALIDDM/data/prediction/test2')

    args = parser.parse_args()
    main(args)




