import argparse
from unittest.mock import patch
from ALIDDM_utils import *
from classes import *
import pandas as pd
import GlobVar as GV
from Agent_class import *

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from torch.utils.tensorboard import SummaryWriter
from training import Model,Training,Validation
from monai.losses import DiceCELoss
from torch.optim import Adam
from monai.metrics import DiceMetric
# from pytorchtools import EarlyStopping
from monai.transforms import AsDiscrete
# import numpy as np
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
    GV.SELECTED_JAW = args.jaw
    GV.DEVICE = torch.device(f"cuda:{args.num_device}" if torch.cuda.is_available() else "cpu")
    if GV.SELECTED_JAW == "L":
        GV.CAMERA_POSITION = np.array(sphere_points_L)
    else:
        GV.CAMERA_POSITION = np.array(sphere_points_U)

    #GEN CSV
    # if not os.path.exists(args.):
    #     os.makedirs(out_path)
    # GenDataSplitCSV(args.dir_patients,args.csv_file,args.val_percentage,args.test_percentage)
    # SplitCSV_train_Val('/home/luciacev-admin/Desktop/Baptiste_Baquero/Project/ALIDDM/data/data_split/Upper/data_split.csv',0.13)
    phong_renderer,mask_renderer = GenPhongRenderer(args.image_size,args.blur_radius,args.faces_per_pixel,GV.DEVICE)

    df = pd.read_csv(args.csv_file)

    train_data,val_data = GenDataSet(df,args.dir_patients,FlyByDataset,GV.DEVICE,args.label)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_verts_faces)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_verts_faces)


    agent = Agent(
        renderer=phong_renderer,
        renderer2=mask_renderer,
        radius=args.sphere_radius,
    )

    num_classes = args.num_classes
    model = Model(4,num_classes)

    loss_function = DiceCELoss(to_onehot_y=True,softmax=True)
    optimizer = Adam(model.parameters(), args.learning_rate)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_true = AsDiscrete(to_onehot=True, num_classes=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, num_classes=num_classes)
    metric_values = list()
    nb_val = 0
    write_image_interval = 1

    best_metric = -1
    best_metric_epoch =-1
    writer = SummaryWriter()
    
    for epoch in range(args.max_epoch):
        # label = random.choice(args.label)
        Training(
            train_dataloader=train_dataloader,
            train_data=train_data,
            agent=agent,
            epoch=epoch,
            nb_epoch=args.max_epoch,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            label=args.label,
            writer=writer,
            )

        if (epoch) % args.val_freq == 0:        
            best_metric,best_metric_epoch = Validation(
                val_dataloader=val_dataloader,
                epoch= epoch,
                nb_epoch=args.max_epoch,
                model=model,
                agent=agent,
                label=args.label,
                dice_metric=dice_metric,
                best_metric=best_metric,
                best_metric_epoch=best_metric_epoch,
                nb_val = nb_val,
                writer=writer,
                write_image_interval=write_image_interval,
                post_true=post_true,
                post_pred=post_pred,
                metric_values=metric_values,
                dir_models=args.dir_models
                )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic Landmark Identification on Digital Dental Model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir_project', type=str, help='dataset directory', default='/home/luciacev-admin/Desktop/Baptiste_Baquero/Project/ALIDDM')
    input_param.add_argument('--dir_data', type=str, help='Input directory with all the data', default=parser.parse_args().dir_project+'/data')
    input_param.add_argument('--dir_patients', type=str, help='Input directory with the meshes',default=parser.parse_args().dir_data+'/patients')
    input_param.add_argument('--csv_file', type=str, help='CSV of the data split',default=parser.parse_args().dir_data+'/data_split/Lower/data_splitfold4.csv')


    #Environment
    input_param.add_argument('-j','--jaw',type=str,help="Prepare the data for uper or lower landmark training (ex: L U)", default="L")
    input_param.add_argument('-sr', '--sphere_radius', type=float, help='Radius of the sphere with all the cameras', default=0.2)
    # input_param.add_argument('--label', type=list, help='label of the teeth',default=(["18","19","20","21","22","23","24","25","26","27","28","29","30","31"]))
    # input_param.add_argument('--label', type=list, help='label of the teeth',default=(["2","3","4","5","6","7","8","9","10","11","12","13","14","15"]))

    input_param.add_argument('--label', type=str, help='label of the teeth',default="31")

    #Training data
    input_param.add_argument('--num_device',type=str, help='cuda:0 or cuda:1', default='0')
    input_param.add_argument('--image_size',type=int, help='size of the picture', default=224)
    input_param.add_argument('--blur_radius',type=int, help='blur raius', default=0)
    input_param.add_argument('--faces_per_pixel',type=int, help='faces per pixels', default=1)
    
    input_param.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=10)
    input_param.add_argument('-nc', '--num_classes', type=int, help='number of classes', default=4)

    # input_param.add_argument('-ds', '--data_size', type=int, help='Data size', default=100)

    # input_param.add_argument('-ds', '--data_size', type=int, help='Size of the dataset', default=4)
    #Training param
    input_param.add_argument('-me', '--max_epoch', type=int, help='Number of training epocs', default=300)
    input_param.add_argument('-vf', '--val_freq', type=int, help='Validation frequency', default=1)
    input_param.add_argument('-vp', '--val_percentage', type=int, help='Percentage of data to keep for validation', default=10)
    input_param.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for test', default=20)
    input_param.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=1e-4)
    # input_param.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=0)

    # parser.add_argument('--nbr_pictures',type=int,help='number of pictures per tooth', default=5)
   
    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('--dir_models', type=str, help='Output directory with all the networks',default=parser.parse_args().dir_data+'/models/Lower/models_csv4')

    
    args = parser.parse_args()
    main(args)