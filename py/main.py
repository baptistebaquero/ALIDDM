import argparse
import datetime
from ALIDDM_utils import *
from classes import *
import pandas as pd
import GlobVar as GV
from Agent_class import *

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from torch.utils.tensorboard import SummaryWriter
from training import Model,Training
from monai.metrics import DiceMetric
# from pytorchtools import EarlyStopping

def main(args):
    
    #GEN CSV
    # GenDataSplitCSV(args.dir_patients,args.csv_file,args.val_percentage,args.test_percentage)
    phong_renderer = GenPhongRenderer(args.image_size,args.blur_radius,args.faces_per_pixel,GV.DEVICE)

    GV.SELECTED_JAW = args.jaw

    df = pd.read_csv(args.csv_file)
    # df_train = df.loc[df['for'] == "train"]

    train_data,val_data = GenDataSet(df,args.dir_patients,FlyByDataset,GV.DEVICE)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_verts_faces)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_verts_faces)

    # loss_function = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    target = GV.LANDMARKS[GV.SELECTED_JAW][1]

    agent = Agent(
        renderer=phong_renderer,
        target=target,
        device=GV.DEVICE,
        save_folder=args.dir_data,
        radius=args.sphere_radius,
    )

    model = Model(GV.DEVICE)

    loss_function = monai.losses.DiceCELoss(to_onehot_y=True,softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    best_metric = -1
    writer = SummaryWriter()
    # early_stopping=EarlyStopping(patience=20, verbose=True, path=args.out)
    # early_stopping = EarlyStopping(patience=20, verbose=True)
    
    for epoch in range(args.max_epoch):
        print('-------- TRAINING --------')          
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
            writer=writer
        )

        # if (epoch) % args.val_freq == 0:
        #     print('-------- VALIDATION --------')
        #     print('---------- epoch :', epoch,'----------')
        #     Validation(
        #         val_dataloader=val_dataloader,
        #         epoch= epoch,
        #         model=model,
        #         agent=agent,
        #         label=args.label,
        #         dice_metric=dice_metric,
        #         best_metric=best_metric,
        #         early_stopping=early_stopping,
        #         writer=writer
        #     )
            
            # if early_stopping.early_stop == True :
            #     print('-------- ACCURACY --------')
            #     Accuracy(agents=agents,
            #             test_dataloader=test_dataloader,
            #             agents_ids=agents_ids,
            #             min_variance = args.min_variance,
            #             loss_function=loss_function,
            #             device=device
            #             )

            #     break




    # for batch, (V, F, RI, CN, LP, MR, SF) in enumerate(train_dataloader):
    #     textures = TexturesVertex(verts_features=CN)
    #     meshes = Meshes(
    #         verts=V,   
    #         faces=F, 
    #         textures=textures
    #     ) # batchsize
       
    #     position_agent = a.position_agent(RI,V,args.label,GV.DEVICE)
    #     # PlotMeshAndSpheres(meshes,position_agent,0.02,[1,1,1])       

    #     img_batch =  a.GetView(meshes)
    #     PlotAgentViews(img_batch)


    #     lst_landmarks = Get_lst_landmarks(LP,GV.LABEL[args.label])
    #     meshes_2 = Generate_land_Mesh(lst_landmarks,GV.DEVICE)
    #     img_batch_2 =  a.GetView(meshes_2)
    #     PlotAgentViews(img_batch_2)
       
       
        # meshes = Generate_Mesh(V,F,CN,lst_landmarks,GV.DEVICE)
        # dic = {"teeth_landmarks_meshes": meshes}
        # plot_fig(dic)

        

    # print(a.positions)

    # PlotMeshAndSpheres(meshes,a.positions,0.02,[1,0,0])
    # PlotAgentViews(a.GetView(meshes))

    # PlotDatasetWithLandmark(target,train_dataloader)
    
    # print(landmark_pos)

    # print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    # writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic Landmark Identification on Digital Dental Model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir_project', type=str, help='dataset directory', default='/Users/luciacev-admin/Desktop/ALIDDM_BENCHMARK')
    input_param.add_argument('--dir_data', type=str, help='Input directory with all the data', default=parser.parse_args().dir_project+'/data')
    input_param.add_argument('--dir_patients', type=str, help='Input directory with the meshes',default=parser.parse_args().dir_data+'/patients')
    input_param.add_argument('--csv_file', type=str, help='CSV of the data split',default=parser.parse_args().dir_data+'/data_split.csv')

    # input_group.add_argument('--dir_cash', type=str, help='Output directory of the training',default=parser.parse_args().dir_data+'/Cash')
    input_param.add_argument('--dir_model', type=str, help='Output directory of the training',default= parser.parse_args().dir_data+'/ALI_models_'+datetime.datetime.now().strftime("%Y_%d_%m"))

    #Environment
    input_param.add_argument('-j','--jaw',type=str,help="Prepare the data for uper or lower landmark training (ex: L U)", default="L")
    input_param.add_argument('-sr', '--sphere_radius', type=float, help='Radius of the sphere with all the cameras', default=0.3)
    input_param.add_argument('--label', type=str, help='label of the teeth',default="18")
   
    #Training data
    input_param.add_argument('--image_size',type=int, help='size of the picture', default=60)
    input_param.add_argument('--blur_radius',type=int, help='blur raius', default=0)
    input_param.add_argument('--faces_per_pixel',type=int, help='faces per pixels', default=1)
    
    input_param.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=2)
    # input_param.add_argument('-ds', '--data_size', type=int, help='Data size', default=100)

    # input_param.add_argument('-ds', '--data_size', type=int, help='Size of the dataset', default=4)
    #Training param
    input_param.add_argument('-me', '--max_epoch', type=int, help='Number of training epocs', default=1)
    input_param.add_argument('-vf', '--val_freq', type=int, help='Validation frequency', default=1)
    input_param.add_argument('-vp', '--val_percentage', type=int, help='Percentage of data to keep for validation', default=10)
    input_param.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for test', default=20)
    input_param.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=1e-4)
    # input_param.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=0)

    # parser.add_argument('--nbr_pictures',type=int,help='number of pictures per tooth', default=5)
   
    # output_params = parser.add_argument_group('Output parameters')
    # output_params.add_argument('--out', type=str, help='Output directory with all the 2D pictures')
    
    args = parser.parse_args()
    main(args)