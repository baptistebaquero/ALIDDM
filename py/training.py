from monai.networks.nets import UNet
from Agent_class import *
from ALIDDM_utils import *

def Model(device):
    
    net = UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        # num_res_units=2,
    ).to(device)

    return net

def Training(train_dataloader,train_data,agent,epoch,nb_epoch,model,optimizer,loss_function,label,writer,device):
    print(f"---------- epoch :{epoch + 1}/{nb_epoch} ----------")
    print("-" * 30)
    model.train() # Switch to training mode
    epoch_loss = 0
    step = 0
    for batch, (V, F, RI, CN, LP, MR, SF) in enumerate(train_dataloader):
        step += 1
        print(CN)
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(
                    verts=V,   
                    faces=F, 
                    textures=textures
                    ).to(GV.DEVICE)

        agent.position_agent(RI,V,label,GV.DEVICE)

        images =  agent.GetView(meshes) #[batch,num_ima,channels,size,size]
        

        lst_landmarks = Get_lst_landmarks(LP,GV.LABEL[label])
        meshes_2 = Generate_land_Mesh(lst_landmarks,GV.DEVICE)
        land_images =  agent.GetView(meshes_2) #[batch,num_ima,channels,size,size]
        # print(outputs.shape)
        
        inputs = torch.empty((0)).to(GV.DEVICE)
        y_true = torch.empty((0)).to(GV.DEVICE)
        for i,batch in enumerate(images):
            # print(batch.shape)
            inputs = torch.cat((inputs,batch.to(GV.DEVICE)),dim=0) #[num_im*batch,channels,size,size]
            y_true = torch.cat((y_true,land_images[i].to(GV.DEVICE)),dim=0) #[num_im*batch,channels,size,size]
  
        inputs = inputs.to(dtype=torch.float32)
        y_true = y_true.to(dtype=torch.float32)
       
        # print(y_true.shape)
        y_true_grey = Convert_RGB_to_grey(y_true)
        print(y_true_grey.shape)

        PlotAgentViews(y_true_grey.detach().unsqueeze(0).cpu())

        # print(inputs.type())
        # print(y_true.type())

        optimizer.zero_grad()
        outputs = model(inputs)
        # PlotAgentViews(outputs.detach().unsqueeze(0).cpu())
        loss = loss_function(outputs,y_true)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = int(np.ceil(len(train_data) / train_dataloader.batch_size))
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
      
    epoch_loss /= step
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    writer.add_scalar("training_loss", epoch_loss, epoch + 1)


def Validation(val_dataloader,epoch,model,agent,label,dice_metric,best_metric,early_stopping,writer):
    model.eval()
    with torch.no_grad():
        val_images = None
        val_yp = None
        val_outputs = None
        
        for batch, (V, F, RI, CN, LP, MR, SF) in enumerate(val_dataloader):   
            
            textures = TexturesVertex(verts_features=CN)
            meshes = Meshes(
                        verts=V,   
                        faces=F, 
                        textures=textures
                        )

            agent.position_agent(RI,V,label,GV.DEVICE)

            inputs =  agent.GetView(meshes)
            
            lst_landmarks = Get_lst_landmarks(LP,GV.LABEL[label])
            meshes_2 = Generate_land_Mesh(lst_landmarks,GV.DEVICE)
            output_true =  agent.GetView(meshes_2) 

            outputs_pred= model(inputs)
            dice_metric(y_pred=outputs_pred, y=output_true) 
        
        metric = dice_metric.aggregate().item()
        dice_metric.reset()

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model.pth")
            print("saved new best metric model")
        print("current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(epoch + 1, metric, best_metric, best_metric_epoch))

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(1-metric, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
             

        # writer.add_scalar("validation_mean_dice", metric, epoch + 1)
        #     imgs_output = torch.argmax(val_outputs, dim=1).detach().cpu()
        #     imgs_output = imgs_output.unsqueeze(1)  # insert dim of size 1 at pos. 1
        #     imgs_normals = val_images[:,0:3,:,:]
        #     val_rgb = torch.cat((255*(1-2*val_labels/33),255*(2*val_labels/33-1),val_labels),dim=1) 
        #     out_rgb = torch.cat((255*(1-2*imgs_output/33),255*(2*imgs_output/33-1),imgs_output),dim=1) 
            
        #     val_rgb[:,2,...] = 255 - val_rgb[:,1,...] - val_rgb[:,0,...]
        #     out_rgb[:,2,...] = 255 - out_rgb[:,1,...] - out_rgb[:,0,...]

        #     norm_rgb = imgs_normals

        #     if nb_val % write_image_interval == 0:       
        #         writer.add_images("labels",val_rgb,epoch)
        #         writer.add_images("output", out_rgb,epoch)
        #         writer.add_images("normals",norm_rgb,epoch)
            
            
# def Accuracy(agents,test_dataloader,agents_ids,min_variance,loss_function,device):
#     list_distance = ({ 'obj' : [], 'distance' : [] })
#     with torch.no_grad():
#         for batch, (V, F, CN, LP, MR, SF) in enumerate(test_dataloader):
#             groupe_data = {}
#             radius = 0.02
#             textures = TexturesVertex(verts_features=CN)
#             meshes = Meshes(
#                 verts=V,   
#                 faces=F, 
#                 textures=textures
#             )
       
#                 print('---------- agents id :', aid ,'----------')
#                 agents[aid].reset_sphere_center(V.shape[0])
#                 agents[aid].eval() 
#                 pos_center = agents[aid].search(meshes,min_variance) #[batchsize,3]
#                 # plot_fig(meshes,center_mesh)

#                 lm_pos = torch.empty((0)).to(device)
#                 for lst in LP:
#                     lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)  #[batchsize,3]
#                     # center_mesh = generate_sphere_mesh(pos_center[0],radius,device,0.9)
                
#                 perfect_pos,verts_teeth,faces_teeth,verts_rgb = generate_sphere_mesh(lm_pos[0],radius,device,0.0)
#                 dic = {"teeth_mesh": meshes, 'landmark':perfect_pos}
#                 for index,step in enumerate(agents[aid].position_center_memory):
#                     center_mesh,verts_teeth,faces_teeth,verts_rgb = generate_sphere_mesh(step[0],radius,device,0.9)
#                     dic[str(index)]=center_mesh

#                 plot_fig(dic)
                
#                 for i in range(V.shape[0]):
#                     # loss = torch.sqrt(loss_function(pos_center[i], lm_pos[i]))
#                     list_distance['obj'].append(str(aid))
#                     # list_distance['distance'].append(float(loss.item()))
#                     scale_surf = SF[i]
#                     # print('scale_surf :', scale_surf)
#                     mean_arr = MR[i]
#                     # print('mean_arr :', mean_arr)
#                     agent_pos = pos_center[i]
#                     # print('landmark_pos before rescaling :', agent_pos)
#                     new_pos_center = Upscale(agent_pos,mean_arr,scale_surf)#(landmark_pos/scale_surf) + mean_arr
#                     # print('pos_center after rescaling :', new_pos_center)
#                     landmark_pos = Upscale(lm_pos[i],scale_surf,mean_arr)
#                     # print('d',LP[i][aid])
#                     # new_landmark_pos = Upscale(LP[i][aid],mean_arr,scale_surf)
#                     # print('m',mean_arr)
#                     # print('s',scale_surf)
#                     # print('u',new_landmark_pos)
#                     new_pos_center=new_pos_center.cpu()
#                     landmark_pos=landmark_pos.cpu()
#                     distance = np.linalg.norm(new_pos_center-landmark_pos)
#                     # print('distance between prediction and real landmark :',distance)
#                     list_distance['distance'].append(distance)
#                     coord_dic = {"x":landmark_pos[0],"y":landmark_pos[1],"z":landmark_pos[2]}
#                     # print(coord_dic)
#                     groupe_data[f'Lower_O-{aid+1}']=coord_dic
#                     # print(groupe_data)
#                     # print(PS[i])
#                     # dic_patients[PS[i]]=groupe_data
#                 # writer.add_scalar('distance',loss)

#             # print(list_distance)
        
#         sns.violinplot(x='obj',y='distance',data=list_distance)
#         plt.show()