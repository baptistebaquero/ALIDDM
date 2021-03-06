from monai.networks.nets import UNet
from monai.networks.nets import UNETR

from Agent_class import *
from ALIDDM_utils import *
from monai.data import decollate_batch
import random

def Model(in_channels,out_channels):
    
    net = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=4
    ).to(GV.DEVICE)

    return net
    
 


def Training(train_dataloader,train_data,agent,epoch,nb_epoch,model,optimizer,loss_function,lst_label,writer):
    print('-------- TRAINING --------')          
    print(f"---------- epoch :{epoch + 1}/{nb_epoch} ----------")
    print("-" * 30)
    # print('label :',label)
    model.train() # Switch to training mode
    epoch_loss = 0
    step = 0

    for batch, (S, V, F, RI, CN, LP, MR, SF) in enumerate(train_dataloader):
        step += 1
        jaw_loss = 0
        optimizer.zero_grad()
        for label in lst_label: 
        # label = random.choice(list_label)
            # optimizer.zero_grad()

            print("Training on tooth num :", label)
            textures = TexturesVertex(verts_features=CN)
            meshes = Meshes(
                        verts=V,   
                        faces=F, 
                        textures=textures
                        ).to(GV.DEVICE)
            
            agent.position_agent(RI,V,label)

            images =  agent.GetView(meshes) #[batch,num_ima,channels,size,size]
            
            meshes_2 = Gen_mesh_patch(S,V,F,CN,LP,label)
        
            land_images =  agent.GetView(meshes_2,rend=True) #[batch,num_ima,channels,size,size]
            # PlotAgentViews(land_images.detach().unsqueeze(0).cpu())

            inputs = torch.empty((0)).to(GV.DEVICE)
            y_true = torch.empty((0)).to(GV.DEVICE)
            for i,batch in enumerate(images):
                inputs = torch.cat((inputs,batch.to(GV.DEVICE)),dim=0) #[num_im*batch,channels,size,size]
                y_true = torch.cat((y_true,land_images[i].to(GV.DEVICE)),dim=0) #[num_im*batch,channels,size,size] channels=1

            inputs = inputs.to(dtype=torch.float32)
            y_true = y_true.to(dtype=torch.float32)
        
            # optimizer.zero_grad()
            outputs = model(inputs)
            # PlotAgentViews(outputs.detach().unsqueeze(0).cpu())
            loss = loss_function(outputs,y_true)
            jaw_loss += loss

        jaw_loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = int(np.ceil(len(train_data) / train_dataloader.batch_size))
        print(f"{step}/{epoch_len}, train_loss: {jaw_loss.item():.4f}")
    

    epoch_loss /= step
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    writer.add_scalar("training_loss", epoch_loss, epoch + 1)
    

def Validation(val_dataloader,epoch,nb_epoch,model,agent,lst_label,dice_metric,best_metric,best_metric_epoch,writer,post_pred,post_true,metric_values,nb_val,write_image_interval,dir_models):
    print('-------- VALIDATION --------')
    print(f"---------- epoch :{epoch + 1}/{nb_epoch} ----------")
    print("-" * 30)
    nb_val += 1 
    model.eval()
    with torch.no_grad():
        step = 0
        final_metric = 0
        for batch, (S, V, F, RI, CN, LP, MR, SF) in enumerate(val_dataloader):  
            step += 1 
            for label in lst_label: 
                # label = random.choice(list_label)
                print('Step :',step," Validation on tooth num :", label)
                textures = TexturesVertex(verts_features=CN)
                meshes = Meshes(
                            verts=V,   
                            faces=F, 
                            textures=textures
                            )

                agent.position_agent(RI,V,label)

                images =  agent.GetView(meshes)
                
                # lst_landmarks = Get_lst_landmarks(LP,GV.LABEL[label])
                meshes_2 = Gen_mesh_patch(S,V,F,CN,LP,label)

                land_images =  agent.GetView(meshes_2,rend=True) 
                
                inputs = torch.empty((0)).to(GV.DEVICE)
                y_true = torch.empty((0)).to(GV.DEVICE)

                for i,batch in enumerate(images):
                    inputs = torch.cat((inputs,batch.to(GV.DEVICE)),dim=0) #[num_im*batch,channels,size,size]
                    y_true = torch.cat((y_true,land_images[i].to(GV.DEVICE)),dim=0) #[num_im*batch,channels,size,size]

                inputs = inputs.to(dtype=torch.float32)
                y_true = y_true.to(dtype=torch.float32)

                outputs_pred = model(inputs)
                
                val_pred_outputs_list = decollate_batch(outputs_pred)                
                val_pred_outputs_convert = [
                    post_pred(val_pred_outputs_tensor) for val_pred_outputs_tensor in val_pred_outputs_list
                ]
                
                val_true_outputs_list = decollate_batch(y_true)
                val_true_outputs_convert = [
                    post_true(val_true_outputs_tensor) for val_true_outputs_tensor in val_true_outputs_list
                ]
            
                dice_metric(y_pred=val_pred_outputs_convert, y=val_true_outputs_convert)
                final_metric += dice_metric.aggregate().item()

        metric = final_metric
        # metric = dice_metric.aggregate().item()
        dice_metric.reset()
        metric_values.append(metric)
        
        if not os.path.exists(dir_models):
            os.makedirs(dir_models)
        
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(dir_models,f"best_metric_model.pth"))
            print("saved new best metric model")
        print("current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(epoch + 1, metric, best_metric, best_metric_epoch))

        writer.add_scalar("validation_mean_dice", metric, epoch + 1)
      
        # imgs_true = torch.cat((y_true,torch.zeros(y_true[:,1,:,:].unsqueeze(1).shape).to(GV.DEVICE)),dim=1)
        # imgs_out = torch.cat((outputs_pred,torch.zeros(outputs_pred[:,1,:,:].unsqueeze(1).shape).to(GV.DEVICE)),dim=1)
        
        inputs = inputs[:,:-1,:,:]
        val_pred = torch.empty((0)).to(GV.DEVICE)
        for image in outputs_pred:
            val_pred = torch.cat((val_pred,post_pred(image).unsqueeze(0).to(GV.DEVICE)),dim=0)

        if nb_val %  write_image_interval == 0:       
            writer.add_images("input",inputs,epoch)
            writer.add_images("true",y_true,epoch)
            writer.add_images("output",val_pred[:,1:,:,:],epoch)
            
    writer.close()

    return best_metric,best_metric_epoch
