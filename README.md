# ALIDDM
 
Contributors: Baptiste Baquero, Juan Prieto, Maxime Gillot, Lucia Cevidanes

## What is it?
ALIDDM is an approach to capture 2D views from intra oral scan and use the generated images to train deep learning algorithms or make a prediction when you have a trained model. 

## How it works?
It scales the mesh to the unit sphere, it captures 2D views from 5 different viewpoints, and it's doing this for each tooth. For each camera the neural  network segment patches on the surface to finally recover the final position after an average of every coordinates point in the patch and an upsampling step to the original scale.


<img width="575" alt="results" src="https://user-images.githubusercontent.com/83285614/160682839-c33312b1-0e55-48e5-8b9b-844932e4d48e.png" title="results and accuracy">
<img width="549" alt="training" src="https://user-images.githubusercontent.com/83285614/160683370-b31110b1-ecc6-4f1d-9e35-e79c80a1daca.png" title= "Patches' prediction" >


## Running the training code:

```bash
python3 main.py --dir_project 'project directory' --dir_data 'data directory' --dir_patients 'patients directory' --csv_file 'csv file' --jaw 'U or L' --label 'tooth number' --batch_size 'default=10' --max_epoch 'default=300' --dir_models 'Output directory with all the networks'
```
```
All the parameters :
   --dir_project : dataset directory
   --dir_data : Input directory with all the data
   --dir_patients : Input directory with the meshes
   --csv_file : CSV of the data split
   
   --jaw : Prepare the data for uper or lower landmark training (ex: L U), default="L"
   --sphere_radius : Radius of the sphere with all the cameras, default=0.2
   --label : label of the teeth
   --num_device : cuda:0 or cuda:1, default='0'
   --image_size : size of the picture, default=224
   --blur_radius : blur raius, default=0
   --faces_per_pixel : faces per pixels, default=1
   --batch_size : Batch size, default=10
   --num_classes : number of classes, default=4
   --max_epoch : Number of training epocs, default=300
   --val_freq : Validation frequency, default=1
   --val_percentage : Percentage of data to keep for validation, default=10
   --test_percentage : Percentage of data to keep for test, default=20
   --learning_rate : Learning rate, default=1e-4
   --dir_models : Output directory with all the networks
```

## Running the prediction code:
prerequisites : Good model's orientation and segmentation of each tooth with Universal labelling. 
```bash
python3 prediction.py --vtk_dir 'path of the 3D model' --model_U --model_L --jaw --sphere_radius --out_path 'path where jsonfile is saved'
```
```


