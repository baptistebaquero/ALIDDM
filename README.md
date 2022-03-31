# ALIDDM
 
Contributors: Baptiste Baquero, Juan Prieto, Maxime Gillot, Lucia Cevidanes

## What is it?
ALIDDM is an approach to capture 2D views from intra oral scan and use the generated images to train deep learning algorithms or make a prediction once you have a trained model. 

## How it works?
It scales the mesh to the unit sphere, it captures 2D views from 5 different viewpoints, and it's doing this for each tooth. For each camera the neural  network segment patches on the surface to finally recover the final position after an average of every coordinates point in the patch and an upsampling step to the original scale.


<img width="575" alt="results" src="https://user-images.githubusercontent.com/83285614/160682839-c33312b1-0e55-48e5-8b9b-844932e4d48e.png" title="results and accuracy">
<img width="549" alt="training" src="https://user-images.githubusercontent.com/83285614/160683370-b31110b1-ecc6-4f1d-9e35-e79c80a1daca.png" title= "Patches' prediction" >


## Running the training code:

```bash
python3 main.py --dir_project 'project directory' --dir_data 'data directory' --dir_patients 'patients directory' --csv_file 'csv file' --jaw 'U or L' --label 'tooth number' --batch_size 'default=10' --max_epoch 'default=300' --dir_models 'Output directory with all the networks'
```

## Running the prediction code:
prerequisites : Good model's orientation and segmentation of each tooth with Universal labelling. 
```bash
python3 prediction.py --vtk_dir 'path of the 3D model' --model_U --model_L --jaw --sphere_radius --out_path 'path where jsonfile is saved'
```

