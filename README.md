# ALIDDM
 
Contributors: Baptiste Baquero, Juan Prieto, Maxime Gillot, Lucia Cevidanes

## What is it?
ALIDDM is an approach to capture 2D views from intra oral scan and use the generated images to train deep learning algorithms or make a prediction once you have a trained model. 

## How it works?
It scales the mesh to the unit sphere, it captures 2D views from 5 different viewpoints, and it's doing this for each tooth. For each camera the neural  network segment patches on the surface to finally recover the final position after an average of every coordinates point in the patch and an upsampling step to the original scale.


![plot](/ALIDDM/Documentation/results.png)

![Alt text](/ALIDDM/Documentation/results.png "Results")
<img src="/ALIDDM/Documentation/results.png)" alt="Alt text" title="Results">


## Running the training code:

```bash
python3 main.py --dir_project 'project directory' --dir_data 'data directory' --dir_patients 'patients directory' --csv_file 'csv file' --jaw 'U or L' --label 'tooth number' --batch_size 'default=10' --max_epoch 'default=300' --dir_models 'Output directory with all the networks'
```

## Running the prediction code:
prerequisites : Good model's orientation and segmentation of each tooth with Universal labelling. 
```bash
python3 prediction.py --vtk_dir 'path of the 3D model' --model_U --model_L --jaw --sphere_radius --out_path 'path where jsonfile is saved'
```

