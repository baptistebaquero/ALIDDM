# ALIDDM
 
Contributors: Baptiste Baquero, Juan Prieto, Maxime Gillot, Lucia Cevidanes

## What is it?
ALIDDM is an approach to capture 2D views from intra oral scan and use the generated images to train deep learning algorithms or make a prediction once you have a trained model. 

## How it works?
It scales the mesh to the unit sphere, it captures 2D views from 5 different viewpoints, and it's doing this for each tooth. For each camera the neural  network segment patches on the surface to finally recover the final position after an average of every coordinates point in the patch and an upsampling step to the original scale.


![plot](/home/luciacev-admin/Desktop/Baptiste_Baquero/Project/ALIDDM/Documentation/results.png)


## Running the code:
