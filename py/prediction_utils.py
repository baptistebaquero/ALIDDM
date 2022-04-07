import json
from posixpath import basename
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import numpy as np
import glob

def Upscale(landmark_pos,mean_arr,scale_factor):
    new_pos_center = (landmark_pos.cpu()/scale_factor) + mean_arr
    return new_pos_center

def GenControlePoint(groupe_data):
    lm_lst = []
    false = False
    true = True
    id = 0
    for landmark,data in groupe_data.items():
        id+=1
        controle_point = {
            "id": str(id),
            "label": landmark,
            "description": "",
            "associatedNodeID": "",
            "position": [float(data["x"]), float(data["y"]), float(data["z"])],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": true,
            "locked": true,
            "visibility": true,
            "positionStatus": "preview"
        }
        lm_lst.append(controle_point)

    return lm_lst

def WriteJson(lm_lst,out_path):
    false = False
    true = True
    file = {
    "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
    "markups": [
        {
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "locked": false,
            "labelFormat": "%N-%d",
            "controlPoints": lm_lst,
            "measurements": [],
            "display": {
                "visibility": false,
                "opacity": 1.0,
                "color": [0.4, 1.0, 0.0],
                "selectedColor": [1.0, 0.5000076295109484, 0.5000076295109484],
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": false,
                "pointLabelsVisibility": true,
                "textScale": 3.0,
                "glyphType": "Sphere3D",
                "glyphScale": 1.0,
                "glyphSize": 5.0,
                "useGlyphScale": true,
                "sliceProjection": false,
                "sliceProjectionUseFiducialColor": true,
                "sliceProjectionOutlinedBehindSlicePlane": false,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": false,
                "snapMode": "toVisibleSurface"
            }
        }
    ]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        # print(file)
        json.dump(file, f, ensure_ascii=False, indent=4)

    f.close

def ReadJson(fiducial_path):
    lm_dic = {}
    with open(fiducial_path) as f:
            data = json.load(f)
    markups = data["markups"][0]["controlPoints"]
    for markup in markups:
        lm_dic[markup["label"]] = {"x":markup["position"][0],"y":markup["position"][1],"z":markup["position"][2]}
    return lm_dic

def ResultAccuracy(fiducial_dir):

    error_dic = {"labels":[], "error":[]}
    patients = {}
    normpath = os.path.normpath("/".join([fiducial_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(img_fn) and ".json" in img_fn:
            baseName = os.path.basename(img_fn)
            patient = os.path.dirname(os.path.dirname(img_fn))
            # print(img_fn)

            num_label_pred = os.path.basename(img_fn).split('_')[1][1:]
            # print(baseName,patient,num_label_pred)
            if patient not in patients.keys():
                patients[patient] = {"Upper":{},"Lower":{}}
            if "_Pred" in baseName:
                if "Upper_" in baseName :
                    patients[patient]["Upper"][f"pred_{num_label_pred}"]=img_fn
                elif "Lower_" in baseName :
                    patients[patient]["Lower"][f"pred_{num_label_pred}"]=img_fn
               
            else:
                if "Upper_" in baseName :
                    patients[patient]["Upper"][f"target"]=img_fn
                elif "Lower_" in baseName :
                    patients[patient]["Lower"][f"target"]=img_fn
                
    fail = 0
    max = 0
    mean = 0
    nbr_pred = 0
    error_lst = []
    f = open(os.path.join(fiducial_dir,"Result.txt"),'w')
    #  print(patients['/Users/luciacev-admin/Desktop/test_accuracy/data/Patients /P10'])
    print(patients)
    for patient,fiducials in patients.items():
        print("Results for patient",patient)
        f.write("Results for patient "+ str(patient)+"\n")
        
        for group,targ_res in fiducials.items():
            print(" ",group,"landmarks:")
            f.write(" "+ str(group)+" landmarks:\n")
            print(targ_res.keys())
            for pat in range(1,163):
                if f"pred_{pat}" in targ_res.keys():
                    target_lm_dic = ReadJson(targ_res["target"])
                    pred_lm_dic = ReadJson(targ_res[f"pred_{pat}"])
                    for lm,t_data in target_lm_dic.items():
                        if lm in pred_lm_dic.keys():
                            a = np.array([float(t_data["x"]),float(t_data["y"]),float(t_data["z"])])
                            p_data = pred_lm_dic[lm]
                            b = np.array([float(p_data["x"]),float(p_data["y"]),float(p_data["z"])])
                            # print(a,b)
                            dist = np.linalg.norm(a-b)
                            if dist > max: max = dist
                            if dist < 5:
                                nbr_pred+=1
                                mean += dist
                                error_dic["labels"].append(lm)
                                error_dic["error"].append(dist)
                                error_lst.append(dist)
                            else:
                                fail +=1
                            print("  ",lm,"error = ", dist)
                            f.write("  "+ str(lm)+" error = "+str(dist)+"\n")
        f.write("\n")
        f.write("\n")

    print(fail,'fail')
    print("STD :", np.std(error_lst))
    print('Error max :',max)
    print('Mean error',mean/nbr_pred)
    
    f.close
    return error_dic


def PlotResults(data):
    sns.set_theme(style="whitegrid")
    # data = {"labels":["B","B","N","N","B","N"], "error":[0.1,0.5,1.6,1.9,0.3,1.3]}    

    # print(tips)
    ax = sns.violinplot(x="labels", y="error", data=data, cut=0)
    plt.show()

def remove_extra_faces(F,num_faces,RI,label):
    last_num_faces =[]
    # print(num_faces)
    # print(RI)
    # print(len(num_faces))
    for face in num_faces:
        # print('label :',label)
        # print('face :',face.item())
        # print('RI.shape :',RI.shape)
        # print(RI.squeeze(0)[int(face.item())])
        # print(F.shape)
        # print(F.squeeze(0)[int(face.item())])
        vertex_color = F.squeeze(0)[int(face.item())]
        # print(vertex_color)
        for vert in vertex_color:
            # print("vert :",vert)
            # print(RI.squeeze(0)[vert])
            if RI.squeeze(0)[vert] == label:
                last_num_faces.append(face)
            # else:
            #     print('wrong label')
    return last_num_faces
