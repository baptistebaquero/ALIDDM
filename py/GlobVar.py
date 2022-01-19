import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Lower = []
Upper = []
# ALL
id_lst = range(1,43)
# CUSTOM
# id_lst = [1,25]

for i in id_lst:
    Lower.append("Lower_O-"+str(i))
    Upper.append("Upper_O-"+str(i))

LANDMARKS = {
    "L":Lower,
    "U":Upper
}

global SELECTED_JAW
SELECTED_JAW = "L"

# MOVEMENT_MATRIX_6 = torch.tensor([
#     [1,0,0],  # MoveLeft
#     [-1,0,0], # MoveRight
#     [0,1,0],  # MoveBack
#     [0,-1,0], # MoveFront
#     [0,0,1],  # MoveUp
#     [0,0,-1], # MoveDown
# ])

LABEL = {
    "15" : LANDMARKS["U"][0:3],
    "14" : LANDMARKS["U"][3:6],
    "13" : LANDMARKS["U"][6:9],
    "12" : LANDMARKS["U"][9:12],
    "11" : LANDMARKS["U"][12:15],
    "10" : LANDMARKS["U"][15:18],
    "9" : LANDMARKS["U"][18:21],
    "8" : LANDMARKS["U"][21:24],
    "7" : LANDMARKS["U"][24:27],
    "6" : LANDMARKS["U"][27:30],
    "5" : LANDMARKS["U"][30:33],
    "4" : LANDMARKS["U"][33:36],
    "3" : LANDMARKS["U"][36:39],
    "2" : LANDMARKS["U"][39:42],

    "18" : LANDMARKS["L"][0:3],
    "19" : LANDMARKS["L"][3:6],
    "20" : LANDMARKS["L"][6:9],
    "21" : LANDMARKS["L"][9:12],
    "22" : LANDMARKS["L"][12:15],
    "23" : LANDMARKS["L"][15:18],
    "24" : LANDMARKS["L"][18:21],
    "25" : LANDMARKS["L"][21:24],
    "26" : LANDMARKS["L"][24:27],
    "27" : LANDMARKS["L"][27:30],
    "28" : LANDMARKS["L"][30:33],
    "29" : LANDMARKS["L"][33:36],
    "30" : LANDMARKS["L"][36:39],
    "31" : LANDMARKS["L"][39:42],

    


    # "2" : ['Lower_O-1','Lower_O-2','Lower_O-3'],
    # "3" : ['Lower_O-4','Lower_O-5','Lower_O-6'],
    # "4" : ['Lower_O-7','Lower_O-8','Lower_O-9'],
    # "5" : ['Lower_O-10','Lower_O-11','Lower_O-12'],
    # "6" : ['Lower_O-13','Lower_O-14','Lower_O-15'],
    # "7" : ['Lower_O-16','Lower_O-17','Lower_O-18'],
    # "8" : ['Lower_O-19','Lower_O-20','Lower_O-21'],
    # "9" : ['Lower_O-22','Lower_O-23','Lower_O-24'],
    # "10" : ['Lower_O-25','Lower_O-26','Lower_O-27'],
    # "11" : ['Lower_O-28','Lower_O-29','Lower_O-30'],
    # "12" : ['Lower_O-31','Lower_O-32','Lower_O-33'],
    # "13" : ['Lower_O-34','Lower_O-35','Lower_O-36'],
    # "14" : ['Lower_O-37','Lower_O-38','Lower_O-39'],
    # "15" : ['Lower_O-40','Lower_O-41','Lower_O-42'],

    # "18" : ['Upper_O-1','Upper_O-2','Upper_O-3'],
    # "19" : ['Upper_O-4','Upper_O-5','Upper_O-6'],
    # "20" : ['Upper_O-7','Upper_O-8','Upper_O-9'],
    # "21" : ['Upper_O-10','Upper_O-11','Upper_O-12'],
    # "22" : ['Upper_O-13','Upper_O-14','Upper_O-15'],
    # "23" : ['Upper_O-16','Upper_O-17','Upper_O-18'],
    # "24" : ['Upper_O-19','Upper_O-20','Upper_O-21'],
    # "25" : ['Upper_O-22','Upper_O-23','Upper_O-24'],
    # "26" : ['Upper_O-25','Upper_O-26','Upper_O-27'],
    # "27" : ['Upper_O-28','Upper_O-29','Upper_O-30'],
    # "28" : ['Upper_O-31','Upper_O-32','Upper_O-33'],
    # "29" : ['Upper_O-34','Upper_O-35','Upper_O-36'],
    # "30" : ['Upper_O-37','Upper_O-38','Upper_O-39'],
    # "31" : ['Upper_O-40','Upper_O-41','Upper_O-42']
}




