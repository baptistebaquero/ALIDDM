import sys
import vtk
import argparse
import glob
import os

def main(args):  
    STL_normpath = os.path.normpath("/".join([args.dir_data_stl,'**','']))
    list_patient = []
    for stlfile in sorted(glob.iglob(STL_normpath, recursive=True)):
        if os.path.isfile(stlfile) and True in [ext in stlfile for ext in [".stl"]]:
            if True in [typ in stlfile for typ in ["mand"]]:
                list_patient.append(stlfile)

    # print(list_patient)
    # print(len(list_patient))

    for num,patient in enumerate(list_patient):
        
        filename = patient 
        a = vtk.vtkSTLReader()
        a.SetFileName(filename)
        a.Update()
        a = a.GetOutput()

        # Write the .vtk file
        new_namefile = f"Lower_new_{num+1}.vtk"
        outfilename = os.path.join(args.dir_data_vtk,new_namefile)
        b = vtk.vtkPolyDataWriter()
        b.SetFileName(outfilename)
        b.SetInputData(a)
        b.Update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='separAte all the teeth from a vtk file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir_data_stl', type=str, help='Input directory with stl models', default='/Users/luciacev-admin/Desktop/Digital Dental Models')

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('--dir_data_vtk', type=str, help='Output directory', default='/Users/luciacev-admin/Desktop/new_data_ALIDDM')
    
    args = parser.parse_args()
    main(args)