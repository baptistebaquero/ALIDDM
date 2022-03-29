import argparse
from prediction_utils import *

def main(args):
    error_dic = ResultAccuracy(args.fiducial_dir)
    PlotResults(error_dic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic Landmark Identification on Digital Dental Model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--fiducial_dir', type=str, help='directory with the target data and predicted data', default='/home/luciacev-admin/Desktop/Baptiste_Baquero/Project/ALIDDM/data/prediction/unique_model_pred_csv1')
    
    args = parser.parse_args()
    main(args)