import pandas as pd 
import numpy as np 
import argparse

parser = argparse.ArgumentParser(description='Splits data into train/test based on study_id', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--csv', type=str, help='CSV file, each row is a sample', required=True)
parser.add_argument('--folds', type=int, help='Split data in folds', default=0)

args = parser.parse_args()

fname = args.csv
df = pd.read_csv(fname)

df = df.sample(frac=1.0).reset_index(drop=True)

if args.folds > 0:
    samples = int(len(df.index)/args.folds)
else:
    samples = 0

print("total:", len(df.index), "testing samples per split:", samples)

if args.folds == 0:

    iloc_arr = np.arange(0, len(df))
    
    iloc_test = iloc_arr[0:samples]
    iloc_train = iloc_arr[samples:]

    df_train = df.iloc[iloc_train]
    df_test = df.iloc[iloc_test]

    if samples > 0:

        train_fn = fname.replace('.csv', '_train.csv')
        df_train.to_csv(train_fn, index=False)

        eval_fn = fname.replace('.csv', '_test.csv')
        df_test.to_csv(eval_fn, index=False)
    else:

        split_fn = fname.replace('.csv', '_split.csv')
        df_train.to_csv(split_fn, index=False)
        
else:

    start_iloc = 0
    end_iloc = samples
    for i in range(args.folds):

        iloc_arr = np.arange(0, len(df))

        iloc_test = iloc_arr[start_iloc:end_iloc]

        df_train = df.iloc[np.isin(iloc_arr, iloc_test, invert=True)]
        df_train = df_train.reset_index(drop=True)
        df_test = df.iloc[iloc_test]
        df_test = df_test.reset_index(drop=True)

        df_test["for"] = ["test"]*len(df_test.index)

        df_train["for"] = ["train"]*len(df_train.index)

        train_fn = fname.replace('.csv', 'fold' + str(i) + '_train.csv')
        df_train.to_csv(train_fn, index=False)

        eval_fn = fname.replace('.csv', 'fold' + str(i) + '_test.csv')
        df_test.to_csv(eval_fn, index=False)

        df_fold = pd.concat([df_train, df_test])
        full_fn = fname.replace('.csv', 'fold' + str(i) + '.csv')
        print("Writing:", full_fn)
        df_fold.to_csv(full_fn, index=False)

        start_iloc += samples
        end_iloc += samples