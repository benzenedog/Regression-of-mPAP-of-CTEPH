import pandas as pd 
import numpy as np
import itertools
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


DF = None


# 0, age
# 1, gender
# 2, mPAP
# 3, BNP
# 4, TRPG
# 5, CTR
# 6, avascular_area
# 7, 2nd arc
def load(args):
    global DF

    if args.dataset == 0:
        csv = "2021_1218_main_NA_replaced_with_mean.csv"
    else:
        csv = "2021_1218_main_NA_excluded.csv"
    print("Target CSV:", csv)
    
    if DF is None:
        print("DF is None. loading file ...")
        df = pd.read_csv(csv)
        DF = df.copy()
        print("Loading is Done")
        
    df = DF.copy()

    if args.log > 0: 
        #log_BNP
        df.iloc[:,3] = np.log(df.iloc[:,3].values)
    if args.log > 1: 
        #log_BNP and log_CTR
        df.iloc[:,5] = np.log(df.iloc[:,5].values)

    if args.scale == 0:
        pass
    elif args.scale == 1:
        sc = StandardScaler()
        df.iloc[:,:] = sc.fit_transform(df.values)
    else: 
        sc = MinMaxScaler()
        df.iloc[:,:] = sc.fit_transform(df.values)

    return df


def train_test(model, seed, args, F):
    M = args.model

    df = load(args)
    N = df.shape[0]
    indices = list(range(N))

    T = args.t
    tr, te = train_test_split(indices, test_size=T, random_state=seed)
    tr_df = df.iloc[tr, F]
    te_df = df.iloc[te, F]

    X = tr_df.iloc[:,1:].values  
    Y = tr_df.iloc[:,0].values
    
    if M == 1:
        automl_settings = {
            "metric": 'mse',
            "task": 'regression',
            "time_budget": 180,
            }
        model.fit(X, Y, **automl_settings)
    else:
        model.fit(X, Y)

    XX = te_df.iloc[:,1:].values  
    YY = te_df.iloc[:,0].values  
    pp = model.predict(XX)

    pp.shape = YY.shape

    r2 = r2_score(YY, pp)
    rmse = np.sqrt(mean_squared_error(YY, pp))
    mae = mean_absolute_error(YY, pp)

    return [r2, rmse, mae]


def list_combinations():
    FF = [0,1,3,4,5,6,7]
    TT = [2]

    l =[]
    for i in [1,2,3,4,5,6,7]:
        l = l + list(itertools.combinations(FF, i))

    return [ TT+list(i) for i in l ]


def get_autokeras():
    s = "my_" + str(random.randint(0, 100000))
    ss = "autokeras_dir" 
    print("new autokeras model:", s, ss)
    return StructuredDataRegressor(max_trials=12, overwrite=True, seed=42, project_name=s, directory=ss)


def main(args):
    N = args.n
    M = args.model

    if args.all_fts > 0:
        if M != 0:
            print("argument error: model != 0", args)
            exit()
        ff = list_combinations()
    else:        
        if args.f > 0:
            print("using optimal features")
            ff = [[2, 0,3,4,5]]
        else:
            ff = [[2, 0,1,3,4,5,6,7]]

    print("GT and Features:", ff)

    models = []
    if M == 0:
        models.append(("LinearRegression",LinearRegression()))
        models.append(("k-Nearest Neighbors",KNeighborsRegressor()))
        models.append(("Support Vector Machine(linear)",LinearSVR(random_state=0)))
        models.append(("Support Vector Machine(rbf)",SVR(kernel='rbf')))
        models.append(("Decision Tree",DecisionTreeRegressor(random_state=0)))
        models.append(("Random Forest",RandomForestRegressor(random_state=0)))
        models.append(("XGBoost", XGBRegressor(random_state=0)))
    elif M == 1:
        models.append( ("FLAML", AutoML()) )
    elif M == 2:
        models.append( ("TPOT", TPOTRegressor(generations=5, population_size=50, random_state=42, verbosity=2)) )
    elif M == 3:
        models.append( ("AutoSklearn", AutoSklearnRegressor(time_left_for_this_task=180, seed=42)) )
    elif M == 4:
        models.append( ("AutoKeras", get_autokeras()) )
    else:
        print("args error")
        exit()
        
    print(models)

    print("*"*80)

    for name, m in models:
        for f in ff:
            print(name, "; features =", f)
            vv = []
            if args.split_seed > -1:
                i = args.split_seed
                #print("starting model: seed =", i)
                v = train_test(m, i, args, f) 
                print("Metrics:", v, ":", name, f)
            else:
                for i in range(N):
                    #print("starting model: seed =", i)
                    v = train_test(m, i, args, f) 
                    if args.verbose > 0:
                        print("R2:", v[0], ":", name, f, v)
                    vv.append(v)
                avg = np.array(vv).mean(axis=0)
                print("Mean R2:", avg[0], ":", name, f)
                print("Means:", avg.tolist(), ":", name, f)

    return


######################
# Main
######################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("-t", type=float, default=0.333)
    parser.add_argument("-f", type=int, default=0)
    parser.add_argument("--dataset", type=int, default=0)
    parser.add_argument("--log", type=int, default=1)
    parser.add_argument("--scale", type=int, default=0)
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--all-fts", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=-1)    
    parser.add_argument("--verbose", type=int, default=0)
        
    args = parser.parse_args()
    print(args)

    M = args.model
    if M == 0:
        pass
    elif M == 1:
        import flaml
        from flaml import AutoML
    elif M == 2:
        from tpot import TPOTRegressor
    elif M == 3:
        from autosklearn.regression import  AutoSklearnRegressor
    else:
        from autokeras import StructuredDataRegressor

    main(args)


