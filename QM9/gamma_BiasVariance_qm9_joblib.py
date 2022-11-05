###########################################################
# 
# 1 - Importing libs
# 
#############################################################
import logging
from statistics import mean
from tokenize import Double
import warnings
import os
import psutil
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import sys; sys.path.append('/usr/local/qmmlpack/python')

import qmmlpack

from pymatgen.core.periodic_table import Element

from joblib import Parallel, delayed

from random import choices

import sklearn
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

from scipy.stats import kurtosis, gamma

from cmlkit.engine import save_yaml, read_yaml
from cmlkit.model import Model
from cmlkit import Dataset
from cmlkit.dataset import Subset

###########################################################
# 
# 2 - Main sequence
# 
#############################################################
runtimes = {}
t0 = datetime.now()

def main_sequence():

    #log, config <- configs
    log, config, new_runtimes = load_cofigurations() 
    runtimes.update(new_runtimes)
    log.info(f"Configurations have been read in {runtimes['load_config']}.")

    #ds_qm9 <- read_qm9
    ds_qm9, dataQM9, folds, new_runtimes = read_qm9(log, config)
    runtimes.update(new_runtimes)
    log.info(f"Dataset was mounted in {runtimes['read_qm9']}.")

    #model <- read_model(model_to_analyse)
    model = read_model(config[config["test_ref"]])
    log.info("Model has been read.")

    #decomposition <- kfold_bias_variance_decomp
    decomposition, average_vch, new_runtimes = kfold_bias_variance_decomp(log, config, model, ds_qm9, 
                                                            folds, config["fold_to_process"])
    runtimes.update(new_runtimes)
    log.info(f"Decomposition finished in {runtimes['total_kfold_bias_variance_decomp']}.")
    
    # check if all folds were processed and statics can be calculates
    folds_processed, decomposition, x_average_vch = read_folds_results(config, folds, ds_qm9)
    if len(folds_processed) == config[config["test_ref"]]["num_folds"]:
        # statistical analysis
        statistics = statistical_analysis(config, decomposition, dataQM9)
        log.info("Statistics has been calculated.")

        runtimes["total run time"] = str(datetime.now() - t0)
        
        #save_results    
        save_results(log, decomposition, dataQM9, average_vch, runtimes, statistics, config)
    else:
        log.info(f"You have folds {folds_processed} processed. You need all {config[config['test_ref']]['num_folds']} folds processed to calculate statistics and save results.")
        


###########################################################
# 
# 3 - Bias Variance predict and decompsition
# 
#############################################################

def kfold_bias_variance_decomp(log, config, model, ds, folds, fold_to_process):

    t = datetime.now()

    test_config = config[config["test_ref"]]
    num_folds=test_config["num_folds"]
    num_rounds=test_config["num_rounds"]
    column_y =  test_config["column_y"]
    len_y = len(ds.p[column_y])
    error = np.zeros(len_y)
    bias = np.zeros(len_y)
    var = np.zeros(len_y)
    predictions = np.zeros((len_y, num_rounds))
    avch_fold = []
    avch_round = []
    avch_train = []
    avch_test = []

    # Define range of folds to use according to "--fold" parameter
    if (fold_to_process < 0):
        folds_to_process = range(1, num_folds + 1)
    else:
        folds_to_process = range(fold_to_process, fold_to_process + 1)
    
    runtimes = {}
    for fold in folds_to_process:
        train = folds[f"train_{fold}"]
        test = folds[f"test_{fold}"]
        t0 = datetime.now()
        log.info(f"Start processing fold {fold} of {num_folds}")
        ds_train = Subset.from_dataset(ds, idx=train, name="train")
        ds_test = Subset.from_dataset(ds, idx=test, name="test")
        error[test], bias[test], var[test], predictions[test], average_vch, new_runtimes = bias_variance_decomp(
                                                                log, test_config, fold,
                                                                model,
                                                                ds_train,
                                                                ds_test,
                                                                num_rounds = num_rounds)
        avch_fold += list(np.full((10), fold))
        avch_round += list(average_vch[0])
        avch_train += list(average_vch[1])
        avch_test += list(average_vch[2])

        total_fold = str(datetime.now() - t0)
        new_runtimes.update({"total_folds:": total_fold})
        runtimes.update({f"fold{fold}": {**new_runtimes}})
        log.info(f"Fold {fold} of {num_folds} finished in {total_fold}")
    
        # save results for use when all fols were processed
        save_fold_results(config, fold, error, bias, var, predictions, average_vch)
   
    runtimes.update({"total_kfold_bias_variance_decomp": str(datetime.now() - t)})
    averages_vch = {"folds": avch_fold, "round": avch_round, "train": avch_train, "test": avch_test}

    return (error, bias, var, predictions), averages_vch, runtimes

def exec_range(num_round, num_rounds, log, test_config, ds_train, model, column_y, ds_test):
        try:
            t0 = datetime.now()
            log.info(f"Start processing round {num_round+1} of {num_rounds}")

            ds_boot = _draw_bootstrap_sample(test_config, ds_train)
            avch_train = np.mean(ds_boot.p["VCH"])
            avch_test = np.mean(ds_test.p["VCH"])
            pred = model.train(ds_boot, column_y).predict(ds_test)
            
            #all_pred[i] = pred
            runtimes.update({f"round{num_round+1}": str(datetime.now() - t0)})
            log.info(f"Round {num_round+1} of {num_rounds} finished in {runtimes[f'round{num_round+1}']}")
        except TypeError:
            log.info(f"Type error: {sys.exc_info()[0]}")  
        except Exception as e:
            log.info(f"Generic error: {sys.exc_info()[0]}")  
        
        return (num_round, pred, avch_train, avch_test)

def bias_variance_decomp(log, test_config, count_fold, model, ds_train, ds_test, 
                         num_rounds=20, random_seed=None):

    log.info(f"Starting {num_rounds}-round procesing of fold {count_fold}.")
    t = datetime.now()
    runtimes = {}

    column_y =  test_config["column_y"]
    all_pred = np.zeros((num_rounds, len(ds_test.p[column_y])), dtype=np.double)
    avch_round = np.zeros((num_rounds) ,dtype=np.int)
    avch_train = np.zeros((num_rounds) ,dtype=np.double)
    avch_test = np.zeros((num_rounds) ,dtype=np.double)

    x1 = datetime.now()

    preds = Parallel(n_jobs=-1, verbose=4)(delayed(exec_range)(num_round, num_rounds, log, test_config, \
                                                               ds_train, model, column_y, ds_test) \
                                           for num_round in range(num_rounds))
    
    
    for i, pred, vch_train, vch_test in preds:
        all_pred[i] = pred    
        avch_round[i] = i
        avch_train[i] = vch_train
        avch_test[i] = vch_test

    x2 = datetime.now() - x1

    loss = np.mean((all_pred - ds_test.p[column_y])**2,axis=0)
    
    main_predictions = np.mean(all_pred, axis=0)

    bias = (main_predictions - ds_test.p[column_y])**2
    var = np.mean((main_predictions - all_pred)**2,axis=0) 
    runtimes["total_bias_variance_decomp"] = str(datetime.now() - t)
    
    return loss, bias, var, np.transpose(all_pred), (avch_round, avch_train, avch_test), runtimes


def _draw_bootstrap_sample(test_config, ds_train):
    
    sample_indices = np.arange(len(ds_train.p[test_config["column_y"]]))
    bootstrap_indices = gamma_sample(x = sample_indices, 
                                    w = ds_train.p["VCH"],
                                    size = test_config['qtd_sample_qm9'],
                                        a = test_config['gamma_a'], 
                                        variance= test_config['gamma_variance'])
    return Subset.from_dataset(ds_train, bootstrap_indices)
    

def gamma_sample(x, w, size, a, loc = 0, variance = 0):

    scale = np.sqrt(variance/a)
    weights = gamma.pdf(w, a, loc, scale)
    return choices(x, weights=weights, cum_weights=None, k=size)


###########################################################
# 
# 4 - General Functions
# 
#############################################################

def save_fold_results(config, fold, error, bias, var, predictions, average_vch):
    file_name = os.path.join(config["path"], f"error_fold_{fold}_{config['test_ref']}.npy")
    np.save(file_name, error)
    file_name = os.path.join(config["path"], f"bias_fold_{fold}_{config['test_ref']}.npy")
    np.save(file_name, bias)
    file_name = os.path.join(config["path"], f"var_fold_{fold}_{config['test_ref']}.npy")
    np.save(file_name, var)
    file_name = os.path.join(config["path"], f"predictions_fold_{fold}_{config['test_ref']}.npy")
    np.save(file_name, predictions)
    file_name = os.path.join(config["path"], f"average_vch_fold_{fold}_{config['test_ref']}.npy")
    np.save(file_name, average_vch)
    
def read_folds_results(config, folds_train_test, ds):
    decomposition = []
    folds_processed = []
    average_vch = []
    folds = range(1, config[config["test_ref"]]["num_folds"] + 1)

    # basic check for folds processed
    for fold in folds:
        file_name = os.path.join(config["path"], f"error_fold_{fold}_{config['test_ref']}.npy")
        if os.path.isfile(file_name):
            folds_processed.append(fold)
    
    # process files if all folds where processed
    if len(folds_processed) == len(folds):
        folds_processed = []
        num_rounds = config[config["test_ref"]]["num_rounds"]
        column_y =  config[config["test_ref"]]["column_y"]
        len_y = len(ds.p[column_y])
        error = np.zeros(len_y)
        bias = np.zeros(len_y)
        var = np.zeros(len_y)
        predictions = np.zeros((len_y, num_rounds))
        for fold in folds:
            all_files_exist = True
            file_name = os.path.join(config["path"], f"error_fold_{fold}_{config['test_ref']}.npy")
            if os.path.isfile(file_name):
                file_error = np.load(file_name)
                error = merge_data(fold, folds_train_test, error, file_error)
            else:
                all_files_exist = False
            file_name = os.path.join(config["path"], f"bias_fold_{fold}_{config['test_ref']}.npy")
            if os.path.isfile(file_name):
                file_bias = np.load(file_name)
                bias = merge_data(fold, folds_train_test, bias, file_bias)
            else:
                all_files_exist = False
            file_name = os.path.join(config["path"], f"var_fold_{fold}_{config['test_ref']}.npy")
            if os.path.isfile(file_name):
                file_var = np.load(file_name)
                var = merge_data(fold, folds_train_test, var, file_var)
            else:
                all_files_exist = False
            file_name = os.path.join(config["path"], f"predictions_fold_{fold}_{config['test_ref']}.npy")
            if os.path.isfile(file_name):
                file_predictions = np.load(file_name)
                predictions = merge_data(fold, folds_train_test, predictions, file_predictions)
            else:
                all_files_exist = False
            #file_name = os.path.join(config["path"], f"average_vch_fold_{fold}_{config['test_ref']}.npy")
            #if os.path.isfile(file_name):
            #    file_average_vch = np.load(file_name)
            #    average_vch = merge_data(fold, folds_train_test, predictions, file_average_vch)
            #else:
            #    all_files_exist = False
            if all_files_exist:
                folds_processed.append(fold)

        decomposition = (error, bias, var, predictions)

    return folds_processed, decomposition, average_vch

def merge_data(fold, folds, destination, origin):
    test_index = f"test_{fold}"

    for index in folds[test_index]:
        destination[index] = origin[index]

    return destination

def statistical_analysis(config, decomposition, dataQM9):
    y = dataQM9[config[config["test_ref"]]["column_y"]]
    
    squared_error = decomposition[0]
    bias = decomposition[1]
    variance = decomposition[2]
    predictions = decomposition[3]
    
    y_hat = np.mean(predictions, axis=1)

    loss = y_hat - y
    
    mse = np.mean(squared_error)
    rmse = mse ** 0.5
    mean_bias = np.mean(bias)
    mean_variance = np.mean(variance)
    r2 = metrics.r2_score(y, y_hat)
    mae = metrics.mean_absolute_error(y, y_hat)
    mean_signed_error = np.mean(loss)
    rmsd = np.sqrt(np.sum( np.mean(  np.square(predictions - mean_signed_error) ,axis=1 ) ) / (len(y) - 1)) 
    q95 = np.percentile(abs(loss), 95)
    c1 = sum(1 for v in loss if v <= 1) / len(loss)
    skew = 0 #skew(abs(erro))
    kurt = kurtosis(loss, fisher=False) # false => Pearson’s definition is used (normal ==> 3.0)
    #W_erro = shapiro(erro)
    #W_erro_abs = shapiro(abs(erro))
    #W_erro_2 = shapiro(erro**2)
    
    # inclui no dataframe
    statistics = {'Mean Squared Error (MSE)': mse, "RMSE": rmse, 'Mean Bias': mean_bias, \
                  'Mean Variance': mean_variance, 'R2 score':r2, 'MAE': mae, "Mean Signed Error":mean_signed_error, \
                  "RMSD":rmsd, "Q95": q95, "C(1)": c1, "Skew":skew, "Kurtosis":kurt}

    return statistics

def load_cofigurations():
    t0 = datetime.now()

    # defining Configuration file to use
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--config_file', "-c", type= str, default= "config_test.yml", # "config.yml",
        help= "Configuration file name.")
    parser.add_argument('--fold', "-f", type= int, default= -1, # "config.yml",
        help= "Number of fold to prcess. Must be greater than 1.")
    args = parser.parse_args()

    # read configuration file
    config_file = os.path.join("", args.config_file)
    config  = read_yaml(config_file)

    #name and path to log file
    log_file_name = f"Log_{config['test_ref']}_{datetime.now()}.log"
    logFile = os.path.join(config["path"], log_file_name)

    # Cofiguring log
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG) 
    logFormatter = logging.Formatter('"%(asctime)s [%(levelname)-5.5s]  %(message)s"')
    fh = logging.FileHandler(logFile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logFormatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logFormatter)
    ch.setLevel(logging.DEBUG)
    log.addHandler(ch)

    log.info("Start!")

    # validate the number of fold to process
    if args.fold > 0 and args.fold <= config[config['test_ref']]["num_folds"]:
        config.update({"fold_to_process": args.fold})
    else:
        config.update({"fold_to_process": -1})


    # avoid warning messages
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    # logging versions
    log.info("Package version:")
    log.info("SYS: %s" % (sys.executable+ " " + sys.version))    
    log.info("Sklearn: %s" % sklearn.__version__)
    #log.info("Scipy: %s" % scipy.__version__)
    log.info("Numpy: %s" % np.__version__)
    log.info("Pandas: %s" % pd.__version__)
    #log.info("Cmlkit: %s" % cmlkit.__version__)
    log.info("Memory:")
    log.info(f"{psutil.virtual_memory()}") 
    
    period = {"load_config": str(datetime.now() - t0)}

    return log, config, period

def read_qm9(log, config):

    t0 = datetime.now()

    # reading qm9
    log.info("Reading qm9")
    fileQM9 = os.path.join(config["path"], config["fileQM9"])
    dataQM9 = pd.read_pickle(fileQM9)

    # reduce tha base when required
    #qtd_sample_qm9 =  config[config["test_ref"]]["qtd_sample_qm9"]
    #qtd_sample_qm9 = 1000
    #if qtd_sample_qm9 > 0:
    #    dataQM9 = dataQM9.sample(qtd_sample_qm9, replace = False)
    
    # creating dataset with column_y as "p" property
    log.info("Mounting dataset")
    data_z = np.array([np.array([Element(k).number for k in x[0]]) for x in dataQM9['Coordinate']])
    data_r = np.array([x[1] for x in dataQM9['Coordinate']])
    ds = Dataset(data_z, data_r)
    column_y =  config[config["test_ref"]]["column_y"]
    y = np.asarray(dataQM9[column_y])
    
    #ds.p = {column_y: y}
    ds.p = {column_y: y, "VCH": np.asarray(dataQM9["VCH"])}
    
    # read or generate folds
    file_name = os.path.join(config["path"], f"folds_{config['test_ref']}.yml")
    if os.path.isfile(file_name):
        folds = read_yaml(file_name)
    else:
        kf = KFold(config[config['test_ref']]["num_folds"], shuffle=True)
        folds = {}
        count_fold = 1
        for train, test in kf.split(ds.p[column_y]):
            folds.update({f"train_{count_fold}": train.tolist()})
            folds.update({f"test_{count_fold}": test.tolist()})
            count_fold = count_fold + 1
        save_yaml(file_name, folds)

    period = {"read_qm9": str(datetime.now() - t0)}
    
    return ds, dataQM9, folds, period

def read_model(test_config):
    
    model = Model.from_config(test_config["model"])
    return model

def save_results(log, decomposition, dataQM9, average_vch, runtimes, statistics, config):
    
    log.info("Saving results.")

    col_erro = decomposition[0]
    col_bias = decomposition[1]
    col_variance = decomposition[2]
    col_predictions = decomposition[3]
    data2 = dataQM9.copy()
    data2["squared_error_"+config["test_ref"]] = col_erro
    data2["bias_"+config["test_ref"]] = col_bias
    data2["variance_"+config["test_ref"]] = col_variance
    data2["predictions_"+config["test_ref"]] = col_predictions.tolist()
    file_name = os.path.join(config["path"], "qm9_estudo_erro_"+config["test_ref"]+".pkl")
    data2.to_pickle(file_name)
    file_name_csv = os.path.join(config["path"], "qm9_estudo_erro_"+config["test_ref"]+".csv")
    data2.to_csv(file_name_csv)

    df = pd.DataFrame(average_vch)
    file_name = os.path.join(config["path"], "average_vch_"+config["test_ref"]+".pkl")
    df.to_pickle(file_name)

    statistics_and_runtimes = {"statistics": statistics, "runtimes": runtimes}
    file_name = os.path.join(config["path"], "statistics_and_runtimes_"+config["test_ref"]+".pkl")
    save_yaml(file_name, statistics_and_runtimes)
    log.info(f"statistics_and_runtimes: {statistics_and_runtimes}")
    log.info("Finished!")


if __name__ == "__main__":
    main_sequence()