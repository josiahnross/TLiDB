import configs
import numpy as np
import os
from utils import Logger, loadState
def GetSavedSourceModelDirectory(dataset, model, task):
    model_config_dict = configs.__dict__[f"{model}_config"]
    modelName = model_config_dict["model"]
    return f"/mnt/bhd/josiahnross/TransferLearningResearchProject/TLiDB/logs_and_models/PRETRAINED_SourceTasks/{dataset}/{modelName}/{task}/"
    
def GetSavedCsvDataDirectory(dataset, model):
    model_config_dict = configs.__dict__[f"{model}_config"]
    modelName = model_config_dict["model"]
    return f"/mnt/bhd/josiahnross/TransferLearningResearchProject/TLiDB/logs_and_models/PRETRAINED_SourceTasks/{dataset}/{modelName}/"

def SaveElementIntoDataCSV(path: str, sourceTask: str, targetTask:str, value):
    csvData = np.genfromtxt(path, delimiter=',', dtype=str)
    row = np.where(csvData[:, 0] == sourceTask)[0]
    col = np.where(csvData[0, :] == targetTask)[0]
    csvData[row, col] = str(value)
    np.savetxt(path, csvData, delimiter=',', fmt='%s')

def CreateNewEmptyCSV(emptyCsvPath, path):
    csvData = np.genfromtxt(emptyCsvPath, delimiter=',', dtype=str)
    np.savetxt(path, csvData, delimiter=',', fmt='%s')

def GetOrMakeEvalDataCSV(dataset, model, splitSeed, splitPercent, ):
    splitPercentDataPath = GetSavedCsvDataDirectory(dataset, model)
    if not os.path.exists(splitPercentDataPath):
        os.makedirs(splitPercentDataPath)
    splitPercentDataPath += f"data_seed_{splitSeed}_splitPercent{splitPercent}.csv"
    if not os.path.exists(splitPercentDataPath):
        CreateNewEmptyCSV(f"empty{dataset}.csv", splitPercentDataPath)
    return splitPercentDataPath

def SaveHyperparameterntoCSV(path, task, lr, ebs, validation, epochs):
    csvData = np.genfromtxt(path, delimiter=',', dtype=str)
    col = np.where(csvData[0, :] == task)[0]
    csvData[1, col] = str(lr)
    csvData[2, col] = str(ebs)
    csvData[3, col] = str(validation)
    csvData[4, col] = str(epochs)
    np.savetxt(path, csvData, delimiter=',', fmt='%s')

def GetSavedHyperparameterCSVDirectory(dataset, model, seed):
    model_config_dict = configs.__dict__[f"{model}_config"]
    modelName = model_config_dict["model"]
    return f"/mnt/bhd/josiahnross/TransferLearningResearchProject/TLiDB/logs_and_models/PRETRAINING/{dataset}/{modelName}/seed.{seed}/"

def GetTempModelSavePath(dataset, model, task, seed):
    model_config_dict = configs.__dict__[f"{model}_config"]
    modelName = model_config_dict["model"]
    return f"/mnt/bhd/josiahnross/TransferLearningResearchProject/TLiDB/logs_and_models/PRETRAINED_{dataset}/{modelName}/seed.{seed}/{task}/"

def GetTempModelSaveLR_EBS(dataset, model, task, seed, lr, ebs):
    model_config_dict = configs.__dict__[f"{model}_config"]
    modelName = model_config_dict["model"]
    return f"/mnt/bhd/josiahnross/TransferLearningResearchProject/TLiDB/logs_and_models/PRETRAINED_{dataset}/{modelName}/seed.{seed}/{task}/LR_{lr}_EBS_{ebs}/"

def LoadModelStateIfExists(path: str, logger: Logger):
    if os.path.exists(path):
        return loadState(path, logger)
    return None

def CreatePathIfNotExist(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def GetOverallLogger(path:str, name:str):
    CreatePathIfNotExist(path)
    if name is not None:
        overallLogDir = path + f"overallLogger{name}.txt"
        if not os.path.exists(overallLogDir):
            overallLogger = Logger(overallLogDir, mode='w')
        else:
            overallLogger = Logger(overallLogDir, mode='a')
            overallLogger.write(f"\n\nRestarting Overall Logger\n\n")
    else:
        loggerCount = 0
        overallLogDir = path + f"overallLogger{loggerCount}.txt"
        while os.path.exists(overallLogDir):
            loggerCount+=1
            overallLogDir = path + f"overallLogger{loggerCount}.txt"
        overallLogger = Logger(overallLogDir, mode='w')
    return overallLogger

def LoadModelIfExitstsWithLastUpdateEpoch(path:str, logger:Logger, bestVal:float =0, lastImproveEpoch:int=0):
    modelState = LoadModelStateIfExists(path + "last_model.pt", logger)
    if modelState is not None:
        bestModelState = LoadModelStateIfExists(path + "best_model.pt", None)
        bestVal = bestModelState['best_val_metric']
        lastImproveEpoch = bestModelState['epoch']
        del bestModelState
        if logger is not None:
            logger.flush()
        return modelState, bestVal, lastImproveEpoch
    return None, bestVal, lastImproveEpoch 