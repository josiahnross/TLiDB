import configs
import numpy as np
import os
def GetSavedSourceModelDirectory(dataset, model, task):
    model_config_dict = configs.__dict__[f"{model}_config"]
    modelName = model_config_dict["model"]
    return f"./logs_and_models/PRETRAINED_SourceTasks/{dataset}/{modelName}/{task}/"
    
def GetSavedCsvDataDirectory(dataset, model):
    model_config_dict = configs.__dict__[f"{model}_config"]
    modelName = model_config_dict["model"]
    return f"./logs_and_models/PRETRAINED_SourceTasks/{dataset}/{modelName}/"

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
    return f"./logs_and_models/PRETRAINING/{dataset}/{modelName}/seed.{seed}/"

def GetTempModelSavePath(dataset, model, task, seed):
    model_config_dict = configs.__dict__[f"{model}_config"]
    modelName = model_config_dict["model"]
    return f"/mnt/bhd/josiahnross/TransferLearningResearchProject/TLiDB/PRETRAINED_{dataset}/{modelName}/seed.{seed}/{task}/"

def GetTempModelSaveLR_EBS(dataset, model, task, seed, lr, ebs):
    model_config_dict = configs.__dict__[f"{model}_config"]
    modelName = model_config_dict["model"]
    return f"/mnt/bhd/josiahnross/TransferLearningResearchProject/TLiDB/PRETRAINED_{dataset}/{modelName}/seed.{seed}/{task}/LR_{lr}_EBS_{ebs}/"