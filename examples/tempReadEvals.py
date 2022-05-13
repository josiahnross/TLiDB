from asyncio import tasks
from run_experimentV2 import EvalModel, TrainModel
from utils import Logger, get_savepath_dir_minimal, loadState
import torch
import os
import configs
import numpy as np
from Config import Config

def GetSavedSourceModelDirectory(dataset, model, task):
    model_config_dict = configs.__dict__[f"{model}_config"]
    modelName = model_config_dict["model"]
    return f"./logs_and_models/PRETRAINED_SourceTasks/{dataset}/{modelName}/{task}/"
    
def GetSavedCsvDataDirectory(dataset, model):
    model_config_dict = configs.__dict__[f"{model}_config"]
    modelName = model_config_dict["model"]
    return f"./logs_and_models/PRETRAINED_SourceTasks/{dataset}/{modelName}/"

def SaveElementIntoCSV(path: str, sourceTask: str, targetTask:str, value):
    csvData = np.genfromtxt(path, delimiter=',', dtype=str)
    row = np.where(csvData[:, 0] == sourceTask)[0]
    col = np.where(csvData[0, :] == targetTask)[0]
    csvData[row, col] = str(value)
    np.savetxt(path, csvData, delimiter=',', fmt='%s')

def CreateNewEmptyCSV(emptyCsvPath, path):
    csvData = np.genfromtxt(emptyCsvPath, delimiter=',', dtype=str)
    np.savetxt(path, csvData, delimiter=',', fmt='%s')

if __name__ == "__main__":
    tasksHyperParams = {
        'emory_emotion_recognition': (1e-5, 30), 
        'reading_comprehension': (1e-5, 10), 
        'character_identification': (1e-5, 10),
        'question_answering': (1e-4, 120), 
        'personality_detection': (1e-5, 60),
        'relation_extraction': (1e-5, 10),
        'MELD_emotion_recognition': (1e-5, 30)
    }
    sourceTasks = [
        'emory_emotion_recognition', 
        'reading_comprehension', 
        'character_identification',
        'question_answering', 
        'personality_detection',
        'relation_extraction',
        'MELD_emotion_recognition'
    ]
    targetTasks = [
        'personality_detection',
        'relation_extraction',
        'emory_emotion_recognition', 
        'reading_comprehension', 
        'question_answering', 
        'MELD_emotion_recognition',
        'character_identification',
    ]
    splitPercents = [
        1
    ]
    startSplitPercentIndex = 0
    startSourceIndex = 0
    startTargetIndex = 2
    seed = 12345
    splitSeed = 31415
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    model = "bert"
    dataset = "Friends"
    maxEpochs = 40
    maxNotImprovingGap = 4
    emptyCsvPath = f"empty{dataset}.csv"
    for splitPercentIndex in range(startSplitPercentIndex, len(splitPercents), 1):
        splitPercent = splitPercents[splitPercentIndex]
        splitPercentDataPath = GetSavedCsvDataDirectory(dataset, model)
        if not os.path.exists(splitPercentDataPath):
            os.makedirs(splitPercentDataPath)
        splitPercentDataPath += f"data_seed_{splitSeed}_splitPercent{splitPercent}.csv"
        if not os.path.exists(splitPercentDataPath):
            CreateNewEmptyCSV(emptyCsvPath, splitPercentDataPath)
        
        currentSourceTaskStartIndex = startSourceIndex if splitPercentIndex == startSplitPercentIndex else 0
        for stIndex in range(currentSourceTaskStartIndex, len(sourceTasks), 1):
            st = sourceTasks[stIndex]
            sourceSavePath = GetSavedSourceModelDirectory(dataset, model, st)
            if not os.path.exists(sourceSavePath):
                os.makedirs(sourceSavePath)
            
            currentTargetTaskStartIndex = startTargetIndex if stIndex == startSourceIndex else 0
            for ttIndex in range(currentTargetTaskStartIndex, len(targetTasks), 1):
                tt = targetTasks[ttIndex]
                if tt == st:
                    continue
                lr, bs = tasksHyperParams[tt]
                targetSavePath = sourceSavePath+ f"FINETUNED_{dataset}.{tt}/seed.{seed}/LR.{lr}_EBS.{bs}/"
                modelState = loadState(targetSavePath + "best_model.pt", None)
                config = Config(model, [st], [dataset], [tt], [dataset], 1, do_train=False, do_finetune=True, eval_best=True, gpu_batch_size=10,
                learning_rate=lr, effective_batch_size=bs, saved_model_dir=sourceSavePath)
                config.seed= seed
                evalMetrics = EvalModel(config, None, None, modelState)
                SaveElementIntoCSV(splitPercentDataPath, st, tt, evalMetrics[0])