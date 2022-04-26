from asyncio import tasks
from run_experimentV2 import TrainModel
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
        #'emory_emotion_recognition', 
        #'reading_comprehension', 
        #'character_identification',
        #'question_answering', 
        #'personality_detection',
        # 'relation_extraction',
        'MELD_emotion_recognition'
    ]
    targetTasks = [
        # 'personality_detection',
        # 'relation_extraction',
        # 'emory_emotion_recognition', 
        'reading_comprehension', 
        'question_answering', 
        'MELD_emotion_recognition',
        'character_identification',
    ]
    splitPercents = [
        0.2,
        0.4,
        0.6,
        0.8
    ]
    startPercentIndex = 0
    startSourceIndex = 0
    startTargetIndex = 0
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
    for splitPercentIndex in range(startPercentIndex, len(splitPercents), 1):
        splitPercentDataPath = GetSavedCsvDataDirectory(dataset, model)
        if not os.path.exists(splitPercentDataPath):
            os.makedirs(splitPercentDataPath)
        splitPercentDataPath += f"data_seed_{splitSeed}_splitPercent{splitPercent}.csv"
        if not os.path.exists(splitPercentDataPath):
            CreateNewEmptyCSV(emptyCsvPath, splitPercentDataPath)
        
        splitPercent = splitPercents[startPercentIndex]
        currentSourceTaskStartIndex = startSourceIndex if splitPercentIndex == startPercentIndex else 0
        for stIndex in range(currentSourceTaskStartIndex, len(sourceTasks), 1):
            st = sourceTasks[stIndex]
            sourceSavePath = GetSavedSourceModelDirectory(dataset, model, st)
            if not os.path.exists(sourceSavePath):
                os.makedirs(sourceSavePath)

            sourceLoggerPath = os.path.join(sourceSavePath, 'logMinimal.txt')
            if not os.path.exists(sourceLoggerPath):
                taskMinimallLogger = Logger(sourceLoggerPath, mode='w')
                taskMinimallLogger.write(f"Starting Source Task: {st}")
            else:
                taskMinimallLogger = Logger(sourceLoggerPath, mode='a')
                taskMinimallLogger.write(f"\n\nRestarting Source Task: {st}")
            currentTargetTaskStartIndex = startTargetIndex if stIndex == startSourceIndex else 0
            for ttIndex in range(currentTargetTaskStartIndex, len(targetTasks), 0):
                tt = targetTasks[ttIndex]
                if tt == st:
                    continue
                lr, bs = tasksHyperParams[tt]
                taskMinimallLogger.write(f"\n\nTarget Task: {tt}   LR:{lr}   EBS: {bs}\n")
                taskMinimallLogger.flush()
                config = Config(model, [st], [dataset], [tt], [dataset], 1, do_train=False, do_finetune=True, eval_best=True, gpu_batch_size=10,
                learning_rate=lr, effective_batch_size=bs, saved_model_dir=sourceSavePath)
                config.saved_model_dir = sourceSavePath
                config.seed= seed
                config.save_last = True
                modelState = loadState(sourceSavePath + "best_model.pt", taskMinimallLogger)
                modelAlgorithm = None
                last_best_metric = None
                last_best_metric_index = 0
                for i in range(maxEpochs):
                    config.num_epochs = i + 1
                    best_val_metric, modelState, modelAlgorithm = TrainModel(config, taskMinimallLogger,modelAlgorithm, modelState,i!=0, 
                                                targetSplitSeed= -1 if splitPercent >= 1 else splitSeed, targetSplitPercent= -1 if splitPercent >= 1 else splitSeed)
                    if last_best_metric == None or best_val_metric > last_best_metric:
                        last_best_metric = best_val_metric
                        last_best_metric_index = i
                    elif i-last_best_metric_index >= maxNotImprovingGap:
                        taskMinimallLogger.write(f"Target Task: {tt} Finished in {i} Epochs, Stopped Due To Lack of Improvement\n")
                        break
                SaveElementIntoCSV(splitPercentDataPath, st, tt, last_best_metric)
            taskMinimallLogger.write(f"Done with all listed target taks for source task: {st}")