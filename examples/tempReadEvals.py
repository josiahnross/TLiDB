from asyncio import tasks
from run_experimentV2 import EvalModel, TrainModel
from utils import Logger, get_savepath_dir_minimal, loadState
import torch
import os
import configs
import numpy as np
from Config import Config
from newUtils import *
if __name__ == "__main__":
    tasksHyperParams = {
        ('bert', 'emory_emotion_recognition'): (1e-5, 30), 
        ('bert', 'reading_comprehension'): (1e-5, 10), 
        ('bert', 'character_identification'): (1e-5, 10),
        ('bert', 'question_answering'): (1e-4, 120), 
        ('bert', 'personality_detection'): (1e-5, 60),
        ('bert', 'relation_extraction'): (1e-5, 10),
        ('bert', 'MELD_emotion_recognition'): (1e-5, 30),
        ('bert', 'masked_language_modeling'): (1e-5, 10),
        
        ('t5', 'emory_emotion_recognition'): (1e-4, 120), 
        ('t5', 'reading_comprehension'): (1e-4, 60), 
        ('t5', 'character_identification'): (1e-05, 10),
        ('t5', 'question_answering'): (1e-5, 10), 
        ('t5', 'personality_detection'): (1e-4, 120),
        ('t5', 'relation_extraction'): (1e-4, 10),
        ('t5', 'MELD_emotion_recognition'): (1e-4, 10)
    }
    sourceTasks = [
        'emory_emotion_recognition', 
        'reading_comprehension', 
        'character_identification',
        #'question_answering', 
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
        #0.2,
        #0.4,
        0.6,
        #0.8,
        #1
    ]
    startSplitPercentIndex = 0
    startSourceIndex = 0
    startTargetIndex = 0
    seed = 12345
    splitSeed = 31415
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    model = "t5"
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
                continue
                # os.makedirs(sourceSavePath)
            
            currentTargetTaskStartIndex = startTargetIndex if stIndex == startSourceIndex else 0
            for ttIndex in range(currentTargetTaskStartIndex, len(targetTasks), 1):
                tt = targetTasks[ttIndex]
                # if tt == st:
                #     continue
                lr, bs = tasksHyperParams[(model, tt)]
                if splitPercent == 1:
                    if tt == st:
                        targetSavePath = sourceSavePath
                    else:
                        targetSavePath = sourceSavePath + f"FINETUNED_{dataset}.{tt}/seed.{seed}/LR.{lr}_EBS.{bs}/"
                else:
                    if tt == st:
                        targetSavePath = sourceSavePath + f"splitSeed.{splitSeed}/splitPercent_{splitPercent}/"
                    else:
                        continue
                        targetSavePath = sourceSavePath + f"FINETUNED_{dataset}.{tt}/seed.{seed}/splitSeed.{splitSeed}/SplitPercent.{splitPercent}_LR.{lr}_EBS.{bs}/"
                targetSavePath += "best_model.pt"
                # print(f"Got here 1   {targetSavePath}")
                if not os.path.exists(targetSavePath):
                    continue
                # print("Got here 2")
                modelState = loadState(targetSavePath, None)
                config = Config(model, [st], [dataset], [tt], [dataset], 1, do_train=False, do_finetune=True, eval_best=True, gpu_batch_size=5,
                learning_rate=lr, effective_batch_size=bs, saved_model_dir=sourceSavePath)
                config.seed= seed
                evalMetrics = EvalModel(config, None, None, modelState)
                SaveElementIntoDataCSV(splitPercentDataPath, st, tt, evalMetrics[0])
                del modelState