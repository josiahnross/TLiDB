from asyncio import tasks
from run_experimentV2 import EvalModel, TrainModel, TrainSourceModel
from utils import Logger, get_savepath_dir_minimal, loadState, append_to_save_path_dir
import torch
import os
import configs
import numpy as np
from Config import Config
from newUtils import *
import traceback
from datetime import datetime
import sys

def TrainNoTransfer(model, dataset, task, splitPercentDataPath, sourceSavePath, seed, splitSeed, splitPercent, maxNotImprovingGap, 
                    maxEpochs, taskMinimallLogger, learningRate, effectiveBatchSize, loadLastSavedModel, gpu_batch_size):
    
    taskMinimallLogger.write(f"\n\nNo Transfer Task: {task}  Split Seed: {splitSeed}  SplitPercent: {splitPercent}  LR:{lr}   EBS: {bs}\n")
    taskMinimallLogger.flush()
    
    config = Config(model, [task], [dataset], [task], [dataset], 1, eval_best=True, gpu_batch_size=gpu_batch_size, learning_rate=learningRate, effective_batch_size=effectiveBatchSize)
    # config.saved_model_dir = sourceSavePath
    config.seed= seed
    config.save_last = True
    sourceSplitSeed= -1 if splitPercent >= 1 else splitSeed
    sourceSplitPercent= -1 if splitPercent >= 1 else splitPercent
    modelSavePath = sourceSavePath
    if sourceSplitSeed >= 0 and sourceSplitPercent >= 0: 
        modelSavePath += f"splitSeed.{splitSeed}/splitPercent_{splitPercent}/"
    
    if not os.path.exists(modelSavePath):
            os.makedirs(modelSavePath)
    startEpoch = 0         
    modelState = None
    last_best_metric = None
    last_best_metric_index = 0
    if loadLastSavedModel:
        modelState, last_best_metric, last_best_metric_index = LoadModelIfExitstsWithLastUpdateEpoch(modelSavePath, taskMinimallLogger, last_best_metric, last_best_metric_index)
        # modelState = LoadModelStateIfExists(modelSavePath + "last_model.pt", taskMinimallLogger)
        # modelState = loadState(modelSavePath + "last_model.pt", taskMinimallLogger)
        if modelState is not None:
            startEpoch = modelState['epoch'] + 1
            # last_best_metric = modelState['best_val_metric']
            # bestModelState = loadState(modelSavePath + "best_model.pt", taskMinimallLogger)
            # last_best_metric_index = bestModelState['epoch']
    modelAlgorithm = None
    if startEpoch - last_best_metric_index >= maxNotImprovingGap:
        taskMinimallLogger.write(f"No Transfer Task: {st} Already Finished\n")
    else:
        for i in range(startEpoch, maxEpochs, 1):
            config.num_epochs = i + 1
            best_val_metric, modelState, modelAlgorithm = TrainSourceModel(config, taskMinimallLogger,modelAlgorithm, modelState, save_path_dir=modelSavePath,
                                        targetSplitSeed=splitSeed, targetSplitPercent=splitPercent)
            if last_best_metric == None or best_val_metric > last_best_metric:
                last_best_metric = best_val_metric
                last_best_metric_index = i
            elif i-last_best_metric_index >= maxNotImprovingGap:
                taskMinimallLogger.write(f"No Transfer Task: {st} Finished in {i} Epochs, Stopped Due To Lack of Improvement\n")
                break
        taskMinimallLogger.flush()
        modelState = loadState(modelSavePath + "best_model.pt", taskMinimallLogger)
        evalMetrics = EvalModel(config, taskMinimallLogger, None, modelState)
        SaveElementIntoDataCSV(splitPercentDataPath, st, st, evalMetrics[0])
    taskMinimallLogger.flush()

if __name__ == "__main__":
    tasksHyperParams = {
        ('bert', 'emory_emotion_recognition'): (1e-5, 30), 
        ('bert', 'reading_comprehension'): (1e-5, 10), 
        ('bert', 'character_identification'): (1e-5, 10),
        ('bert', 'question_answering'): (1e-4, 120), 
        ('bert', 'personality_detection'): (1e-5, 60),
        ('bert', 'relation_extraction'): (1e-5, 10),
        ('bert', 'MELD_emotion_recognition'): (1e-5, 30),
        
        ('t5', 'emory_emotion_recognition'): (1e-4, 120), 
        ('t5', 'reading_comprehension'): (1e-4, 60), 
        ('t5', 'character_identification'): (0, 0),
        ('t5', 'question_answering'): (1e-5, 10), 
        ('t5', 'personality_detection'): (1e-4, 120),
        ('t5', 'relation_extraction'): (1e-4, 10),
        ('t5', 'MELD_emotion_recognition'): (1e-4, 10)
    }
    sourceTasks = [
        #'emory_emotion_recognition', 
        #'reading_comprehension', 
        # 'character_identification',
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
        #'character_identification',
    ]
    splitPercents = [
        0.2,
        0.4,
        0.6,
        0.8
    ]
    startSplitPercentIndex = 0
    startSourceIndex = 1
    startTargetIndex = 3
    loadLastSavedModel = True
    trainNoTransfer = True
    gpu_batch_size = 5
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
    hasException = False
    loggerCount = 0
    basePath = GetSavedCsvDataDirectory(dataset, model)
    # CreatePathIfNotExist(basePath + "logs/")
    overallLogDir = basePath + f"Logs/"
    
    overallLogger = GetOverallLogger(overallLogDir, sys.argv[1] if sys.argv[1] else None)
    # while os.path.exists(overallLogDir):
    #     loggerCount+=1
    #     overallLogDir = basePath + f"logs/transferOverallLogger{loggerCount}.txt"
    # overallLogger = Logger(overallLogDir, mode='w')


    timeData = datetime.now().strftime("%m/%d %H:%M:%S")
    overallLogger.write(f"Model: {model}  Dataset: {dataset}  Time: {timeData}\n")
    overallLogger.write(f"SplitPercents (start:{startSplitPercentIndex}): {splitPercents}\nSourceTasks (start:{startSourceIndex}): {sourceTasks}\nTargetTasks (start:{startTargetIndex}):{targetTasks}\n")
    overallLogger.write(f"maxNotImprovingGap: {maxNotImprovingGap}  maxEpochs: {maxEpochs}  loadLastSavedModel:{loadLastSavedModel}  trainNoTransfer:{trainNoTransfer}  seed:{seed}  splitSeed:{splitSeed}\n")
    for splitPercentIndex in range(startSplitPercentIndex, len(splitPercents), 1):
        splitPercent = splitPercents[splitPercentIndex]

        splitPercentDataPath = GetOrMakeEvalDataCSV(dataset, model, splitSeed, splitPercent)      

        currentSourceTaskStartIndex = startSourceIndex if splitPercentIndex == startSplitPercentIndex else 0
        for stIndex in range(currentSourceTaskStartIndex, len(sourceTasks), 1):
            st = sourceTasks[stIndex]
            sourceSavePath = GetSavedSourceModelDirectory(dataset, model, st)
            if not os.path.exists(sourceSavePath):
                os.makedirs(sourceSavePath)
            sourceLoggerPath = sourceSavePath
            if splitPercent != 1:
                sourceLoggerPath += f'splitSeed.{splitSeed}/'
                if not os.path.exists(sourceLoggerPath):
                    os.makedirs(sourceLoggerPath)
                sourceLoggerPath += f'splitPercent_{splitPercent}_'
            
            sourceLoggerPath += 'logMinimal.txt'
            if not os.path.exists(sourceLoggerPath):
                taskMinimallLogger = Logger(sourceLoggerPath, mode='w')
                taskMinimallLogger.subLogger = overallLogger
                taskMinimallLogger.write(f"\nStarting Source Task: {st} Split Percent: {splitPercent} Split Seed: {splitSeed}")
            else:
                taskMinimallLogger = Logger(sourceLoggerPath, mode='a')
                taskMinimallLogger.subLogger = overallLogger
                taskMinimallLogger.write(f"\n\nRestarting Source Task: {st} Split Percent: {splitPercent} Split Seed: {splitSeed}")
            taskMinimallLogger.flush()
            if trainNoTransfer: 
                lr, bs = tasksHyperParams[(model, st)]
                torch.cuda.empty_cache()
                try:
                    TrainNoTransfer(model, dataset, st, splitPercentDataPath, sourceSavePath, seed,splitSeed, splitPercent, maxNotImprovingGap, 
                                maxEpochs, taskMinimallLogger, lr, bs, loadLastSavedModel and splitPercentIndex == startSplitPercentIndex and stIndex == startSourceIndex, gpu_batch_size)
                except Exception as e:
                        hasException = True
                        taskMinimallLogger.write(f"\n\n ERROR: {e}\n")
                        taskMinimallLogger.write(traceback.format_exc())
                        taskMinimallLogger.flush()
                        break
            currentTargetTaskStartIndex = startTargetIndex if stIndex == startSourceIndex and  splitPercentIndex == startSplitPercentIndex else 0
            for ttIndex in range(currentTargetTaskStartIndex, len(targetTasks), 1):
                torch.cuda.empty_cache()
                tt = targetTasks[ttIndex]
                if tt == st:
                    continue
                lr, bs = tasksHyperParams[(model, tt)]
                taskMinimallLogger.write(f"\n\nTarget Task: {tt}   LR:{lr}   EBS: {bs}\n")
                taskMinimallLogger.flush()
                config = Config(model, [st], [dataset], [tt], [dataset], 1, do_train=False, do_finetune=True, eval_best=True, gpu_batch_size=gpu_batch_size,
                learning_rate=lr, effective_batch_size=bs, saved_model_dir=sourceSavePath)
                config.saved_model_dir = sourceSavePath
                config.seed= seed
                config.save_last = True
                targetSplitSeed= -1 if splitPercent >= 1 else splitSeed
                targetSplitPercent= -1 if splitPercent >= 1 else splitPercent
                modelSavePath = sourceSavePath
                startEpoch = 0
                last_best_metric = None
                last_best_metric_index = 0
                if loadLastSavedModel and ttIndex == startTargetIndex and stIndex == startSourceIndex and splitPercentIndex == startSplitPercentIndex:
                    appenedSavePath = append_to_save_path_dir(modelSavePath, config.target_datasets, config.target_tasks, config.few_shot_percent, config.seed, 
                                                        config.learning_rate, config.effective_batch_size, targetSplitSeed, targetSplitPercent) + "/"
                    modelState, last_best_metric, last_best_metric_index = LoadModelIfExitstsWithLastUpdateEpoch(appenedSavePath, taskMinimallLogger, last_best_metric, last_best_metric_index)
                    # modelState = loadState(appenedSavePath + "/last_model.pt", taskMinimallLogger)
                    if modelState is not None:
                        startEpoch = modelState['epoch'] + 1
                    else:
                        modelState = loadState(modelSavePath + "best_model.pt", taskMinimallLogger)
                else:                                   
                    modelState = loadState(modelSavePath + "best_model.pt", taskMinimallLogger)
                modelAlgorithm = None
                if startEpoch - last_best_metric_index >= maxNotImprovingGap:
                    taskMinimallLogger.write(f"No Source Task: {st} Target Task: {tt} Already Finished\n")
                else:
                    for i in range(startEpoch, maxEpochs, 1):
                        config.num_epochs = i + 1
                        try:
                            best_val_metric, modelState, modelAlgorithm = TrainModel(config, taskMinimallLogger,modelAlgorithm, modelState, i!=0, 
                                                        targetSplitSeed=targetSplitSeed, targetSplitPercent=targetSplitPercent)
                        except Exception as e:
                            hasException = True
                            taskMinimallLogger.write(f"\n\n ERROR: {e}\n")
                            taskMinimallLogger.write(traceback.format_exc())
                            taskMinimallLogger.flush()
                            break
                        if last_best_metric == None or best_val_metric > last_best_metric:
                            last_best_metric = best_val_metric
                            last_best_metric_index = i
                        elif i-last_best_metric_index >= maxNotImprovingGap:
                            taskMinimallLogger.write(f"Target Task: {tt} Finished in {i} Epochs, Stopped Due To Lack of Improvement\n")
                            break
                    if hasException:
                        break
                    appenedSavePath = append_to_save_path_dir(modelSavePath, config.target_datasets, config.target_tasks, config.few_shot_percent, config.seed, 
                                                            config.learning_rate, config.effective_batch_size, targetSplitSeed, targetSplitPercent)
                    modelState = loadState(appenedSavePath + "/best_model.pt", taskMinimallLogger)
                    evalMetrics = EvalModel(config, taskMinimallLogger, None, modelState)
                    SaveElementIntoDataCSV(splitPercentDataPath, st, tt, evalMetrics[0])
            if hasException:
                break
            taskMinimallLogger.write(f"Done with all listed target taks for source task: {st}")
        if hasException:
            break