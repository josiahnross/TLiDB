from run_experimentV2 import TrainSourceModel, EvalModel
from utils import Logger, get_savepath_dir_minimal, loadState, save_state
import torch
import os
import configs
import numpy as np
from Config import Config
from newUtils import *



if __name__ == "__main__":
    tasks = [
        # 'emory_emotion_recognition', 
        # 'reading_comprehension', 
        #'character_identification',
        #'question_answering', 
        # 'personality_detection'
        'relation_extraction',
        # 'MELD_emotion_recognition'
    ]
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

    learningRates = [1e-5] #[1e-4, 1e-5, 1e-6]
    effectiveBatchSizes = [10]#[10, 30, 60, 120]
    percentToRemove = 0.45
    epochsPerRemove = 2
    startEpochGroup = 0

    modelDataSetDir = GetSavedHyperparameterCSVDirectory(dataset, model, seed)
    modelDataSetCSVPath = modelDataSetDir + f"HyperparameterInfos.csv"
    modelDataSetLogsDir = modelDataSetDir + "logs/"
    if not os.path.exists(modelDataSetLogsDir):
        os.makedirs(modelDataSetLogsDir)
    
    loggerCount = 0
    overallLogDir = modelDataSetLogsDir + f"overallLogger{loggerCount}.txt"
    while os.path.exists(overallLogDir):
        loggerCount+=1
        overallLogDir = modelDataSetLogsDir + f"overallLogger{loggerCount}.txt"
    overallLogger = Logger(overallLogDir, mode='w')
    for t in tasks:
        # tempConfig = Config("bert", [t], ["Friends"], [t], ["Friends"], 0, do_train=True, eval_best=True, gpu_batch_size=10)
        # savePathDir = get_savepath_dir_minimal(tempConfig.source_datasets, tempConfig.source_tasks, seed, tempConfig.log_and_model_dir, 
        #                                         tempConfig.model, tempConfig.multitask)   
        savePathDir = GetTempModelSavePath(dataset, model, t, seed)
        if not os.path.exists(savePathDir):
            os.makedirs(savePathDir)
        taskMinimallLoggerPath = savePathDir + 'logMinimal.txt'
        if not os.path.exists(taskMinimallLoggerPath):
            taskMinimallLogger = Logger(taskMinimallLoggerPath, mode='w')
            taskMinimallLogger.write(f"Starting Hyperparameter Search  Task: {t}  Seed: {seed}")
        else:
            taskMinimallLogger = Logger(taskMinimallLoggerPath, mode='a')
            taskMinimallLogger.write(f"\n\nRestarting Hyperparameter Search  Task: {t}  Seed: {seed}")
        results = {}
        for bs in effectiveBatchSizes:
            for lr in learningRates:
                results[(lr, bs)] = (0, 0, None, None)
        bestFinishedInfo = None
        config = Config(model, [t], [dataset], None, None, 1, eval_best=True, 
                gpu_batch_size=10, learning_rate=0, effective_batch_size=0)
        config.seed= seed
        config.save_last = True
        for i in range(startEpochGroup, int(maxEpochs/epochsPerRemove), 1):
            for lr, bs in results.keys():
                # taskMinimallLogger.write(f"Starting Epoch: {epochsPerRemove*i} LR: {lr} EBS: {bs}\n")
                # taskMinimallLogger.flush()
                config.learning_rate = lr
                config.effective_batch_size = bs
                config.num_epochs = (i+1) + epochsPerRemove
                (prevBest, prevImproveEpoch, modelState, modelAlgorithm) = results[(lr, bs)]
                savePathWithLR_EBS = GetTempModelSaveLR_EBS(dataset, model, t, seed, lr, bs)
                if not os.path.exists(savePathWithLR_EBS):
                    os.makedirs(savePathWithLR_EBS) 
                best_val_metric, modelState, modelAlgorithm = TrainSourceModel(config, taskMinimallLogger,modelAlgorithm, modelState, save_path_dir=savePathWithLR_EBS)
                if best_val_metric > prevBest:
                    prevBest = best_val_metric
                    prevImproveEpoch = i * epochsPerRemove
                results[(lr, bs)] = (prevBest, prevImproveEpoch, modelState, modelAlgorithm)
            paramsToRemove = int(len(results) * percentToRemove)
            sortedParams = sorted(results, key=lambda k: results[k][0])
            worstParams = sortedParams[0:paramsToRemove]
            for p in worstParams:
                del results[p]
            worstParams = []
            for k in results.keys():
                (val, lastImproveEpoch) = results[k]
                if i-lastImproveEpoch >= maxNotImprovingGap:
                    if bestFinishedInfo is None or bestFinishedInfo[2] < val:
                        bestFinishedInfo = (k[0], k[1], val, lastImproveEpoch)
                    worstParams.append(k)
            for p in worstParams:
                del results[p]
            if len(results) > 0:
                taskMinimallLogger.write(f"\n\n Remaining Hyperparameters: {results}\n\n")
            else:
                SaveHyperparameterntoCSV(modelDataSetCSVPath, t, bestFinishedInfo[0], bestFinishedInfo[1], bestFinishedInfo[2],
                                                                             bestFinishedInfo[3])
                
                savePathWithLR_EBS = GetTempModelSaveLR_EBS(dataset, model, t, seed, bestFinishedInfo[0], bestFinishedInfo[1])
                modelState = loadState(savePathWithLR_EBS + "best_model.pt", taskMinimallLogger)

                # Save best model in good spot 
                convenientModelPath = GetSavedSourceModelDirectory(dataset, model, t)
                save_state(modelState, os.path.join(convenientModelPath,f"best_model.pt"),taskMinimallLogger)

                # Eval model and save result in CSV
                evalMetrics = EvalModel(config, taskMinimallLogger, None, modelState)
                
                splitPercentDataPath = GetOrMakeEvalDataCSV(dataset, model, splitSeed, 1)  
                SaveElementIntoDataCSV(splitPercentDataPath, t, t, evalMetrics[0])
                taskMinimallLogger.write(f"\n\n Finished Task: {t} \n\n")
                break
            taskMinimallLogger.flush()

    print(results)