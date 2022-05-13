from run_experimentV2 import TrainSourceModel, EvalModel
from utils import Logger, get_savepath_dir_minimal, loadState, save_state, load_algorithmFromState
import torch
import os
import configs
import numpy as np
from Config import Config
from newUtils import *
import sys
import traceback

if __name__ == "__main__":
    tasks = [
        #'emory_emotion_recognition', 
        #'reading_comprehension', 
        #'character_identification',
        #'question_answering', 
        #'personality_detection',
        #'relation_extraction',
        #'MELD_emotion_recognition',
        'masked_language_modeling'
    ]
    seed = 12345
    splitSeed = 31415
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    model = "bert"
    dataset = "Friends"
    maxEpochs = 40
    maxNotImprovingGapGroup = 2
    resumeFromLastModel = False

    learningRates = [1e-4, 1e-5]
    effectiveBatchSizes = [10, 30, 60, 120]
    percentToRemove = 0.45
    epochsPerRemove = 2
    startEpochGroup = 0

    modelDataSetDir = GetSavedHyperparameterCSVDirectory(dataset, model, seed)
    modelDataSetCSVPath = modelDataSetDir + f"HyperparameterInfos.csv"
    modelDataSetLogsDir = modelDataSetDir + "logs/"
    if not os.path.exists(modelDataSetLogsDir):
        os.makedirs(modelDataSetLogsDir)
    
    hasException = False
    overallLogger = GetOverallLogger(modelDataSetLogsDir, sys.argv[1] if sys.argv[1] else None)
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
            taskMinimallLogger.subLogger = overallLogger
            taskMinimallLogger.write(f"Starting Hyperparameter Search  Task: {t}  Seed: {seed}\n\n")
        else:
            taskMinimallLogger = Logger(taskMinimallLoggerPath, mode='a')
            taskMinimallLogger.subLogger = overallLogger
            taskMinimallLogger.write(f"\n\nRestarting Hyperparameter Search  Task: {t}  Seed: {seed}\n\n")
        taskMinimallLogger.flush()
        results = {}
        for bs in effectiveBatchSizes:
            for lr in learningRates:
                results[(lr, bs)] = (0, 0, None, None)
        bestFinishedInfo = None
        config = Config(model, [t], [dataset], [t], [dataset], 1, eval_best=True, 
                gpu_batch_size=5, learning_rate=0, effective_batch_size=0)
        config.seed= seed
        config.save_last = True
        # config.resume = resumeFromLastModel
        for i in range(startEpochGroup, int(maxEpochs/epochsPerRemove), 1):
            for lr, bs in results.keys():
                torch.cuda.empty_cache()
                taskMinimallLogger.write("\n")
                config.learning_rate = lr
                config.effective_batch_size = bs
                config.num_epochs = (i+1) * epochsPerRemove
                (prevBest, prevImproveEpochGroup, modelState, modelAlgorithm) = results[(lr, bs)]
                savePathWithLR_EBS = GetTempModelSaveLR_EBS(dataset, model, t, seed, lr, bs)
                if not os.path.exists(savePathWithLR_EBS):
                    os.makedirs(savePathWithLR_EBS) 
                if modelState is None and resumeFromLastModel:
                    modelState, prevBest, prevImproveEpochGroup = LoadModelIfExitstsWithLastUpdateEpoch(savePathWithLR_EBS, taskMinimallLogger, prevImproveEpochGroup)
                    if modelState is not None:
                        prevImproveEpochGroup = int(prevImproveEpochGroup/epochsPerRemove)
               
                try:
                    best_val_metric, modelState, modelAlgorithm = TrainSourceModel(config, taskMinimallLogger,modelAlgorithm, modelState, save_path_dir=savePathWithLR_EBS)
                                                                            #, targetSplitSeed=splitSeed, targetSplitPercent=0.2)
                except Exception as e:
                    hasException = True
                    taskMinimallLogger.write(f"\n\n ERROR: {e}\n")
                    taskMinimallLogger.write(traceback.format_exc())
                    taskMinimallLogger.flush()
                    break
                if best_val_metric > prevBest:
                    prevBest = best_val_metric
                    prevImproveEpochGroup = i
                results[(lr, bs)] = (prevBest, prevImproveEpochGroup, modelState, modelAlgorithm)
            if hasException:
                break
            paramsToRemove = int(len(results) * percentToRemove)
            sortedParams = sorted(results, key=lambda k: results[k][0])
            worstParams = sortedParams[0:paramsToRemove]
            for p in worstParams:
                del results[p]
            worstParams = []
            remainingHyperparameters = {}
            for k in results.keys():
                val, lastImproveEpochGroup, _, _ = results[k]
                if i - lastImproveEpochGroup >= maxNotImprovingGapGroup:
                    if bestFinishedInfo is None or bestFinishedInfo[2] < val:
                        bestFinishedInfo = (k[0], k[1], val, (lastImproveEpochGroup+1)*epochsPerRemove)
                    worstParams.append(k)
                else:
                    remainingHyperparameters[k] = (val, lastImproveEpochGroup)
            for p in worstParams:
                del results[p]
            if len(results) > 0:
                taskMinimallLogger.write(f"\n\n Remaining Hyperparameters: {remainingHyperparameters}\n\n")
            else:
                SaveHyperparameterntoCSV(modelDataSetCSVPath, t, bestFinishedInfo[0], bestFinishedInfo[1], bestFinishedInfo[2],
                                                                             bestFinishedInfo[3])
                
                savePathWithLR_EBS = GetTempModelSaveLR_EBS(dataset, model, t, seed, bestFinishedInfo[0], bestFinishedInfo[1])
                modelState = loadState(savePathWithLR_EBS + "best_model.pt", taskMinimallLogger)

                # Save best model in good spot 
                convenientModelPath = GetSavedSourceModelDirectory(dataset, model, t)
                if not os.path.exists(convenientModelPath):
                    os.makedirs(convenientModelPath)  
                save_state(modelState, os.path.join(convenientModelPath,f"best_model.pt"),taskMinimallLogger)

                # Eval model and save result in CSV
                evalMetrics = EvalModel(config, taskMinimallLogger, None, modelState)
                
                splitPercentDataPath = GetOrMakeEvalDataCSV(dataset, model, splitSeed, 1)  
                SaveElementIntoDataCSV(splitPercentDataPath, t, t, evalMetrics[0])
                taskMinimallLogger.write(f"\n\n Finished Task: {t} \n\n")
                break
            taskMinimallLogger.flush()
    