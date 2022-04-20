from run_experiment import Config, main
from utils import Logger, get_savepath_dir_minimal
import torch
import os
if __name__ == "__main__":
    tasks = [
        # 'emory_emotion_recognition', 
        # 'reading_comprehension', 
        #'character_identification',
        #'question_answering', 
        # 'personality_detection'
        #'relation_extraction',
         'MELD_emotion_recognition'
    ]
    seed = 12345
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    learningRates = [1e-4, 1e-5, 1e-6]
    effectiveBatchSizes = [10, 30, 60, 120]
    percentToRemove = 0.45
    epochsPerRemove = 2
    for t in tasks:
        tempConfig = Config("bert", [t], ["Friends"], [t], ["Friends"], 0, do_train=True, eval_best=True, gpu_batch_size=10)
        savePathDir = get_savepath_dir_minimal(tempConfig.source_datasets, tempConfig.source_tasks, seed, tempConfig.log_and_model_dir, 
                                                tempConfig.model, tempConfig.multitask)   
        if not os.path.exists(savePathDir):
            os.makedirs(savePathDir)
        taskMinimallLogger = Logger(os.path.join(savePathDir, 'logMinimal.txt'), mode='w')
        results = {}
        for bs in effectiveBatchSizes:
            for lr in learningRates:
                results[(lr, bs)] = 0
        for i in range(9):
            for lr, bs in results.keys():
                # taskMinimallLogger.write(f"Starting Epoch: {epochsPerRemove*i} LR: {lr} EBS: {bs}\n")
                # taskMinimallLogger.flush()
                config = Config("bert", [t], ["Friends"], [t], ["Friends"], epochsPerRemove*(i+1), do_train=True, eval_best=True, gpu_batch_size=10,
                learning_rate=lr, effective_batch_size=bs)
                config.seed= seed
                config.save_last = True
                config.resume = True # i != 0 
                best_val_metric = main(config, taskMinimallLogger)
                results[(lr, bs)] = max(best_val_metric, results[(lr, bs)])
            paramsToRemove = int(len(results) * percentToRemove)
            sortedParams = sorted(results, key=results.get)
            worstParams = sortedParams[0:paramsToRemove]
            for p in worstParams:
                del results[p]
            print(f"Remaining Hyperparameters: {results}")
            taskMinimallLogger.write(f"\n\n Remaining Hyperparameters: {results}\n\n")
            taskMinimallLogger.flush()

    print(results)