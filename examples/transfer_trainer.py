from asyncio import tasks
from run_experimentV2 import TrainModel
from utils import Logger, get_savepath_dir_minimal, loadState
import torch
import os
import configs
from Config import Config

def GetSavedSourceModelDirectory(dataset, model, task):
    model_config_dict = configs.__dict__[f"{model}_config"]
    modelName = model_config_dict["model"]
    return f"./logs_and_models/PRETRAINED_SourceTasks/{dataset}/{modelName}/{task}/"

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
        # 'reading_comprehension', 
        #'character_identification',
        #'question_answering', 
        # 'personality_detection',
        # 'relation_extraction',
        # 'MELD_emotion_recognition'
    ]
    targetTasks = [
        'personality_detection',
        'relation_extraction',
        'emory_emotion_recognition', 
        'reading_comprehension', 
        'question_answering', 
        'MELD_emotion_recognition'
        'character_identification',
    ]
    seed = 12345
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    model = "bert"
    dataset = "Friends"
    maxEpochs = 40
    maxNotImprovingGap = 4
    for st in sourceTasks:
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
        for tt in targetTasks:
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
                best_val_metric, modelState, modelAlgorithm = TrainModel(config, taskMinimallLogger,modelAlgorithm, modelState,i!=0)
                if last_best_metric == None or best_val_metric > last_best_metric:
                    last_best_metric = best_val_metric
                    last_best_metric_index = i
                elif i-last_best_metric_index >= maxNotImprovingGap:
                    taskMinimallLogger.write(f"Target Task: {tt} Finished in {i} Epochs, Stopped Due To Lack of Improvement\n")
                    break
        taskMinimallLogger.write(f"Done with all listed target taks for source task: {st}")