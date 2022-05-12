from run_experiment import Config, main
from utils import Logger, append_to_save_path_dir

# if __name__ == '__main__':
#     config = Config("bert", ["reading_comprehension"], ["Friends"], ["reading_comprehension"], 
#         ["Friends"], 1, do_train=True, eval_best=True, gpu_batch_size=5)
#     config.frac = 0.1
#     main(config, Logger())

# if __name__ == '__main__':
#     config = Config("bert", ["masked_language_modeling"], ["Friends"], ["masked_language_modeling"], 
#         ["Friends"], 1, do_train=True, eval_best=True, gpu_batch_size=5)
#     config.frac = 0.1
#     main(config, Logger())

# if __name__ == '__main__':
#     config = Config("bert", ["masked_language_modeling"], ["Friends"], ["masked_language_modeling"], 
#         ["Friends"], 4, do_train=True, eval_best=True, gpu_batch_size=5, resume=True)
#     print("best val metric: " + main(config, Logger()))

import os
from run_experiment import Config, main
import torch

from utils import Logger, get_savepath_dir
if __name__ == "__main__":
    task_config = {
        'emory_emotion_recognition':
        {
            'learning_rate': 1e-5,
            'effective_batch_size': 30,
            'num_epochs': 3
        },
        'reading_comprehension':
        {
            'learning_rate': 1e-5,
            'effective_batch_size': 10,
            'num_epochs': 3
        },
        'character_identification':
        {
            'learning_rate': 1e-5,
            'effective_batch_size': 10,
            'num_epochs': 9
        },
        'question_answering':
        {
            'learning_rate': 1e-4,
            'effective_batch_size': 120,
            'num_epochs': 2
        }, 
        'personality_detection':
        {
            'learning_rate': 1e-5,
            'effective_batch_size': 60,
            'num_epochs': 9
        },
        'relation_extraction':
        {
            'learning_rate': 1e-5,
            'effective_batch_size': 10,
            'num_epochs': 18
        },
        'MELD_emotion_recognition':
        {
            'learning_rate': 1e-5,
            'effective_batch_size': 30,
            'num_epochs': 3
        }
    }

    logger = Logger("transfer_log.txt")


    seed = 12345
    results = {}
    s  = "masked_language_modeling"
    logger.write("Starting source task " + s + "\n")
    logger.flush()
    sourceConfig = Config("bert", [s], ["Friends"], [s], ["Friends"], num_epochs=5, do_finetune=True, eval_best=True, gpu_batch_size=10)
    print(sourceConfig.learning_rate)

    source_model_dir = get_savepath_dir(sourceConfig.source_datasets, sourceConfig.source_tasks, sourceConfig.seed, sourceConfig.log_and_model_dir, 
        sourceConfig.model, sourceConfig.few_shot_percent, sourceConfig.learning_rate, sourceConfig.effective_batch_size, sourceConfig.multitask)

    for t in task_config.keys():
        logger.write("Starting target task " + t + "\n")
        logger.flush()
        config = Config("bert", [s], ["Friends"], [t], ["Friends"], num_epochs=task_config[t]['num_epochs'], do_finetune=True, eval_best=True, gpu_batch_size=10,
                learning_rate=task_config[t]['learning_rate'], effective_batch_size=task_config[t]['effective_batch_size'],
                saved_model_dir=source_model_dir, resume=True)
        save_path_dir = append_to_save_path_dir(source_model_dir, config.target_datasets, config.target_tasks, config.few_shot_percent, config.seed, 
                                                        config.learning_rate, config.effective_batch_size)
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)
        taskLogger = Logger(os.path.join(save_path_dir, 'logMinimal.txt'), mode='w')
        metric = main(config, taskLogger)
        results[(s,t)] = metric
        logger.write("Finished training target task " + t + ".  Results:\n" + results[(s,t)])
        logger.flush()
    logger.write("All done, complete results:\n" + results)
                