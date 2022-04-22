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
    for s in task_config.keys():
        logger.write("Starting source task " + s + "\n")
        logger.flush()
        sourceConfig = Config("bert", [s], ["Friends"], [s], ["Friends"], num_epochs=task_config[s]['num_epochs'], do_finetune=True, eval_best=True, gpu_batch_size=10,
            learning_rate=task_config[s]['learning_rate'], effective_batch_size=task_config[s]['effective_batch_size'])

        source_model_dir = get_savepath_dir(sourceConfig.source_datasets, sourceConfig.source_tasks, sourceConfig.seed, sourceConfig.log_and_model_dir, 
            sourceConfig.model, sourceConfig.few_shot_percent, sourceConfig.learning_rate, sourceConfig.effective_batch_size, sourceConfig.multitask)

        for t in task_config.keys():
            logger.write("Starting target task " + t + "\n")
            logger.flush()
            metric_at_each_epoch = []
            for i in range(task_config[t]['num_epochs']):
                config = Config("bert", [s], ["Friends"], [t], ["Friends"], num_epochs=task_config[t]['num_epochs'], do_finetune=True, eval_best=True, gpu_batch_size=10,
                        learning_rate=task_config[t]['learning_rate'], effective_batch_size=task_config[t]['effective_batch_size'],
                        saved_model_dir=source_model_dir, resume=True)
                save_path_dir = get_savepath_dir(config.train_datasets, config.train_tasks, config.seed, config.log_and_model_dir, 
                    config.model, config.few_shot_percent, config.learning_rate, config.effective_batch_size, config.multitask)
                taskLogger = Logger(os.path.join(save_path_dir, 'logMinimal.txt'), mode='w')
                metric = main(config, taskLogger)
                metric_at_each_epoch.append(metric)
                if metric < min(metric_at_each_epoch[-5:]):
                    break
            results[(s,t)] = metric_at_each_epoch
            logger.write("Finished training target task after " + str(i + 1) + "epochs.  Results:\n" + results[(s,t)])
            logger.flush()
    logger.write("All done, complete results:\n" + results)
                