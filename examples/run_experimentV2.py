from Config import Config
from algorithms.initializer import initialize_algorithm
from train import train,evaluate
import os
from utils import GetAlgorithmState, Logger, load_datasets_split, load_algorithm, load_algorithmFromState, log_config, \
        set_seed, log_dataset_info, get_savepath_dir, append_to_save_path_dir


def EvalModel(config: Config, minimalLogger: Logger, algorithm, modelState):

    assert(config.target_datasets and config.target_tasks),"Must specify target datasets and tasks to finetune"
    datasets = {}
    
    datasets['test'] = load_datasets_split("test",config.target_tasks, config.target_datasets, config)
    # log configuration and dataset info
    # log_config(config,eval_logger)
    # log_dataset_info(datasets, eval_logger)

    if algorithm is None:
        # initialize algorithm
        algorithm = initialize_algorithm(config, datasets)  
    
    epoch, best_val_metric = load_algorithmFromState(algorithm, modelState, None)
    evalMetrics = evaluate(algorithm, datasets, config, minimalLogger, epoch)
    minimalLogger.flush()
    return evalMetrics


def TrainModel(config: Config, minimalLogger: Logger, algorithm, modelState, isModelStartedFinetune: bool,
                targetSplitSeed: int=-1, targetSplitPercent: float=-1):
    # if multitask, then train on both source+target tasks, and dev is target only
    if config.multitask:
        config.train_datasets = config.source_datasets+config.target_datasets
        config.train_tasks = config.source_tasks+config.target_tasks
        config.dev_datasets = config.target_datasets
        config.dev_tasks = config.target_tasks
    # if training only on source tasks, then train/dev are the same
    else:
        config.train_datasets = config.source_datasets
        config.train_tasks = config.source_tasks
        config.dev_datasets = config.source_datasets
        config.dev_tasks = config.source_tasks

    if config.target_datasets and config.target_tasks:
        # always finetune and evaluate on target tasks
        config.finetune_datasets = config.target_datasets
        config.finetune_tasks = config.target_tasks
        config.eval_datasets = config.target_datasets
        config.eval_tasks = config.target_tasks

    # create save path based only on train data
    config.save_path_dir = get_savepath_dir(config.train_datasets, config.train_tasks, config.seed, config.log_and_model_dir, 
    config.model, config.few_shot_percent, config.learning_rate, config.effective_batch_size, config.multitask)
    logger = None
    if config.do_train:
        # Initialize logs
        if os.path.exists(config.save_path_dir) and \
            (config.resume or ((config.do_finetune or config.do_eval) and (not config.do_train))):
            # if explicitly resuming, or running eval only then append to logger
            resume=True
            mode='a'
        else:
            resume=False
            mode='w'

        if not os.path.exists(config.save_path_dir):
            os.makedirs(config.save_path_dir)

        if config.debug:
            logger = Logger(mode='w')
        else:
            logger = Logger(os.path.join(config.save_path_dir, 'log.txt'), mode)

    epoch_offset = 0
    training_best_val_metric = None
    set_seed(config.seed)
    if config.do_train:
        datasets = {}
        
        # load datasets for training
        datasets['train'] = load_datasets_split("train",config.train_tasks, config.train_datasets, config, targetSplitSeed, targetSplitPercent)
        datasets['dev'] = load_datasets_split("dev",config.dev_tasks, config.dev_datasets, config, targetSplitSeed, targetSplitPercent)

        # log configuration and dataset info
        logger.write("TRAINING\n")
        log_config(config,logger)
        log_dataset_info(datasets, logger)

        if algorithm == None:
            # initialize algorithm
            algorithm = initialize_algorithm(config, datasets)

        # try to resume training from a saved model
        resume_success = False
        if resume:
            if os.path.exists(os.path.join(config.save_path_dir, 'last_model.pt')):
                prev_epoch, best_val_metric = load_algorithmFromState(algorithm, modelState, logger) 
                #prev_epoch, best_val_metric =  load_algorithm(algorithm, os.path.join(config.save_path_dir, 'last_model.pt'),logger)
                epoch_offset = prev_epoch + 1
                logger.write(f"Resuming training from epoch {prev_epoch} with best validation metric {best_val_metric}\n")
                resume_success = True
            else:
                logger.write("No previous model found, starting from scratch\n")

        # if not resuming, or if resuming but no previous model found, then train from scratch
        if not resume_success:
            epoch_offset=0
            best_val_metric = None
        
        training_best_val_metric = train(algorithm, datasets, config, logger, minimalLogger, epoch_offset, best_val_metric)
        epoch_offset += config.num_epochs

    if config.do_finetune:
        assert(config.target_datasets and config.target_tasks),"Must specify target datasets and tasks to finetune"
        datasets = {}
        # get the pre-trained model path
        if config.saved_model_dir:
            # if user explcitily specified a pretrained model to finetune, then use that
            config.save_path_dir = config.saved_model_dir
        elif config.do_train or (config.train_datasets and config.train_tasks):
            # Do nothing, this means we already have a save_dir_path and a model saved there
            pass
        else:
            raise ValueError("To run fine-tuning, use:\n--saved_model_dir to specify the pre-trained model OR\
                \n--train_datasets and --train_tasks to specify the pretraining datasets and tasks")
        
        # if fine tuning, set fine-tune train, and fine-tune dev to the same tasks
        config.finetune_train_tasks = config.finetune_tasks
        config.finetune_train_datasets = config.finetune_datasets
        config.finetune_dev_tasks = config.finetune_tasks
        config.finetune_dev_datasets = config.finetune_datasets

        # load datasets for finetuning
        datasets['train'] = load_datasets_split("train",config.finetune_train_tasks,config.finetune_train_datasets,config, splitSeed=targetSplitSeed, splitPercent=targetSplitPercent)
        datasets['dev'] = load_datasets_split("dev", config.finetune_dev_tasks, config.finetune_dev_datasets, config, splitSeed=targetSplitSeed, splitPercent=targetSplitPercent)
        if algorithm == None:
            # initialize algorithm
            algorithm = initialize_algorithm(config, datasets)
        
        # update save path with fine-tuning details
        config.save_path_dir = append_to_save_path_dir(config.save_path_dir, config.finetune_datasets, config.finetune_tasks, config.few_shot_percent, config.seed, 
                                                        config.learning_rate, config.effective_batch_size, targetSplitSeed, targetSplitPercent)

        # always load best pretrained model
        # model_path = os.path.join(config.save_path_dir, 'best_model.pt')
        # is_best = True
        # load_algorithm(algorithm, model_path, logger)
        
        # create new logger for fine-tuning
        if not os.path.exists(config.save_path_dir):
            os.makedirs(config.save_path_dir)

        if config.debug:
            finetune_logger = Logger(mode='w')
        else:
            loggerPath = os.path.join(config.save_path_dir, 'log.txt')
            if os.path.exists(loggerPath):
                finetune_logger = Logger(loggerPath, mode="a")
                finetune_logger.write("\nCONTINUEING FINETUNING\n")
            else:
                finetune_logger = Logger(loggerPath, mode="w")
                finetune_logger.write("FINETUNING\n")

        
        saved_prev_epoch, saved_best_val_metric = load_algorithmFromState(algorithm, modelState, finetune_logger) 
        if isModelStartedFinetune:
            epoch_offset = saved_prev_epoch
            best_val_metric = saved_best_val_metric
        else:
            epoch_offset = 0
            best_val_metric = None

        # note the fine-tuning in the pretrained model log
        finetune_logger.write(f"FINETUNING at {config.save_path_dir}\n")

        # log configuration and dataset info
        finetune_logger.write(f"Loaded pretrained model from state\n")
        log_config(config,finetune_logger)
        log_dataset_info(datasets, finetune_logger)

        training_best_val_metric = train(algorithm, datasets, config, finetune_logger, minimalLogger, epoch_offset, best_val_metric)
        epoch_offset = config.num_epochs
        finetune_logger.close()

    if logger != None:
        logger.close()
    return training_best_val_metric, GetAlgorithmState(algorithm, epoch_offset, training_best_val_metric), algorithm

