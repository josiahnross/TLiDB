import torch
from algorithms.initializer import initialize_algorithm
import configs
from train import train,evaluate
import os
from utils import Logger, load_datasets_split, load_algorithm, log_config, \
        set_seed, log_dataset_info, get_savepath_dir, append_to_save_path_dir
import argparser

class Config:
    frac = 1.0
    debug = False
    generate_during_training = False

    # config for experimentation ease
    model_config = None

    # general args
    cpu_only = False
    seed = -1
    log_and_model_dir = "./logs_and_models"
    saved_model_dir = None
    data_dir = "../TLiDB/data"
    num_workers = 4

    # model args
    model = None
        
    # training args
    do_train = False
    do_finetune = False
    num_epochs = 10
    effective_batch_size = 60
    gpu_batch_size = 20
    learning_rate = 3e-5
    fp16 = False
    max_grad_norm = 1.0
    save_best = True
    save_last = False
    imbalanced_task_weighting = True

    # evaluation args
    do_eval = False
    eval_best = False
    eval_last = False
    
    # task args
    source_tasks = []
    source_datasets = []
    target_tasks = []
    target_datasets = []

    # TTiDB args
    multitask = False
    few_shot_percent = None
    
    # algorithm args
    optimizer = "Adam"
    weight_decay = 0.0

    # misc. args
    progress_bar = True
    save_pred = False
    resume = False

    output_type = None
    model_type = None
    generation_config = None
    device = None

    def __init__(self, model_config: str, source_tasks: list, source_datasets: list, target_tasks: list, target_datasets: list, num_epochs: int, 
        gpu_batch_size: int = 5, do_finetune: bool = False, do_train: bool = False, do_eval: bool = False, eval_best = True, 
        learning_rate = -1, effective_batch_size = -1, saved_model_dir = None, resume = False):
        self.model_config = model_config
        self.source_tasks = source_tasks
        self.source_datasets = source_datasets
        self.target_tasks = target_tasks
        self.target_datasets = target_datasets
        self.num_epochs = num_epochs
        self.gpu_batch_size = gpu_batch_size
        self.do_finetune = do_finetune
        self.do_train = do_train
        self.do_eval = do_eval
        self.eval_best = eval_best
        self.saved_model_dir = saved_model_dir
        self.resume = resume

        model_config_dict = configs.__dict__[f"{model_config}_config"]
        self.model = model_config_dict["model"]
        self.optimizer = model_config_dict["optimizer"]
        self.learning_rate = model_config_dict["learning_rate"]
        self.fp16 = model_config_dict["fp16"]
        self.effective_batch_size = model_config_dict["effective_batch_size"]
        self.max_dialogue_length = model_config_dict["max_dialogue_length"]

        if learning_rate > 0:
            self.learning_rate = learning_rate
        if effective_batch_size > 0:
            self.effective_batch_size = effective_batch_size

        if "bert" in self.model:
            self.output_type = "categorical"
            self.output_type = "categorical"
            self.model_type = "Encoder"
        elif "gpt" in self.model:
            self.output_type = "token"
            self.model_type = "Decoder"
            self.generation_config = configs.GPT2_generation_config
        elif "t5" in self.model:
            self.output_type = "token"
            self.model_type = "EncoderDecoder"
            self.generation_config = configs.t5_generation_config
        else:
            raise ValueError(f"Model {self.model} not supported")

        if not self.cpu_only:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

def main(config: Config, minimalLogger: Logger):
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


    training_best_val_metric = None
    set_seed(config.seed)
    if config.do_train:
        datasets = {}
        
        # load datasets for training
        datasets['train'] = load_datasets_split("train",config.train_tasks, config.train_datasets, config)
        datasets['dev'] = load_datasets_split("dev",config.dev_tasks, config.dev_datasets, config)

        # log configuration and dataset info
        logger.write("TRAINING\n")
        log_config(config,logger)
        log_dataset_info(datasets, logger)

        # initialize algorithm
        algorithm = initialize_algorithm(config, datasets)

        # try to resume training from a saved model
        resume_success = False
        if resume:
            if os.path.exists(os.path.join(config.save_path_dir, 'last_model.pt')):
                prev_epoch, best_val_metric = load_algorithm(algorithm, os.path.join(config.save_path_dir, 'last_model.pt'),logger)
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

    if config.do_finetune:
        assert(config.target_datasets and config.target_tasks),"Must specify target datasets and tasks to finetune"
        datasets = {}
        # get the pre-trained model path
        if config.do_train or (config.train_datasets and config.train_tasks):
            # Do nothing, this means we already have a save_dir_path and a model saved there
            pass
        elif config.saved_model_dir:
            # if user explcitily specified a pretrained model to finetune, then use that
            config.save_path_dir = config.saved_model_dir
        else:
            raise ValueError("To run fine-tuning, use:\n--saved_model_dir to specify the pre-trained model OR\
                \n--train_datasets and --train_tasks to specify the pretraining datasets and tasks")
        
        # if fine tuning, set fine-tune train, and fine-tune dev to the same tasks
        config.finetune_train_tasks = config.finetune_tasks
        config.finetune_train_datasets = config.finetune_datasets
        config.finetune_dev_tasks = config.finetune_tasks
        config.finetune_dev_datasets = config.finetune_datasets

        # load datasets for finetuning
        datasets['train'] = load_datasets_split("train",config.finetune_train_tasks,config.finetune_train_datasets,config)
        datasets['dev'] = load_datasets_split("dev", config.finetune_dev_tasks, config.finetune_dev_datasets, config)

       # initialize algorithm
        algorithm = initialize_algorithm(config, datasets)

        # always load best pretrained model
        model_path = os.path.join(config.save_path_dir, 'best_model.pt')
        is_best = True
        load_algorithm(algorithm, model_path, logger)
        epoch_offset = 0
        best_val_metric = None

        # update save path with fine-tuning details
        config.save_path_dir = append_to_save_path_dir(config.save_path_dir, config.finetune_datasets, config.finetune_tasks, config.few_shot_percent, config.seed)
        
        # note the fine-tuning in the pretrained model log
        logger.write(f"FINETUNING at {config.save_path_dir}\n")
        
        # create new logger for fine-tuning
        if not os.path.exists(config.save_path_dir):
            os.makedirs(config.save_path_dir)
        if config.debug:
            finetune_logger = Logger(mode='w')
        else:
            finetune_logger = Logger(os.path.join(config.save_path_dir, 'log.txt'), mode="w")

        # log configuration and dataset info
        finetune_logger.write("FINETUNING\n")
        finetune_logger.write(f"Loaded pretrained model from {model_path}\n")
        log_config(config,finetune_logger)
        log_dataset_info(datasets, finetune_logger)

        train(algorithm, datasets, config, finetune_logger, epoch_offset, best_val_metric)
        finetune_logger.close()

    if config.do_eval:
        assert(config.target_datasets and config.target_tasks),"Must specify target datasets and tasks to finetune"
        datasets = {}
        # If coming from training/fine-tuning, 
        #   this means we already have a save_dir_path from training/fine-tuning and a model saved there
        if config.do_finetune or config.do_train:
            pass
        elif config.saved_model_dir:
            # if user explcitily specified a model to evaluate, then use that
            config.save_path_dir = config.saved_model_dir
        elif (config.finetune_datasets and config.finetune_tasks) and (config.train_datasets and config.train_tasks):
            # Given all the datasets and tasks, we can infer the path to the fine-tuned model
            config.save_path_dir = append_to_save_path_dir(config.save_path_dir, config.finetune_datasets, config.finetune_tasks, config.few_shot_percent, config.seed)
        else:
            raise ValueError("To run evaluation, use:\n--saved_model_dir to specify the model to evaluate OR\
                \n--finetune_datasets and --finetune_tasks and --train_datasets and --train_tasks to infer the path to the model")

        # ensure user has specified a model to evaluate
        assert(not(config.eval_last and config.eval_best)), "cannot evaluate both last and best models"
        assert(config.eval_last or config.eval_best), "must evaluate at least one model"
        
        # create logger for evaluation
        if config.debug:
            eval_logger = Logger(mode='w')
        else:
            eval_logger = Logger(os.path.join(config.save_path_dir, 'log.txt'), mode="a")
        eval_logger.write("EVALUATING\n")
        
        # load datasets for evaluation
        # TODO Ask to make sure this splits properly
        datasets['test'] = load_datasets_split("test",config.eval_tasks, config.eval_datasets, config)
        # log configuration and dataset info
        log_config(config,eval_logger)
        log_dataset_info(datasets, eval_logger)

        # initialize algorithm
        algorithm = initialize_algorithm(config, datasets)  
        
        # load evaluation model
        if config.eval_last:
            eval_model_path = os.path.join(config.save_path_dir, "last_model.pt")
            is_best = False
        else:
            eval_model_path = os.path.join(config.save_path_dir, 'best_model.pt')
            is_best = True

        epoch, best_val_metric = load_algorithm(algorithm, eval_model_path,eval_logger)
        evaluate(algorithm, datasets, config, eval_logger, epoch, is_best)

        eval_logger.close()
    logger.close()
    return training_best_val_metric

if __name__ == "__main__":
    config = argparser.parse_args()
    main(config)
