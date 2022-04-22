import torch
import configs
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