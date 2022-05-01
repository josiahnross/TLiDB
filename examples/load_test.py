from run_experiment import Config, main
from utils import Logger


config = Config("bert", ["reading_comprehension"], ["Friends"], ["reading_comprehension"], 
    ["Friends"], 1, do_train=True, eval_best=True, gpu_batch_size=3)

main(config, Logger())
