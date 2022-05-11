from run_experiment import Config, main
from utils import Logger

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

if __name__ == '__main__':
    config = Config("bert", ["masked_language_modeling"], ["Friends"], ["emory_emotion_recognition"], 
        ["Friends"], 1, do_train=True, eval_best=True, gpu_batch_size=5)
    main(config, Logger())

