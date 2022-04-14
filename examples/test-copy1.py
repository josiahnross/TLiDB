from run_experiment import Config, main
import torch
if __name__ == "__main__":
    tasks = [
        #'emory_emotion_recognition', 
        'reading_comprehension', 
        # 'character_identification',
        # 'question_answering', 
        # 'personality_detection'
        # 'relation_extraction',
        # 'MELD_emotion_recognition'
    ]
    seed = 12345
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    learningRates = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    effectiveBatchSizes = [10, 30, 60, 120]
    percentToRemove = 0.4
    for t in tasks:
        results = {}
        for bs in effectiveBatchSizes:
            for lr in learningRates:
                results[(lr, bs)] = 0
        for i in range(0, 6, 1):
            for lr, bs in results.keys():
                config = Config("bert", [t], ["Friends"], [t], ["Friends"], 3*(i+1), do_train=True, eval_best=True, gpu_batch_size=10,
                learning_rate=lr, effective_batch_size=bs)
                config.seed= seed
                config.save_last = True
                config.resume = True # i != 0 
                best_val_metric = main(config)
                results[(lr, bs)] = max(best_val_metric, results[(lr, bs)])
            paramsToRemove = int(len(results) * percentToRemove)
            sortedParams = sorted(results, key=results.get)
            worstParams = sortedParams[0:paramsToRemove]
            for p in worstParams:
                del results[p]
            print(f"Remaining Hyperparameters: {results}")

    print(results)