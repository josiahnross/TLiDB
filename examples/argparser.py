import argparse
import torch
import configs

incompatible_with_fp16 = ["t5-base"]
incompatible_with_generation = ["bert"]

def parse_args():
    parser = argparse.ArgumentParser()
    
    # debugging args
    parser.add_argument("--frac", type=float, default=1.0,
        help="Convenience parameter that scales down dataset size to specified fraction, for debugging purposes")
    parser.add_argument("--debug", action="store_true", help="Setup training for debugging. For example, no logging")
    parser.add_argument("--generate_during_training", action="store_true", help="If true, generative models will generate\
        text during training, normally they only calculate loss")

    # config for experimentation ease
    parser.add_argument("--model_config",type=str, default=None)

    # general args
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--log_and_model_dir", type=str, default="./logs_and_models")
    parser.add_argument("--saved_model_dir", type=str, default=None, help="To load a saved model for fine-tuning or evaluation")
    parser.add_argument("--data_dir", type=str, default="../TLiDB/data",
        help="The directory where data is stored, by default we place it inside the TLiDB package in a data folder")
    parser.add_argument("--num_workers", type=int, default=4)

    # model args
    parser.add_argument("--model", type=str)
        
    # training args
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_finetune", action="store_true")
    parser.add_argument("-e", "--num_epochs", type=int, default=10)
    parser.add_argument("--effective_batch_size", type=int, default=60)
    parser.add_argument("--gpu_batch_size", type=int, default=20)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_best", type=bool, default=True)
    parser.add_argument("--save_last", type=bool, default=False)
    parser.add_argument("--imbalanced_task_weighting", type=bool, default=True)

    # evaluation args
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_best", action="store_true")
    parser.add_argument("--eval_last", action="store_true")
    
    # task args
    parser.add_argument("--source_tasks", type=str,nargs='+',required=True)
    parser.add_argument("--source_datasets", type=str,nargs='+',required=True)
    parser.add_argument("--target_tasks", type=str,nargs='+')
    parser.add_argument("--target_datasets", type=str,nargs='+')

    # TTiDB args
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--few_shot_percent", type=float, default=None)
    
    # algorithm args
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # misc. args
    parser.add_argument("--progress_bar", type=bool, default=True)
    parser.add_argument("--save_pred",action="store_true",help="Save predictions for each task")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")

    args = parser.parse_args()

    if args.debug:
        # send debugging logs and models to different directory
        args.log_and_model_dir = "./debug_logs_and_models"

    if args.model_config is not None:
        for attr, value in configs.__dict__[f"{args.model_config}_config"].items():
            setattr(args, attr, value)

    if "bert" in args.model:
        setattr(args, "output_type", "categorical")
        setattr(args, "model_type", "Encoder")
    elif "gpt" in args.model:
        setattr(args, "output_type", "token")
        setattr(args, "model_type", "Decoder")
        setattr(args, "generation_config", configs.GPT2_generation_config)
    elif "t5" in args.model:
        setattr(args, "output_type", "token")
        setattr(args, "model_type", "EncoderDecoder")
        setattr(args, "generation_config", configs.t5_generation_config)
    else:
        raise ValueError(f"Model {args.model} not supported")

    if not args.cpu_only:
        setattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    else:
        setattr(args, "device", "cpu")

    if args.num_workers > 1:
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = "true"

    assert(not(args.model in incompatible_with_fp16 and args.fp16)), f"Cannot use fp16 with model {args.model}"
    assert(not(args.model in incompatible_with_generation and\
            any(['generation' in t for t in args.source_tasks+args.target_tasks]))), f"Cannot perform generation tasks with model {args.model}"

    return args
