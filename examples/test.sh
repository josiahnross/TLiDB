python run_experiment.py --do_train --model_config bert --source_tasks personality_detection --source_datasets Friends --do_finetune --do_eval --eval_best --target_tasks emory_emotion_recognition --target_datasets Friends --gpu_batch_size 5 --num_epochs 3