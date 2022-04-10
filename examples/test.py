from run_experiment import Config, main

if __name__ == "__main__":

    config = Config("bert", ["personality_detection"], ["Friends"], ["emory_emotion_recognition"], ["Friends"], 3, do_finetune=True, do_eval=True)

    main(config)