# touhou-gpt-bot
TensorFlow 2 implementation of GPT-2 for use as a chatbot in Discord servers.

1. [Setup](#setup)
	1. [Software](#software)
	2. [Hardware](#hardware)
2. [Acknowledgement](#acknowledgement)


## Setup <a name="setup"></a>

### Software <a name="software"></a>

```
virtualenv -p /usr/bin/python3.6 venv
. venv/bin/activate

pip install -r requirements.txt
```

### Hardware <a name="hardware"></a>

#### GPU
Normally, a GPU is not required for GPT2, but it is highly recommended as it can take several hours to fine tune models.
GPT2 is very GPU memory intensive. Here is the minimal requirements for models of different sizes:

* 124M: 11GB (1080Ti, 2080Ti etc)
* 355M: 24GB (RTX Titan, RTX Quadro 6000, Tesla V100 etc)
* 774M: 48GB (RTX Quadro 8000)
* 1558M: seems not possible on a single GPU.

## Acknowledgement <a name="acknowledgement"></a>

This project would not be possible without the guidance and inspiration from these repositories:

[OpenAI GPT2](https://github.com/openai/gpt-2): For pre-trained GPT2 models and examples of running inference with them.

[OpenAI ln-human-preferences](https://github.com/openai/lm-human-preferences): For example of data loader for the `cnn-dailymail` dataset.

[minimaxir](https://github.com/minimaxir/gpt-2-simple): For examples of fine-tuning GPT2 models in TensorFlow 1.14.

[CyberZHG](https://github.com/CyberZHG/keras-gpt-2): For examples of Keras implementation of GPT2 graph and restoring weights from checkpoints.

[lambdal](https://github.com/lambdal/gpt-tf2-keras): This is the repository that I based this project around. I really could not have made this possible without this repo!

Notice: This repo __does not__ implement the RL based fine-tuning algorithm as described in [this blog](https://openai.com/blog/fine-tuning-gpt-2/). In contrast, we fine-tune the transformer layers using additional datasets for each new application.
