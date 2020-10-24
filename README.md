# gpt-tf2
TensorFlow 2 implementation of GTP2 for fine-tuning on a single GPU.

1. [Setup](#setup)
	1. [Software](#software)
	2. [Hardware](#hardware)
2. [Acknowledgement](#acknowledgement)
3. [Examples](#examples)
	1. [Text Generation](#text-generation)
	2. [Text Summarization](#text-summarization)


## Setup <a name="setup"></a>

### Software <a name="software"></a>

```
virtualenv -p /usr/bin/python3.6 venv
. venv/bin/activate

pip install -r requirements.txt
```

### Hardware <a name="hardware"></a>

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

Notice: This repo __does not__ implement the RL based fine-tuning algorithm as described in [this blog](https://openai.com/blog/fine-tuning-gpt-2/). In contrast, we fine-tune the transformer layers using additional datasets for each new application.


## Examples <a name="examples"></a>

### Text Generation <a name="text-generation"></a>

The first application is to fine-tune GPT2 to generate text of a particular "style." We have two examples here: the screenplay of `Kill Bill,` and the first five books of `A Song of Ice and Fire`. The training data is stored as a single `.txt` file. 

For testing, we condition the text generation by the starter sentence `She picked up the sword` and see if the fine-tuned model can create any exciting output.

First, let see what the pre-trained model (no fine-tuning) produces:

```
python inference.py \
--model_dir=models/124M/ \
--nucleus \
--top_p=0.8 \
--temperature=1.0 \
--output_length=200 \
--starter='She picked up the sword'
```

The paragraph below is an example output: the English is largely fluent, however it presents made-up characters and somewhat semi-coherent story.

```
She picked up the sword and went out again, and a second later he entered into the castle.

Yikes.

The moment the shadow disappeared, Qin Yu discovered that there was no sign of the sword before he arrived there.

"Bastard bastard. Don't underestimate me." An old man with a nose wrinkled his brows, "I'll personally help you escape from here!"

That was as expected. A normal and upright person would be strong, but his strength was limited. It wasn't a difficult matter. This didn't have any meaning, just to be able to survive, but had to be said.

Qin Yu couldn't help but take a deep breath. "Bastard!"

Qin Yu held onto the sword as he rushed forward.

He was starting to get worried.

In the end, when Qin Yu left Qin Yu's side, Qin Yu had become ill and would eventually pass away, with no further mention
```

To fine-tune GPT2 for text generation, we specify the model (size, pre-trained ckpt, json files for model hyperparameters and the encoder, the byte-pair-encoding of the vocabulary) and the training data (path to the text file and the type of loader).

The following command fine-tunes the 355M model on `Kill Bill` for five epochs, where each epoch has 2000 pieces (1024 tokens each) of text randomly sampled from the screenplay. 

```
python finetune.py \
--model_dir=models/124M/ \
--output_name=killbill_124M_5x2000.h5 \
--dataset_path=dataset/killbill.txt \
--data_loader=text \
--num_epoch=5 \
--decay_epochs="4,5" \
--steps_per_epoch=2000
```

To test the fine-tuned model, we generate 200 tokens using `nucleus sampling` with `top_p=1.0` and `temperature=1.0`:

```
python inference.py \
--model_dir=models/124M/
--custom_model=output/killbill_355M_5x2000.h5 \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--starter='She picked up the sword'
```

This is the result:

```
She picked up the sword, and is slowly dragging it over her head.

TILL...

They lock eyes...

...Yuki yells to her in Japanese;



                    YUKI (JAPANESE)
          ...You can kill yourself.

Yuki giggles.



                      YUKI (JAPANESE)
          Just don't make me kill you.



                    THE BRIDE (JAPANESE)
          Okay, I want to see how    good you really are.
```

As you can see, the output start looks a lot more like a screenplay, with the correct format and characters from `Kill Bill` (THE BRIDE and YUKI). 

### Text Summarization <a name="text-summarization"></a>


Our next experiment fine-tunes the GPT2 model for text summarization. The original OpenAI GPT2 blog demonstrated that a pre-trained model (without fine-tuning) has a certain ability to [summarize](https://openai.com/blog/better-language-models/#task5), indicating the training dataset contains some examples where the model learned the concept of "summarization" and the certain keywords to triggle the summarization process. 

As described in the [original paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)(Section 3.6), one only need to add `TL;DR:` to the end of the starter text
for "zero-shot" text summarization:

```
python inference.py \
--model_dir=models/355M/ \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--starter="In the Second Age of Middle-earth, the lords of Elves, Dwarves, and Men are given Rings of Power. Unbeknownst to them, the Dark Lord Sauron forges the One Ring in Mount Doom, infusing into it a great part of his power to dominate, through it and at a distance, the other Rings, so he might conquer Middle-earth. A final alliance of men and elves battles Sauron\'s forces in Mordor, where Prince Isildur of Gondor severs Sauron\'s finger, and the Ring with it, thereby destroying his physical form. With Sauron\'s first defeat, the Third Age of Middle-earth begins. Unfortunately, the Ring\'s influence corrupts Isildur, and, rather than destroy the Ring, Isildur takes it for himself. Isildur is later killed by Orcs, and the Ring is lost for 2,500 years, until it is found by Gollum, who owns it for five centuries. The Ring is then found by a hobbit named Bilbo Baggins, who turns invisible when he puts it on, but is unaware of its history. TL;DR:"
```

However, our experience of the output is somewhat disappointing. For example, these are a couple of "summaries" generated from the above LOTR text:

```
TL;DR: far darker than we know, one of the oldest, largest tongue- ancient and vast in history even. 
The encirzation stars symbol, thought to be any rectangles, scrimshaw, or awkwardly spaced (less variohed to).be issued by our language. 
Its making power achievable by all through magic consumption. identity is the consuming power from sacrificial outpourings. 
Rivendell entertainment; stunt-star beford bounty awarded its Moron. against. Anyone may enables its,'production. 
Treason involves.' but He speculated whon, if was this power utilized. it goes as if he, than unfairly. 
around culture goes by Danny Moby had Pergola ate mystery haunceaced Tickets Yes
```  

```
TL;DR: after the adventure designs Lost and found the would, in Baggins & Gimli & Galadriel will go there first and only Sulando the housebonder will, 
run from there when the 3sus take Telemachus and not. If absorbed by, is he unaware by the 4sing evil in his form at. 
Or Barliman thepotent only, wolves v\ a danger. Terror Axits his lives,d { dont confissions even rhodes char Youoglas, 
through onin he himself demands 7 with it to. 1861 would seven hoursa in which they would an out is going to fight speedtic happenspses there. 
theirs Put'sunder where some always been Since the days to know not Known bear into dyes anymore. prior disclose knowledge Knowing of the Lies. 
The key' lies and arrayed. It, thereafter of thingmouth yet refuses, will endure Miracles up without spelling Lem. lesions and roots.
```

As you can see, the results are too long, non-factual, and only vaguely related to the starter. They are more like text generation instead of summarization. Notice that the generated text has some key important LOTR concepts that do not exist in the starter text, such as "Rivendell" (an elf city), "Gimli" (a dwarf character), "Galadriel" (an elf character) and etc. This indicates the original training data have articles about LOTR, and the pre-trained model actually memorized these concepts, and pull them out during the inference (probably through the attention mechanism)

Next, let's see if the performance can be improved by fine-tuning. We use the [cnn-dailymail](https://cs.nyu.edu/~kcho/DMQA/) dataset, which has 290k news articles and a few "highlights" for each article. We create the ground-truth training data by concatenating each article with one of its highlights (randomly picked). Here are some examples:

___Ground Truth One___

```
If you turn to the Bible -- Isaiah Chapter 35, Verse 8 -- you will see a passage that in part says, "A highway shall be there, and a road, and it shall be called the Highway of Holiness."

Churchgoers in six states have held prayer sessions along the side of Interstate 35.

Now, is it possible that this "highway" mentioned in Chapter 35 is actually Interstate 35 that runs through six U.S. states, from southern Texas to northern Minnesota? Some Christians have faith that is indeed the case.

... Truncated for brevity ...

But on the side of the road, the prayerful aren't going to change their minds. Holy highways and nude clubs, they believe, are not a combination God has in mind. E-mail to a friend
TL;DR:


I-35 runs from southern Texas to northern Minnesota<|endoftext|>

```

___Ground Truth Two___

```
(InStyle) -- It all boils down to this. It doesn't really matter all that much what hot, nubile French maverick has set the fashion world on fire. Or which Milanese visionary has a new fabric technique discovered during a life-changing trip to Angkor Wat that's sure to bring back sixties minimalism with a twist. Or that so-and-so has signed a deal to develop boutique spa hotels around the globe in former monasteries. Because, in the end, he's Ralph Lauren, and we're not.

Ralph Lauren has his eye on China and Japan.

... Truncated for brevity ...


Get a FREE TRIAL issue of InStyle - CLICK HERE!

Copyright © 2007 Time Inc. All rights reserved.
TL;DR:


Ralph Lauren began as tie salesman from the Bronx
```

To fine-tune the `355M` model, we point the `dataset_path` to the [preprocessed cnn-dailymail dataset] and specify `cnndm` as the loader. Here we fine-tune the model for five epoch and 2000 steps per epoch. We also decreased the initial learning rate to `0.0001` to avoid gradient overflow.

```
python finetune.py \
--model_dir=models/355M/ \
--output_name=cnndm_355M_5x2000.h5 \
--dataset_path=/home/ubuntu/data/summarization \
--data_loader=cnndm \
--base_lr=0.0001 \
--num_epoch=5 \
--decay_epochs="4,5" \
--steps_per_epoch=2000
```

Here are some summaries from the fine-tuned model. They are significantly better than the pre-trained model: the results are much more concise and associated with the starter text.

___Fine-tuned Example One___

```
In the Second Age of Middle-earth, the lords of Elves, Dwarves, and Men are given Rings of Power. Unbeknownst to them, 
the Dark Lord Sauron forges the One Ring in Mount Doom, infusing into it a great part of his power to dominate, 
through it and at a distance, the other Rings, so he might conquer Middle-earth. A final alliance of men and elves battles Sauron's forces in Mordor, 
where Prince Isildur of Gondor severs Sauron's finger, and the Ring with it, thereby destroying his physical form. With Sauron's first defeat, 
the Third Age of Middle-earth begins. Unfortunately, the Ring's influence corrupts Isildur, and, rather than destroy the Ring, 
Isildur takes it for himself. Isildur is later killed by Orcs, and the Ring is lost for 2,500 years, until it is found by Gollum, 
who owns it for five centuries. The Ring is then found by a hobbit named Bilbo Baggins, who turns invisible when he puts it on, 
but is unaware of its history.
TL;DR:

# Result 1
The Ring is believed to have been lost primarily for 25 years


# Result 2
Excess Ring content contributed to Final Age of Middle-earth

# Result 3
The ring is found by Gollum
```

___Fine-tuned Example Two___

```
TensorFlow [1] is an interface for expressing machine learning algorithms, and an implementation for executing such algorithms. 
A computation expressed using TensorFlow can be executed with little or no change on a wide variety of heterogeneous systems, 
ranging from mobile devices such as phones and tablets up to large-scale distributed systems of hundreds of machines and 
thousands of computational devices such as GPU cards. The system is flexible and can be used to express a wide variety of algorithms, 
including training and inference algorithms for deep neural network models, and it has been used for conducting research and for deploying 
machine learning systems into production across more than a dozen areas of computer science and other fields, including speech recognition, 
computer vision, robotics, information retrieval, natural language processing, geographic information extraction, 
and computational drug discovery. This paper describes the TensorFlow interface and an implementation of that interface that we 
have built at Google. The TensorFlow API and a reference implementation were released as an open-source package under the 
Apache 2.0 license in November, 2015 and are available at www.tensorflow.org.
TL;DR:

# Result 1
TensorFlow software was a section of Google's foundation software suite

# Result 2
(1) TensorFlow interface for machine learning algorithms and an implementation of that interface

# Result 3
TensorFlow was built to express computer learning processes
```
