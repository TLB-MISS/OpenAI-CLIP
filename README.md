# INTRODUCTION
--------------------------------------------------------------------------
This is a modification of the OpenAI-CLIP repo of moein-shariatnia.

The current training dataset supports flicker-8k or flicker-30k, and the image encoder supports Resnet50 or ViT(*vit_base_patch16_384*).

Text encoder supports only DistilBert like moein-shariatnia.

# ENVIRONTMENT SETTING
--------------------------------------------------------------------------
```
$ virtualenv .venv --python=python3.6
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

# EXECUTTION
--------------------------------------------------------------------------
+ Pretrain
```
$ python3 pretrain.py
```

+ Inference
```
$ python3 inference.py --qeury={YOUR QUERY}
```

# CAUTION
--------------------------------------------------------------------------
You must set(or check) some options in config.py before pretrain & inference
> ex1) *dataset*(*"8k"* or *"30k"*): Train dataset(flicker-8k or flicker-30k)

> ex2) *model_name*(*"resnet50"* or *"vit_base_patch16_384"*): Type of image encoder

> ex3) *pretrain*(*True* or *False*): Decide whether to learn by loading pretrain versions of text encoder(DistilBert) and image encoder(resnet50 or ViT)

> ex4) *batch_size*: Set according to the capacity of the machine
