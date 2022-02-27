# INTRODUCTION
--------------------------------------------------------------------------
This is a modification of the OpenAI-CLIP repo of moein-shariatnia(https://github.com/moein-shariatnia/OpenAI-CLIP).

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

+ Inference image from text query 
```
$ python3 inference_text_to_image.py --query={YOUR QUERY}
```

+ Inference text from image query
```
$ python3 inference_image_to_text.py --query_text={YOUR .txt FILE} --query_image={YOUR .jpg or .png FILE}
```

# CAUTION
--------------------------------------------------------------------------
You must set(or check) some options in config.py before pretrain & inference
> ex1) *dataset*(*"8k"* or *"30k"*): Train dataset(flicker-8k or flicker-30k)

> ex2) *model_name*(*"resnet50"* or *"vit_base_patch16_384"*): Type of image encoder

> ex3) *pretrained*(*True* or *False*): Decide whether to learn by loading pretrain versions of text encoder(DistilBert) and image encoder(resnet50 or ViT)

> ex4) *batch_size*: Set according to the capacity of the machine
