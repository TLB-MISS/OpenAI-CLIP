import torch

debug = False
num_workers = 4
head_lr = 1e-3
image_encoder_lr = 1e-4
text_encoder_lr = 1e-5
weight_decay = 1e-3
patience = 1
factor = 0.8
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_encoder_model = "distilbert-base-uncased"
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 200

# related with image encoder
# image size & batch size & image_embedding
model_name = "vit_base_patch16_384" # "resnet50" or "vit_base_patch16_384"
if model_name == "resnet50":
    size = 224
    batch_size = 32
    image_embedding = 2048
elif model_name == "vit_base_patch16_384":
    size = 384
    batch_size = 8
    image_embedding = 1000
else:
    raise Exception("Does not support other than resnet50 or vit_base_patch16_384")

# related with dataset
# dataset path & model path
dataset = "8k" # "8k" or "30k"
if dataset == "8k":
    image_path = "/cmsdata/ssd0/sangwon/flickr_dataset/flickr8k/Images"
    captions_path = "/cmsdata/ssd0/sangwon/flickr_dataset/flickr8k/captions.txt"
    model_path = "./8k_best.pt"
elif dataset == "30k":
    image_path = "/cmsdata/ssd0/sangwon/flickr_dataset/flickr30k/flickr30k_images"
    captions_path = "/cmsdata/ssd0/sangwon/flickr_dataset/flickr30k/results.csv"
    model_path = "./30k_best.pt"
else:
    raise Exception("Does not support other than flickr8k or flickr30k")
model_path = "./" + model_name + "_" + dataset + "_best.pt"


pretrained = True # for both image encoder and text encoder
trainable = True # for both image encoder and text encoder
temperature = 1.0

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1
