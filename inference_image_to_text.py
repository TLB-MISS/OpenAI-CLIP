import gc
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import argparse

import config as CFG
from dataset import get_transforms
from pretrain import build_loaders, make_train_valid_dfs
from CLIP import CLIPModel

def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(CFG.model_path, map_location=CFG.device))
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

def get_embeddings(text_file, image_file):
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(CFG.model_path, map_location=CFG.device))
    model.eval()

    # get text query
    f = open(text_file)
    query = f.readlines()
    query = list(map(lambda x:x.replace("\n", ""), query))

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer(query, padding=True, truncation=True, max_length=CFG.max_length)
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }

    # get image
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transforms = get_transforms()
    image = transforms(image=image)['image']
    image = torch.tensor(image).permute(2, 0, 1).float()
    image = torch.reshape(image, ((1,) + image.shape))

    # get text & image embedding
    with torch.no_grad():
        image_features = model.image_encoder(image.to(CFG.device))
        image_embeddings = model.image_projection(image_features)
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    # find match
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    dot_similarity = dot_similarity.T
    _, indices = torch.topk(dot_similarity.squeeze(0), 3)
    matches = [query[idx] for idx in indices]
    print(matches)
        

def find_matches(text_embeddings, image_embeddings, query):
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    _, indices = torch.topk(dot_similarity.squeeze(0), 3)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    
    plt.show()
    query_ = query.replace(" ","_")
    result_filename = query_ + "_" + CFG.model_name + "_" + CFG.dataset + "_result.png"
    plt.savefig(result_filename)

def main(args):
    if (args.query_text is None) or (args.query_image is None):
        raise Exception("Usage :[python3 inference_image_to_text.py --query_text={YOUR QUERY TEXT FILE} --query_image={YOUR QUERY IMAGE FILE}]")


    get_embeddings(args.query_text, args.query_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_text', type=str)
    parser.add_argument('--query_image', type=str)
    args = parser.parse_args()
    main(args)
