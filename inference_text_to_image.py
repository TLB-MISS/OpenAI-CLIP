import gc
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import argparse
import pandas as pd

import config as CFG
from pretrain import build_loaders
from CLIP import CLIPModel

def make_dfs():
    if CFG.dataset == "8k":
      dataframe = pd.read_csv(f"{CFG.captions_path}")
    elif CFG.dataset == "30k":
      dataframe = pd.read_csv(f"{CFG.captions_path}", sep='|')
      dataframe.columns = ["image", "comment_number", "caption"]
      dataframe = dataframe.drop(["comment_number"],axis=1)
    else:
      raise Exception("Does not support other than flickr8k or flickr30k")

    dataframe = dataframe.dropna()
    dataframe.insert(0, "id", dataframe.index)
    return dataframe

def get_image_embeddings(df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    data_loader = build_loaders(df, tokenizer, mode="valid")
     
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    res_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            res_image_embeddings.append(image_embeddings)
    return model, torch.cat(res_image_embeddings)

def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    _, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
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
    if args.query is None:
        raise Exception("Usage :[python3 inference_text_to_image.py --query={YOUR QUERY}]")

    df = make_dfs()
    model, image_embeddings = get_image_embeddings(df, CFG.model_path)
    find_matches(model,
                image_embeddings,
                query=args.query,
                image_filenames=df['image'].values,
                n=9)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str)
    args = parser.parse_args()
    main(args)
