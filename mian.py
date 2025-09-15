import random
import warnings
from transformers import  logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, ViTImageProcessor
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
import pandas as pd
import numpy as np
from tools.dataloader import FNDDataset
from model.model import MultiModalMoE
from tqdm.auto import tqdm
from tools.tools import setup_seed
import logging
import pandas as pd
import numpy as np
import sys
from transformers import logging as transformers_logging

setup_seed(111)

transformers_logging.set_verbosity_error()


warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("output_log.txt"), 
        logging.StreamHandler(sys.stdout)   
    ]
)

def log_message(message):
    logging.info(message)



vit_path = 'D:\daima\VIT'
processor = ViTImageProcessor.from_pretrained(vit_path)

tokenizer = BertTokenizer.from_pretrained(model_path)
def collate_fn(batch, max_len=128):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    captions = [item["caption"] for item in batch]
    ORCs = [item["ORC"] for item in batch]
    labels = [item["label"] for item in batch]
    labels = torch.tensor(labels, dtype=torch.long)

    image_inputs = processor(
        images=images, 
        return_tensors="pt",
        padding=True,     
        truncation=True,    
        return_token_type_ids=False
    )

    text_encoding = tokenizer(
        texts,
        truncation = True,
        padding='max_length',
        max_length=max_len,
        return_length=True,
        return_tensors='pt'
    )
    
    caption_encoding = tokenizer(
        captions,
        truncation = True,
        padding='max_length',
        max_length=max_len,
        return_length=True,
        return_tensors='pt'
    )

    ORC_encoding = tokenizer(
        ORCs,
        truncation = True,
        padding='max_length',
        max_length=max_len,
        return_length=True,
        return_tensors='pt'
    )

    return {
        "text_encoding": text_encoding,
        "image_inputs":image_inputs,
        "caption_encoding": caption_encoding,
        "ORC_encoding": ORC_encoding,
        "labels": labels
    }


train_dataset = FNDDataset(csv_path = r"dataset\weibo\train.csv", 
                           caption_path = r"dataset\weibo\Caption.csv", 
                           ORC_path = r"dataset\weibo\ORC.csv", 
                           image_path = r"dataset\weibo\img", 
                           dataname = "Weibo")
test_dataset = FNDDataset(csv_path = r"dataset\weibo\test.csv", 
                          caption_path = r"dataset\weibo\Caption.csv", 
                          ORC_path = r"dataset\weibo\ORC.csv", 
                          image_path = r"dataset\weibo\img", 
                          dataname = "Weibo")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,collate_fn=collate_fn)
val_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 20

total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


def train_epoch(model, data_loader, optimizer, device, scheduler):
    print("train")
    model.train()
    predictions = []
    true_labels = []
    losses = []
    for batch in tqdm(data_loader):
        text_encoding = batch['text_encoding'].to(device)
        image_inputs = batch['image_inputs'].to(device)
        caption_encoding = batch['caption_encoding'].to(device)
        ORC_encoding = batch['ORC_encoding'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
                text=text_encoding,
                image_inputs = image_inputs,
                caption=caption_encoding,
                orc=ORC_encoding,
                labels = labels
        )

        logits = outputs["final_logit"]+0.1*outputs["adv_loss"]

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average=None)
    
    metrics = {
        'accuracy': accuracy,
        'precision_neg': precision[0],
        'recall_neg': recall[0],
        'f1_neg': f1[0],
        'precision_pos': precision[1],
        'recall_pos': recall[1],
        'f1_pos': f1[1]
    }
    
    return np.mean(losses),metrics