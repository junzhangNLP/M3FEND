import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import pandas as pd
from transformers import BertTokenizer
import ast
import warnings
default_size = (224, 224)
class FNDDataset(Dataset):
    def __init__(self, csv_path, caption_path, ORC_path, dataname, image_path,caption_path_video = None):
        self.data = []

        if dataname == "Weibo":
            self.text_data = pd.read_csv(csv_path)
            self.caption_data = pd.read_csv(caption_path,sep=";")
            self.ORC_data = pd.read_csv(ORC_path,sep=";")

            for _, row in self.text_data.iterrows():
                images= []
                ORCs = []
                for i in ast.literal_eval(row["image_files"]):
                    if not self.caption_data[self.caption_data["filename"] == i].empty:
                        images.append(str(self.caption_data[self.caption_data["filename"] == i]["content"]))
                        
                        ORCs.append(str(self.ORC_data[self.ORC_data["filename"] == i]["content"])) 
                        filepath = os.path.join(image_path, i)
                        
                        
                ORC_list = [
                    s for s in ORCs 
                    if not s.replace('\n', '').replace('\r', '').isdigit() and len(s) >= 5
                ]
                images_result = " ".join(images)
                ORC_result = " ".join(ORC_list)
                self.data.append({"text":row["content"],
                                "image":image,
                                "caption":images_result,
                                "ORC":ORC_result,
                                "label":row["label"],
                                })
        elif dataname == "X":
            self.text_data = pd.read_csv(csv_path)
            self.caption_data = pd.read_csv(caption_path,sep=";")
            self.ORC_data = pd.read_csv(ORC_path,sep=";")

            for _, row in self.text_data.iterrows():
                images= []
                ORCs = []
                for i in ast.literal_eval(row["image_files"]):
                    if not self.caption_data[self.caption_data["filename"] == i].empty:
                        images.append(str(self.caption_data[self.caption_data["filename"] == i]["content"]))
                        ORCs.append(str(self.ORC_data[self.ORC_data["filename"] == i]["content"])) 
                        filepath = os.path.join(image_path, i)

                ORC_list = [
                    s for s in ORCs 
                    if not s.replace('\n', '').replace('\r', '').isdigit() and len(s) >= 5
                ]
                images_result = " ".join(images)
                ORC_result = " ".join(ORC_list)
                self.data.append({"text":row["content"],
                                "image":image,
                                "caption":images_result,
                                "ORC":ORC_result,
                                "label":row["label"],
                                })
        elif dataname == "Fakeddit":
            self.text_data = pd.read_csv(csv_path)
            self.caption_data = pd.read_csv(caption_path,sep=";")
            self.ORC_data = pd.read_csv(ORC_path,sep=";")

            for _, row in self.text_data.iterrows():
                images= []
                ORCs = []
                for i in ast.literal_eval(row["image_files"]):
                    if not self.caption_data[self.caption_data["filename"] == i].empty:
                        images.append(str(self.caption_data[self.caption_data["filename"] == i]["content"]))
                        ORCs.append(str(self.ORC_data[self.ORC_data["filename"] == i]["content"])) 
                        filepath = os.path.join(image_path, i)

                        
                ORC_list = [
                    s for s in ORCs 
                    if not s.replace('\n', '').replace('\r', '').isdigit() and len(s) >= 5
                ]
                images_result = " ".join(images)
                ORC_result = " ".join(ORC_list)
                self.data.append({"text":row["content"],
                                "image":None,
                                "caption":images_result,
                                "ORC":ORC_result,
                                "label":row["label"],
                                })
        elif dataname == "AMG":
            self.text_data = pd.read_csv(csv_path)
            self.caption_data = pd.read_csv(caption_path,sep=";")
            self.ORC_data = pd.read_csv(ORC_path,sep=";")
            self.video_data = pd.read_csv(caption_path_video,sep=";")

            self.caption_data['clean_id'] = self.caption_data['filename'].apply(
                lambda x: os.path.splitext(x)[0] if pd.notna(x) else None)
            self.ORC_data['clean_id'] = self.ORC_data['filename'].apply(
                lambda x: os.path.splitext(x)[0] if pd.notna(x) else None)
            self.video_data['clean_id'] = self.video_data['filename'].apply(
                lambda x: os.path.splitext(x)[0] if pd.notna(x) else None)

            for id, row in self.text_data.iterrows():
    
                caption_match = self.caption_data[self.caption_data["clean_id"].astype(int) == row["id"]]

                orc_match = self.ORC_data[self.ORC_data["clean_id"].astype(int) == row["id"]]

                video_match = self.video_data[self.video_data["clean_id"].astype(int) == row["id"]]
                if caption_match.empty and not video_match.empty:
                    caption = video_match["content"].iloc[0]
                elif not caption_match.empty and  video_match.empty:
                    caption = caption_match["content"].iloc[0]
                elif caption_match.empty and video_match.empty:
                    caption = "nan"
             
                
                if not orc_match.empty:
                    orc = str(orc_match["content"].iloc[0])
                    if orc.replace('\n', '').replace('\r', '').isdigit() or len(orc) < 5:
                        orc = "nan"
                else:
                    orc = "nan"
                support_exts = ['.jpg', '.jpeg', '.png', '.mp4']
                found_path = None
                for root, _, files in os.walk(image_path):
                    for file in files:
                        # 分离文件名和扩展名
                        name, ext = os.path.splitext(file)
                        if name == str(row["id"]) and ext.lower() in support_exts:
                            found_path = os.path.join(root, file)
                            break
                    if found_path:
                        break

                if found_path:
                    if found_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    elif found_path.lower().endswith('.mp4'):
                        cap = cv2.VideoCapture(found_path)
                        ret, frame = cap.read()  # 读取第一帧
                        cap.release()
                        if not ret:
                            img = Image.new('RGB', (224, 224), (0, 0, 0))
                        else:
                        # 转换BGR→RGB，并转为PIL Image
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(frame_rgb)
                
                self.data.append({"text":str(row["content"]),
                                "image":img,
                                "caption":str(caption),
                                "ORC":str(orc),
                                "label":row["label"]
                                })
                


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'text': item["text"],  
            "image":item["image"],
            'caption': item['caption'], 
            'ORC': item['ORC'],  
            'label': int(item['label'])
        }



