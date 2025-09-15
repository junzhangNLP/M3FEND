import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import  ViTModel
import math
class ModalityExpert(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        
    def forward(self, raw_input):
        raise NotImplementedError

class VisionExpert(ModalityExpert):

    def __init__(self, pre_path, output_dim=768):
        super().__init__(output_dim=output_dim)
        

        self.vit = ViTModel.from_pretrained(pre_path)
        

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),  
            nn.Linear(output_dim, 2) 
        )
        
    def forward(self, images):

        outputs = self.vit(
            pixel_values=images["pixel_values"],
        )
        

        features = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(features)
        
        return features, logits

class TextExpert(ModalityExpert):

    def __init__(self,pre_path):
        super().__init__(output_dim=768)
        self.bert = BertModel.from_pretrained(pre_path)

        self.classifier = nn.Sequential(
            nn.Dropout(0.01),
            nn.Linear(768, 2)
        )
    def forward(self,text):
        return self.bert(input_ids=text["input_ids"], attention_mask=text["attention_mask"]).last_hidden_state[:, 0, :],self.classifier(self.bert(input_ids=text["input_ids"], attention_mask=text["attention_mask"]).last_hidden_state[:, 0, :])

class VisualConceptExpert(ModalityExpert):

    def __init__(self,pre_path):
        super().__init__(output_dim=768)
        self.bert = BertModel.from_pretrained(pre_path)

        self.classifier = nn.Sequential(
            nn.Dropout(0.01),
            nn.Linear(768, 2)
        )
    def forward(self, image):
        return self.bert(input_ids=image["input_ids"], attention_mask=image["attention_mask"]).last_hidden_state[:, 0, :],self.classifier(self.bert(input_ids=image["input_ids"], attention_mask=image["attention_mask"]).last_hidden_state[:, 0, :])

class OCRAnalysisExpert(ModalityExpert):

    def __init__(self,pre_path):
        super().__init__(output_dim=768)
        self.bert = BertModel.from_pretrained(pre_path)

        self.classifier = nn.Sequential(
            nn.Dropout(0.01),
            nn.Linear(768, 2)
        )
    def forward(self, orc):
        return self.bert(input_ids=orc["input_ids"], attention_mask=orc["attention_mask"]).last_hidden_state[:, 0, :],self.classifier(self.bert(input_ids=orc["input_ids"], attention_mask=orc["attention_mask"]).last_hidden_state[:, 0, :])

class MoMOExpert(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        
    def forward(self, raw_input):
        raise NotImplementedError

class TMOVExpert(MoMOExpert):

    def __init__(self,input_dim,hidden_dim,output_dim,hidden_dropout_prob):
        super().__init__(output_dim)
        self.num_heads = 4
        self.head_dim = input_dim // 4
        

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        

        self.feature_head = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim) 
        )
        

        self.classifier = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(output_dim, 2)
        )
        
        
    def forward(self,text_feature):

        batch_size = text_feature.size(0)
        text, visual = torch.chunk(text_feature, 2, dim=1)  
        

        q = self.q_proj(text).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(visual).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(visual).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = (attn_weights @ v).transpose(1, 2).reshape(batch_size, -1)
        attn_output = self.out_proj(context)
        
        

        fused_feature = torch.cat([text.squeeze(1), attn_output.squeeze(1)], dim=1)
        transformed = self.feature_head(fused_feature)
        
        features = self.feature_head(fused_feature)
        
        logits = self.classifier(features)
        return features, logits

class TMOOExpert(MoMOExpert):

    def __init__(self,input_dim,hidden_dim,output_dim,hidden_dropout_prob):
        super().__init__(output_dim)
        self.num_heads = 4
        self.head_dim = input_dim // 4
        

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        

        self.feature_head = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)  
        )
        

        self.classifier = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(output_dim, 2)
        )
        
        
    def forward(self,text_feature):

        batch_size = text_feature.size(0)
        text, visual = torch.chunk(text_feature, 2, dim=1)  
        

        q = self.q_proj(text).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(visual).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(visual).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = (attn_weights @ v).transpose(1, 2).reshape(batch_size, -1)
        attn_output = self.out_proj(context)
        
        

        fused_feature = torch.cat([text.squeeze(1), attn_output.squeeze(1)], dim=1)
        transformed = self.feature_head(fused_feature)
        
        features = self.feature_head(fused_feature)
        
        logits = self.classifier(features)
        return features, logits
    
class TMOIExpert(MoMOExpert):

    def __init__(self,input_dim,hidden_dim,output_dim,hidden_dropout_prob):
        super().__init__(output_dim)
        self.num_heads = 4
        self.head_dim = input_dim // 4
        

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        

        self.feature_head = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim) 
        )
        

        self.classifier = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(output_dim, 2)
        )
        
        
    def forward(self,text_feature):

        batch_size = text_feature.size(0)
        text, visual = torch.chunk(text_feature, 2, dim=1) 
        

        q = self.q_proj(text).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(visual).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(visual).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = (attn_weights @ v).transpose(1, 2).reshape(batch_size, -1)
        attn_output = self.out_proj(context)
        
        

        fused_feature = torch.cat([text.squeeze(1), attn_output.squeeze(1)], dim=1)
        transformed = self.feature_head(fused_feature)
        
        features = self.feature_head(fused_feature)
        
        logits = self.classifier(features)
        return features, logits
    
class TaskDrivenAdversarialGate(nn.Module):

    def __init__(self, feat_dim, num_experts):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_experts = num_experts
        

        self.gate_network = nn.Sequential(
            nn.Linear(feat_dim * 3, 64), 
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )

        self.importance_predictor = nn.ModuleList([
            nn.Linear(feat_dim, 1) for _ in range(3)  
        ])
        
    def forward(self, visual_feat, ocr_feat,image_feat, task_loss=None):

        gate_input = torch.cat([visual_feat, ocr_feat,image_feat], dim=1)
        raw_gates = self.gate_network(gate_input)  # [B, num_experts]
        
        if not self.training or task_loss is None:
            return F.softmax(raw_gates, dim=1)
        

        modalities = [visual_feat, ocr_feat,image_feat]
        importance_scores = []
        
        for i, feat in enumerate(modalities):

            grad = torch.autograd.grad(
                outputs=task_loss,
                inputs=feat,
                retain_graph=True,
                create_graph=True  
            )[0]  # [B, feat_dim]
            

            importance = 1 / (grad.norm(dim=1, keepdim=True) + 1e-6)  # [B, 1]
            importance_scores.append(importance)
        

        pred_importance = [
            torch.sigmoid(self.importance_predictor[i](feat)) 
            for i, feat in enumerate(modalities)
        ]
        

        adv_loss = sum(
            F.mse_loss(pred, true.detach()) 
            for pred, true in zip(pred_importance, importance_scores)
        )
        

        weighted_gates = raw_gates * torch.cat(importance_scores, dim=1)  
        final_gates = F.softmax(weighted_gates, dim=1)
        
        return final_gates, adv_loss

class MultiModalMoE(nn.Module):
    """多模态混合专家系统"""
    def __init__(self,input_dim,hidden_dim,output_dim,hidden_dropout_prob,pre_path,vit_path):
        super().__init__()

        self.modality_experts = nn.ModuleDict({
            "vit": VisionExpert(vit_path),
            "text": TextExpert(pre_path),
            "visual": VisualConceptExpert(pre_path),
            "ocr": OCRAnalysisExpert(pre_path)
        })

        self.adversarial_gate = TaskDrivenAdversarialGate(
            feat_dim=input_dim, 
            num_experts=3  
        )

        self.classifier = nn.Linear(output_dim, 2)

        self.Fusemodal_experts = nn.ModuleDict({
            "TV": TMOVExpert(input_dim,hidden_dim,output_dim,hidden_dropout_prob),
            "TO": TMOOExpert(input_dim,hidden_dim,output_dim,hidden_dropout_prob),
            "TI": TMOIExpert(input_dim,hidden_dim,output_dim,hidden_dropout_prob),
        })


        self.classfenlei = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, text,image_inputs, caption, orc, labels):

        text_feat,t_logit = self.modality_experts["text"](text)
        image_feat,t_logit = self.modality_experts["vit"](image_inputs)
        visual_feat,v_logit = self.modality_experts["visual"](caption)
        ocr_feat,o_logit = self.modality_experts["ocr"](orc)


        TV_feat, TVfuse_logit= self.Fusemodal_experts["TV"](torch.cat([text_feat,visual_feat], dim=1))
        TO_feat, TOfuse_logit = self.Fusemodal_experts["TO"](torch.cat([text_feat,ocr_feat], dim=1))
        TI_feat, TIfuse_logit = self.Fusemodal_experts["TI"](torch.cat([text_feat,image_feat], dim=1))

        task_loss = None
        if labels is not None:

            temp_logits = self.classifier(0.5 * (TV_feat + TO_feat + TI_feat))
            task_loss = F.cross_entropy(temp_logits, labels)
        

        if self.training and labels is not None:
            gate_weights, adv_loss = self.adversarial_gate(visual_feat, ocr_feat, image_feat, task_loss)
        else:
            gate_weights = self.adversarial_gate(visual_feat, ocr_feat, image_feat)  
            adv_loss = None
        fused = TV_feat + TO_feat+ TI_feat

        fused_feat = gate_weights[:, 0:1] * TV_feat + gate_weights[:, 1:2] * TO_feat+ gate_weights[:, 2:3] * TI_feat

        output = self.classfenlei(fused_feat)
        
        return {
            "final_logit": output,
            "fused_feat": fused_feat,
            "fused": fused,
            "adv_loss": adv_loss,
        }