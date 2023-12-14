import torch
import torch.nn as nn
from typing import Optional
from torch.nn import CrossEntropyLoss
from transformers import CLIPConfig, CLIPModel, CLIPProcessor, CLIPPreTrainedModel


class CLIPForImageClassification(CLIPPreTrainedModel):
    def __init__(self, config: CLIPConfig, num_labels: int):
        super().__init__(config)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.clip.parameters():
            p.requires_grad=False
        self.num_labels = num_labels
        self.criterion = torch.nn.CrossEntropyLoss()
        self.head = nn.Linear(self.config.projection_dim, num_labels)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        ):
        clip_outputs = self.clip(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.head(clip_outputs.image_embeds)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.criterion(logits, labels)

        return {"loss": loss, "logits": logits}
