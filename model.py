from transformers import * 
import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss

#tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
#model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased') 


class UDA(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = XLNetModel(config) 
        self.lm_loss = nn.Linear(config.d_model, config.vocab_size, bias=True) 
        self.init_weights() 
    

