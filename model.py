from transformers import XLNetTokenizer, XLNetLMHeadModel 
import torch 

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')