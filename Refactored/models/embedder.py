import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from models.datasets import TextDataset
from transformers import AutoTokenizer, BertModel
from tqdm import tqdm
from typing import List


logger = logging.getLogger(__name__)


class BERTEmbedder:
    """Generates embeddings for text data using a pre-trained BERT model (google-bert/bert-base-uncased)."""
    
    def __init__(self, model_name: str = "google-bert/bert-base-uncased", batch_size: int = 64):
        """Initializes the BERTEmbedder with a specified model and batch size and locks the model to evaluation mode."""
        self.batch_size = batch_size
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    
    def generate_embeddings(self, text_list: List[str]) -> np.ndarray:
        """Generates embeddings for a list of text strings using the BERT model.
        Args:
            text_list (List[str]): A list of text strings for which to generate embeddings.
        Returns:
            np.ndarray: A matrix of embeddings for the input texts.
        """
        dataset = TextDataset(text_list)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_embeddings = []

        with torch.no_grad():
            for batch_texts in tqdm(dataloader, desc="Extracting"):
                
                encoded_input = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                ).to(self.device)
                
                
                model_output = self.model(**encoded_input)
                token_embeddings = model_output.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                all_embeddings.append(batch_embeddings.cpu().numpy())

        final_matrix = np.vstack(all_embeddings)
        return final_matrix