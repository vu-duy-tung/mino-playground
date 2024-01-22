import random
import typing
from collections import Counter
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class IMDBBertDataset(Dataset):
    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    UNK = '[UNK]'
    
    MASK_PERCENTAGE = 0.15 # How much words to mask
    
    MASKED_INDICES_COLUMN = 'masked_indices'
    TARGET_COLUMN = 'indices'
    NSP_TARGET_COLUMN = 'is_next'
    TOKEN_MASK_COLUMN = 'token_mask'
    
    OPTIMAL_LENGTH_PERCENTILE = 70
    
    def __init__(self, path, ds_from=None, ds_to=None, should_include_text=False):
        self.ds: pd.Series = pd.read_csv(path)['review']
        if ds_from is not None or ds_to is not None:
            self.ds = self.ds[ds_from: ds_to]
            
        self.tokenizer = get_tokenizer('basic_english')
        self.counter = Counter()
        self.vocab = None
        self.optimal_sentence_length = None
        self.should_include_text = should_include_text

        if should_include_text:
            self.columns = ['masked_sentence', self.MASKED_INDICES_COLUMN, 'sentence', self.TARGET_COLUMN,
                            self.TOKEN_MASK_COLUMN,
                            self.NSP_TARGET_COLUMN]
        else:
            self.columns = [self.MASKED_INDICES_COLUMN, self.TARGET_COLUMN, self.TOKEN_MASK_COLUMN,
                            self.NSP_TARGET_COLUMN]
            
        self.df = self.prepare_dataset()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return
    
    def _update_length(self, sentences: typing.List[str], lengths: typing.List[str]):
        for v in sentences:
            l = len(v.split())
            lengths.append(l)
        return lengths
    
    def _find_optimal_sentence_length(self, lengths: typing.List[int]):
        arr = np.array(lengths)
        return int(np.percentile(arr, self.OPTIMAL_LENGTH_PERCENTILE))
    
    def _fill_vocab(self):
        self.vocab = vocab(self.counter, min_freq=2)
        self.vocab.insert_token(self.CLS, 0)  
        self.vocab.insert_token(self.PAD, 1)  
        self.vocab.insert_token(self.MASK, 2)  
        self.vocab.insert_token(self.SEP, 3)  
        self.vocab.insert_token(self.UNK, 4)  
        self.vocab.set_default_index(4)
        
    def _mask_sentence(self, sentence: typing.List[str]):
        """Replace MASK_PERCENTAGE of words with MASK token or random word token

        Args:
            sentence (typing.List[str]): sentence to mask

        Returns:
            (processed sentence, inverse token mask)
        """
        
        len_s = len(sentence)
        inverse_token_mask = [True for _ in range(max(len_s, self.optimal_sentence_length))]
        
        mask_amount = round(len_s * self.MASK_PERCENTAGE)
        for _ in range(mask_amount):
            i = random.randint(0, len_s - 1)
            
            if random.random() < 0.8:
                sentence[i] = self.MASK
            else:
                j = random.randint(5, len(self.vocab) - 1)
                sentence[i] = self.vocab.lookup_token(j)
                
            inverse_token_mask[i] = False
            
        return sentence, inverse_token_mask
    
    def _preprocess_sentence(self, sentence: typing.List[str], should_mask: bool=True):
        inverse_token_mask = True
        if should_mask:
            sentence, inverse_token_mask = self._mask_sentence(sentence)
        sentence, inverse_token_mask = self._pad_sentence([self.CLS] + sentence, [True] + inverse_token_mask)
                
        return sentence, inverse_token_mask
        
    def _create_item(self, first: typing.List[str], second: typing.List[str], target: int):
        # create masked sentence item
        updated_first, first_mask = self._preprocess_sentence(first.copy())
        updated_second, second_mask = self._preprocess_sentence(second.copy())
        nsp_sentence = updated_first + [self.SEP] + updated_second
        nsp_indices = self.vocab.lookup_indices(nsp_sentence)
        inverse_token_mask = first_mask + [True] + second_mask
        
        
        
    def prepare_dataset(self) -> pd.DataFrame:
        sentences = []
        nsp = []
        sentence_lens = []
        
        # Split dataset on sentences
        for review in self.ds:
            review_sentences = review.split(". ")
            sentences += review_sentences
            self._update_length(review_sentences, sentence_lens)
        self.optimal_sentence_length = self._find_optimal_sentence_length(sentence_lens)
        
        print("Create vocabulary")
        for sentence in tqdm(sentences):
            tokens = self.tokenizer(sentence)
            self.counter.update(tokens)
            
        self._fill_vocab()
        
        print("Preprocessing dataset")
        cnt = 0
        for review in tqdm(self.ds):
            cnt += 1
            review_sentences = review.split('. ')
            if len(review_sentences) > 1:
                for i in range(len(review_sentences) - 1):
                    first, second = self.tokenizer(review_sentences[i]), self.tokenizer(review_sentences[i+1])
                    nsp.append(self._create_item(first, second, 1))
                    
                    # False NSP item
                    first, second = self._select_false_nsp_sentences(sentences)
                    first, second = self.tokenizer(first), self.tokenizer(second)
                    nsp.append(self._create_item(first, second, 0))
        df = pd.DataFrame(nsp, columns=self.columns)
        
    
        


if __name__ ==  "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    print(BASE_DIR)
    
    data = IMDBBertDataset(path='data/IMDB Dataset.csv')