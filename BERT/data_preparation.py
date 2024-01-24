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
        item = self.df.iloc[index]

        inp = torch.Tensor(item[self.MASKED_INDICES_COLUMN]).long()
        token_mask = torch.Tensor(item[self.TOKEN_MASK_COLUMN]).bool()
        
        mask_target = torch.Tensor(item[self.TARGET_COLUMN]).long()
        mask_target = mask_target.masked_fill_(token_mask, 0)
        
        attention_mask = (inp == self.vocab[self.PAD]).unsqueeze(0)
        
        if item[self.NSP_TARGET_COLUMN] == 0:
            t = [1, 0]
        else:
            t = [0, 1]
        
        nsp_target = torch.Tensor(t)
        
        return (
            inp.to(device),
            attention_mask.to(device),
            
        )
    
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
    
    def _pad_sentence(self, sentence: typing.List[str], inverse_token_mask: typing.List[bool]=None):
        len_s = len(sentence)
        
        if len_s >= self.optimal_sentence_length:
            s = sentence[:self.optimal_sentence_length]
        else:
            s = sentence + [self.PAD] * (self.optimal_sentence_length - len_s)
            
        # inverse token mask should be padded as well
        if inverse_token_mask:
            len_m = len(inverse_token_mask)
            if len_m >= self.optimal_sentence_length:
                inverse_token_mask = inverse_token_mask[:self.optimal_sentence_length]
            else:
                inverse_token_mask = inverse_token_mask + [True] * (self.optimal_sentence_length - len_s)
        
        return s, inverse_token_mask
    
    def _preprocess_sentence(self, sentence: typing.List[str], should_mask: bool=True):
        inverse_token_mask = [True] * len(sentence)
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
        
        # Create sentence item without masking random words
        first, _ = self._preprocess_sentence(first.copy(), should_mask=False)
        second, _ = self._preprocess_sentence(second.copy(), should_mask=False)
        original_nsp_sentence = first + [self.SEP] + second
        original_nsp_indices = self.vocab.lookup_indices(original_nsp_sentence)
        
        if self.should_include_text:
            return (
                nsp_sentence,
                nsp_indices,
                original_nsp_sentence,
                original_nsp_indices,
                inverse_token_mask,
                target
            ) 
        else:
            return (
                nsp_indices,
                original_nsp_indices,
                inverse_token_mask,
                target
            )
            
    def _select_false_nsp_sentences(self, sentences: typing.List[str]):
        """ Select sentences to create false nsp item
        Args:
            sentences: list of all sentences
            
        Returns:
            Tuple of two sentences and the second one is not the next sentence of the first one
        
        """
        number_of_sentences = len(sentences)
        
        while True:
            first_sentence_idx = random.randint(0, number_of_sentences-1)
            second_sentence_idx = random.randint(0, number_of_sentences-1)
            if abs(first_sentence_idx - second_sentence_idx) > 1:
                break
        
        if first_sentence_idx > second_sentence_idx:
            first_sentence_idx, second_sentence_idx = second_sentence_idx, first_sentence_idx
        
        return sentences[first_sentence_idx], sentences[second_sentence_idx]
        
        
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
        
        return df
        
    
        


if __name__ ==  "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    print(BASE_DIR)
    
    data = IMDBBertDataset(path='data/IMDB Dataset.csv', ds_from=0, ds_to=50000, should_include_text=True)
    
    print(type(data.df))
    print(data.df.columns)
    print(data[0])