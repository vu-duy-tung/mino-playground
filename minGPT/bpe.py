import os
import json
import regex as re
import requests
import torch

def bytes_to_unicode():
    """
    Return a dictionary of bytes with their corresponding unicode string
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    return d


def get_pairs(word):
    """
    Return all bigrams as set of tuples, of all consecutive elements in the iterable word
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def get_file(local_file, remote_file):
    """
    Download remote_file to local_file if necessary
    """
    if not os.path.isfile(local_file):
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        open(local_file, "wb").write(response.content)



class Encoder():
    def __init__(self, encoder, bpe_merges):
        self.bytes_encoder = bytes_to_unicode()
        self.bytes_decoder = {v : k for k, v in self.bytes_encoder.items()}

        self.encoder = encoder  # token -> bpe index
        self.decoder = {v : k for k, v in self.encoder.items()}

        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    

    def bpe(self, tokens) -> list[str]:
        pairs = list(get_pairs(tokens))

        while len(pairs) > 0:
            new_tokens = []
            pair = min(pairs, key = lambda pair : self.bpe_ranks.get(pair, float('inf')))
            if self.bpe_ranks.get(pair, float('inf')) == float('inf'):
                break

            i = 0
            while i < len(tokens):
                if tokens[i] == pair[0] and i+1 <= len(tokens) - 1 and tokens[i+1] == pair[1]:
                    new_tokens.append(''.join(pair))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
            pairs = get_pairs(tokens)

        return tokens
    

    def encode(self, text) -> list[int]:
        tokens = re.findall(self.pat, text)
        
        new_tokens = []
        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.bytes_encoder[byte] for byte in token_bytes)
            new_tokens.append(token_translated)
        
        bpe_tokens = self.bpe(new_tokens)
        bpe_indexes = [self.encoder[token] for token in bpe_tokens]
        return bpe_indexes


def get_encoder():
    """
    Return an instance of GPT BPE Encoder/Decoder
    and handle caching of "database" file
    """ 
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.cache', 'minGPT')
    os.makedirs(cache_dir, exist_ok=True)

    # load mapping between token -> bpe index
    encoder_local_file = os.path.join(cache_dir, "encoder.json")
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
    get_file(encoder_local_file, encoder_remote_file)

    token2index = json.load(open(encoder_local_file)) # this will incude 256 byte tokens, 50,000 merged tokens, and <|endoftext|> token
    

    # load vocab.bpe
    vocab_local_file = os.path.join(cache_dir, "vocab.bpe")
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    get_file(vocab_local_file, vocab_remote_file)
    print(vocab_local_file)
    with open(vocab_local_file, 'r', encoding='utf-8') as f:
        bpe_data = f.read()

    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]

    encoder = Encoder(token2index, bpe_merges)
    return encoder

enc = get_encoder()
print(enc.encode("Hello World!!! Today is perfect cause I got 100 in the final exam???"))


# byte_encoder = bytes_to_unicode()
# pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
# text = "Hello World!!! Today is perfect cause I got 100 in the final exam???"
# tokens = re.findall(pat, text)
# print(tokens)

# for token in tokens:
#     token_bytes = token.encode('utf-8')
#     token_translated = ''.join([byte_encoder[b] for b in token_bytes])
#     print(token_translated)