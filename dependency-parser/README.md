# Name
## Description

## Requirements
Create Conda Environment:
```bash
conda create -n nlp_dp_proj python=3.10
conda activate nlp_dp_proj
```

Install packages
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install matplotlib gensim scipy==1.10.1 tqdm pyconll
```


## Getting started
Install data
```bash
git clone https://github.com/datquocnguyen/VnDT.git
```
Run training
```bash
cd scr
python main.py
```

Run evaluation
```bash
cd src
python testing.py
```

## Contributors

## References