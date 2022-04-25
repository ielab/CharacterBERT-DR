# CharacterBERT-DR
The offcial repository for [CharacterBERT and Self-Teaching for Improving the Robustness of Dense Retrievers on Queries with Typos](https://arxiv.org/pdf/2204.00716.pdf), Shengyao Zhuang and Guido Zuccon, SIGIR2022
> Note: this repository is in refactoring.


## Installation
`pip install --editable .`

> Note: The current code base has been tested with, `torch==1.8.1`, `faiss-cpu==1.7.1`, `transformers==4.9.2`, `datasets==1.11.0`, `textattack=0.3.4`

## Preparing data and model

### Typo queries and dataset
All the queries and qrels used in our paper are in the `/data` folder.

### Download CharacterBERT
Install google drive download tool: `pip install gdown`

Download [CharacterBERT](https://github.com/helboukkouri/character-bert/tree/0c1f5c2622950988833a9d95e29bc26864298592#pre-trained-models) trained with general domain:

`gdown https://docs.google.com/uc?id=11-kSfIwSWrPno6A4VuNFWuQVYD8Bg_aZ`

`tar -xf general_character_bert.tar.xz`


## Train

### CharacterBERT + ST training
```
python -m tevatron.driver.train \
--model_name_or_path bert-base-uncased \
--character_bert_path ./general_character_bert \
--output_dir model_msmarco_characterbert \
--save_steps 20000 \
--dataset_name Tevatron/msmarco-passage \
--fp16 \
--per_device_train_batch_size 16 \
--learning_rate 5e-6 \
--max_steps 150000 \
--dataloader_num_workers 10 \
--cache_dir ./cache \
--logging_steps 150 \
--character_query_encoder True \
--self_teaching True
```