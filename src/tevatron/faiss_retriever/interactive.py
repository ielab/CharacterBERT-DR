import torch
import numpy as np
import glob
from itertools import chain
from tqdm import tqdm

from tevatron.arguments import ModelArguments, InteractiveArguments
from .retriever import BaseFaissIPRetriever
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from tevatron.modeling import DenseOutput, DenseModelForInference, CharacterIndexer

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def load_collection(collection_path):
    collection = {}
    with open(collection_path, 'r') as f:
        for line in tqdm(f, desc="Loading collection"):
            docid, text = line.strip().split('\t')
            collection[docid] = text
    return collection


def search_queries(retriever, q_reps, p_lookup, args):
    all_scores, all_indices = retriever.search(q_reps, args.depth)
    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def main():
    parser = HfArgumentParser((InteractiveArguments, ModelArguments))
    args, model_args = parser.parse_args_into_dataclasses()
    args: InteractiveArguments
    model_args: ModelArguments

    index_files = glob.glob(args.passage_reps)
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = DenseModelForInference.build(
        model_name_or_path=model_args.model_name_or_path,
        model_args=model_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if model_args.character_query_encoder:
        indexer: CharacterIndexer = CharacterIndexer()

    p_reps_0, p_lookup_0 = torch.load(index_files[0])
    retriever = BaseFaissIPRetriever(p_reps_0.float().numpy())

    shards = chain([(p_reps_0, p_lookup_0)], map(torch.load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
    look_up = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps.float().numpy())
        look_up += p_lookup

    logger.info('Loading collection')
    collection = load_collection(args.collection)

    while True:
        query = input("Type your query: ")

        if model_args.character_query_encoder:
            encoded_query = tokenizer.basic_tokenizer.tokenize(query)
            encoded_query = ['[CLS]', *encoded_query, '[SEP]']
            encoded_query = indexer.as_padded_tensor(
                [encoded_query],
                maxlen=128
            )
        else:
            encoded_query = tokenizer.encode_plus(
                query,
                max_length=128,
                truncation='only_first',
                padding=False,
                return_token_type_ids=False,
                return_tensors='pt'
            )

        model_output: DenseOutput = model(query=encoded_query)

        logger.info('Index Search Start')
        all_scores, psg_indices = search_queries(retriever, model_output.q_reps.float().numpy(), look_up, args)
        logger.info('Index Search Finished')

        logger.info(f'The target query: {query}')
        logger.info(f'Returned results:')
        for i, (score, pid) in enumerate(zip(all_scores[0], psg_indices[0])):
            print(f'{i}, id: {pid}, score: {score} \n {collection[pid]} \n')


if __name__ == '__main__':
    main()