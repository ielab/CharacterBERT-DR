from textattack.transformations import WordSwapNeighboringCharacterSwap, \
    WordSwapRandomCharacterDeletion, WordSwapRandomCharacterInsertion, \
    WordSwapRandomCharacterSubstitution, WordSwapQWERTY
from textattack.augmentation import Augmenter
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation import MinWordLength, StopwordModification

from tqdm import tqdm
from argparse import ArgumentParser
import random


STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                      "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                      'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                      'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                      'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                      'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                      'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                      'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                      'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                      "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                      "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                      'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                      "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
                      'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])


class FixWordSwapQWERTY(WordSwapQWERTY):
    def _get_replacement_words(self, word):
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = len(word) - (1 + self.skip_last_char)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            i = random.randrange(start_idx, end_idx + 1)
            if len(self._get_adjacent(word[i])) == 0:
                candidate_word = (
                    word[:i] + random.choice(list(self._keyboard_adjacency.keys())) + word[i + 1:]
                )
            else:
                candidate_word = (
                    word[:i] + random.choice(self._get_adjacent(word[i])) + word[i + 1:]
                )
            candidate_words.append(candidate_word)
        else:
            for i in range(start_idx, end_idx + 1):
                for swap_key in self._get_adjacent(word[i]):
                    candidate_word = word[:i] + swap_key + word[i + 1 :]
                    candidate_words.append(candidate_word)

        return candidate_words


def read_query_lines(path_to_query):
    query_lines = []
    with open(path_to_query, 'r') as f:
        contents = f.readlines()

    for line in tqdm(contents, desc="Loading query"):
        qid, query = line.strip().split("\t")
        query_lines.append((qid, query))
    return query_lines


def write_query_file(qids, queries, output_path):
    query_lines = []
    for i in range(len(qids)):
        query_lines.append(str(qids[i]) + "\t" + queries[i] + "\n")
    with open(output_path, "a+") as f:
        f.writelines(query_lines)


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_file', required=True)
    parser.add_argument('--save_to', required=True)
    args = parser.parse_args()


    query_lines = read_query_lines(args.query_file)
    transformation = CompositeTransformation([
        WordSwapRandomCharacterDeletion(),
        WordSwapNeighboringCharacterSwap(),
        WordSwapRandomCharacterInsertion(),
        WordSwapRandomCharacterSubstitution(),
        FixWordSwapQWERTY(),
    ])
    constraints = [MinWordLength(3), StopwordModification(STOPWORDS)]
    augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0)
    qids = []
    typo_queires = []
    for qid, query in tqdm(query_lines, desc="Making typo queries"):
        while True:
            typo_query = augmenter.augment(query)[0]
            if typo_query != query and typo_query.lower() == query:
                continue
            break
        typo_query = typo_query.lower()
        qids.append(qid)
        typo_queires.append(typo_query)
    write_query_file(qids, typo_queires, args.save_to)


if __name__ == '__main__':
    main()