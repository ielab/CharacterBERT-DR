# Install pyspellchecker: pip install pyspellchecker

from tqdm import tqdm
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
import argparse

spell = SpellChecker()

parser = argparse.ArgumentParser()
parser.add_argument('--query_file')
parser.add_argument('--save_to')
args = parser.parse_args()

with open(args.query_file, 'r') as f, \
        open(args.save_to, 'a+') as wf:
    lines = f.readlines()
    for line in tqdm(lines):
        qid, qry = line.strip().split('\t')
        words = word_tokenize(qry)
        correct_qry = ''
        for word in words:
            misspelled = spell.unknown([word])
            if len(misspelled) != 0:
                word = spell.correction(word)
            correct_qry += word + " "
        wf.write(qid + '\t' + correct_qry + '\n')

