# Queries and Judgements

This folder contains the query and qrel files used in our paper, including our new DL-typo dataset.

In each folder, we include a qrel file, a query file and its corresponding typo query files.

For dl2019, dl2020, and msmarco dev we provide 10 synthetically generated typo query files which we used in our paper.
If you want to generate typo queries by yourself, run: 

`python make_typo_queries.py --query_file path-to-query --save_to save-path`

For DL-typo and msmarco dev, we also provided spellchecker corrected query files, `.ms` means corrected by [MS Bing Spell Check API](https://docs.microsoft.com/en-us/azure/cognitive-services/bing-spell-check/overview)
`.py` means corrected by [pyspellchecker](https://github.com/barrust/pyspellchecker). If you want to do spell check by yourself, run: 

`python xx_spellchecker.py --query_file path-to-query --save_to save-path`
