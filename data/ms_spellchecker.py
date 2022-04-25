import requests
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--query_file')
parser.add_argument('--save_to')
args = parser.parse_args()

# To set up ms checker api
# follow https://docs.microsoft.com/en-us/azure/cognitive-services/bing-spell-check/overview
api_key = "3ded996efae74cd3b22e0306930998d5"
endpoint = "https://api.bing.microsoft.com/v7.0/SpellCheck"
params = {
    'mkt':'en-us',
    'mode':'proof'
    }
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Ocp-Apim-Subscription-Key': api_key,
    }

with open(args.query_file, 'r') as f, \
        open(args.save_to, 'a+') as wf:
    lines = f.readlines()
    for i, line in tqdm(enumerate(lines)):

        qid, query = line.strip().split('\t')
        data = {'text': query}
        while True:
            try:
                response = requests.post(endpoint, headers=headers, params=params, data=data)
                json_response = response.json()
                if 'flaggedTokens' not in json_response:
                    continue
                break
            except Exception as e:
                print("An exception occurred: ", e)
                print(qid, query)

        # json_response = response.json()

        ft = json_response['flaggedTokens']
        correct_query = ""
        if len(ft) != 0:
            current_offset = 0
            for flag in ft:
                offset = flag["offset"]
                correct_query += query[current_offset:offset]
                correct_query += flag["suggestions"][0]["suggestion"]
                current_offset = len(flag["token"]) + offset
            correct_query += query[current_offset:]
            correct_query = correct_query.lower()
        else:
            correct_query = query
        wf.write(qid + '\t' + correct_query + '\n')

