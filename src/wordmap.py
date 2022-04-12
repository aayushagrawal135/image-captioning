import os
import json
import argparse


class WordMap:
    def __init__(self, args: argparse):
        word_map_filename = os.path.join(args.data_folder, f"WORDMAP_{args.data_name}.json")
        with open(word_map_filename, "r") as f:
            self.word_map = json.load(f)

    def get(self):
        return self.word_map
        pass

    def __len__(self):
        return len(self.word_map)

    def __getitem__(self, item):
        return self.word_map[item]
