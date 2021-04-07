from pathlib import Path
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, columns=None):
        if columns == None:
            self.columns = [
                "Id",
                "Form",
                "Lemma",
                "POS",
                "XPos",
                "Morph",
                "Head",
                "Rel",
                "_",
                "_",
            ]
        else:
            self.columns = columns

    def load_data(self, path):
        df = pd.read_csv(path, sep="\t", header=None, skip_blank_lines=False)
        df.columns = self.columns
        df = df.iloc[:, :8]
        return df

    def read_conll(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
            sentences = []
            sentence = []
            for line in lines:
                if line == "\n":
                    sentences.append(pd.DataFrame(sentence))
                    sentence = []
                else:
                    line = line.strip("\n")
                    sentence.append(line.split("\t")[:8])
            return sentences

    def to_conll_df(self, filename, df_splits):
        df_splits = [
            split.astype("str").append(pd.Series(dtype="object"), ignore_index=True)
            for split in df_splits
        ]
        df = pd.concat(df_splits, axis=0)

        df.to_csv(filename + ".conll06", index=False, sep="\t", header=False)

    def to_conll(self, path, df_splits):
        df_splits = [split.astype("str") for split in df_splits]
        with open(path, "w") as f:
            for split in df_splits:
                rows = split.values.tolist()
                for row in rows:
                    f.write("\t".join(row))
                    f.write("\n")
                f.write("\n")
