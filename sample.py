from pathlib import Path

import pandas as pd
import numpy as np

import dp.config
import dp.core
import dp.data_loader
import dp.dataset
import dp.models.eisner

def data_handler():
    dl = dp.data_loader.DataLoader()
    ds = dp.dataset.Dataset()

    df_splits = dl.read_conll(Path(dp.config.ENG_TRAIN))

    sentences = ds.sentences_from_splits(df_splits)

    df_splits = ds.splits_from_sentences(sentences)

    dummy_splits = []
    for split in df_splits:
        split["dummy1"] = "_"
        split["dummy2"] = "_"
        dummy_splits.append(split)

    dl.to_conll(dp.config.OUTPUT / Path("dummy.conll06"), dummy_splits)

def model():
    ml_score = np.array([[-99, 9, 10, 9], [np.inf, -99, 20, 3], [np.inf, 30, -99, 30], [np.inf,11,0 ,-99]])
    no_tokens = 4

    eis = dp.models.eisner.Eisner()
    eis.fit(no_tokens, ml_score)

    eis.execute_backtrack(0, no_tokens-1)

    print(eis.backtrack)

def run():
    data_handler()
    model()

if __name__ == "__main__":
    run()
