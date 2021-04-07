import pandas as pd

from . import core


class Dataset:
    def make_df_splits(self, df):
        df_splits = np.split(df, df[df.isnull().all(1)].index)
        df_splits = [
            split[split.notnull().all(1).values].astype({"Id": int, "Head": int})
            for split in df_splits
        ]
        return df_splits[:-1]

    def sentences_from_splits(self, df_splits, test_sentence):
        """df_splits: list of pd.DataFrame"""

        sentences = []
        for split_no, splits in enumerate(df_splits):
            sentences.append(
                core.Sentence(
                    split_no,
                    data=df_splits[split_no].values.tolist(),
                    test_sentence=test_sentence,
                )
            )

        return sentences

    def splits_from_sentences(self, sentences):
        """sentences: list of Sentence objects"""

        return [pd.DataFrame(sentence.data) for sentence in sentences]
