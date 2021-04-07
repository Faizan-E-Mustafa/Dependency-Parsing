from pathlib import Path

DATA_ROOT = "data"
OUTPUT = DATA_ROOT / Path("outputs")
ENG_TRAIN_1K = DATA_ROOT / Path(
    "english/train/wsj_train.only-projective.first-1k.conll06"
)
ENG_TRAIN_5K = DATA_ROOT / Path(
    "english/train/wsj_train.only-projective.first-5k.conll06"
)
ENG_TEST = DATA_ROOT / Path("english/test/wsj_test.conll06.blind")
ENG_DEV_GOLD = DATA_ROOT / Path("english/dev/wsj_dev.conll06.gold")
ENG_TRAIN_FULL = DATA_ROOT / Path("english/train/wsj_train.only-projective.conll06")

DE_TRAIN_FULL = DATA_ROOT / Path("german/train/tiger-2.2.train.only-projective.conll06")
DE_DEV_GOLD = DATA_ROOT / Path("german/dev/tiger-2.2.dev.conll06.gold")
DE_TEST = DATA_ROOT / Path("german/test/tiger-2.2.test.conll06.blind")
