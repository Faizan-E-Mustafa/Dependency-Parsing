import numpy as np
import itertools
from collections import defaultdict
from collections import Counter
from tqdm import tqdm
import pickle
import bz2

import dp


class Eisner:
    """Implementation of Eisner’s Algorithm"""

    def __init__(self):

        self.backtrack = defaultdict(dict)  # backtrack info for all 4 arrays
        self.backtrack_execution_result = {}  # {Dep : Head}

    def _initialize_arrays(self, no_tokens):
        import copy

        self.o_r = np.zeros((no_tokens, no_tokens))
        lower_triangle = np.tril_indices(no_tokens, -1)
        self.o_r[lower_triangle] = -9999

        self.o_l = copy.deepcopy(self.o_r)
        self.c_r = copy.deepcopy(self.o_r)
        self.c_l = copy.deepcopy(self.o_r)

    def fit(self, no_tokens, ml_score):
        """
        no_tokens(int): number of tokens in sentence.
        ml_score(np,array): scores from ml model.
        """

        self._initialize_arrays(no_tokens)
        self.backtrack = defaultdict(dict)  # backtrack info for all 4 arrays
        self.backtrack_execution_result = {}

        for m in range(1, no_tokens + 1):
            for s in range(no_tokens - m - 1 + 1):
                t = s + m

                # calculations for o_r
                result = [
                    self.c_l[s, q] + self.c_r[q + 1, t] + ml_score[t, s]
                    for q in range(s, t)
                ]
                max_index = np.argmax(result)
                max_value = result[max_index]
                self.o_r[s, t] = max_value
                q = s + max_index
                self.register_backtrack(s, t, q, matrix_type="o_r")

                # calculations for o_l
                result = [
                    self.c_l[s, q] + self.c_r[q + 1, t] + ml_score[s, t]
                    for q in range(s, t)
                ]
                max_index = np.argmax(result)
                max_value = result[max_index]
                self.o_l[s, t] = max_value
                q = s + max_index
                self.register_backtrack(s, t, q, matrix_type="o_l")

                # calculations for c_r
                result = [self.c_r[s, q] + self.o_r[q, t] for q in range(s, t)]
                max_index = np.argmax(result)
                max_value = result[max_index]
                self.c_r[s, t] = max_value
                q = s + max_index
                self.register_backtrack(s, t, q, matrix_type="c_r")

                # calculations for c_l
                result = [self.o_l[s, q] + self.c_l[q, t] for q in range(s + 1, t + 1)]
                max_index = np.argmax(result)
                max_value = result[max_index]
                self.c_l[s, t] = max_value
                q = s + 1 + max_index
                self.register_backtrack(s, t, q, matrix_type="c_l")

        return self.c_l[0][no_tokens - 1]

    def register_backtrack(self, s, t, q, matrix_type):
        self.backtrack[matrix_type][(s, t)] = q

    def execute_backtrack(self, i, j, matrix_type="c_l"):
        """Get backtract and generate dependency relations."""

        if i == j:
            # print(f"[RETURN] {matrix_type}(i = {i} , j = {j})")
            # print("=+" * 15)
            return None

        status, direction = matrix_type.split("_")
        q = self.backtrack[matrix_type][(i, j)]
        # print(f"{matrix_type}(i = {i} , j = {j} , q = {q})")
        # closed
        if status == "c":
            if direction == "r":
                self.execute_backtrack(q, j, matrix_type="o_r")
                self.execute_backtrack(i, q, matrix_type="c_r")
            elif direction == "l":
                self.execute_backtrack(q, j, matrix_type="c_l")
                self.execute_backtrack(i, q, matrix_type="o_l")

        # open
        elif status == "o":
            # print(i, j)
            if direction == "r":
                # j is head of i
                self.backtrack_execution_result[i] = j
            elif direction == "l":
                self.backtrack_execution_result[j] = i

            self.execute_backtrack(i, q, matrix_type="c_l")
            self.execute_backtrack(q + 1, j, matrix_type="c_r")

    def make_potential_trees(self, sentences):
        potential_trees = {}
        for sentence in sentences:
            if len(sentence) in potential_trees.keys():
                continue
            invalid_indices = [(i, i) for i in range(len(sentence) + 1)] + [
                (i, 0) for i in range(len(sentence) + 1)
            ]  # diagnol and coodinate where root is depenedent.

            potential_coordinates = [
                dp.core.Arc(head, dep)
                for dep in range(len(sentence) + 1)
                for head in range(len(sentence) + 1)
                if (head, dep) not in invalid_indices
            ]
            potential_tree = dp.core.Tree(potential_coordinates)
            potential_trees[len(sentence)] = potential_tree
        return potential_trees


class EisnerPerceptron:
    def __init__(
        self,
        vocab_size,
        eisner,
        feature_extraction,
        evaluation,
        dev_gold_sentences=None,
    ):
        self.vocab_size = vocab_size
        self.eisner = eisner
        self.feature_extraction = feature_extraction
        self.dev_gold_sentences = dev_gold_sentences
        self.evaluation = evaluation
        if dev_gold_sentences:
            self.dev_potential_trees = self.eisner.make_potential_trees(
                dev_gold_sentences
            )

    def _intitialize_model_scores(self, sentence_len):
        model_score = np.zeros((sentence_len + 1, sentence_len + 1))
        model_score[:, 0] = np.inf
        np.fill_diagonal(model_score, -9999)
        return model_score

    def train(self, sentences, trees, potential_trees, epochs, path):

        self.weight = np.zeros((self.vocab_size, 1))

        potential_tree_features_container = []
        for sentence in sentences:
            potential_tree = potential_trees[len(sentence)]
            potential_tree_features = self.feature_extraction.extract_feature_full(
                sentence, potential_tree
            )
            potential_tree_features_container.append(potential_tree_features)

        tree_features_container = []
        for sentence, tree in zip(sentences, trees):
            tree_features = self.feature_extraction.extract_feature_full(sentence, tree)
            tree_features_container.append(tree_features)

        dev_best = 0

        for epoch in range(epochs):
            result = []

            for sentence, tree, potential_tree_features, tree_features in tqdm(
                zip(
                    sentences,
                    trees,
                    potential_tree_features_container,
                    tree_features_container,
                ),
                total=len(sentences),
            ):
                potential_tree = potential_trees[len(sentence)]
                model_score = self._intitialize_model_scores(len(sentence))

                for arc, feature in zip(potential_tree.arcs, potential_tree_features):

                    model_score[(arc.head, arc.dep)] = sum(
                        [self.weight[idx] for idx in feature]
                    )

                self.eisner.fit(len(sentence) + 1, model_score)
                self.eisner.execute_backtrack(0, len(sentence))
                predicted_tree = self.eisner.backtrack_execution_result
                predicted_tree = dp.core.Tree.from_eisner(sentence, predicted_tree)

                if not (predicted_tree == tree):

                    prediction_features = self.feature_extraction.extract_feature_full(
                        sentence, predicted_tree
                    )

                    potential_tree_features = list(
                        itertools.chain.from_iterable(tree_features)
                    )
                    prediction_features = list(
                        itertools.chain.from_iterable(prediction_features)
                    )

                    features_counts = Counter(potential_tree_features)
                    prediction_features_counts = Counter(prediction_features)

                    for k, v in features_counts.items():
                        self.weight[k] = self.weight[k] + v
                    for k, v in prediction_features_counts.items():
                        self.weight[k] = self.weight[k] - v

                result.append(self.evaluation.UAS(tree, predicted_tree))

            if self.dev_gold_sentences:

                dev_result = self.test(
                    self.dev_potential_trees, self.dev_gold_sentences, load_from_path=""
                )

                if dev_result > dev_best:
                    print("Valid Score improved. Saving model .....")
                    dev_best = dev_result
                    name = "_".join(self.feature_extraction.use_templates)
                    filename_weight = path / f"weights_{name}.pkl"
                    filename_vocab = path / f"vocab_{name}.pkl"

                    dp.core.save_zipped_pickle(self.weight, filename_weight)
                    if epoch == 0:
                        dp.core.save_zipped_pickle(
                            self.feature_extraction.vocab, filename_vocab
                        )

                print(f"Epoch = {epoch+1} Train  = {np.mean(result)} Dev: {dev_result}")

    def load_saved(self, path):

        name = "_".join(self.feature_extraction.use_templates)
        filename_weight = path / f"weights_{name}.pkl"
        filename_vocab = path / f"vocab_{name}.pkl"

        self.weight = dp.core.load_zipped_pickle(filename_weight)
        self.feature_extraction.vocab = dp.core.load_zipped_pickle(filename_vocab)

    def test(
        self,
        dev_potential_trees,
        dev_gold_sentences=None,
        load_from_path="",
        test=False,
    ):
        predicted_trees = []
        if load_from_path:
            self.load_saved(load_from_path)

        dev_gold_trees = [s.to_tree() for s in dev_gold_sentences]

        result = []

        for sentence, gold_tree in tqdm(
            zip(dev_gold_sentences, dev_gold_trees), total=len(dev_gold_sentences),
        ):
            potential_tree = dev_potential_trees[len(sentence)]

            model_score = self._intitialize_model_scores(len(sentence))

            potential_tree_features = self.feature_extraction.extract_feature_full(
                sentence, potential_tree
            )

            for arc, feature in zip(potential_tree.arcs, potential_tree_features):
                model_score[(arc.head, arc.dep)] = sum(
                    [self.weight[idx] for idx in feature]
                )

            self.eisner.fit(len(sentence) + 1, model_score)
            self.eisner.execute_backtrack(0, len(sentence))
            predicted_tree = self.eisner.backtrack_execution_result
            predicted_tree = dp.core.Tree.from_eisner(sentence, predicted_tree)
            if test:
                predicted_trees.append(predicted_tree)
            else:
                result.append(self.evaluation.UAS(gold_tree, predicted_tree))
        if test:
            return predicted_trees
        else:
            return np.mean(result)
