import numpy as np
import dp
from tqdm import tqdm


class Configuration:
    def __init__(self, stack, buffer, arcs):
        self.stack = stack
        self.buffer = buffer
        self.arcs = arcs  # set(dep,head)
        self.stack_top = stack[-1] if stack else stack
        self.buffer_front = buffer[0] if buffer else buffer

    @property
    def is_terminal(self):
        return self.buffer == []

    def do_left_arc(self):
        if self.can_left_arc():

            new_stack = self.stack[:-1]
            new_buffer = self.buffer
            #             import pdb; pdb.set_trace()
            self.arcs.add((self.buffer_front, self.stack_top))
            new_arcs = self.arcs.copy()
            return Configuration(new_stack, new_buffer, new_arcs), "left"
        else:
            raise Exception("Condition voilated for left arc")

    def can_left_arc(self):
        return self.stack_top != 0

    def do_right_arc(self):

        new_stack = self.stack[:-1]
        new_buffer = [self.stack_top] + self.buffer[1:]
        self.arcs.add((self.stack_top, self.buffer_front))
        new_arcs = self.arcs.copy()
        return Configuration(new_stack, new_buffer, new_arcs), "right"

    def can_shift(self):
        return len(self.buffer) >= 2 or self.stack == []

    def do_shift(self):
        if self.can_shift():
            new_stack = self.stack + [self.buffer_front]
            new_buffer = self.buffer[1:]
            new_arcs = self.arcs.copy()
            return Configuration(new_stack, new_buffer, new_arcs), "shift"
        else:
            raise Exception("Condition voilated for shift")


class Oracle:
    def __init__(self, start_config, correct_arcs):
        self.start_config = start_config
        self.correct_arcs = correct_arcs

    def execute(self):
        correct_sequence = []
        configs_sequence = []
        config = self.start_config
        # configs_sequence.append(config)
        while not (config.is_terminal):

            if self._should_left_arc(config, self.correct_arcs):
                trans_config, trans_name = config.do_left_arc()
            elif self._should_right_arc(config, self.correct_arcs):
                trans_config, trans_name = config.do_right_arc()
            else:
                trans_config, trans_name = config.do_shift()

            config = trans_config
            configs_sequence.append(config)
            correct_sequence.append(trans_name)
            # print(
            #     f"{trans_name} - stack {config.stack}\t - buffer {config.buffer}\t - arcs {config.arcs}"
            # )
            # print("=+" * 10)

        assert config.arcs == self.correct_arcs
        return correct_sequence, configs_sequence

    def _should_left_arc(self, config, correct_arcs):
        head = config.buffer_front
        dep = config.stack_top

        if head == [] or dep == []:  # takes care of last step
            return False

        if (head, dep) in correct_arcs:
            return True
        else:
            return False

    def _should_right_arc(self, config, correct_arcs):
        head = config.stack_top
        dep = config.buffer_front

        if head == [] or dep == []:  # takes care of last step
            return False

        if (head, dep) in correct_arcs and self._has_all_children(
            dep, config.arcs, correct_arcs
        ):
            return True
        else:
            return False

    def _has_all_children(self, buffer_front, predicted_arcs, correct_arcs):
        correct_head = [arc for arc in correct_arcs if arc[0] == buffer_front]
        correct_head_len = len(correct_head)

        if len(predicted_arcs) == 0:
            predict_head_len = 0
        else:
            predict_head = [arc for arc in predicted_arcs if arc[0] == buffer_front]
            predict_head_len = len(predict_head)

        return correct_head_len == predict_head_len


class ArcStandard:
    def __init__(self, start_config, model_scores):
        self.start_config = start_config
        self.model_scores = model_scores
        self.arc_standard_configs_sequence = []

    def execute(self):
        config = self.start_config
        score_extract_id = 0
        while not (config.is_terminal):
            current_model_scores = self._get_scores(score_extract_id)
            # if current_model_scores == None:
            #     break
            valid_trans = self._find_first_valid(
                current_model_scores, score_extract_id, config
            )

            last_trans_condition = config.stack == [] and config.buffer == [0]

            for trans_name in valid_trans:
                try:
                    if trans_name == "shift" or last_trans_condition:
                        trans_config, trans_name = config.do_shift()
                        break
                    elif trans_name == "left":
                        trans_config, trans_name = config.do_left_arc()
                        break
                    elif trans_name == "right":
                        trans_config, trans_name = config.do_right_arc()
                        break
                except Exception as e:
                    # print(e)
                    continue

            config = trans_config
            score_extract_id += 1
            self.arc_standard_configs_sequence.append(config)

            # print(
            #     f"{trans_name} - stack {config.stack}\t - buffer {config.buffer}\t - arcs {config.arcs}"
            # )
            # print("=+" * 10)

        return config

    def predict_next_config(self, config):
        score_extract_id = 0
        current_model_scores = self._get_scores(score_extract_id)

        valid_trans = self._find_first_valid(
            current_model_scores, score_extract_id, config
        )

        last_trans_condition = config.stack == [] and config.buffer == [0]

        for trans_name in valid_trans:
            try:
                if trans_name == "shift" or last_trans_condition:
                    trans_config, trans_name = config.do_shift()
                    break
                elif trans_name == "left":
                    trans_config, trans_name = config.do_left_arc()
                    break
                elif trans_name == "right":
                    trans_config, trans_name = config.do_right_arc()
                    break
            except Exception as e:
                # print(e)
                continue

        return trans_config

    def _get_scores(self, score_extract_id):
        # try:
        return self.model_scores[score_extract_id]
        # except:
        #     return None
        # terminal config not reached, probably model_scores are zeros and we do shift only.

    def _find_first_valid(self, current_model_scores, score_extract_id, config):
        # import pdb; pdb.set_trace()

        desc_predict_trans_idx = list((-np.array(current_model_scores)).argsort())
        valid_trans = []
        for predict_trans_idx in desc_predict_trans_idx:

            predict_trans_name = {0: "left", 1: "right", 2: "shift"}.get(
                predict_trans_idx
            )

            if np.count_nonzero(current_model_scores) == 0 and config.can_shift():
                entry = "shift"

            elif predict_trans_name == "shift" and config.can_shift():
                entry = predict_trans_name

            elif predict_trans_name == "left" and config.can_left_arc():
                entry = predict_trans_name

            elif predict_trans_name == "right":
                entry = predict_trans_name
            try:
                valid_trans.append(entry)
            except Exception as e:
                # print(e)
                continue

        return valid_trans


import logging

logging.basicConfig(filename="arc_stand.log", filemode="w", level=logging.DEBUG)


class ArcStandardPerceptron:
    def __init__(self, evaluation, dev_sentences):
        self.evaluation = evaluation
        self.dev_sentences = dev_sentences
        if dev_sentences:
            self.dev_trees = [s.to_tree() for s in dev_sentences]

    def _make_configs_from_oracle(self, sentences, trees):
        correct_sequences, configs_sequences = [], []

        for sentence, tree in zip(sentences, trees):
            stack = [0]
            buffer = list(range(1, len(sentence) + 1))
            arcs = set()
            correct_arcs = {(arc.head, arc.dep) for arc in tree.arcs}
            start_config = Configuration(stack, buffer, arcs)

            o = Oracle(start_config, correct_arcs)
            correct_sequence, configs_sequence = o.execute()
            assert len(correct_sequence) == len(configs_sequence)
            correct_sequences.append(correct_sequence)
            configs_sequences.append(configs_sequence)

        return correct_sequences, configs_sequences

    def _extract_features(self, fe, sentences, configs):
        features = []

        for i, sentence in enumerate(sentences):
            sentence_features = []
            sentence_configs = configs[i]
            for config in sentence_configs:
                vector = fe.extract_feature_arcstandard(sentence, config)
                sentence_features.append(vector)
            features.append(sentence_features)

        return features

    def _apply_arcstandard(self, sentence, model_scores):
        stack = [0]
        buffer = list(range(1, len(sentence) + 1))
        arcs = set()
        start_config = Configuration(stack, buffer, arcs)

        arcstand = ArcStandard(start_config, model_scores)
        final_config = arcstand.execute()
        return dp.core.Tree.from_arcstandard(final_config.arcs)

    def train(self, sentences, trees, template, epochs):
        import itertools

        correct_sequences, configs_sequences = self._make_configs_from_oracle(
            sentences, trees
        )

        self.fe = dp.feature_extraction.FeatureExtraction(
            sentences,
            None,
            template,
            "arc_standard",
            use_templates=["nivre", "nivre_bigram"],
            configs=configs_sequences,
        )
        # import pdb; pdb.set_trace()
        features = self._extract_features(self.fe, sentences, configs_sequences)

        # import pdb; pdb.set_trace()
        # logging.debug(f"len(features) = {len(features)} - {features[0]}")

        vocab_size = len(self.fe.vocab)
        self.weights = np.zeros((vocab_size, 3))
        for epoch in tqdm(range(epochs), total=epochs):
            uas_result = []
            index = 0
            for sentence, tree in zip(sentences, trees):
                predicted_sequence = []
                model_scores = []
                sentence_features = features[index]
                # print(sentence_features)
                sentence_correct_sequence = correct_sequences[index]
                assert len(sentence_features) == len(sentence_correct_sequence)
                for feature, y in zip(sentence_features, sentence_correct_sequence):

                    sparse_feature = np.zeros((vocab_size, 1)).flatten().astype(int)
                    sparse_feature.put(feature, np.ones(len(feature)))

                    result = self.weights.T @ sparse_feature.reshape((vocab_size, 1))
                    y_hat_arg = np.argmax(result)

                    y_hat = {0: "left", 1: "right", 2: "shift"}.get(y_hat_arg)
                    y_arg = {"left": 0, "right": 1, "shift": 2}.get(y)

                    predicted_sequence.append(y_hat)
                    model_scores.append(result.flatten().tolist())

                    if y_hat_arg != y_arg:

                        self.weights[:, y_arg] = (
                            self.weights[:, y_arg] + sparse_feature.flatten()
                        )
                        self.weights[:, y_hat_arg] = (
                            self.weights[:, y_hat_arg] - sparse_feature.flatten()
                        )
                    # logging.debug(f"[W] {sum(weights)}")
                    # print(f"[W] {sum(weights)}")
                # import pdb; pdb.set_trace()

                index += 1
                # print(model_scores)
                # print("----------------")
                predicted_tree = self._apply_arcstandard(sentence, model_scores)
                # import pdb;pdb.set_trace()
                # print(sum(self.weights))
                uas_result.append(self.evaluation.UAS(tree, predicted_tree))

            print(f"Epoch = {epoch} - UAS {np.mean(uas_result)}")
            if self.dev_sentences:
                self.test(self.dev_sentences, self.dev_trees)

        return self.weights

    def predict_next_config(self, config, model_scores):
        arcstand = ArcStandard(config, model_scores)
        next_config = arcstand.predict_next_config(config)
        next_config.arcs.update(config.arcs)
        return next_config

    def test(self, sentences, trees):

        vocab_size = len(self.fe.vocab)

        uas_result = []

        for sentence, tree in zip(sentences, trees):
            predicted_sequence = []

            stack = [0]
            buffer = list(range(1, len(sentence) + 1))
            arcs = set()
            start_config = Configuration(stack, buffer, arcs)

            config = start_config
            for _ in range(len(sentence)):
                model_scores = []
                # import pdb; pdb.set_trace()
                feature_vector = self.fe.extract_feature_arcstandard(sentence, config)

                sparse_feature = np.zeros((vocab_size, 1)).flatten().astype(int)
                sparse_feature.put(feature_vector, np.ones(len(feature_vector)))

                result = self.weights.T @ sparse_feature.reshape((vocab_size, 1))
                y_hat_arg = np.argmax(result)

                y_hat = {0: "left", 1: "right", 2: "shift"}.get(y_hat_arg)

                predicted_sequence.append(y_hat)
                model_scores.append(result.flatten().tolist())
                # import pdb; pdb.set_trace()
                next_config = self.predict_next_config(config, model_scores)
                config = next_config
                if config.is_terminal:
                    break

            predicted_tree = dp.core.Tree.from_arcstandard(config.arcs)
            # print(config.arcs)

            uas_result.append(self.evaluation.UAS(tree, predicted_tree))
        print(f"DEV - UAS {np.mean(uas_result)}")
