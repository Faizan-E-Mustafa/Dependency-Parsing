from tqdm import tqdm


class FeatureExtraction:
    def __init__(
        self, sentences, potential_trees, template, algo_name, use_templates, configs,
    ):
        self.template = template
        self.use_templates = use_templates

        if algo_name == "eisner":
            self._make_vocab(sentences, potential_trees, template)
        elif algo_name == "arc_standard":

            self._make_vocab_arcstandard(sentences, configs, template)

    def extract_feature(self, sentence, arc):
        arc_vocab = []
        for template_name in self.use_templates:
            temp_arc_vocab = self.template.eisner_execute_template(
                template_name, arc.head, arc.dep, sentence
            )
            arc_vocab.extend(temp_arc_vocab)
        #             print(arc_vocab)
        sparse_vector = self._arc_to_sparse_vector(arc_vocab)

        return sparse_vector

    def extract_feature_full(self, sentence, tree):
        arc_sparse_vectors = []
        for arc in tree.arcs:
            sparse_vector = self.extract_feature(sentence, arc)
            arc_sparse_vectors.append(sparse_vector)
        return arc_sparse_vectors

    def extract_feature_arcstandard(self, sentence, config):
        config_vocab = []
        for template_name in self.use_templates:
            temp_config_vocab = self.template.arcstandard_execute_template(
                template_name, config, sentence
            )

            config_vocab.extend(temp_config_vocab)
        # import pdb; pdb.set_trace()
        sparse_vector = self._arc_to_sparse_vector(config_vocab)
        return sparse_vector

    def extract_feature_arcstandard_full(self, sentence, configs):
        config_sparse_vectors = []
        for config in configs:
            sparse_vector = self.extract_feature_arcstandard(sentence, config)
            config_sparse_vectors.append(sparse_vector)
        return config_sparse_vectors

    def _arc_to_sparse_vector(self, arc_vocab):
        arc_vocab_indexes = []

        for entry in arc_vocab:
            if entry in self.vocab:
                arc_vocab_indexes.append(self.vocab.get(entry))

            else:  # test
                # arc_vocab_indexes.append(self.vocab.get("__NULL__"))
                continue
                # raise Exception

        return arc_vocab_indexes

    def _make_vocab(self, sentences, potential_trees, template):
        self.vocab = dict()

        for sentence in tqdm(sentences, total=len(sentences)):
            tree = potential_trees[len(sentence)]
            for arc in tree.arcs:
                for template_name in self.use_templates:
                    vocab = template.eisner_execute_template(
                        template_name, arc.head, arc.dep, sentence
                    )
                    for word in vocab:
                        if word in self.vocab:
                            continue
                        self.vocab[word] = self.vocab.get(word, len(self.vocab))

    def _make_vocab_arcstandard(self, sentences, configs, template):
        self.vocab = dict()
        for sentence, sentence_config in zip(sentences, configs):
            for sentence_config in sentence_config:
                for template_name in self.use_templates:
                    vocab = template.arcstandard_execute_template(
                        template_name, sentence_config, sentence
                    )

                    for word in vocab:
                        if word in self.vocab:
                            continue
                        self.vocab[word] = self.vocab.get(word, len(self.vocab))

        self.vocab["__NULL__"] = self.vocab.get("__NULL__", len(self.vocab))
