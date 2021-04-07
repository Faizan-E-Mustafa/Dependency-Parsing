import bz2
import pickle


class Sentence:
    def __init__(self, sentence_id, data, test_sentence=True):
        self.sentence_id = sentence_id
        self.data = data
        self.tokens = []
        self.test_sentence = test_sentence
        self._add_tokens(data)

    def __repr__(self):
        token_text = [t.form for t in self.tokens]
        if len(token_text) < 8:
            return f"Len = {len(token_text)} --> {' '.join(token_text)}"
        else:
            return f"Len = {len(token_text)} --> {' '.join(token_text[:8])} ....."

    def __getitem__(self, idx):
        return self.tokens[idx]

    def __len__(self):
        return len(self.tokens)

    def _add_tokens(self, data):
        for token_id, row in enumerate(data):
            # self.tokens.append(Token(*['0'] + ['']*7))
            if self.test_sentence:
                import pdb

                pdb.set_trace()
                row[6] = "_"  # head is "_"
            self.tokens.append(Token(*row))

    def to_tree(self):
        tree_dictionary = {}
        for token in self.tokens:
            tree_dictionary[token.token_id] = token.head

        return Tree.from_dictionary(tree_dictionary)

    def get_head_info_from(self, predicted_tree):
        predicted_tree = sorted(
            [(arc.head, arc.dep) for arc in predicted_tree.arcs], key=lambda tup: tup[1]
        )
        for i, arc in enumerate(predicted_tree):
            arc_head = arc[0]
            self.data[i][-2] = arc_head
        return Sentence(self.sentence_id, self.data, self.test_sentence)


class Token:
    def __init__(self, token_id, form, lemma, pos, xpos, morph, head, relation):
        self.token_id = token_id
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.morph = morph
        self.head = head
        self.relation = relation
        self.xpos = xpos
        self.keys = [
            "token_id",
            "form",
            "lemma",
            "pos",
            "xpos",
            "morph",
            "head",
            "relation",
        ]

        try:
            direction_score = int(head) - int(token_id)
            self.direction = "right" if direction_score < 0 else "left"
            self.distance = abs(direction_score)
        except:
            pass

    def __repr__(self):
        self.complete_data = [
            self.token_id,
            self.form,
            self.lemma,
            self.pos,
            self.xpos,
            self.morph,
            self.head,
            self.relation,
        ]
        self.complete_data = dict(zip(self.keys, self.complete_data))
        r = [f"{k}:{v}" for k, v in self.complete_data.items()]
        return " | ".join(r)


from collections import Counter


class Tree:
    def __init__(self, arcs):
        """arcs: list of arcs"""

        self.arcs = arcs

    def __eq__(self, other):
        self_arcs = self.arcs
        other_arcs = other.arcs

        if len(self_arcs) != len(other_arcs):
            import pdb

            pdb.set_trace()
            return False

        self_arcs = [arc.mapping for arc in self_arcs]
        other_arcs = [arc.mapping for arc in other_arcs]

        return Counter(self_arcs) == Counter(other_arcs)

    def __len__(self):
        return len(self.arcs)

    def __repr__(self):
        return f"{self.arcs}"

    @classmethod
    def from_dictionary(cls, dictionary):
        cls.arcs = dictionary  # (Dep, Head)
        cls.arcs = [Arc(v, k) for k, v in cls.arcs.items()]  # (Head, Dep)
        return Tree(cls.arcs)

    @classmethod
    def from_eisner(cls, sentence, backtrack_execution_result):
        cls.sentence = sentence
        cls.arcs = backtrack_execution_result  # (Dep, Head)
        cls.arcs = [Arc(v, k) for k, v in cls.arcs.items()]  # (Head, Dep)
        return Tree(cls.arcs)

    @classmethod
    def from_arcstandard(cls, final_config_arcs):
        cls.arcs = [Arc(k, v) for k, v in final_config_arcs]  # (Head, Dep)
        return Tree(cls.arcs)


class Arc:
    def __init__(self, head, dep):
        try:
            self.head = int(head)
            self.dep = int(dep)
        except:
            self.head = head
            self.dep = dep
        self.mapping = (self.head, self.dep)

    def __repr__(self):
        return f"{self.head} -> {self.dep}"


def make_potential_trees(sentences):
    potential_trees = []
    for sentence in sentences:
        invalid_indices = [(i, i) for i in range(len(sentence) + 1)] + [
            (i, 0) for i in range(len(sentence) + 1)
        ]  # diagnol and coodinate where root is depenedent.

        potential_coordinates = [
            Arc(head, dep)
            for dep in range(len(sentence) + 1)
            for head in range(len(sentence) + 1)
            if (head, dep) not in invalid_indices
        ]
        potential_tree = Tree(potential_coordinates)
        potential_trees.append(potential_tree)
    return potential_trees


def save_zipped_pickle(obj, filename):

    with bz2.BZ2File(filename, "wb") as f:
        pickle.dump(obj, f)


def load_zipped_pickle(filename):
    with bz2.BZ2File(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object
