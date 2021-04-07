class Evaluation:
    def UAS(self, gold_tree, predicted_tree):
        gold_arcs = [i.mapping for i in gold_tree.arcs]
        correct = sum(
            [
                True
                for predict_arc in predicted_tree.arcs
                if predict_arc.mapping in gold_arcs
            ]
        )
        return correct / len(gold_tree)
