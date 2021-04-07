class Templates:
    def __init__(self):
        from collections import defaultdict

        self.templates = defaultdict(list)
        self.algo_names = set()
        self.template_names = set()
        self.store_chunks = {}  # for bigram or trigram

    def get_algo_templates(self, algo_name):
        return [v for k, v in self.templates.items() if k[0] == algo_name]

    def add_template(self, algo_name, template_name, template):
        self.templates[(algo_name, template_name)] += template
        self.algo_names.add(algo_name)
        self.template_names.add(template_name)

    def eisner_execute_template(self, template_name, head, dep, sentence):
        algo_name = "eisner"
        template = self.templates[(algo_name, template_name)]
        vocab = []

        direction_score = int(head) - int(dep)
        direction = "right" if direction_score < 0 else "left"
        distance = abs(direction_score)

        if template_name == "unigram":

            for step_name in template:
                #                 import pdb; pdb.set_trace()
                if step_name == "hform":
                    step_name_value = sentence.tokens[head - 1].form
                    entry = (
                        "hform:P_ROOT"
                        if head == 0
                        else f"hform:{step_name_value}+{direction}+{distance}"
                    )
                    vocab.append(entry)

                elif step_name == "hpos":
                    step_name_value = sentence.tokens[head - 1].pos
                    entry = (
                        "hpos:P_ROOT"
                        if head == 0
                        else f"hpos:{step_name_value}+{direction}+{distance}"
                    )
                    vocab.append(entry)
                elif step_name == "hform+hpos":
                    step_name_value = (
                        sentence.tokens[head - 1].form + sentence.tokens[head - 1].pos
                    )
                    entry = (
                        "hform+hpos:P_ROOT"
                        if head == 0
                        else f"hform+hpos:{step_name_value}+{direction}+{distance}"
                    )
                    vocab.append(entry)
                elif step_name == "dform":
                    step_name_value = sentence.tokens[dep - 1].form
                    entry = f"dform:{step_name_value}+{direction}+{distance}"
                    vocab.append(entry)
                elif step_name == "dpos":
                    step_name_value = sentence.tokens[dep - 1].pos
                    entry = f"dpos:{step_name_value}+{direction}+{distance}"
                    vocab.append(entry)
                elif step_name == "dform+dpos":
                    step_name_value = (
                        sentence.tokens[dep - 1].form + sentence.tokens[dep - 1].pos
                    )
                    entry = f"dform+dpos:{step_name_value}+{direction}+{distance}"
                    vocab.append(entry)

                self.store_chunks[step_name] = entry

        if template_name == "bigram":
            for step_name in template:
                step_names = step_name.split("+")
                entry = "+".join(
                    [self.store_chunks[step_name] for step_name in step_names]
                )
                vocab.append(entry)
        return vocab

    def arcstandard_execute_template(self, template_name, config, sentence):

        algo_name = "arc_standard"
        template = self.templates[(algo_name, template_name)]
        vocab = list()

        if template_name == "nivre":
            # TODO: add _NULL_ for when stepname is _
            for step_name in template:
                try:
                    if step_name == "B[0]-form":
                        entry = (
                            "B[0]-form:P_ROOT"
                            if config.buffer[0] == 0
                            else f"B[0]-form:{sentence.tokens[config.buffer[0]-1].form}"
                        )

                    elif step_name == "B[0]-lemma":
                        entry = (
                            "B[0]-lemma:P_ROOT"
                            if config.buffer[0] == 0
                            else f"B[0]-lemma:{sentence.tokens[config.buffer[0]-1].lemma}"
                        )

                    elif step_name == "B[0]-pos":
                        entry = (
                            "B[0]-pos:P_ROOT"
                            if config.buffer[0] == 0
                            else f"B[0]-pos:{sentence.tokens[config.buffer[0]-1].pos}"
                        )

                    #                     elif step_name == "B[0]-xpos":
                    #                         entry = "B[0]-xpos:P_ROOT" if config.buffer[0] == 0 else f"B[0]-xpos:{sentence.tokens[config.buffer[0]-1].xpos}"
                    #                         vocab.add(entry)
                    #                     elif step_name == "B[1]-xpos":
                    #                         entry = "B[1]-xpos:P_ROOT" if config.buffer[1] == 0 else f"B[1]-xpos:{sentence.tokens[config.buffer[1]-1].xpos}"
                    #                         vocab.add(entry)
                    #                     elif step_name == "B[2]-xpos":
                    #                         entry = "B[2]-xpos:P_ROOT" if config.buffer[2] == 0 else f"B[2]-xpos:{sentence.tokens[config.buffer[2]-1].xpos}"
                    #                         vocab.add(entry)
                    #                     elif step_name == "B[3]-xpos":
                    #                         entry = "B[3]-xpos:P_ROOT" if config.buffer[3] == 0 else f"B[3]-xpos:{sentence.tokens[config.buffer[3]-1].xpos}"
                    #                         vocab.add(entry)

                    elif step_name == "S[0]-form":
                        entry = (
                            "S[0]-form:P_ROOT"
                            if config.stack[-1] == 0
                            else f"S[0]-form:{sentence.tokens[config.stack[-1]-1].form}"
                        )

                    elif step_name == "S[0]-pos":
                        entry = (
                            "S[0]-pos:P_ROOT"
                            if config.stack[-1] == 0
                            else f"S[0]-pos:{sentence.tokens[config.stack[-1]-1].pos}"
                        )

                    elif step_name == "S[0]-lemma":
                        entry = (
                            "S[0]-lemma:P_ROOT"
                            if config.stack[-1] == 0
                            else f"S[0]-lemma:{sentence.tokens[config.stack[-1]-1].lemma}"
                        )

                    elif step_name == "B[1]-pos":
                        entry = (
                            "B[1]-pos:P_ROOT"
                            if config.buffer[1] == 0
                            else f"B[1]-pos:{sentence.tokens[config.buffer[1]-1].pos}"
                        )

                    elif step_name == "S[1]-pos":
                        entry = (
                            "S[1]-pos:P_ROOT"
                            if config.stack[-2] == 0
                            else f"S[1]-pos:{sentence.tokens[config.stack[-2]-1].pos}"
                        )

                    elif step_name == "ld(S[0])":
                        stack_value = config.stack[-1]
                        left_most = sorted(
                            [
                                head - dep
                                for head, dep in config.arcs
                                if head == stack_value
                            ],
                            reverse=True,
                        )[0]
                        entry = f"ld(S[0]):{sentence.tokens[left_most-1].pos}"

                    elif step_name == "rd(S[0])":
                        stack_value = config.stack[-1]
                        right_most = [
                            head - dep
                            for head, dep in config.arcs
                            if head == stack_value
                        ][0]
                        entry = f"rd(S[0]):{sentence.tokens[right_most-1].pos}"

                    elif step_name == "ld(B[0])":
                        buffer_value = config.buffer[0]
                        left_most = sorted(
                            [
                                head - dep
                                for head, dep in config.arcs
                                if head == buffer_value
                            ],
                            reverse=True,
                        )[0]
                        entry = f"ld(B[0]):{sentence.tokens[left_most-1].pos}"

                    elif step_name == "rd(B[0])":
                        buffer_value = config.buffer[0]
                        right_most = [
                            head - dep
                            for head, dep in config.arcs
                            if head == buffer_value
                        ][0]
                        entry = f"rd(B[0]):{sentence.tokens[right_most-1].pos}"

                    # import pdb; pdb.set_trace()
                    vocab.append(entry)
                    self.store_chunks[step_name] = entry

                except IndexError as e:
                    entry = (
                        f"{step_name}:__NULL__"  # index Not Found in our stack / buffer
                    )
                    vocab.append(entry)

                    #                     print(f"{step_name} == {e}")
                    continue
        if template_name == "nivre_bigram":
            for step_name in template:
                step_names = step_name.split("+")
                entry = "+".join(
                    [self.store_chunks[step_name] for step_name in step_names]
                )
                vocab.append(entry)

        return vocab
