########## NEW: based on graphs/kg.py from pyrdf2vec

from typing import DefaultDict, Dict, List, Optional, Set, Tuple, Union
from typing import Any

import attr
import rdflib

from pyrdf2vec.graphs.vertex import Vertex

import string
import itertools
import json
import os

from pyrdf2vec.graphs import KG

@attr.s
class KGExtended(KG):
    _dict_nodes = dict()
    _id_node = 0

    _dict_relations = dict()
    _id_relation = 0
    def excel_cols():
        n = 1
        while True:
            yield from (''.join(group) for group in itertools.product(string.ascii_lowercase, repeat=n))
            n += 1
    # _alphabet_list = list(itertools.islice(excel_cols(), 100))
    _alphabet_list = list(itertools.islice(excel_cols(), 100000))

    _list_triples = list()

    _list_targets = list()

    counter = 0

    target_predicate = attr.ib(
        factory=set,
        type=Set[str],
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(str)
        ),
    )

    entities = attr.ib(
        factory=list,
        type=List[Any],
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(str)
        ),
    )

    labels = attr.ib(
        factory=list,
        type=List[str],
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(str)
        ),
    )

    dataset_home = attr.ib(
        default=None,
        type=Optional[str],
        validator=[
            attr.validators.optional(attr.validators.instance_of(str)),
        ],
    )

    dataset_name = attr.ib(
        default=None,
        type=Optional[str],
        validator=[
            attr.validators.optional(attr.validators.instance_of(str)),
        ],
    )

    facts_to_explain_home = attr.ib(
        default=None,
        type=Optional[str],
        validator=[
            attr.validators.optional(attr.validators.instance_of(str)),
        ],
    )

    def __attrs_post_init__(self):
        if self.location is not None:
            self._is_remote = self.location.startswith(
                "http://"
            ) or self.location.startswith("https://")

            if self._is_remote is True:
                self.connector = SPARQLConnector(
                    self.location, cache=self.cache
                )
            elif self.location is not None:
                print("Parsing RDF Graph...")
                g = rdflib.Graph().parse(self.location, format=self.fmt) ##### this parser is non-deterministic it seems
                print(f'Processing {len(g)} facts...')
                for subj, pred, obj in g:
                    self.add_fact(subj, pred, obj)
                    self.counter += 1
                    if self.counter % 10000 == 0:
                        print(f'Processed facts: {self.counter}')
                        # print(f'Processed facts: {self.counter}/{len(rdflib.Graph().parse(self.location, format=self.fmt))}.')

                    subj = Vertex(str(subj))
                    obj = Vertex(str(obj))
                    self.add_walk(
                        subj,
                        Vertex(
                            str(pred), predicate=True, vprev=subj, vnext=obj
                        ),
                        obj,
                    )
        self.generate_kelpie_files(self.dataset_home, self.dataset_name, self.facts_to_explain_home)

    def add_fact(self, subj, pred, obj) -> bool:
        subj, pred, obj = str(subj), str(pred), str(obj)
        if subj not in self._dict_nodes:
            self._dict_nodes[subj] = self._id_node
            self._id_node += 1
        if pred not in self._dict_relations:
            self._dict_relations[pred] = self._alphabet_list[self._id_relation]
            self._id_relation += 1
        if obj not in self._dict_nodes:
            self._dict_nodes[obj] = self._id_node
            self._id_node += 1
        if pred not in self.skip_predicates:
            self._list_triples.append([self._dict_nodes[subj], self._dict_relations[pred], self._dict_nodes[obj]])
        if pred in self.target_predicate:
            self._list_targets.append([self._dict_nodes[subj], self._dict_relations[pred], self._dict_nodes[obj]])
        return True

    def add_targets(self):
        for instance in zip(self.entities, self.labels):
            (target,) = self.target_predicate
            self._dict_relations[target] = self._alphabet_list[self._id_relation]
            self._list_targets.append([self._dict_nodes[instance[0]], self._dict_relations[target], self._dict_nodes[instance[1]]])
        return True

    def generate_kelpie_files(self, dataset_home, dataset_name, facts_to_explain_home):
        self.add_targets()
        self._id_relation += 1
        ##### don't think this is needed anymore, but need to remove lp models or else it gives size mismatch because
        ##### of list of triples but list targets can already be removed
        self._dict_nodes["dummy_node"] = self._id_node
        self._id_node += 1
        (target,) = self.target_predicate
        target_dummy_fact = [self._dict_nodes["dummy_node"], self._dict_relations[target], self._dict_nodes["dummy_node"]]
        self._list_triples.append(target_dummy_fact)
        # self._list_targets.append(target_dummy_fact)
        with open(os.path.join(dataset_home, 'entity2id.json'), 'w') as f:
            json.dump(self._dict_nodes, f, sort_keys=True, indent=4)
        with open(os.path.join(dataset_home, 'relation2id.json'), 'w') as f:
            json.dump(self._dict_relations, f, sort_keys=True, indent=4)
        with open(os.path.join(dataset_home, 'train.txt'), 'w') as f:
            for triple in self._list_triples:
                f.write(str(triple[0]) + '\t' + str(triple[1]) + '\t' + str(triple[2]) + '\n')
        with open(os.path.join(dataset_home, 'valid.txt'), 'w') as f: # JUST THE SAME AS TRAIN, LIKE THIS BECAUSE OF KELPIE CODE
            for triple in self._list_triples:
                f.write(str(triple[0]) + '\t' + str(triple[1]) + '\t' + str(triple[2]) + '\n')
        with open(os.path.join(dataset_home, 'test.txt'), 'w') as f: # JUST THE SAME AS TRAIN, LIKE THIS BECAUSE OF KELPIE CODE
            for triple in self._list_triples:
                f.write(str(triple[0]) + '\t' + str(triple[1]) + '\t' + str(triple[2]) + '\n')
        # with open(os.path.join(facts_to_explain_home, 'rdf2vec_' + dataset_name.lower() + '_all.csv'), 'w') as f:
        #     for triple in self._list_targets:
        #         f.write(str(triple[0]) + '\t' + str(triple[1]) + '\t' + str(triple[2]) + '\n')
        with open(os.path.join(facts_to_explain_home, 'rdf2vec_' + dataset_name.lower() + '_all.csv'), 'w') as f:
            dict_nodes_reverse = {str(v): k for k, v in self._dict_nodes.items()}
            dict_relations_reverse = {str(v): k for k, v in self._dict_relations.items()}
            for triple in self._list_targets:
                f.write(dict_nodes_reverse[str(triple[0])] + '\t' + dict_relations_reverse[str(triple[1])] + '\t' + dict_nodes_reverse[str(triple[2])] + '\n')