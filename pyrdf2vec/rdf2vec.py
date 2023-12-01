from __future__ import annotations

import asyncio
import pickle
import time
from typing import List, Sequence, Tuple
from typing import Optional

import attr

from pyrdf2vec.embedders import Embedder, Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.typings import Embeddings, Entities, Literals, SWalk
from pyrdf2vec.walkers import RandomWalker, Walker

import json


@attr.s
class RDF2VecTransformer:
    """Transforms nodes in a Knowledge Graph into an embedding.

    Attributes:
        _embeddings: All the embeddings of the model.
            Defaults to [].
        _entities: All the entities of the model.
            Defaults to [].
        _is_extract_walks_literals: True if the session must be closed after
            the call to the `transform` function. False, otherwise.
            Defaults to False.
        _literals: All the literals of the model.
            Defaults to [].
        _pos_entities: The positions of existing entities to be updated.
            Defaults to [].
        _pos_walks: The positions of existing walks to be updated.
            Defaults to [].
        _walks: All the walks of the model.
            Defaults to [].
        embedder: The embedding technique.
            Defaults to Word2Vec.
        walkers: The walking strategies.
            Defaults to [RandomWalker(2, None)]
        verbose: The verbosity level.
            0: does not display anything;
            1: display of the progress of extraction and training of walks;
            2: debugging.
            Defaults to 0.

    """

    embedder = attr.ib(
        factory=lambda: Word2Vec(),
        type=Embedder,
        validator=attr.validators.instance_of(Embedder),  # type: ignore
    )

    walkers = attr.ib(
        factory=lambda: [RandomWalker(2)],  # type: ignore
        type=Sequence[Walker],
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(
                Walker  # type: ignore
            ),
            iterable_validator=attr.validators.instance_of(list),
        ),
    )

    verbose = attr.ib(
        kw_only=True,
        default=0,
        type=int,
        validator=attr.validators.in_([0, 1, 2]),
    )

    _is_extract_walks_literals = attr.ib(
        init=False,
        default=False,
        type=bool,
        repr=False,
        validator=attr.validators.instance_of(bool),
    )

    _embeddings = attr.ib(init=False, type=Embeddings, factory=list)
    _entities = attr.ib(init=False, type=Entities, factory=list)
    _literals = attr.ib(init=False, type=Literals, factory=list)
    _walks = attr.ib(init=False, type=List[List[SWalk]], factory=list)

    _pos_entities = attr.ib(init=False, type=List[str], factory=list)
    _pos_walks = attr.ib(init=False, type=List[int], factory=list)

    with_multiprocessing = attr.ib(
        kw_only=True,
        type=Optional[bool],
        default=True,
        validator=attr.validators.instance_of(bool),
    )

    ##### this could have been added to rdf2vec.py but then I don't think it would load models trained with original
    ##### pyrdf2vec library, but need to test
    # _filename = attr.ib(
    #     default=None,
    #     type=Optional[str],
    #     validator=[
    #         attr.validators.optional(attr.validators.instance_of(str)),
    #     ],
    # )

    def fit(
        self, walks: List[List[SWalk]], is_update: bool = False, mimic_entity = None, mimic_init_original = True
    ) -> RDF2VecTransformer:
        """Fits the embeddings based on the provided entities.

        Args:
            walks: The walks to fit.
            is_update: True if the new corpus should be added to old model's
                corpus, False otherwise.
                Defaults to False.

        Returns:
            The RDF2VecTransformer.

        """
        if self.verbose == 2:
            print(self.embedder)

        tic = time.perf_counter()
        self.embedder.fit(walks, is_update, mimic_entity, mimic_init_original)
        toc = time.perf_counter()

        if self.verbose >= 1:
            n_walks = sum([len(entity_walks) for entity_walks in walks])
            print(f"Fitted {n_walks} walks ({toc - tic:0.4f}s)")
            if len(self._walks) != len(walks):
                n_walks = sum(
                    [len(entity_walks) for entity_walks in self._walks]
                )
                print(
                    f"> {n_walks} walks extracted "
                    + f"for {len(self._entities)} entities."
                )
        return self

    def fit_transform(
        self, kg: KG, entities: Entities, is_update: bool = False
    ) -> Tuple[Embeddings, Literals]:
        """Creates a model and generates embeddings and literals for the
        provided entities.

        Args:
            kg: The Knowledge Graph.
            entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.
            is_update: True if the new corpus should be added to old model's
                corpus, False otherwise.
                Defaults to False.

        Returns:
            The embeddings and the literals of the provided entities.

        """
        self._is_extract_walks_literals = True
        self.fit(self.get_walks(kg, entities), is_update)
        return self.transform(kg, entities)

    def get_walks(self, kg: KG, entities: Entities) -> List[List[SWalk]]:
        """Gets the walks of an entity based on a Knowledge Graph and a
        list of walkers

        Args:
            kg: The Knowledge Graph.
            entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.

        Returns:
            The walks for the given entities.

        Raises:
            ValueError: If the provided entities aren't in the Knowledge Graph.

        """
        if kg.skip_verify is False and not kg.is_exist(entities):
            if kg.mul_req:
                asyncio.run(kg.connector.close())
            raise ValueError(
                "At least one provided entity does not exist in the "
                + "Knowledge Graph."
            )

        if self.verbose == 2:
            print(kg)
            print(self.walkers[0])

        walks: List[List[SWalk]] = []
        tic = time.perf_counter()
        for walker in self.walkers:
            walks += walker.extract(kg, entities, self.with_multiprocessing, self.verbose)
        toc = time.perf_counter()

        self._update(self._entities, entities)
        self._update(self._walks, walks)

        if self.verbose >= 1:
            n_walks = sum([len(entity_walks) for entity_walks in walks])
            print(
                f"Extracted {n_walks} walks "
                + f"for {len(entities)} entities ({toc - tic:0.4f}s)"
            )
        if (
            kg._is_remote
            and kg.mul_req
            and not self._is_extract_walks_literals
        ):
            asyncio.run(kg.connector.close())
        return walks

    def transform(
        self, kg: KG, entities: Entities
    ) -> Tuple[Embeddings, Literals]:
        """Transforms the provided entities into embeddings and literals.

        Args:
            kg: The Knowledge Graph.
            entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.

        Returns:
            The embeddings and the literals of the provided entities.

        """
        assert self.embedder is not None
        embeddings = self.embedder.transform(entities)

        tic = time.perf_counter()
        literals = kg.get_literals(entities, self.verbose)
        toc = time.perf_counter()

        self._update(self._embeddings, embeddings)
        if len(literals) > 0:
            self._update(self._literals, literals)

        if kg._is_remote and kg.mul_req:
            self._is_extract_walks_literals = False
            asyncio.run(kg.connector.close())

        if self.verbose >= 1 and len(literals) > 0:
            print(
                f"Extracted {len(literals)} literals for {len(entities)} "
                + f"entities ({toc - tic:0.4f}s)"
            )
        return embeddings, literals

    def save(self, filename: str = "transformer_data") -> None:
        """Saves a RDF2VecTransformer object.

        Args:
            filename: The binary file to save the RDF2VecTransformer object.

        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def _update(self, attr, values) -> None:
        """Updates an attribute with a variable.

        This method is useful to keep all entities, walks, literals and
        embeddings after several online training.

        Args:
            attr: The attribute to update
            var: The new values to add.

        """
        if attr is None:
            attr = values
        elif isinstance(values[0], str):
            for i, entity in enumerate(values):
                if entity not in attr:
                    attr.append(entity)
                else:
                    self._pos_entities.append(attr.index(entity))
                    self._pos_walks.append(i)
        else:
            tmp = values
            for i, pos in enumerate(self._pos_entities):
                attr[pos] = tmp.pop(self._pos_walks[i])
            attr += tmp

    # @staticmethod
    def load(self, filename: str = "transformer_data") -> RDF2VecTransformer:
        """Loads a RDF2VecTransformer object.

        Args:
            filename: The binary file to load the RDF2VecTransformer object.

        Returns:
            The loaded RDF2VecTransformer.

        """

        with open(filename, "rb") as f:
            transformer = pickle.load(f)
            if not isinstance(transformer, RDF2VecTransformer):
                raise ValueError(
                    "Failed to load the RDF2VecTransformer object"
                )
            self._filename = filename
            return transformer






    def fit_transform_external_walks(
        self, walks, entities: Entities, is_update: bool = False, mimic_init_original: bool = True
        ) -> Tuple[Embeddings, Literals]:
        """Creates a model and generates embeddings and literals for the
        provided entities.

        Args:
            kg: The Knowledge Graph.
            entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.
            is_update: True if the new corpus should be added to old model's
                corpus, False otherwise.
                Defaults to False.

        Returns:
            The embeddings and the literals of the provided entities.

        """
        self._is_extract_walks_literals = True
        walks = self.add_walks(walks, entities)
        self.fit(walks, is_update, mimic_entity=entities, mimic_init_original=mimic_init_original)
        return self.transform_no_literals(entities)

    def add_walks(self, walks, entities: Entities):
        """Gets the walks of an entity based on a Knowledge Graph and a
        list of walkers

        Args:
            kg: The Knowledge Graph.
            entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.

        Returns:
            The walks for the given entities.

        Raises:
            ValueError: If the provided entities aren't in the Knowledge Graph.

        """
        # if kg.skip_verify is False and not kg.is_exist(entities):
        #     if kg.mul_req:
        #         asyncio.run(kg.connector.close())
        #     raise ValueError(
        #         "At least one provided entity does not exist in the "
        #         + "Knowledge Graph."
        #     )

        # if self.verbose == 2:
        #     print(kg)
        #     print(self.walkers[0])

        # walks: List[List[SWalk]] = []
        # tic = time.perf_counter()
        # for walker in self.walkers:
        #     walks += walker.extract(kg, entities, self.verbose)
        # toc = time.perf_counter()

        self._update(self._entities, entities)
        self._update(self._walks, walks)

        if self.verbose >= 1:
            n_walks = sum([len(entity_walks) for entity_walks in walks])
            print(
                f"Added {n_walks} walks"
                # + f"for {len(entities)} entities ({toc - tic:0.4f}s)"
            )
        # if (
        #     kg._is_remote
        #     and kg.mul_req
        #     and not self._is_extract_walks_literals
        # ):
        #     asyncio.run(kg.connector.close())
        return walks

    def transform_no_literals(
        self, entities: Entities
        ) -> Tuple[Embeddings, Literals]:
        """Transforms the provided entities into embeddings and literals.

        Args:
            kg: The Knowledge Graph.
            entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.

        Returns:
            The embeddings and the literals of the provided entities.

        """
        assert self.embedder is not None
        embeddings = self.embedder.transform(entities)

        # tic = time.perf_counter()
        # literals = kg.get_literals(entities, self.verbose)
        # toc = time.perf_counter()

        self._update(self._embeddings, embeddings)
        # if len(literals) > 0:
        #     self._update(self._literals, literals)

        # if kg._is_remote and kg.mul_req:
        #     self._is_extract_walks_literals = False
        #     asyncio.run(kg.connector.close())

        # if self.verbose >= 1 and len(literals) > 0:
        #     print(
        #         f"Extracted {len(literals)} literals for {len(entities)} "
        #         + f"entities ({toc - tic:0.4f}s)"
        #     )
        return embeddings

    def _entities_walks_triples(self):
        """
            Dict of entities walks already processed as list of triples using
            get_my_entity_walks_triples method.
        """
        raise NotImplementedError("To implement soon.")

    def get_entities_walks_triples(self):
        """
            Get the data in _entities_walks_triples.
        """
        raise NotImplementedError("To implement soon.")

    ##### getting error
    # def file_path_data():
    #     file_path_data = 'data/AIFB'

    #     return file_path_data

    ##### also in update_rdf2vec...
    def get_idx_of_entity_to_explain(self, entity_to_explain):
        return self._entities.index(entity_to_explain)

    ##### also in update_rdf2vec...
    ##### transform the list of triples in walks to ids
    def merge(self, dict1, dict2):

        res = {**dict1, **dict2}
        return res

    ##### also in update_rdf2vec...
    ##### transform a walk in a list of triples of that walk
    def walks_to_lists_of_simple_walks(self, my_entity_walks):
        """
            Transforms a list of walks in a list of list of triples of those walks e.g. [(a, b, c, d, e), (a, f, g)] to
            [[(a, b, c), (c, d, e)],[(a, f, g)]]
        """

        my_entity_walks_triples = []
        # print('my_entity_walks')
        # print(my_entity_walks)
        for walk in my_entity_walks:
            # print('walk')
            # print(walk)
            walk_triples = []
            for idx in range(2, len(walk), 2):
                walk_triples.append((walk[idx-2], walk[idx-1], walk[idx]))
            my_entity_walks_triples.append(walk_triples)
        
        return my_entity_walks_triples

    ##### also in update_rdf2vec...
    def lists_of_simple_walks_to_list_of_triples(self, my_entity_walks_triples):
        """
            Transforms a list of list of triples of walks to a list of triples
        """

        my_entity_walks_triples_simple_list = [triple for walk_triple in my_entity_walks_triples for triple in walk_triple]

        return my_entity_walks_triples_simple_list

    def get_my_entity_walks_triples(self, entity_to_explain, entity2id_path, relation2id_path, include_entity_walks_in_other_entities=False):
        """
            Transforms a list of triples original names to ids
        """
        entity_and_relation_same_encoding_allowed = True ##### for example, in the INT_DATA entities2id and relation2id both
                                                 ##### use numbers in the dict

        ##### this block was repeated I think, it seems it was using "_hmimic" entitiy instead of original
        # my_entity_walks = self.get_my_entity_walks()
        # my_entity_walks_triples = self.walks_to_lists_of_simple_walks(my_entity_walks)
        # # Get list of triples in walks to feed to kelpie topology-prefilter
        # my_entity_walks_triples_simple_list = self.lists_of_simple_walks_to_list_of_triples(my_entity_walks_triples)

        # load dictionary to convert list of triples in walks to id to be read by kelpie topology_prefilter
        # file_path_data, file_path_facts = get_my_paths()
        # with open(os.path.join(str(self.file_path_data), 'entity2id.json'), 'r') as f:
        with open(entity2id_path, 'r') as f:
            dict_nodes = json.load(f)
        # with open(os.path.join(str(self.file_path_data), 'relation2id.json'), 'r') as f:
        with open(relation2id_path, 'r') as f:
            dict_relations = json.load(f)
        dict_all = self.merge(dict_nodes, dict_relations)

        my_entity_walks = self._walks[self.get_idx_of_entity_to_explain(entity_to_explain)]
        # print(my_entity_walks)
        # raise Exception

        ##### to include walks in other entities walks, this is used in myKelpie/.../random_walks_prefilter to list all
        ##### the walks in which an entity shows up to then filter further
        if include_entity_walks_in_other_entities:
            # other_entities_list = self.transformer._entities[:]
            other_entities_list = self._entities[:]
            other_entities_list.remove(entity_to_explain)
            for entity in other_entities_list:
                for walk in self._walks[self.get_idx_of_entity_to_explain(entity)]:
                    if entity_to_explain in walk: 
                        my_entity_walks.append(walk)

        # print(len(my_entity_walks))
        # raise Exception        
        
        my_entity_walks_triples = self.walks_to_lists_of_simple_walks(my_entity_walks)
        # print('self.get_idx_of_entity_to_explain(entity_to_explain)')
        # print(self.get_idx_of_entity_to_explain(entity_to_explain))        
        # print('self._walks[self.get_idx_of_entity_to_explain(entity_to_explain)]')
        # print(self._walks[self.get_idx_of_entity_to_explain(entity_to_explain)])
        # print('my_entity_walks_triples')
        # print(my_entity_walks_triples)
        my_entity_walks_triples_simple_list = self.lists_of_simple_walks_to_list_of_triples(my_entity_walks_triples)
        my_entity_walks_triples_simple_list_ids = []
        if entity_and_relation_same_encoding_allowed:
            # print('my_entity_walks_triples_simple_list')
            # print(my_entity_walks_triples_simple_list)
            for triple in my_entity_walks_triples_simple_list:
                # print('triple')
                # print(triple)
                # raise Exception
                new_subj = triple[0].replace(triple[0], str(dict_nodes[triple[0]]))
                new_rel = triple[1].replace(triple[1], str(dict_relations[triple[1]]))
                new_obj = triple[2].replace(triple[2], str(dict_nodes[triple[2]]))
                new_triple = tuple([new_subj, new_rel, new_obj])
                my_entity_walks_triples_simple_list_ids.append(new_triple)
        else:
            for triple in my_entity_walks_triples_simple_list:
                new_triple = tuple([item.replace(item, str(dict_all[item])) for item in list(triple)])
                my_entity_walks_triples_simple_list_ids.append(new_triple)
        # my_entity_walks = [my_entity_walks]

        return my_entity_walks_triples_simple_list_ids

    def reload(self) -> RDF2VecTransformer:
        """Loads a RDF2VecTransformer object.

        Args:
            filename: The binary file to load the RDF2VecTransformer object.

        Returns:
            The loaded RDF2VecTransformer.

        """
        try:
            with open(self._filename, "rb") as f:
                transformer = pickle.load(f)
                if not isinstance(transformer, RDF2VecTransformer):
                    raise ValueError(
                        "Failed to load the RDF2VecTransformer object"
                    )
                return transformer
        except:
            raise Exception("RDF2VecTransformer cannot be reloaded without being previously loaded.")