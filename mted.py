import json
import os
from copy import deepcopy
from glob import glob
from itertools import combinations
from typing import Dict, List, Set, Tuple

import streamlit as st

LANGUAGES = ['Russian', 'Portuguese', 'Polish', 'German', 'English', 'Turkish', 'Lithuanian', 'Chinese']
TALK_IDS = ['talk_1927', 'talk_1971', 'talk_1976', 'talk_1978', 'talk_2009', 'talk_2150']


class Sentence:
    """Stores sentence-level information"""

    def __init__(self, sentence: dict, sentence_index: int):
        self.sentence = sentence['sentence']
        self.sentence_index = sentence_index
        self.language = sentence['language']
        self.en_translation = sentence['en_translation']
        self.intra_annotations = []
        self.inter_annotations_as_arg1 = []
        self.inter_annotations_as_arg2 = []

    def __repr__(self) -> str:
        # return self.sentence
        return self.en_translation

    def add_annotation(self, inter_or_intra: str, arg1_or_arg2: str, annot: dict) -> None:
        if inter_or_intra == 'intra':
            self.intra_annotations.append(annot)
        else:
            if arg1_or_arg2 == 'arg1':
                self.inter_annotations_as_arg1.append(annot)
            else:
                self.inter_annotations_as_arg2.append(annot)


class Talk:
    """A language-specific Talk"""

    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            talk = json.load(f)
        self.talk_id = talk['talk_id']
        self.language = talk['language']
        self.annotations = talk['annotations']
        self.sentences = self._load_sentences_from_json(talk['sentences'])

    def _load_sentences_from_json(self, dict_sentences: List[dict]) -> List[Sentence]:
        sentences = [Sentence(sent, index) for index, sent in enumerate(dict_sentences)]
        for annot in self.annotations:
            arg1_sent_index = annot['arg1_sentence_index']
            arg2_sent_index = annot['arg2_sentence_index']
            if annot['inter_or_intra'] == 'intra':
                sentences[arg1_sent_index].add_annotation('intra', None, annot)
            else:
                sentences[arg1_sent_index].add_annotation('inter', 'arg1', annot)
                sentences[arg2_sent_index].add_annotation('inter', 'arg2', annot)
        return sentences


class MultilingualTalk:
    """Stores all language-specific `Talk`s"""

    def __init__(self, talk_id: str):
        self.talk_id = talk_id
        self.talks = {}
        self.pairwise_alignments = {}

    def add_talk(self, talk: Talk) -> None:
        self.talks[talk.language] = talk

    def set_pairwise_alignments(self, en_to_xx_alignments: dict) -> None:
        """Sets sentence-level cross-lingual alignments"""
        # set EN-XX alignments
        for language, aligns in en_to_xx_alignments.items():
            self.pairwise_alignments[('English', language)] = [[set(ens), set(xxs)] for ens, xxs in aligns]
        # set XX-YY using EN as pivot
        for lang1, lang2 in self.get_all_lang_pairs(except_langs=['English']):
            self._set_xx_to_yy_alignments(lang1, lang2)
        # add reverse directions
        reversed_alignments = {}
        for (lang1, lang2), alignments in self.pairwise_alignments.items():
            rev_aligns = [[lang2_inds, lang1_inds] for lang1_inds, lang2_inds in alignments]
            reversed_alignments[(lang2, lang1)] = rev_aligns
        for (lang2, lang1), rev_aligns in reversed_alignments.items():
            self.pairwise_alignments[(lang2, lang1)] = rev_aligns

    def _set_xx_to_yy_alignments(self, xx: str, yy: str) -> None:
        """Sets alignments for XX-YY using EN-XX and EN-YY alignments"""

        def _match_en_indices(ex_ens: Set[int], en_to_yy: List[List[Set[int]]], ey_index: int = 0) \
                              -> Tuple[str, Set[int], int]:
            while ey_index < len(en_to_yy):
                ey_ens, _ = en_to_yy[ey_index]
                if ex_ens == ey_ens:
                    return 'matched', ey_ens, ey_index
                if ex_ens.issubset(ey_ens):
                    return 'ex is in ey', ey_ens, ey_index
                if ey_ens.issubset(ex_ens):
                    return 'ey is in ex', ey_ens, ey_index
                ey_index += 1
            return 'no match', None, None

        def _retrive_xx_to_yy_pairs(en_to_xx: List[List[Set[int]]],
                                    en_to_yy: List[List[Set[int]]],
                                    matched_ens: List[List[int]]) -> List[List[Set[int]]]:
            xxyy_by_ens = {}
            for ens in matched_ens:
                ens = set(ens)
                ens_tuple = tuple(sorted(list(ens)))
                xxyy_by_ens[ens_tuple] = [[], []]
                for ex_ens, ex_xxs in en_to_xx:
                    if ex_ens.issubset(ens):
                        xxyy_by_ens[ens_tuple][0].extend(list(ex_xxs))
                for ey_ens, ey_yys in en_to_yy:
                    if ey_ens.issubset(ens):
                        xxyy_by_ens[ens_tuple][1].extend(list(ey_yys))
            return [[set(xx), set(yy)] for _, (xx, yy) in xxyy_by_ens.items()]

        # en_to_xx: [ [ {en_i, ..}, {xx_j, ..} ], ..]
        en_to_xx = deepcopy(self.pairwise_alignments[('English', xx)])
        en_to_yy = deepcopy(self.pairwise_alignments[('English', yy)])

        # shift the beginning of either list so that the very starting EN indices are the same for both
        first_ex_en = sorted(list(en_to_xx[0][0]))[0]
        first_ey_en = sorted(list(en_to_yy[0][0]))[0]
        start_index = 0
        if first_ex_en >= first_ey_en:
            while sorted(list(en_to_yy[start_index][0]))[0] < first_ex_en:
                start_index += 1
            en_to_yy = en_to_yy[start_index:]
        else:
            while sorted(list(en_to_xx[start_index][0]))[0] < first_ey_en:
                start_index += 1
            en_to_xx = en_to_xx[start_index:]

        ex_ens_buffer = set()
        ey_ens_buffer = set()
        ex_index = 0
        ey_index = 0
        matched_ens = []
        while ex_index < len(en_to_xx):
            ex_ens, _ = en_to_xx[ex_index]
            ex_ens_buffer.update(ex_ens)
            case, ey_ens, returned_ey_index = _match_en_indices(ex_ens_buffer, en_to_yy, ey_index=ey_index)
            if ex_ens_buffer == ey_ens_buffer:
                matched_ens.append(list(ex_ens_buffer))
                ex_ens_buffer.clear()
                ey_ens_buffer.clear()
                ex_index += 1
                continue
            if case == 'matched':
                matched_ens.append(list(ex_ens_buffer))
                ex_ens_buffer.clear()
                ey_ens_buffer.clear()
                ex_index += 1
                continue
            if case == 'ex is in ey':
                ex_index += 1
                continue
            if case == 'ey is in ex':
                ey_ens_buffer.update(ey_ens)
                ey_index = returned_ey_index + 1
                continue
            if case == 'no match':
                ex_ens_buffer.difference_update(ex_ens)
                ex_index += 1
                continue

        self.pairwise_alignments[(xx, yy)] = _retrive_xx_to_yy_pairs(en_to_xx, en_to_yy, matched_ens)

    def get_all_langs(self) -> List[str]:
        return list(self.talks.keys())

    def get_num_langs(self) -> int:
        return len(self.get_all_langs())

    def get_all_lang_pairs(self, except_langs: List[str] = None) -> List[Tuple[str, str]]:
        if except_langs is None:
            langs = []
        else:
            langs = [lang for lang in self.get_all_langs() if lang not in except_langs]
        return combinations(langs, 2)

    def get_pairwise_aligned_relations(self, xx: str, yy: str) -> List[Tuple[List[Dict], List[Dict]]]:
        def _get_relations_for_indices(indices, sentences):
            relations = []
            for index in indices:
                sent = sentences[index]
                rels = sent.intra_annotations + sent.inter_annotations_as_arg1
                relations.extend(rels)
            return relations

        alignments = self.pairwise_alignments[(xx, yy)]
        xx_sentences = self.talks[xx].sentences
        yy_sentences = self.talks[yy].sentences
        aligned_relations = []
        for xx_inds, yy_inds in alignments:
            xx_relations = _get_relations_for_indices(xx_inds, xx_sentences)
            yy_relations = _get_relations_for_indices(yy_inds, yy_sentences)
            aligned_relations.append((xx_relations, yy_relations))
        return aligned_relations

    def get_pairwise_aligned_relation_type_and_senses(self, xx: str, yy: str) -> List[Tuple[Dict, Dict]]:
        def _get_relation_types_and_senses(relations):
            rels_type = []
            rels_first = []
            rels_second = []
            rels_first_and_second = []
            rels_all_three = []
            for rel in relations:
                rel_type = rel['relation_type']
                rels_type.append(rel_type)
                senses = rel.get('sclass1a', 'N/A.N/A.N/A').split('.')[:2]
                if len(senses) < 2:
                    senses.append('N/A')
                assert len(senses) == 2
                rels_first.append(senses[0])
                rels_second.append(senses[1])
                rels_first_and_second.append((senses[0], senses[1]))
                rels_all_three.append((rel_type, senses[0], senses[1]))
            rels_type_sense = {
                'type': rels_type,
                'first': rels_first,
                'second': rels_second,
                'first_and_second': rels_first_and_second,
                'all_three': rels_all_three
            }
            return rels_type_sense

        xx_yy_type_sense = []
        for xx_rels, yy_rels in self.get_pairwise_aligned_relations(xx, yy):
            xx_type_sense = _get_relation_types_and_senses(xx_rels)
            yy_type_sense = _get_relation_types_and_senses(yy_rels)
            xx_yy_type_sense.append((xx_type_sense, yy_type_sense))
        return xx_yy_type_sense


@st.cache
def load_dataset(dataset_dir: str = './dataset') -> Dict[str, MultilingualTalk]:
    talk_alignment_path = os.path.join(dataset_dir, 'cross-lingual_sentence-level_alignments.json')
    with open(talk_alignment_path, 'r') as f:
        talk_alignments = json.load(f)

    mtalks = {}
    for talk_id in TALK_IDS:
        print(f'Loading {talk_id}..')
        mtalk = MultilingualTalk(talk_id)
        talk_paths = glob(os.path.join(dataset_dir, talk_id, f'{talk_id}_*.json'))
        for talk_path in talk_paths:
            mtalk.add_talk(Talk(talk_path))
        mtalk.set_pairwise_alignments(talk_alignments[talk_id])
        mtalks[talk_id] = mtalk
    return mtalks
