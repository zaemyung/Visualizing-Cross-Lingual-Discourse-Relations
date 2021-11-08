import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from apyori import apriori

from mted import LANGUAGES, MultilingualTalk


def get_matched_elements(list_a: List[str], list_b: List[str]) -> List[str]:
    return list((Counter(list_a) & Counter(list_b)).elements())


def merge_dict_and_average(all_match_result: List[Dict[str, float]]) -> Dict[str, float]:
    combined_result = defaultdict(list)
    for res in all_match_result:
        for category in res:
            combined_result[category].append(res[category])
    combined_result = {cat: np.mean(accs) for cat, accs in combined_result.items()}
    return combined_result


def match_paired_relations_type_sense(paired_relations_type_sense: List[Tuple[Dict, Dict]]) -> Dict[str, float]:
    """Returns the accuracy scores of matching each relation type and senses
        E.g. {"type": float,
              "first": float,
              "second": float,
              "first_and_second": float,
              "all_three": float}
    """
    def _calc_relation_matches(xx_type_sense, yy_type_sense):
        acc_results = {}
        for category in xx_type_sense.keys():
            xx_t_s = xx_type_sense[category]
            yy_t_s = yy_type_sense[category]
            if len(xx_t_s) == 0 or len(yy_t_s) == 0:
                continue
            num_matched = get_matched_elements(xx_t_s, yy_t_s)
            accuracy = (2 * len(num_matched)) / (len(xx_t_s) + len(yy_t_s))
            acc_results[category] = accuracy
        return acc_results

    all_match_result = []
    for xx_type_sense, yy_type_sense in paired_relations_type_sense:
        xx_yy_match_result = _calc_relation_matches(xx_type_sense, yy_type_sense)
        all_match_result.append(xx_yy_match_result)
    overall_result = merge_dict_and_average(all_match_result)
    return overall_result


def calculate_pairwise_relation_preservation(mtalks: Dict[str, MultilingualTalk], xx: str, yy: str) -> Dict[str, float]:
    """Calculate the accuracy of matching relations for XX-YY across all talks"""
    all_matched_result = []
    for talk in mtalks.values():
        if xx not in talk.talks or yy not in talk.talks:
            continue
        paired_relations_type_sense = talk.get_pairwise_aligned_relation_type_and_senses(xx, yy)
        matched_result = match_paired_relations_type_sense(paired_relations_type_sense)
        all_matched_result.append(matched_result)
    all_matched_result = merge_dict_and_average(all_matched_result)
    return all_matched_result


def mine_association_rules(mtalks: Dict[str, MultilingualTalk], xx: str, yy: str):
    def _print_rules(rules, filter_identity=True):
        _rules = []
        for record in rules:
            for ordered_stat in record.ordered_statistics:
                _rules.append((ordered_stat.lift, ordered_stat.confidence, ordered_stat.items_base, ordered_stat.items_add))
        _rules = sorted(_rules, reverse=True)
        for lift, confidence, items_base, items_add in _rules:
            items_base = list(items_base)
            items_add = list(items_add)
            if len(items_base) == 0 or len(items_add) == 0:
                continue
            if filter_identity and len(items_base) == len(items_add):
                base_senses = set([sense.split('-', 1)[1] for sense in items_base])
                add_senses = set([sense.split('-', 1)[1] for sense in items_add])
                if base_senses == add_senses:
                    continue
            if lift > 1.0:
                st.write(f'Rule: {items_base} -> {items_add}, Confidence: {confidence:.3f}, Lift: {lift:.3f}')

    transactions_first = []
    for talk in mtalks.values():
        if xx not in talk.talks or yy not in talk.talks:
            continue
        paired_relations_type_sense = talk.get_pairwise_aligned_relation_type_and_senses(xx, yy)
        for xx_type_sense, yy_type_sense in paired_relations_type_sense:
            xx_first = [f'{xx}-{r}' for r in xx_type_sense['first'] if r != 'N/A']
            yy_first = [f'{yy}-{r}' for r in yy_type_sense['first'] if r != 'N/A']
            if len(xx_first) > 0 and len(yy_first) > 0:
                transactions_first.append([*xx_first, *yy_first])

    results_first = list(apriori(transactions_first))
    st.header('Association Rules')
    st.write('Below, we show association rules that are "non-identical" and have lift score greater than "1.0".')
    _print_rules(results_first)


def find_relation_translation_pattern(mtalks: Dict[str, MultilingualTalk], xx: str, yy: str):
    def draw_two_pie_charts(data1, data2, title):
        fig = plt.figure(figsize=(22, 10))
        fig.suptitle(title)
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        plt.pie(x=data1.values(), labels=data1.keys(), autopct="%.1f%%", explode=[0.05]*len(data1), pctdistance=0.5)
        ax2 = plt.subplot2grid((1, 2), (0, 1))
        plt.pie(x=data2.values(), labels=data2.keys(), autopct="%.1f%%", explode=[0.05]*len(data2), pctdistance=0.5)
        st.pyplot(fig)

    st.header('Pie Charts for Relation Divergence')
    st.write('When drawing the following pie charts, we only considered aligned XX-YY pairs that have the same number of relations.')
    all_paired_relations = []
    for talk in mtalks.values():
        if xx not in talk.talks or yy not in talk.talks:
            continue
        paired_relations_type_sense = talk.get_pairwise_aligned_relation_type_and_senses(xx, yy)
        all_paired_relations.extend(paired_relations_type_sense)

    patterns_xx2yy = defaultdict(lambda: defaultdict(Counter))
    patterns_yy2xx = defaultdict(lambda: defaultdict(Counter))
    for xx_rels, yy_rels in all_paired_relations:
        if len(xx_rels) <= 0 or len(yy_rels) <= 0:
            continue
        # we only consider xx-yy pairs that have the same no. of relations
        if len(xx_rels['type']) != len(yy_rels['type']):
            continue

        for xx_r, yy_r in zip(xx_rels['type'], yy_rels['type']):
            patterns_xx2yy['relation_type'][xx_r][yy_r] += 1
            patterns_yy2xx['relation_type'][yy_r][xx_r] += 1
        for xx_r, yy_r in zip(xx_rels['first'], yy_rels['first']):
            patterns_xx2yy['first_sense'][xx_r][yy_r] += 1
            patterns_yy2xx['first_sense'][yy_r][xx_r] += 1
        for xx_r, yy_r in zip(xx_rels['first_and_second'], yy_rels['first_and_second']):
            patterns_xx2yy['first_and_second_sense'][xx_r][yy_r] += 1
            patterns_yy2xx['first_and_second_sense'][yy_r][xx_r] += 1

    for pattern in patterns_xx2yy.keys():
        with st.expander(pattern):
            rels_dict_xx2yy = patterns_xx2yy[pattern]
            rels_dict_yy2xx = patterns_yy2xx[pattern]
            st.subheader(f'{xx} -> {yy}')
            for rel_key in sorted(rels_dict_xx2yy.keys()):
                xx_rel_counter = rels_dict_xx2yy[rel_key]
                yy_rel_counter = rels_dict_yy2xx[rel_key]
                with st.container():
                    draw_two_pie_charts(xx_rel_counter, yy_rel_counter, f'{rel_key} ->')
                    st.markdown('''---''')


def page_overall_patterns(mtalks: Dict[str, MultilingualTalk]) -> None:
    def _print_heatmap(langpair_to_scores):
        rows = []
        cols_inds = []
        for xx in LANGUAGES:
            acc_strs = []
            cols_inds.extend(['', '', xx])
            for yy in LANGUAGES:
                if xx == yy:
                    acc_strs.extend([1.0, 1.0, 1.0])
                else:
                    try:
                        acc_str = langpair_to_scores[(xx, yy)]
                    except KeyError:
                        acc_str = langpair_to_scores[(yy, xx)]
                    acc_strs.extend(acc_str)
            rows.append(acc_strs)
        df = pd.DataFrame(rows, columns=cols_inds, index=LANGUAGES)
        sns.set(font_scale=1)
        fig, ax = plt.subplots(figsize=(12,6))
        ax = sns.heatmap(df, annot=True, cmap='coolwarm_r', vmin=0, vmax=1, cbar=False)
        ax.hlines(list(range(len(LANGUAGES)+1)), *ax.get_xlim(), colors='black')
        ax.vlines([3 * i for i in range(len(LANGUAGES)+1)], *ax.get_ylim(), colors='black')
        st.pyplot(fig)

    st.header('Overall Accuracies')
    st.write('Overall accuracy scores language pairs for matching relations across all talks.\n\n' +
             'For each language pair, each cell represents, from left-to-right, ' +
             'the accuracy of matching `relation_type`; `first_sense`; `first_sense` and `second_sense` (joint).')

    langpair_to_scores = {}
    for xx, yy in itertools.combinations(LANGUAGES, 2):
        scores = calculate_pairwise_relation_preservation(mtalks, xx, yy)
        langpair_to_scores[(xx, yy)] = [scores['type'], scores['first'], scores['first_and_second']]
    _print_heatmap(langpair_to_scores)

    sel_xx = st.selectbox('Select first language', LANGUAGES, index=4)
    sel_yy = st.selectbox('Select second language', LANGUAGES, index=0)
    if sel_xx == sel_yy:
        st.write('First and second langs must be different!')
    else:
        mine_association_rules(mtalks, sel_xx, sel_yy)
        find_relation_translation_pattern(mtalks, sel_xx, sel_yy)
