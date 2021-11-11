import os
from typing import Dict, List

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from mted import LANGUAGES, MultilingualTalk, Sentence
from pairwise_talks import _format_intra_node, _format_sentence_for_node, _render_streamlit_component


def _render_result_network_graph(result: Dict,
                                 width_pixels: int = 1000, height_pixels: int = 1000,
                                 rendering_dir: str = './renderings', use_cache: bool = False) -> None:
    def _add_node(sentence: Sentence, xx_width_pos:int) -> None:
        if sentence is None:
            return
        node_id = f'{xx}-{sentence.sentence_index}'
        sentence_str = sentence.sentence
        if len(sentence.intra_annotations) > 0:
            for r in sentence.intra_annotations:
                class_type = '.'.join(r.get('sclass1a', 'N/A').split('.')[:2])
                rel_type = f'{r["relation_type"]}\n{class_type}'
                sentence_str = _format_intra_node(sentence_str, r['arg1_sentence'], r['arg2_sentence'], rel_type)
        else:
            sentence_str = _format_sentence_for_node(sentence_str)
        G.add_node(node_id, title=sentence_str, group=xx, x=xx_width_pos, y=0, physics=False, value=2)

    def _add_inter_relation_edges(sentence: Sentence) -> None:
        if sentence is None:
            return
        for r in sentence.inter_annotations_as_arg1:
            arg1_node_id = f'{xx}-{r["arg1_sentence_index"]}'
            arg2_node_id = f'{xx}-{r["arg2_sentence_index"]}'
            class_type = '.'.join(r.get('sclass1a', 'N/A').split('.')[:2])
            rel_type = f'{r["relation_type"]}\n{class_type}'
            if r['inter_or_intra'] == 'inter':
                conn = r.get('conn_spanlist_text', 'N/A')
                if conn == 'N/A':
                    conn = r.get('conn1', 'N/A')
                if arg1_node_id in G.get_nodes() and arg2_node_id in G.get_nodes():
                    G.add_edge(arg1_node_id, arg2_node_id, title=f'Connective: "{conn}"', value=1.0, label=rel_type, arrowStrikethrough=True)

    xx = result['language']
    talk_id = result['talk_id']
    query = result['query']
    curr_sent_index = result['sent_index']

    output_graph_path = os.path.join(rendering_dir, f'{xx}_{talk_id}_{curr_sent_index}_{query}.html')
    if not use_cache or not os.path.isfile(output_graph_path):
        before_sentence = result['before_sentence']
        curr_sentence = result['sentence']
        after_sentence = result['after_sentence']

        G = Network(width_pixels, height_pixels, layout=False, directed=True)
        _add_node(before_sentence, -500)
        _add_node(curr_sentence, 0)
        _add_node(after_sentence, 500)
        _add_inter_relation_edges(before_sentence)
        _add_inter_relation_edges(curr_sentence)

        if not os.path.isdir(rendering_dir):
            os.makedirs(rendering_dir)
        G.set_edge_smooth('dynamic')
        G.show(output_graph_path)
        _render_streamlit_component(output_graph_path, width_pixels, height_pixels)


def _render_found_results(results: List[Dict], query: str) -> None:
    def _highlight_sentence(string: str) -> str:
        return string.replace(query, f'❮{query}❯')

    for res in results:
        current_sent = _highlight_sentence(res['sentence'].sentence)
        with st.expander(current_sent):
            res['query'] = query
            _render_result_network_graph(res)


def page_search(mtalks: Dict[str, MultilingualTalk]) -> None:
    col1, col2 = st.columns(2)
    with col1:
        sel_xx = st.selectbox('Select language', LANGUAGES, index=LANGUAGES.index('English'))
    with col2:
        query = st.text_input('Enter search query', max_chars=100)

    talks = []
    for mtalk in mtalks.values():
        if sel_xx in mtalk.talks:
            talks.append(mtalk.talks[sel_xx])

    query = query.strip()
    if len(query) > 0:
        found = []
        for talk in talks:
            for sent_index, sent_instance in enumerate(talk.sentences):
                if query in sent_instance.sentence:
                    assert sent_index == sent_instance.sentence_index
                    before_sentence = talk.sentences[sent_index - 1] if sent_index > 0 else None
                    after_sentence = talk.sentences[sent_index + 1] if sent_index < len(talk.sentences) - 1 else None
                    result = {
                        'talk_id': talk.talk_id,
                        'sent_index': sent_index,
                        'language': talk.language,
                        'before_sentence': before_sentence,
                        'sentence': sent_instance,
                        'after_sentence': after_sentence
                    }
                    found.append(result)
        if len(found) > 0:
            _render_found_results(found, query)
        else:
            st.write('No result found.')
