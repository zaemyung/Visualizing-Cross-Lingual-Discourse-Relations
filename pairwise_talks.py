import os
from itertools import product
from typing import Dict, List, Tuple

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from mted import MultilingualTalk, Sentence


def render_interactive_graph_network(mtalk: MultilingualTalk, xx: str, yy: str,
                                     width_pixels: int = 1700, height_pixels: int = 1500,
                                     rendering_dir: str = './renderings', use_cache: bool = False) -> None:
    def _format_sentence_for_node(sentence: str) -> str:
        tokens = sentence.split()
        new_tokens = []
        for i, tok in enumerate(tokens, start=1):
            new_tokens.append(tok)
            if i % 10 == 0:
                new_tokens.append('<br>')
        return ' '.join(new_tokens)

    def _format_intra_node(whole_sentence: str, arg1_part: str, arg2_part: str, rel_type: str) -> str:
        # surround arg1_part and arg2_part with tag
        whole_sentence = whole_sentence.replace(arg1_part, f'❮{rel_type}-arg1❯{arg1_part}❮/arg1❯')
        whole_sentence = whole_sentence.replace(arg2_part, f'❮{rel_type}-arg2❯{arg2_part}❮/arg2❯')
        whole_sentence = _format_sentence_for_node(whole_sentence)
        return whole_sentence

    def _add_paired_nodes_and_crosslingaul_relations(xx_inds: List[int], yy_inds: List[int],
                                                     xx_width_pos: int, yy_width_pos: int) -> Tuple[int, int]:
        xx_height_pos = -100
        yy_height_pos = 100
        len_xx_inds = len(xx_inds)
        len_yy_inds = len(yy_inds)
        if len_xx_inds < len_yy_inds:
            xx_width_step = float(len_yy_inds) / len_xx_inds
            yy_width_step = 1
        else:
            xx_width_step = 1
            yy_width_step = float(len_xx_inds) / len_yy_inds
        width_spacing = 500

        # add nodes (sentences) first
        for xx_i, sent_index in enumerate(xx_inds, start=1):
            node_id = f'{xx}-{sent_index}'
            formatted_sent = _format_sentence_for_node(xx_sentences[sent_index].en_translation)
            G.add_node(node_id, title=formatted_sent, group=xx, x=xx_width_pos + (xx_i * width_spacing * xx_width_step), y=xx_height_pos, physics=False, value=2)
        for yy_i, sent_index in enumerate(yy_inds, start=1):
            node_id = f'{yy}-{sent_index}'
            formatted_sent = _format_sentence_for_node(yy_sentences[sent_index].en_translation)
            G.add_node(node_id, title=formatted_sent, group=yy, x=yy_width_pos + (yy_i * width_spacing * yy_width_step), y=yy_height_pos, physics=False, value=2)
        # add cross-lingual edges
        for xx_sent_index, yy_sent_index in product(xx_inds, yy_inds):
            xx_node_id = f'{xx}-{xx_sent_index}'
            yy_node_id = f'{yy}-{yy_sent_index}'
            G.add_edge(xx_node_id, yy_node_id, value=0.5, arrowStrikethrough=False)

        xx_cuml_width_pos = xx_width_pos + (xx_i * width_spacing * xx_width_step)
        yy_cuml_width_pos = yy_width_pos + (yy_i * width_spacing * yy_width_step)
        return xx_cuml_width_pos, yy_cuml_width_pos

    def _add_relation_edges(relations: List[Dict], language: str, sentences: List[Sentence]) -> None:
        for r in relations:
            arg1_node_id = f'{language}-{r["arg1_sentence_index"]}'
            arg2_node_id = f'{language}-{r["arg2_sentence_index"]}'
            class_type = '.'.join(r.get('sclass1a', 'N/A').split('.')[:2])
            rel_type = f'{r["relation_type"]}\n{class_type}'
            if r['inter_or_intra'] == 'inter':
                conn = r.get('conn_spanlist_text', 'N/A')
                if conn == 'N/A':
                    conn = r.get('conn1', 'N/A')
                if arg1_node_id in G.get_nodes() and arg2_node_id in G.get_nodes():
                    G.add_edge(arg1_node_id, arg2_node_id, title=f'Connective: "{conn}"', value=1.0, label=rel_type, arrowStrikethrough=True)
            else:
                if arg1_node_id in G.get_nodes():
                    en_sentence = sentences[r["arg1_sentence_index"]].en_translation
                    arg1_en = r['arg1_sentence_en']
                    arg2_en = r['arg2_sentence_en']
                    G.get_node(arg1_node_id)['title'] = _format_intra_node(en_sentence, arg1_en, arg2_en, rel_type)

    def _render_streamlit_component(component_path: str) -> None:
        with open(component_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        components.html(source_code, width=width_pixels, height=height_pixels)

    pairwise_indices = mtalk.pairwise_alignments[(xx, yy)]
    pairwise_relations = mtalk.get_pairwise_aligned_relations(xx, yy)
    assert len(pairwise_indices) == len(pairwise_relations)

    # use sliders to limit the number of paired relations to render
    col1, col2 = st.columns(2)
    with col1:
        lb_sent_index = st.slider('Lower-bound for sentence alignments index:', 0, len(pairwise_indices), 0)
    with col2:
        ub_sent_index = st.slider('Upper-bound for sentence alignments index:', 0, len(pairwise_indices), min(40, len(pairwise_indices)))

    output_graph_path = os.path.join(rendering_dir, f'{mtalk.talk_id}_{xx}-{yy}_{lb_sent_index}-{ub_sent_index}.html')
    if not use_cache or not os.path.isfile(output_graph_path):
        G = Network(width_pixels, height_pixels, layout=False, directed=True)
        pairwise_indices = pairwise_indices[lb_sent_index:ub_sent_index + 1]
        pairwise_relations = pairwise_relations[lb_sent_index:ub_sent_index + 1]
        xx_sentences = mtalk.talks[xx].sentences
        yy_sentences = mtalk.talks[yy].sentences
        xx_cuml_width_pos = 0
        yy_cuml_width_pos = 0
        for xx_inds, yy_inds in pairwise_indices:
            xx_cuml_width_pos, yy_cuml_width_pos = _add_paired_nodes_and_crosslingaul_relations(xx_inds, yy_inds, xx_cuml_width_pos, yy_cuml_width_pos)
        # add {inter, intra}-sentential relation edges after all the nodes are added
        for xx_rels, yy_rels in pairwise_relations:
            _add_relation_edges(xx_rels, xx, xx_sentences)
            _add_relation_edges(yy_rels, yy, yy_sentences)

        if not os.path.isdir(rendering_dir):
            os.makedirs(rendering_dir)
        G.set_edge_smooth('dynamic')
        G.show(output_graph_path)

    st.write('When graph network is not visible, click on any point in the empty box and drag it to upper-left direction a few times.')
    st.write('You can also zoom-in and zoom-out on the graph, and click on the nodes and edges.')
    _render_streamlit_component(output_graph_path)


def page_pairwise_talks(mtalks: Dict[str, MultilingualTalk]) -> None:
    st.header('Interactive Graph Network for Pairwise Talks')
    sel_talk_id = st.selectbox('Select Talk ID', list(mtalks.keys()), index=0)
    st.subheader(sel_talk_id)
    sel_talk = mtalks[sel_talk_id]
    languages = sel_talk.get_all_langs()
    col1, col2 = st.columns(2)
    with col1:
        sel_xx = st.selectbox('Select 1st language', languages, index=languages.index('English'))
    with col2:
        sel_yy = st.selectbox('Select 2nd language', languages, index=languages.index('German'))
    if sel_xx == sel_yy:
        st.write('First and second langs must be different!')
    else:
        render_interactive_graph_network(sel_talk, sel_xx, sel_yy)
