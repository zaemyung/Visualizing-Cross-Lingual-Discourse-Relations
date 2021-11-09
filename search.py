from typing import Dict, List

import pandas as pd
import streamlit as st

from mted import LANGUAGES, MultilingualTalk


def _parse_relations(relations: List[Dict]) -> str:
    parsed = []
    for r in relations:
        class_type = '.'.join(r.get('sclass1a', 'N/A').split('.')[:2])
        rel_type = f'{r["relation_type"]}-{class_type}'
        conn = r.get('conn_spanlist_text', 'N/A')
        if conn == 'N/A':
            conn = r.get('conn1', 'N/A')
        parsed.append(f'{rel_type}-{r["inter_or_intra"]}-{conn}')
    return ';'.join(parsed)


def _render_found_results(results: List[Dict], query: str) -> None:
    def _highlight_sentence(string: str) -> str:
        return string.replace(query, f'❮{query}❯')

    for res in results:
        sent = res['sentence']
        before_sent = res['before_sentence']
        after_sent = res['after_sentence']
        highlighted_sentence = _highlight_sentence(sent.sentence)
        with st.expander(highlighted_sentence):
            parsed_res = {'before_sent': before_sent.sentence,
                          'current_sent': highlighted_sentence,
                          'after_sent': after_sent.sentence,
                          'before_rel': _parse_relations(before_sent.intra_annotations + before_sent.inter_annotations_as_arg1),
                          'current_rel': _parse_relations(sent.intra_annotations + sent.inter_annotations_as_arg1),
                          'after_rel': _parse_relations(before_sent.intra_annotations + before_sent.inter_annotations_as_arg1)}
            res_df = pd.DataFrame.from_dict(parsed_res, orient='index').transpose()
            st.dataframe(res_df)


def page_search(mtalks: Dict[str, MultilingualTalk]) -> None:
    col1, col2 = st.columns(2)
    with col1:
        sel_xx = st.selectbox('Select language', LANGUAGES, index=4)
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
                    before_sentence = talk.sentences[sent_index - 1] if sent_index > 0 else None
                    after_sentence = talk.sentences[sent_index + 1] if sent_index < len(talk.sentences) - 1 else None
                    result = {'talk_id': talk.talk_id,
                              'sent_index': sent_index,
                              'language': talk.language,
                              'before_sentence': before_sentence,
                              'sentence': sent_instance,
                              'after_sentence': after_sentence}
                    found.append(result)
        if len(found) > 0:
            _render_found_results(found, query)
        else:
            st.write('No result found.')
