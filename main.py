import streamlit as st

from mted import load_dataset
from overall_patterns import page_overall_patterns
from pairwise_talks import page_pairwise_talks
from search import page_search

if __name__ == '__main__':
    st.set_page_config(
        page_title='Viz-MTED',
        layout='wide',
        initial_sidebar_state='expanded',
    )

    mtalks = load_dataset()

    st.sidebar.markdown('This is online demo of the paper, "[Visualizing Cross‐Lingual Discourse Relations in Multilingual TED Corpora](https://aclanthology.org/2021.codi-main.16/)" ' +
                        'presented at *[CODI @ EMNLP 2021](https://sites.google.com/view/codi-2021/home)*.')
    st.sidebar.markdown('# Navigation')
    nav_page = st.sidebar.radio('', ('Overall Patterns', 'Pairwise Talks', 'Search'))
    if nav_page == 'Overall Patterns':
        page_overall_patterns(mtalks)
    elif nav_page == 'Pairwise Talks':
        page_pairwise_talks(mtalks)
    else:
        page_search(mtalks)
