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

    st.sidebar.markdown("# Navigation")
    nav_page = st.sidebar.radio('', ('Overall Patterns', 'Pairwise Talks', 'Search'))
    if nav_page == 'Overall Patterns':
        page_overall_patterns(mtalks)
    elif nav_page == 'Pairwise Talks':
        page_pairwise_talks(mtalks)
    else:
        page_search(mtalks)
