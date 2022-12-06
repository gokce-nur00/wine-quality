import streamlit as st
import pickle
import streamlit.components.v1 as components

st.beta_set_page_config(page_title="YAP470 Project APP",
                        page_icon=":wine_glass:")
tree_model = pickle.load(open('classification_model.pkl', 'rb'))
