import streamlit as st
import pickle
import streamlit.components.v1 as components
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Wine ML",
    page_icon="wine2.png"
)

model = pickle.load(open('nn_xgb_model.pkl', 'rb'))
explainer = pickle.load(open('explainer.pkl', 'rb'))
shap_values = pickle.load(open('shap_values.pkl', 'rb'))


def classify(num):
    if num == 1:
        st.success("Predicted Quality: Good")
    else:
        st.warning("Predicted Quality: Bad")
    pass


def main():
    st.title("Wine Quality ML")
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    right, center, left = st.columns(3)
    fixed_acidity = right.slider('Fixed Acidity', 4.0, 16.0, 10.7)
    volatile_acidity = right.slider('Volatile Acidity', 0.10, 2.0, 0.35)
    citric_acid = right.slider('Citric Acid', 0.0, 1.0, 0.53)
    residual_sugar = right.slider('Residual Sugar', 0.0, 16.0, 2.61)
    chlorides = left.slider('Chlorides', 0.0, 1.0, 0.09)
    free_sulfur_dioxide = left.slider('Free Sulphur Dioxide', 0, 75, 5)
    total_sulfur_dioxide = left.slider('Total Sulphur Dioxide', 0, 300, 75)
    density = center.slider('Density', 0.0, 1.50, 0.99)
    ph = center.slider('pH', 0.0, 7.0, 3.15)
    sulphates = center.slider('Sulphates', 0.0, 2.0, 0.65)
    alcohol = center.slider('Alcohol', 0.0, 15.0, 10.92)
    input = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
              chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]]

    scaler = MinMaxScaler()
    #inputs = scaler.fit_transform(inputs)
    classify(model.predict(input))

    pass


if __name__ == "__main__":
    main()
