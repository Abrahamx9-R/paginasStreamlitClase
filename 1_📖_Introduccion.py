import streamlit as st
from streamlit.logger import get_logger
import json
from streamlit_lottie import st_lottie

from utils import footer

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="IA",
        page_icon="💻",
    )

    st.write("# Bienvenido al primer prototipo del curso de Temas Selectos de Física Computacional I (Inteligencia Artificial en Ciencias Exactas ) 👋🔭")

    st.sidebar.success("Selecciona una clase")

    st.markdown(
        """
        
    """
    )
    def load_lottieurl(filepath: str):
        with open(filepath,"r") as f:
            return json.load(f)

    lottie_hello = load_lottieurl("pages/images/circle_shape_morphing_animation.json")

    st_lottie(lottie_hello,speed = 1, reverse=False,loop=True,quality="low",height=600,width=None,key=None,)
    footer()

if __name__ == "__main__":
    run()