import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="IA",
        page_icon="💻",
    )

    st.write("# Bienvenido al cuerso de Temas Selectos de Física Computacional I (Inteligencia Artificial en la Física) 👋🔭")

    st.sidebar.success("Selecciona una clase")

    st.markdown(
        """
        
    """
    )


if __name__ == "__main__":
    run()