import inspect
import textwrap
import base64
import streamlit as st


def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))
def show_text_color(texto,size=8,color='blue'):
    st.markdown(r'''<font size='''+ str(size) + ''' color ='''+color+'''>'''+texto,unsafe_allow_html=True)
    
def show_image_local(ruta,size='60%'):
    file_ = open(ruta, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<center> <img width={size} src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
        unsafe_allow_html=True,
    )   