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
    
def footer():
    file_ = open("componentes/124534-tricube-spinner-1.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    
    local_css("componentes/estilos.css")
    
    file_ = open("componentes/logo_unam_oro.png", "rb")
    contents = file_.read()
    data_url2 = base64.b64encode(contents).decode("utf-8")
    file_.close()



    st.markdown(f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Pie de pagina</title>
    <meta name="viewport" >
     <!--Iconos-->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
</head>
<body>
<!--::::Pie de Pagina::::::-->
    <footer class="pie-pagina after main" tabindex="0">
        <div class="grupo-1">
            <div class="box">
                <h4>Laboratorio de Inteligencia Artificial en Ciencias Exactas (LIACE),</h4>
                <figure>
                    <a href="#">
                        <img src="data:image/gif;base64,{data_url}" alt="Logo de SLee Dw">
                    </a>
                </figure>
            </div>
            <div class="box">
                <h4>Instructores</h4>
                <p2>Doctor José Guadalupe Pérez Ramírez </p>
                <div class="red-social">
                    <a href="mailto:bokhimi@fisica.unam.mx?subject=Clases IA pagina" class="fa fa-envelope"></a>
                    <a href="https://www.youtube.com/channel/UC2U6mx7uYvCYU6F4tWUKJUQ/videos" class="fa fa-youtube"></a>
                </div>
                <p2>Karen Daniela Cruz Hernández  </p>
                <div class="red-social">
                    <a href="karen.daniela@ciencias.unam.mx" class="fa fa-envelope"></a>
                </div>
                <p2>Carlos Emilio Camacho Lorenzana  </p>
                <div class="red-social">
                    <a href="https://www.linkedin.com/in/carlos-emilio-camacho-lorenzana-618bb0128" class="fa fa-linkedin"></a>
                    <a href="https://github.com/LorenzanaGauge" class="fa fa-github"></a>
                    <a href="ccamacholorenzana@ciencias.unam.mx" class="fa fa-envelope"></a>
                </div>
                <p2>Abraham Galindo Ruiz </p>
                <div class="red-social">
                    <a href="https://www.linkedin.com/in/abraham-galindo-ru%C3%ADz-08944a202/" class="fa fa-linkedin"></a>
                    <a href="https://github.com/Abrahamx9-R" class="fa fa-github"></a>
                    <a href="abraham_g_r@ciencias.unam.mx" class="fa fa-envelope"></a>
                </div>
            </div>
            <div class="box">
                <h4>Instituto de Física, UNAM </h4>
                <figure>
                    <a href="#">
                        <img src="data:image/gif;base64,{data_url2}" alt="Logo de SLee Dw">
                    </a>
                </figure>
            </div>
        </div>
    </footer>
</body>
</html>""",unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

