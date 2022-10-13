import streamlit as st
import json
from streamlit_lottie import st_lottie
import numpy as np
import base64
import matplotlib.pyplot as plt
from random import shuffle

from utils import show_image_local, show_text_color

ruta = "pages/images/imagesClase3/"
st.set_page_config(
    page_title="Clase 3",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://www.extremelycoolapp.com/help',
        'Report a Bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title('Clase 3 ')

def load_lottieurl(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

lottie_hello = load_lottieurl("pages/images/morphing.json")

st_lottie(lottie_hello,speed = 1, reverse=False,loop=True,quality="low",height=600,width=None,key=None,)

st.video("https://youtu.be/4kbrh9d1z2o")

st.markdown(r'''<font size=10 color='blue'>Problema a resolver''',unsafe_allow_html=True)

file_ = open("pages/images/imagesClase3/Problema.jpg", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<center> <img width="50%" src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
    unsafe_allow_html=True,
) 

st.markdown(r'''<font size=8 color='red'>

<center> Movimiento de un objeto sin interacción <center>

</font>
    
Nos interesa conocer la distancia que recorre el objeto conforme pasa el tiempo. Las variables que definen nuestro problema son la distancia que recorre el objeto, <font size=4 color='red' > Y(cm)</font> y el tiempo, <font size=4 color='red' > $\textbf{X}$(seg)</font>.''',unsafe_allow_html=True)

file_ = open("pages/images/imagesClase3/car-air-rail.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<center> <img width="60%" src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
    unsafe_allow_html=True,
)

st.markdown(r'''<font size=10 color='blue'>Generación de las mediciones''',unsafe_allow_html=True)

st.markdown('''Como el número de mediciones de distancia contra tiempo en un experimento en el laboratorio es reducido, simularemos el movimiento del carrito en el riel de aire. Así obtendremos un número grande de "mediciones".''')

st.markdown("Se genera un conjunto de numeros aleatorios ($(x_1,y_1),(x_2,y_2),…,(x_n,y_n)$)")

st.code('''# Se importan las librerías de Python

import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

# Para tener un código en donde todos tengamos los mismos valores, se emplea una semilla aleatoria 
np.random.seed(3)''')
np.random.seed(3)

st.code('''# Esta función genera un conjunto de datos que simulan 
# la medición de la distancia de un carrito en un riel de aire
# en la ausencia de una fuerza sobre el carrito.
# Se propone un error en la medición de la distancia

def generador_datos_simple(n_points, distance_0, measuring_time, speed, max_distance_error):
    
    # n_points es el número de puntos que serán generados
    
    x = np.random.random(n_points) * measuring_time
     
    # x es arreglo con m numeros aleatorios entre 0.0 y measure_time
    
    error = np.random.randn(n_points) * max_distance_error 
    
    # e es un error generado aleatoriamente con un valor maximo max_distance_error

    y = distance_0 + speed*x + error 
        
    return x, y
''')

def generador_datos_simple(n_points, distance_0, measuring_time, speed, max_distance_error):
    x = np.random.random(n_points) * measuring_time
    error = np.random.randn(n_points) * max_distance_error 
    y = distance_0 + speed*x + error    
    return x,y

st.code('''# Generacción de las "mediciones" (xi,yi)
n_points = 1000
distance_0 = 100.0
measuring_time = 100.0
speed = 20.0
max_distance_error = 100

x, y = generador_datos_simple(n_points, distance_0, measuring_time, speed, max_distance_error)

print("x type", type(x), "x shape", x.shape)
print("y type", type(y), "y shape", y.shape)''')

st.sidebar.write('Generacion datos')
n_points = st.sidebar.number_input("Numero de datos",min_value=1,value=1000,format="%i")
distance_0 = st.sidebar.number_input("Distacia 0",value=100.0,format="%f")
measuring_time = st.sidebar.number_input("Medidor de tiempo",value=100.0,format="%f")
speed = st.sidebar.number_input("Velocidad",value=20.0,format="%f")
max_distance_error = st.sidebar.number_input("Maxima distancia de error",value=100.0,format="%f")

x,y= generador_datos_simple(n_points, distance_0, measuring_time, speed, max_distance_error)

st.write("$x type", type(x), "x shape", x.shape)
st.write("$y type", type(y), "y shape", y.shape)

fig, ax = plt.subplots()
ax.set_title('Datos',fontdict = {'fontsize':18})
ax.set_ylabel('Y(cm)',fontdict = {'fontsize':16})
ax.set_xlabel('X(seg)',fontdict = {'fontsize':16})
ax.scatter(x,y,s=6)

st.pyplot(fig)

file_ = open("pages/images/imagesClase3/Mediciones.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<center> <img width="50%" src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
    unsafe_allow_html=True,
)

st.markdown(r'''<font size=10 color ='blue'>Se generan histogramas de las variables $\textbf{X}$ y $Y$''',unsafe_allow_html=True)

st.code('''plt.figure(figsize=(13,5))
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

plt.subplot(1, 2, 1)
plt.hist(x, bins=30, edgecolor='black', alpha=0.5)
plt.xlabel('tiempo(s)', fontsize=16)
plt.ylabel('frecuencia', fontsize=16)


plt.subplot(1, 2, 2)
plt.hist(y, bins=30, edgecolor='black', alpha=0.5)
plt.xlabel('distancia(cm)', fontsize=16)
plt.ylabel('frecuencia', fontsize=16);''')

fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(10,5)

axs[0].set_xlabel('tiempo(s)', size=16)
axs[0].set_ylabel('frecuencia', size=16)
axs[0].xaxis.set_tick_params(labelsize=6)
axs[0].yaxis.set_tick_params(labelsize=6)
axs[0].hist(x, bins=30, edgecolor='black', alpha=0.5)

axs[1].set_xlabel('distancia(cm)', size=16)
axs[1].set_ylabel('frecuencia', size=16)
axs[1].xaxis.set_tick_params(labelsize=6)
axs[1].yaxis.set_tick_params(labelsize=6)
axs[1].hist(y, bins=30, edgecolor='black', alpha=0.5)

st.pyplot(fig)

st.markdown(r'''<font size=10 color ='blue'>En busca de una función que describa la dependencia de $Y$ con $\textbf{X}$ ''',unsafe_allow_html=True)

st.markdown('''Para encontrar la dependencia entre estas variables, proponemos una relación lineal, definida por la siguiente función:''')

st.latex('\hat Y= F(X, W, b) = b + W X ')

st.markdown('''en donde $b$ y $W$ son parámetros que determinan la función, $ \hat Y$ son los valores que genera la función.
    
Esta funcion es derivable respecto a todas sus variables, $X, W, b$.
La letra $W$ se emplea como abreviación de la palabra en ingles "weight", porque se relaciona con la importancia que tiene la variable independiente $X$ en el valor de la función $F$. La letra $b$ es la abreviación de la palabra "bias" en ingles (sesgo en español). ''')

file_ = open("pages/images/imagesClase3/AI-correlation-function.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<center> <img width="30%" src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
    unsafe_allow_html=True,
)

st.markdown(r'''<font size=6 color ='blue'>Variando $\textbf{W}$ y $b$ para encontrar la función que mejor describa la dependencia de $Y$ con la variable $\textbf{X}$  ''',unsafe_allow_html=True)

st.markdown(r'''Para encontrar la función que debe describir la dependencia entra $\textbf{X}$ y $Y$, es necesario generar una métrica que nos cuantifique la diferencia entre los valores $Y$ medidos y los valores $ \hat Y$ que genera la función, para valores específicos de $\textbf{W}$ y $b$.

En el presente caso, se propone la siguiente métrica: para cada muestra $(x_i, y_i)$ se evalua $F(x_i,W,b)$ y se compara con el correspondiente valor $y_i$, la diferencia entre estos valores se eleva al cuadrado.''')
st.latex('(F(x_i,W,b)-y_i)^{2}')
st.markdown(r'''Finalmente se calcula el promedio de este valor sobre todas las mediciones, el cual se define como error cuadrático medio (MSE, por sus siglas en ingles, Mean Squared Error). 
Si m es el número de mediciones, el error cuadrático medio queda como:''')

st.latex('MSE = \dfrac {1}{m}∑_{i=1}^{m}(F(x_i,W,b)-y_i)^{2} ')

file_ = open("pages/images/imagesClase3/Metrica.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<center> <img width="30%" src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
    unsafe_allow_html=True,
)

st.markdown(r'''<font size=6 color ='blue'>En el siguiente código implementa la generación del error cuadrático medio dada una función específica definida por los pesos iniciales w = weight_0, y b = bias_0.  ''',unsafe_allow_html=True)

st.markdown(r'''<font size=6 color ='red'>Es importante poner atención en los valores iniciales de w y b.''',unsafe_allow_html=True)

st.code('''#Initializing the variables of the function f

weight_0 = 10.0
bias_0 = 100.0''')

weight_0 = 10.0
bias_0 = 100.0

st.markdown(r'''<font size=6 color ='red'>Se grafica la correspondiente función $F(X,W,b)$, junto con los puntos que representan a las mediciones''',unsafe_allow_html=True)

st.code('''#The following arrays are generated for plotting the Function F(x, weight_0, bias_0)
x_ = np.arange(0.0, measuring_time, 0.1)
y_ = weight_0*x_ + bias_0


# Using this function F, the MSE is calculated by comparing the calculated and measured values

residuo = 0

for i in range(len(x)):

    residuo += (y[i] - weight_0*x[i] - bias_0)**2

residuo = residuo/len(x)

print('residuo:', residuo)

# Samples and function F are plotted
plt.figure(figsize=(13,8))

#Plotting function
plt.plot(x_, y_, color='green', lw=4, label='F(x, w, b)')
plt.legend()
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.ylabel('Y(cm)', fontsize=16)
plt.xlabel('X(seg)', fontsize=16)

#Plotting samples
plt.scatter(x, y)

plt.show()''')

x_ = np.arange(0.0, measuring_time, 0.1)
y_ = weight_0*x_ + bias_0
residuo = 0
for i in range(len(x)):
    residuo += (y[i] - weight_0*x[i] - bias_0)**2
residuo = residuo/len(x)
st.write('$residuo:', residuo)

fig, ax = plt.subplots()
ax.plot(x_, y_, color='green', lw=4, label='F(x, w, b)')
ax.legend()
ax.set_ylabel('Y(cm)', fontsize=16)
ax.set_xlabel('X(seg)', fontsize=16)
ax.scatter(x, y)

st.pyplot(fig)

st.markdown(r'''<font size=6 color ='blue'>Se actualizan los valores de los parámetros W y b buscando reducir el error cuadrático medio.''',unsafe_allow_html=True)

st.markdown(r'''<font size=6 color ='blue'>Se emplea el método de gradiente descendente para realizar esta actualización.
    
<font size=6 color ='blue'>Para mas informacion colsulta Cauchy_gradient-descent.pdf que esta en el archivo zip''',unsafe_allow_html=True)

st.markdown(r'''Los parámetros $W$ y $b$ se deben actualizar, de manera que el MSE, disminuya. 

Esto se realiza usando el método de gradiente descendente:''')

st.latex(r"W_{new} = W_{old} - \alpha \dfrac{\partial MSE(W, b)}{\partial W}")
st.latex(r"b_{new} = b_{old} - \alpha \dfrac{\partial MSE(W, b)}{\partial b}")

st.write(r'''$\alpha$, como vimos en la clase anterior, depende de los valores del gradiente; y nos permite usar este (el gradiente) para actualizar los valores de los parámetros $W$ y de $b$. Para distinguier la $\alpha$ de los parámetros que pueden variar durante el ajuste, $W$ y $b$, se le da el nombre de hiperparámetro del modelo.


Dado que el error cuadrático medio está definido mediante la relación siguiente:''')

st.latex('MSE = \dfrac{1}{m}∑_{i=1}^{m}(F(x_i,W,b) - y_i)^{2} = \dfrac{1}{m}∑_{i=1}^{m}(W x_i + b -y_i)^2')

st.markdown('las derivadas quedan como:')

st.latex(r'\dfrac{\partial MSE(W, b)}{\partial W} = \dfrac{2}{m}∑_{i=1}^{m}[(W x_i + b -y_i)(x_i)]')

st.latex(r'\dfrac{\partial MSE(W, b)}{\partial b} = \dfrac{2}{m}∑_{i=1}^{m}[(W x_i + b -y_i)]')

file_ = open("pages/images/imagesClase3/W-b-update.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<center> <img width="70%" src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
    unsafe_allow_html=True,
)

st.code('''#Function to update the parameters weight and bias

def update_parameters(x, y, weight, bias, alfa, num_iteraciones):
    
    # inicializacion de parametros

    alfa = alfa
    residuo = 0
    d_w = 0.0
    d_b = 0.0
    m = len(x)

    # Especificaciones de las graficas
    plt.figure(figsize=(22,8))    
    
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)
    
    ax1.scatter(x, y)

    ax1.set_title('Y vs X', size=24)
    ax1.set_xlabel('X', size=20)
    ax1.set_ylabel('Y', size=20)

    # Recta generada con los parametros iniciales

    y_ = weight*x + bias
    
    ax1.plot(x, y_, color='green', lw=4 )
    
    weights = []
    biases = []
    residuos = []
    
    for i in range(num_iteraciones):

        # calculo de derivadas y el residuo

        for i in range(m):

            r = (y[i]-weight*x[i] - bias)**2

            d_w += 2*(weight*x[i] + bias- y[i])*x[i]
            
            d_b += 2*(weight*x[i]+bias-y[i])

            residuo += r

        residuo /= m

        d_w /= m

        d_b /= m
        
        weights.append(weight)
        biases.append(bias)
        residuos.append(residuo)
        
        # Actualizacion de los parametros

        weight = weight - alfa*d_w
        bias = bias - alfa*d_b


        # Recta generada con la actualizacion de los parametros
        
        y_ = weight*x + bias
        
        ax1.plot(x, y_, lw=4 )

        # Grafica de los residuos como funcion de uno de los parametros (el peso)
        
        ax2.scatter(weight, residuo)
        ax2.set_title('MSE vs weight', size=24)
        ax2.set_xlabel('weight', size=22)
        ax2.set_ylabel('MSE', size=22)

        # Grafica de los residuos como funcion de uno de los parametros (el bias)
        
        ax3.scatter(bias, residuo)
        ax3.set_title('MSE vs bias', size=24)
        ax3.set_xlabel('bias', size=22)
        ax3.set_ylabel('MSE', size=22)


    return weights, biases, residuos''')

def update_parameters(x, y, weight, bias, alfa, num_iteraciones):
    alfa = alfa
    residuo = 0
    d_w = 0.0
    d_b = 0.0
    m = len(x)
    fig = plt.figure(figsize=(22,8))    
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)
    ax1.scatter(x, y)
    ax1.set_title('Y vs X', size=24)
    ax1.set_xlabel('X', size=20)
    ax1.set_ylabel('Y', size=20)
    y_ = weight*x + bias
    ax1.plot(x, y_, color='green', lw=4 )
    weights = []
    biases = []
    residuos = []
    for i in range(num_iteraciones):
        for i in range(m):
            r = (y[i]-weight*x[i] - bias)**2
            d_w += 2*(weight*x[i] + bias- y[i])*x[i]
            d_b += 2*(weight*x[i]+bias-y[i])
            residuo += r
        residuo /= m
        d_w /= m
        d_b /= m
        weights.append(weight)
        biases.append(bias)
        residuos.append(residuo)
        weight = weight - alfa*d_w
        bias = bias - alfa*d_b
        y_ = weight*x + bias   
        ax1.plot(x, y_, lw=4 )
        ax2.scatter(weight, residuo)
        ax2.set_title('MSE vs weight', size=24)
        ax2.set_xlabel('weight', size=22)
        ax2.set_ylabel('MSE', size=22)
        ax3.scatter(bias, residuo)
        ax3.set_title('MSE vs bias', size=24)
        ax3.set_xlabel('bias', size=22)
        ax3.set_ylabel('MSE', size=22)
    st.pyplot(fig)
    return weights, biases, residuos

st.code('''# inicializacion de parametros

weight = weight_0 
bias = bias_0 
alfa = 0.00001
num_iter = 100
weights, biases, residuos = update_parameters(x, y, weight, bias, alfa, num_iter)''')

weight = weight_0 
bias = bias_0 
alfa = 0.00001
num_iter = 100
weights, biases, residuos = update_parameters(x, y, weight, bias, alfa, num_iter)

st.markdown(r'''<font size=6 color ='blue'><center>Se grafica del error cuadrático medio, MSE, como función de cada iteración en que se modificaron  $b$, y $W$.</center>''',unsafe_allow_html=True)

st.markdown('Esto nos permite ver de manera rápida como varía el error cuadrático medio conforme se van variando los parámetros W y b. Esta gráfica es muy útil sobre todo cuando el número de parámetros W y b es grande. Es una forma sencilla de ver la variación del error cuadrático conforme se varían los parámetros.')

st.code('''plt.figure(figsize=(12, 7))
plt.scatter(range(num_iter), residuos, color='blue')
plt.title('MSE vs Iteración', size=20)
plt.xlabel('Iteración', size=20)
plt.ylabel('MSE', size=20);''')

fig, ax = plt.subplots()
ax.scatter(range(num_iter), residuos, color='blue',s=6)
ax.legend()
ax.set_title('MSE vs Iteración')
ax.set_xlabel('Iteración', fontsize=16)
ax.set_ylabel('MSE', fontsize=16)
st.pyplot(fig)

st.markdown(r'''<font size=10 color ='blue'>Resumen del ajuste''',unsafe_allow_html=True)

file_ = open("pages/images/imagesClase3/Ajuste-datos.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<center> <img width="70%" src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
    unsafe_allow_html=True,
)

st.markdown(r'''<font size=10 color ='blue'><center> Alternativa para analizar la evolución del ajuste''',unsafe_allow_html=True)

st.markdown(r'''<font size=5 color ='blue'>Ahora enriquecemos la técnica que hemos descrito. Se separa una parte del total de las mediciones para evaluar al final el ajuste que se obtiene.''',unsafe_allow_html=True)

st.markdown('El total de las mediciones son divididas en dos grupos: uno con el 90 % de las mediciones (mediciones_ajustar) y el segundo con el restante 10 % (mediciones_prueba)')

file_ = open("pages/images/imagesClase3/Mediciones-ajustar-prueba.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<center> <img width="70%" src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
    unsafe_allow_html=True,
)

st.markdown('A continuación desarrollamos el código para reordenar los datos al azar, y separlos en dos partes')

st.markdown('''La funcion shuffle reordena de forma aleatoria la posicion de un conjunto de datos en una lista.

     x = [ 1, 5, 7, 3, 8]
     shuffle(x) = [5, 8, 3, 1, 7]

 La funcion zip permite hacer conjuntos ordenados de datos combinando dos vectores de igual dimensión, por ejemplo

     x = [1, 2, 3]
     y = [5, 6, 7]
    
     zip(x, y) = ((1, 5), (2, 6), (3, 7))
    
 De esta manera, junto con la funcion shuffle, se asegura que los datos correspondientes a $x$ y $y$ intercambian su posición de la misma manera.
 

Por otra parte, la operación zip(*a) separa los datos "a" que inicialmente están mezclados
    
a = [[1,2,3],['a','b','c'],[7,8,9]]$$$$
b = zip(*a)
    
for c in b:
    print(c)
    
(1, 'a', 7)
(2, 'b', 8)
(3, 'c', 9)''')

st.code('''# Los datos se cambian de posicion aleatoriamente

from random import shuffle

c = list(zip(x, y)) 

shuffle(c)
    
(x, y) = zip(*c)

print("Número total de mediciones:", len(x))

# Los datos se dividen

muestras_ajustar = (x[0:int(0.90*len(x))], y[0:int(0.90*len(y))])
print("Número de mediciones para ajustar:", len(muestras_ajustar[0]))

muestras_test = (x[int(0.90*len(x)):], y[int(0.90*len(y)):])
print("Número de mediciones para evaluar el ajuste", len(muestras_test[0]))

plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.scatter(muestras_ajustar[0], muestras_ajustar[1])
plt.title('Mediciones_ajustar', size=20)
plt.xlabel('Tiempo(s)', size =15)
plt.ylabel('Distancia(cm)', size =15)
plt.subplot(1, 2, 2)
plt.scatter(muestras_test[0], muestras_test[1], color='red')
plt.title('Mediciones_prueba', size=20)
plt.xlabel('Tiempo(s)', size =15)
plt.ylabel('Distancia(cm)', size =15);''')

c = list(zip(x, y)) 
shuffle(c) 
(x, y) = zip(*c)
st.write("$Número total de mediciones:", len(x))
muestras_ajustar = (x[0:int(0.90*len(x))], y[0:int(0.90*len(y))])
st.write("Número de mediciones para ajustar:", len(muestras_ajustar[0]))

muestras_test = (x[int(0.90*len(x)):], y[int(0.90*len(y)):])
st.write("Número de mediciones para evaluar el ajuste", len(muestras_test[0]))

fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(10,5)
axs[0].set_title('Mediciones_ajustar')
axs[0].set_xlabel('Tiempo(s)', size=16)
axs[0].set_ylabel('Distancia(cm)', size=16)
axs[0].xaxis.set_tick_params(labelsize=6)
axs[0].yaxis.set_tick_params(labelsize=6)
axs[0].scatter(muestras_ajustar[0], muestras_ajustar[1],s=6)

axs[1].set_title('Mediciones_prueba')
axs[1].set_xlabel('Tiempo(s)', size=16)
axs[1].set_ylabel('Distancia(cm)', size=16)
axs[1].xaxis.set_tick_params(labelsize=6)
axs[1].yaxis.set_tick_params(labelsize=6)
axs[1].scatter(muestras_test[0], muestras_test[1], color='red',s=6)

st.pyplot(fig)

st.markdown(r'''<font size=10 color ='blue'>Normalizacion de las muestras que se emplearan para el ajuste''',unsafe_allow_html=True)

st.markdown('Si se normalizan los datos con 3 veces la desviacion estandar, el 99.7 de los datos tendrán valores entre -1 y 1. El rango con que se normaliza tambien se puede variar entre una desviación o dos desviaciones estandar. En el primer caso el 68 % de los datos tendrán valores entre -1 y 1, mientras que en el segundo caso este rango correspondera al 95 % de los datos.')

file_ = open("pages/images/imagesClase3/Standard-deviation.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<center> <img width="50%" src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
    unsafe_allow_html=True,
)

st.markdown(r'''<font size=6 color ='red'>Normalmente los datos se normalizan con sólo una desviación estandar. Esto es lo que aplicaremos para el resto del curs''',unsafe_allow_html=True)

st.markdown(r'''<font size=6 color ='blue'>Los valores se normalizan usando solo una desviación estandar, de manera que el 68 % de los datos quedan con valores entre -1 y +1.''',unsafe_allow_html=True)

st.code('''x_ajustar = muestras_ajustar[0]
y_ajustar = muestras_ajustar[1]''')

x_ajustar = muestras_ajustar[0]
y_ajustar = muestras_ajustar[1]

st.code('''mean_distance = np.mean(y_ajustar)
std_distance = np.std(y_ajustar)

ajustar_y = (y_ajustar-mean_distance)/std_distance

mean_time = np.mean(x_ajustar)
std_time = np.std(x_ajustar)

ajustar_x = (x_ajustar-mean_time)/std_time''')

mean_distance = np.mean(y_ajustar)
std_distance = np.std(y_ajustar)
ajustar_y = (y_ajustar-mean_distance)/std_distance
mean_time = np.mean(x_ajustar)
std_time = np.std(x_ajustar)
ajustar_x = (x_ajustar-mean_time)/std_time


fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(10,5)

axs[0].set_xlabel('Tiempo(unidades relativas)', size=16)
axs[0].set_ylabel('Frecuencia', size=16)
axs[0].xaxis.set_tick_params(labelsize=6)
axs[0].yaxis.set_tick_params(labelsize=6)
axs[0].hist(ajustar_x, bins=30,color='blue', edgecolor='b', alpha=0.5)

axs[1].set_xlabel('Distancia(unidades relativas)', size=16)
axs[1].set_ylabel('Frecuencia', size=16)
axs[1].xaxis.set_tick_params(labelsize=6)
axs[1].yaxis.set_tick_params(labelsize=6)
axs[1].hist(ajustar_y, bins=30,color='blue', edgecolor='b', alpha=0.5)

st.pyplot(fig)

st.markdown(r'''<font size=6 color ='blue'>Se vuelve a correr el ajuste de datos empleando el metodo de gradiente descendente, pero usando sólo el conjunto de datos correspondiente al 90 %''',unsafe_allow_html=True)

st.code('''weight_0 = np.random.random()
bias_0 = np.random.random()
print(weight_0, bias_0)

alfa = 0.04       #relación de ajuste
num_iter = 100    #número de iteraciones

weights, biases, MSE = update_parameters(ajustar_x, ajustar_y, weight_0, bias_0, alfa, num_iter)''')


weight_0 = np.random.random()
bias_0 = np.random.random()
st.write("$",weight_0, bias_0)
alfa = 0.04 
num_iter = 100
weights, biases, MSE = update_parameters(ajustar_x, ajustar_y, weight_0, bias_0, alfa, num_iter)

st.markdown(r'''<font size=6 color ='blue'>Graficar el error cuadrático medio obtenido como función del número de iteración''',unsafe_allow_html=True)

st.code('''plt.figure(figsize=(11, 6))
plt.scatter(range(num_iter), MSE, color='blue')
plt.title('MSE vs iteración', size=20)
plt.xlabel('iteración', size=15)
plt.ylabel('MSE', size=15);
''')

fig, ax = plt.subplots()
ax.set_title('MSE vs iteración',fontdict = {'fontsize':18})
ax.set_ylabel('MSE',fontdict = {'fontsize':16})
ax.set_xlabel('iteración',fontdict = {'fontsize':16})
ax.scatter(range(num_iter), MSE, color='blue',s=6)

st.pyplot(fig)

st.code('''def MSE(x, y, weight, bias):
    
    r = 0

    m = len(x)
    
    for i in range(m):

        r += (y[i]-weight*x[i] - bias)**2

    r /= m
    
    return r''')

def MSE(x, y, weight, bias):    
    r = 0
    m = len(x)
    for i in range(m):
        r += (y[i]-weight*x[i] - bias)**2
    r /= m
    return r

st.code('''print("El MSE mínimo durante el ajuste es =  %.5f" \
      %MSE(ajustar_x, ajustar_y, weights[-1], biases[-1]))
''')
st.write("$El MSE mínimo durante el ajuste es =  %.5f" \
      %MSE(ajustar_x, ajustar_y, weights[-1], biases[-1]))

st.markdown(r'''<font size=6 color ='blue'>Con los valores optimos de $b$ y $W$, se calcula el error cuadrático medio que se obtiene con el 10 % de los datos restantes.''',unsafe_allow_html=True)

st.write('Para calcular el residuo asociado a las muestras_test, estas se normalizan con los parámetros (valore medio y desviación estandard) utilizados para normalizar los datos de las muestras_ajustar.')

st.code('''test_y = (muestras_test[1]-mean_distance)/std_distance

test_x = (muestras_test[0]-mean_time)/std_time

print("MSE_prueba =  %.5f" %MSE(test_x, test_y, weights[-1], biases[-1]))''')

test_y = (muestras_test[1]-mean_distance)/std_distance
test_x = (muestras_test[0]-mean_time)/std_time
st.write("$MSE_prueba =  %.5f" %MSE(test_x, test_y, weights[-1], biases[-1]))

st.markdown(r'''<font size=6 color ='blue'><center>Otra forma de evaluar el ajuste obtenido.''',unsafe_allow_html=True)

st.write('''En este caso, las muestras seleccionadas para hacer el ajuste se dividen en dos grupos:

El 90 % (este valor es solo un ejemplo) de ellas se emplea para hacer el ajuste, mientras que
    
el 10 % restante se emplean para evaluar el error cuadrático medio obtenido en cada iteración del ajuste.''')


file_ = open("pages/images/imagesClase3/Mediciones-ajustar-valida.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<center> <img width="60%" src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
    unsafe_allow_html=True,
)

st.markdown(r'Modificamos la función que se tiene para actualizar $W$ y $b$ en cada ciclo. Esta nueva función recibe la proporción (val_ratio) de los datos de ajuste que se emplean para validar.')

st.code('''#Function to update weight and bias

def update_parameters_1(x, y, weight, bias, alfa, iteraciones, val_ratio=0.1):
    
    # inicializacion de parametros
    
    #x = np.squeeze(x)
    #y = np.squeeze(y)
    alfa = alfa
    residuo = 0
    d_w = 0.0
    d_b = 0.0
    m = len(x)

    # Especificaciones de las graficas
    plt.figure(figsize=(13,8)) 
    plt.title('MSE vs iteración', size=24)
    plt.xlabel('iteración', size=18)
    plt.ylabel('MSE', size=18)
    
    ajustar_ratio = int((1.0-val_ratio)*len(x))  
   
    samples_ajustar = (x[0:ajustar_ratio], y[0:ajustar_ratio])
    samples_val = (x[ajustar_ratio:], y[ajustar_ratio:])
    x = samples_ajustar[0]
    y = samples_ajustar[1]
    x_val = samples_val[0]
    y_val = samples_val[1]
       
    weights = []
    biases = []
    residuos = []
    residuos_val = []
    
    m_ajustar = len(x)
    m_val = len(x_val)
    
    for i in range(iteraciones):

        # calculo de derivadas y el residuo
        residuo = 0.0
        residuo_val = 0.0
        
        for j in range(m_ajustar):

            r = (y[j]-weight*x[j] - bias)**2

            d_w += 2*(weight*x[j] + bias- y[j])*x[j]
            
            d_b += 2*(weight*x[j]+bias-y[j])

            residuo += r

        residuo /= m_ajustar

        d_w /= m_ajustar

        d_b /= m_ajustar
        
        #calculo del residuo de las muestras de valoración
        
        for j in range(m_val):

            r = (y_val[j]-weight*x_val[j] - bias)**2

            #r += np.squeeze(r)
            residuo_val += r

        residuo_val /= m_val
                      
        weights.append(weight)
        biases.append(bias)
        residuos.append(residuo)
        residuos_val.append(residuo_val)
        
        # Actualizacion de los parametros

        weight = weight - alfa*d_w
        bias = bias - alfa*d_b
        
        plt.scatter(i, residuo, color='blue')
        
        plt.scatter(i, residuo_val, color='orange')
        
        plt.legend(['Ajuste', 'Validation'], loc='upper right')


    return weights, biases, residuos, residuos_val''')


#Function to update weight and bias

def update_parameters_1(x, y, weight, bias, alfa, iteraciones, val_ratio=0.1):
    alfa = alfa
    residuo = 0
    d_w = 0.0
    d_b = 0.0    
    fig, ax = plt.subplots()
    ax.set_title('MSE vs iteración',fontdict = {'fontsize':18})
    ax.set_ylabel('MSE',fontdict = {'fontsize':16})
    ax.set_xlabel('iteración',fontdict = {'fontsize':16})    
    ajustar_ratio = int((1.0-val_ratio)*len(x))  
    samples_ajustar = (x[0:ajustar_ratio], y[0:ajustar_ratio])
    samples_val = (x[ajustar_ratio:], y[ajustar_ratio:])
    x = samples_ajustar[0]
    y = samples_ajustar[1]
    x_val = samples_val[0]
    y_val = samples_val[1]
    weights = []
    biases = []
    residuos = []
    residuos_val = []
    m_ajustar = len(x)
    m_val = len(x_val)
    for i in range(iteraciones):
        residuo = 0.0
        residuo_val = 0.0
        for j in range(m_ajustar):
            r = (y[j]-weight*x[j] - bias)**2
            d_w += 2*(weight*x[j] + bias- y[j])*x[j]
            d_b += 2*(weight*x[j]+bias-y[j])
            residuo += r
        residuo /= m_ajustar
        d_w /= m_ajustar
        d_b /= m_ajustar
        for j in range(m_val):
            r = (y_val[j]-weight*x_val[j] - bias)**2
            residuo_val += r
        residuo_val /= m_val            
        weights.append(weight)
        biases.append(bias)
        residuos.append(residuo)
        residuos_val.append(residuo_val)
        weight = weight - alfa*d_w
        bias = bias - alfa*d_b
        ax.scatter(i, residuo, color='blue',s=6)
        ax.scatter(i, residuo_val, color='orange',s=6)
        ax.legend(['Ajuste', 'Validation'], loc='upper right')

    st.pyplot(fig)
    return weights, biases, residuos, residuos_val

st.code('''weight_0 = np.random.random()
bias_0 = np.random.random()
alfa = 0.04
num_iter = 100
validacion_ratio = 0.1

weights, biases, mse, mse_val = update_parameters_1 \
        (ajustar_x, ajustar_y, weight_0, bias_0, alfa, num_iter, validacion_ratio)
''')

weight_0 = np.random.random()
bias_0 = np.random.random()
alfa = 0.04
num_iter = 100
validacion_ratio = 0.1
weights, biases, mse, mse_val = update_parameters_1 \
        (ajustar_x, ajustar_y, weight_0, bias_0, alfa, num_iter, validacion_ratio)

show_text_color('Con los valores óptimos obtenidos para $W$ y $b$, evaluamos ahora el residuo que se obtiene las muestras_prueba.',size=6)

st.code('print("MSE_prueba =  %.5f" %MSE(test_x, test_y, weights[-1], biases[-1]))')

st.write('''"$MSE_prueba =  %.5f" %MSE(test_x, test_y, weights[-1], biases[-1])''')

show_text_color('Inferencia')

st.write("Dado el tiempo t (no empleado en las mediciones) podemos obtener una predicción (inferencia) del valor de la distancia que recorre el carrito.")

st.code('''def inference(t, w, b, mean_t, std_t, mean_y, std_y) :
    
    t = (t-mean_t) / std_t 
    d = w * t + b
    
    d = d * std_y + mean_y
    
    return d''')

def inference(t, w, b, mean_t, std_t, mean_y, std_y) :
    t = (t-mean_t) / std_t 
    d = w * t + b
    d = d * std_y + mean_y
    return d

st.code('''#El tiempo esta dado en las unidades con los que se recolectaron los datos, segundos
tiempos = [1.65, 32.20, 43.5, 84.7]

for t in tiempos :
    distance = inference(t,weights[-1], biases[-1], mean_time, std_time, mean_distance, std_distance)

    print ("Para el tiempo de {0:5.2f} s la distancia inferida es {1:6.2f} cm ".format(t,distance))''')

tiempos = [1.65, 32.20, 43.5, 84.7]
for t in tiempos :
    distance = inference(t,weights[-1], biases[-1], mean_time, std_time, mean_distance, std_distance)
    st.write("$Para el tiempo de {0:5.2f} s la distancia inferida es {1:6.2f} cm ".format(t,distance))

st.markdown(r'''Vemos entonces que con el ajuste de una función al conjunto de mediciones $(x_i, y_i)$ podemos hacer predicciones del valor de y dado el valor de x. 
    
Es decir, el sistema desarrollado "aprendió" la correlación que hay entre las variables $X$ y $Y$, y por ello, puede hacer inferencias.''')

show_text_color('<center>Inteligencia Artificial <center>',size=10)

show_text_color('''Pasaremos conceptualmente de la nomenclatura de ajuste de datos a la nomenclatura que se emplea en Inteligencia Artificial''',size=6)

st.markdown('En inteligencia artificial, un sistema es "inteligente" cuando despues de ser entrenado con información que le es suministrada, es capaz de hacer inferencias (predicciones).')

show_text_color('Dada la dinámica del desarrollo de la inteligencia artificial a nivel mundial, en el presente curso, emplearemos la nomenclatura estandar que se emplea en el idioma Ingles.',size=6)

st.markdown(r'''Al buscar resolver un problema en la vida real, después de analizarlo debemos proponer las variables que consideramos lo describen.
    
El paso siguiente es cuantificar estas variables y asignarles valores.

El problema puede definir de manera destacada una de las variables, una en la cual se tenga particular interés, esta variable la definimos como la variable $Y$, la cual considermos depende del resto de las variables que identifican al problema, estas variables serán las variables independientes $\textbf X$.''')

show_text_color('El ejemplo que vimos de ajuste de una función a un conjunto de puntos $(x_i, y_i)$ lo traduciremos a un sistema de aprendizaje artificial.',size=4)

show_image_local(ruta+"AI-Problem.png")

show_text_color('<center> Problema: Predecir la distancia que recorre un objeto con el tiempo, cuando tiene un movimiento rectilineo y uniforme')

st.markdown(r'''Lo que nos interesa es la distancia, la cual la identificaremos con la variable $Y$
    
Proponemos que esta distancia depende solo del tiempo, el cual lo identificaremos con la variable $X$.''')

show_text_color('Analizaremos un ejemplo:')

st.markdown(r'''Tenemos un cuerpo, del cual se han obtenido $m$ muestras (mediciones) de estas variables. Cada muestra de este cuerpo la identificamos por la dupla $(x_i, y_i)$.
    
Generaremos un sistema de aprendizaje, en donde proponemos que la función $F(\textbf {X},\textbf W,b)$ describe la relación entre estas variables. 
    
Esta función puede tener diferentes formas. Por ejemplo:''')

st.latex(r'F(\textbf {X}, \textbf {W}, b) = b + \textbf {WX}')
st.markdown('o bien:')
st.latex(r'''F(\textbf {X},\textbf {W},b) = 1.7159*tanh(\textbf {WX}+b)''')
st.markdown('o bien: ')
st.latex(r'F(\textbf {X},\textbf{W},b) = Artificial-Neural-Network (\textbf {ANN})')
st.markdown('o bien: ')
st.latex(r'F(\textbf {X},\textbf{W},b) = Supported-Vector-Machine(SVM)')
st.markdown('o bien: ')
st.latex(r'F(\textbf {X},\textbf{W},b) = Decision-Tree ')
st.markdown('o bien: ')
st.latex(r'F(\textbf {X},\textbf{W},b) = Decision-Random-Forest ')
st.markdown('entre otras.')

show_image_local(ruta+"AI-correlation-function.png",size='35%')

st.markdown('Dada la simplicidad de nuestros datos, para nuestro sistema de aprendizaje proponemos que la relación entre las variables que describen nuestro sistema es una relación lineal, descrita por la función:')
st.latex(r'F(\textbf{X},\textbf{W}, b) = b + \textbf{WX}')
st.markdown('En la nomenclatura de inteligencia artificial, esto significa que nuestro problema se resuelve con una "regresión lineal" (Linear regression). Nomenclatura muy desafortunada. Esto es porque el los valores de las variables son no discretos.')

show_text_color('Este es el origen de el término regresion: "Se regresa al tamaño anterior" en estatura, en tamaño de la semilla. El término no indica nada sobre el hecho de que la variable Y toma valores continuos (no discretos).',size=4)
show_text_color('Para mas informacion hacerca de este tema consulta en el archivo zip la literatura Origen del término regresion: Francis Galton, 1877 y Origen del término regresión: W.F. Stanly, 1885',size=4)

show_text_color('Estos son los datos que describen el movimiento de nuestro cuerpo')

st.code('''# Generacción de las muestras (xi,yi)
n_points = 1000
distance_0 = 100.0
measuring_time = 100.0
speed = 20.0
max_distance_error = 100

x, y = generador_datos_simple(n_points, distance_0, measuring_time, speed, max_distance_error)

print("x type", type(x), "x shape", x.shape)
print("y type", type(y), "y shape", y.shape)''')

x,y= generador_datos_simple(n_points, distance_0, measuring_time, speed, max_distance_error)

st.write("$x type", type(x), "x shape", x.shape)
st.write("$y type", type(y), "y shape", y.shape)

fig, ax = plt.subplots()
ax.set_title('Datos',fontdict = {'fontsize':18})
ax.set_ylabel('Y(cm)',fontdict = {'fontsize':16})
ax.set_xlabel('X(seg)',fontdict = {'fontsize':16})
ax.scatter(x,y,s=6)

st.pyplot(fig)

show_text_color('Se generan histogramas de las variables $X$ y $Y$')

st.code('''plt.figure(figsize=(13,5))
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

plt.subplot(1, 2, 1)
plt.hist(x, bins=30, edgecolor='black', alpha=0.5)
plt.xlabel('X(segundos)', fontsize=16)
plt.ylabel('frecuencia', fontsize=16)


plt.subplot(1, 2, 2)
plt.hist(y, bins=30, edgecolor='black', alpha=0.5)
plt.xlabel('Y(cm)', fontsize=16)
plt.ylabel('frecuencia', fontsize=16);''')

fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(10,5)

axs[0].set_xlabel('X(segundos)', size=16)
axs[0].set_ylabel('frecuencia', size=16)
axs[0].xaxis.set_tick_params(labelsize=6)
axs[0].yaxis.set_tick_params(labelsize=6)
axs[0].hist(x, bins=30, edgecolor='black', alpha=0.5)

axs[1].set_xlabel('Y(cm)', size=16)
axs[1].set_ylabel('frecuencia', size=16)
axs[1].xaxis.set_tick_params(labelsize=6)
axs[1].yaxis.set_tick_params(labelsize=6)
axs[1].hist(y, bins=30, edgecolor='black', alpha=0.5)

st.pyplot(fig)

show_text_color('El total de los datos son divididos en dos grupos: uno con el 90 % de los datos para realizar el aprendizaje, y el segundo con el restante 10 % para probar la calidad del aprendizaje.',size=6)

show_image_local(ruta+"AI-train-val-test-data.png")

st.code('''# Los datos se cambian de posicion aleatoriamente

from random import shuffle

c = list(zip(x, y)) 

shuffle(c)
    
(x, y) = zip(*c)

#print(len(x), len(y))

# Los datos se dividen

samples_train = (x[0:int(0.90*len(x))], y[0:int(0.90*len(y))])
#print(len(samples_train[0]), len(samples_train[1]))

samples_test = (x[int(0.90*len(x)):], y[int(0.90*len(y)):])

plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.scatter(samples_train[0], samples_train[1])
plt.title('Samples_train', size=20)
plt.xlabel('X(s)', size =15)
plt.ylabel('Y(cm)', size =15)
plt.subplot(1, 2, 2)
plt.scatter(samples_test[0], samples_test[1], color='red')
plt.title('Samples_test', size=20)
plt.xlabel('X(s)', size =15)
plt.ylabel('Y(cm)', size =15);''')

c = list(zip(x, y)) 
shuffle(c)
(x, y) = zip(*c)
samples_train = (x[0:int(0.90*len(x))], y[0:int(0.90*len(y))])
samples_test = (x[int(0.90*len(x)):], y[int(0.90*len(y)):])
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.scatter(samples_train[0], samples_train[1])
plt.title('Samples_train', size=20)
plt.xlabel('X(s)', size =15)
plt.ylabel('Y(cm)', size =15)
plt.subplot(1, 2, 2)
plt.scatter(samples_test[0], samples_test[1], color='red')
plt.title('Samples_test', size=20)
plt.xlabel('X(s)', size =15)
plt.ylabel('Y(cm)', size =15);

fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(10,5)

axs[0].set_title('Samples_train')
axs[0].set_xlabel('X(s)', size=16)
axs[0].set_ylabel('Y(cm)', size=16)
axs[0].xaxis.set_tick_params(labelsize=6)
axs[0].yaxis.set_tick_params(labelsize=6)
axs[0].scatter(samples_train[0], samples_train[1],s=6)

axs[1].set_title('Samples_test')
axs[1].set_xlabel('X(s)', size=16)
axs[1].set_ylabel('Y(cm)', size=16)
axs[1].xaxis.set_tick_params(labelsize=6)
axs[1].yaxis.set_tick_params(labelsize=6)
axs[1].scatter(samples_test[0], samples_test[1], color='red',s=6)

st.pyplot(fig)

show_text_color('Para las muestras de entrenamiento, las variables se normalizan empleando su promedio y su desviación estandar.',size=5)

st.code('''x_train = samples_train[0]
y_train = samples_train[1]
x_test = samples_test[0]
y_test = samples_test[1]''')

x_train = samples_train[0]
y_train = samples_train[1]
x_test = samples_test[0]
y_test = samples_test[1]

st.code('''y_mean = np.mean(y_train)
y_std = np.std(y_train)

train_y = (y_train-y_mean)/y_std

x_mean = np.mean(x_train)
x_std = np.std(x_train)

train_x = (x_train-x_mean)/x_std''')

y_mean = np.mean(y_train)
y_std = np.std(y_train)
train_y = (y_train-y_mean)/y_std
x_mean = np.mean(x_train)
x_std = np.std(x_train)
train_x = (x_train-x_mean)/x_std

st.code('''plt.figure(figsize=(13,5))
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)


plt.subplot(1, 2, 1)
plt.hist(train_x, bins=30,color='blue', edgecolor='b', alpha=0.5)
plt.xlabel('X', fontsize=14)
plt.ylabel('Frecuencia',fontsize=14)


plt.subplot(1, 2, 2)
plt.hist(train_y, bins=30,color='blue', edgecolor='b', alpha=0.5)
plt.xlabel('Y',fontsize=14)
plt.ylabel('Frecuencia',fontsize=14);''')

fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(10,5)

axs[0].set_xlabel('X', size=16)
axs[0].set_ylabel('Frecuencia', size=16)
axs[0].xaxis.set_tick_params(labelsize=6)
axs[0].yaxis.set_tick_params(labelsize=6)
axs[0].hist(train_x, bins=30,color='blue', edgecolor='b', alpha=0.5)

axs[1].set_xlabel('Y', size=16)
axs[1].set_ylabel('Frecuencia', size=16)
axs[1].xaxis.set_tick_params(labelsize=6)
axs[1].yaxis.set_tick_params(labelsize=6)
axs[1].hist(train_y, bins=30,color='blue', edgecolor='b', alpha=0.5)

st.pyplot(fig)

st.markdown('''Las muestras seleccionadas para hacer el entrenamiento se dividen en dos grupos:

El 90 % (este valor es solo un ejemplo) de ellas se emplea para hacer el entrenamiento,
    
mientras que el 10 % restante se emplean para evaluar la evolución del mismo en cada época (iteración).
''')

show_image_local(ruta+"train_data.png")

st.markdown(r'''Generamos el sistema de aprendizaje que contiene la arquitectura $F(X,W,b)$, así como la métrica que emplearemos para el entrenamiento. 
    
Ejecutaremos el entrenamiento y generaremos las graficas de la función de pérdida (error) como función de la época. En el futuro, estas acciones se programan por separado.  ''')

show_text_color('Sea $F(\textbf{X},\textbf{W},b)$, la función que describe el sistema de aprendizaje. En donde $\textbf{W}$ y $b$, son los parámetros que la definen.',size=4)

show_image_local(ruta+'AI-correlation-function.png',size='35%')

st.markdown(r'''
Durante el entrenamiento los valores de $W$ y $b$ son actualizados. Esto se realiza definiendo una métrica para obtener la función de pérdida, la cual compara los valores de la función $F(x_i,\textbf{W},b)$ con los valores $y_i$ para cada una de las muestras.

La siguiente es un ejemplo de este tipo de métricas:''')
st.latex(r'(F(x_i,\textbf{w},b)-y_i)^{2}')
st.markdown('''Esto se hace para cada una de las muestras para obtener, en el presente caso, la función de pérdida (loss en Inglés). A esta función también se le conoce con el nombre de función de costo. 

Si m es el número de muestras, la función de pérdida queda como:''')

st.latex(r'J(W, b) = \dfrac{1}{m}∑_{i=1}^{m}(F(x_i,\textbf{W},b)-y_i)^{2}')

st.markdown(r'Cuando con los valores de $\bf W$ y $b$, generamos la pérdida (loss) y los actualizamos con la las derivadas de esta, decimos que se ejecutó una época.')

st.markdown(r'Los parámetros $\bf W$ y $b$ se pueden actualizar usando el método de gradiente descendente:')
st.markdown('para mas informacion consulta la bibliografia disponible en el archivo zip, Cauchy, Gradiente Descendente')
st.latex(r'\textbf {W}_{new} = \textbf{W}_{old} - \alpha \dfrac{\partial J(W, b)}{\partial \textbf{W}}')
st.latex(r'b_{new} = b_{old} - \alpha \dfrac{\partial J(W, b)}{\partial b}')

st.markdown(r'$\alpha$, ahora se le llama relación de aprendizaje, y es un hiperparámetro del modelo del modelo de aprendizaje. $\alpha$ controla la rapidez con que se varían los valores de los parametros $\bf W$ y $b$.')

show_image_local(ruta+'AI-W-b-update.png')
show_text_color('<center>El entrenamiento se realiza sólo con los datos destinados para ello ',size=10)
show_text_color('Es muy importante evaluar el entrenamiento en cada época. ',size=6)

st.markdown('''Esto se realiza separando una porción (10 %) de los datos destinados inicialmente para el entrenamiento. Con ellos se obtiene un valor de la funcion de pérdida en cada época, llamado pérdida en la validacion, el cual se grafica junto con el valor de la función de pérdida asociada al entrenamiento. 
    
El 90 % de resto de los datos destinados inicialmente para el entrenamiento se emplean para realizar el entrenamiento.''')

st.code('''#Function to update weight and bias

def training(x_train, y_train, weight, bias, alpha, epochs, val_ratio=0.1):
    
    # inicializacion de parametros

    alpha = alfa
    d_w = 0.0
    d_b = 0.0

    # Especificaciones de las graficas
    plt.figure(figsize=(13,8)) 
    plt.title('Pérdida vs Epoca', size=24)
    plt.xlabel('Época', size=20)
    plt.ylabel('Pérdida', size=20)
    
    train_ratio = int((1.0-val_ratio)*len(x_train))  
   
    samples_train = (x_train[0:train_ratio], y_train[0:train_ratio])
    samples_val = (x_train[train_ratio:], y_train[train_ratio:])
    x = samples_train[0]
    y = samples_train[1]
    x_val = samples_val[0]
    y_val = samples_val[1]
       
    weights = []
    biases = []
    costs = []
    costs_val = []
    
    m_train = len(x)
    m_val = len(x_val)
    
    for i in range(epochs):

        # calculo de derivadas y el residuo
        cost = 0.0
        cost_val = 0.0
        
        for j in range(m_train):

            r = (y[j]-weight*x[j] - bias)**2

            d_w += 2*(weight*x[j] + bias- y[j])*x[j]
            
            d_b += 2*(weight*x[j]+bias-y[j])

            cost += r

        cost /= m_train

        d_w /= m_train

        d_b /= m_train
        
        #calculo del residuo de las muestras de valoración
        
        for j in range(m_val):

            cost_val += (y_val[j] - weight*x_val[j] - bias)**2

        cost_val /= m_val
               
       
        weights.append(weight)
        biases.append(bias)
        costs.append(cost)
        costs_val.append(cost_val)
        
        # Actualizacion de los parametros

        weight = weight - alpha*d_w
        bias = bias - alpha*d_b
        
        plt.scatter(i, cost, color='green')
        
        plt.scatter(i, cost_val, color='orange')
        plt.legend(['Training', 'Validation'], loc='upper right')

    return weights, biases, costs, costs_val''')


def training(x_train, y_train, weight, bias, alpha, epochs, val_ratio=0.1):
    alpha = alfa
    d_w = 0.0
    d_b = 0.0
    fig, ax = plt.subplots()
    ax.set_title('Pérdida vs Epoca',fontdict = {'fontsize':18})
    ax.set_ylabel('Pérdida',fontdict = {'fontsize':16})
    ax.set_xlabel('Época',fontdict = {'fontsize':16}) 
    train_ratio = int((1.0-val_ratio)*len(x_train))  
    samples_train = (x_train[0:train_ratio], y_train[0:train_ratio])
    samples_val = (x_train[train_ratio:], y_train[train_ratio:])
    x = samples_train[0]
    y = samples_train[1]
    x_val = samples_val[0]
    y_val = samples_val[1]
    weights = []
    biases = []
    costs = []
    costs_val = []
    m_train = len(x)
    m_val = len(x_val)
    for i in range(epochs):
        cost = 0.0
        cost_val = 0.0
        for j in range(m_train):
            r = (y[j]-weight*x[j] - bias)**2
            d_w += 2*(weight*x[j] + bias- y[j])*x[j]           
            d_b += 2*(weight*x[j]+bias-y[j])
            cost += r
        cost /= m_train
        d_w /= m_train
        d_b /= m_train
        for j in range(m_val):
            cost_val += (y_val[j] - weight*x_val[j] - bias)**2
        cost_val /= m_val
        weights.append(weight)
        biases.append(bias)
        costs.append(cost)
        costs_val.append(cost_val)
        weight = weight - alpha*d_w
        bias = bias - alpha*d_b
        ax.scatter(i, cost, color='green',s=6)
        ax.scatter(i, cost_val, color='orange',s=6)
        ax.legend(['Training', 'Validation'], loc='upper right')
    st.pyplot(fig)
    return weights, biases, costs, costs_val

show_image_local(ruta+'Training.png')
st.code('''weight_0 = np.random.random()
bias_0 = np.random.random()
alpha = 0.04
num_epochs = 100
validation_ratio = 0.1

weights, biases, cost, cost_val = training \
        (train_x, train_y, weight_0, bias_0, alpha, num_epochs, validation_ratio)''')

weight_0 = np.random.random()
bias_0 = np.random.random()
alpha = 0.04
num_epochs = 100
validation_ratio = 0.1
weights, biases, cost, cost_val = training \
        (train_x, train_y, weight_0, bias_0, alpha, num_epochs, validation_ratio)

show_text_color('Inferencia (predicción)')

show_image_local(ruta+'Predictions.png')

st.markdown('Dado el tiempo $X$ el sistema de aprendizaje puede predecir (inferir) del valor de la distancia ($Y$) para ese tiempo.')

show_text_color('Recuerdese que para realizar el entrenamiento del sistema de aprendizaje, tanto la variable dependiente, $Y$, como la variable independiente, $X$, fueron normalizadas.',size='5')

st.code('''def loss(x, y, weight, bias):
    
    r = 0

    m = len(x)
    
    for i in range(m):

        r += (y[i]-weight*x[i] - bias)**2

    r /= m
    
    return r''')

def loss(x, y, weight, bias):
    r = 0
    m = len(x)
    for i in range(m):
        r += (y[i]-weight*x[i] - bias)**2
    r /= m
    return r

st.code('''test_y = (y_test-y_mean)/y_std

test_x = (x_test-x_mean)/x_std

print("loss_prueba =  %.5f" %loss(test_x, test_y, weights[-1], biases[-1]))''')
test_y = (y_test-y_mean)/y_std
test_x = (x_test-x_mean)/x_std
st.write("$loss_prueba =  %.5f" %loss(test_x, test_y, weights[-1], biases[-1]))

st.markdown('''Para poder explotar lo aprendido por el sistema de aprendizaje con un conjunto de muestras nuevas, es necesario reescalar el valor de la variable independiente, $X$, de cada muestra nueva. 
    
Una vez inferido, para una muestra, el valor de la variable dependiente, $Y$, es necesario reescalarlo para tener un valor que se pueda comparar con lo valores originales en las muestras empleadas para el entrenamiento.''')
st.code('''def inference(x, w, b, x_train_mean, x_train_std, y_train_mean, y_train_std) :
    
    # reescalando la variable x
    x = (x - x_train_mean) / x_train_std
    
    y = w * x + b
    
    # reescalando el valor inferido
    y = y * y_train_std + y_train_mean
    
    return y''')

def inference(x, w, b, x_train_mean, x_train_std, y_train_mean, y_train_std) :
    x = (x - x_train_mean) / x_train_std
    y = w * x + b
    y = y * y_train_std + y_train_mean
    return y

st.code('''tiempos = [1.65, 32.20, 43.5, 84.7]

for t in tiempos :
    distancia = inference(x=t,w=weights[-1], b=biases[-1], \
                          x_train_mean = x_mean, x_train_std = x_std,\
                          y_train_mean = y_mean, y_train_std = y_std)

    print ("Para el tiempo de {0:5.2f} s la distancia inferida es {1:6.2f} cm ".\
           format(t,distancia))''')

tiempos = [1.65, 32.20, 43.5, 84.7]
for t in tiempos :
    distancia = inference(x=t,w=weights[-1], b=biases[-1], \
                          x_train_mean = x_mean, x_train_std = x_std,\
                          y_train_mean = y_mean, y_train_std = y_std)
    st.write("$Para el tiempo de {0:5.2f} s la distancia inferida es {1:6.2f} cm ".\
           format(t,distancia))

show_text_color('<center> Resumen de la solución de nuestro problema empleando Inteligencia Artificial')

col1, col2 = st.columns(2)

with col1:
   show_image_local(ruta+'AI-complete-system.png')

with col2:
   show_image_local(ruta+'Predictions.png')