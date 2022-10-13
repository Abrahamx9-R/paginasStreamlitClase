import base64
from matplotlib.backend_bases import RendererBase
import numpy as np
import json
from streamlit_lottie import st_lottie
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils import show_code
np.random.seed(3)

st.set_page_config(
    page_title="Clase 2",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://www.extremelycoolapp.com/help',
        'Report a Bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

def generador_datos_simple(n_points, distance_0, measuring_time, speed, max_distance_error):
    # Esta función genera un conjunto de datos que simulan 
    # la medición de la distancia de un carrito en un riel de aire
    # en la ausencia de una fuerza sobre el carrito.
    # Se propone un error en la medición de la distancia
    """
      n_points: number of point that will be generated, integer
      distance_0 : initial distantce (at time zero) 
      measuring_time: the time inteval used for the measurement
      speed : carś speed
      max_distance_error: Maximum error measuring distance
      
    """
    
    # n_points es el número de puntos que serán generados
    
    x = np.random.random(n_points) * measuring_time
     
    # x es arreglo con n_points numeros aleatorios entre 0.0 y measuring_time
    
    error = np.random.randn(n_points) * max_distance_error 
    
    # error es un error generado aleatoriamente con un valor maximo max_distance_error

    y = distance_0 + speed*x + error 
        
    return x, y

def plot_x_y_y__(x, y, y_, points=True):
    # Function for plotting (x,y) and (x,y_)
    
    fig, ax = plt.subplots()
    ax.set_ylabel('Y(cm)',fontdict = {'fontsize':16})
    ax.set_xlabel('X(seg)',fontdict = {'fontsize':16})
    ax.scatter(x,y,s=6)
    ax.plot(x, y_, color='green', lw=3, label='F(X, W, b)')
    ax.legend(fontsize=6)
    

    st.pyplot(fig)

def update_weights_biases(x, y, weight, bias, delta_weight, delta_bias):
    
    weight = weight + delta_weight
    
    bias = bias + delta_bias

    #The following date are for constructing the F(x,weight, bias)
    
    y_ = weight*x + bias
       
    residuo = np.mean((y - y_)**2)
    
    #print('residuo: {:10.2f}'.format(residuo))   
        
    return weight, bias, y_, residuo
st.markdown("# Clase 2")

def load_lottieurl(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

lottie_hello = load_lottieurl("pages/images/toaster.json")

st_lottie(lottie_hello,speed = 1, reverse=False,loop=True,quality="low",height=600,width=None,key=None,)

with open("pages/notebooks/Clase-2.zip", "rb") as fp:
    btn = st.download_button(
        label="Descarga notebook",
        data=fp,
        file_name='Clase-2.zip',
        mime='paginas/notebooks/',
    )
st.write('<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/40e0f5ea89eb43c398084f457933c6fa" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>',
    unsafe_allow_html=True,)

st.sidebar.header("Clase 2")
st.write(
    "## Ajuste de un conjunto de puntos con una función"
)
st.write(
    "####  Generación de las muestras"
)
st.write(
    "Se genera un conjunto de $m$ numeros aleatorios ($(x_1,y_1),(x_2,y_2),…,(x_m,y_m)$)"
)
st.code('''# Importamos las bibliotecas necesarias

import numpy as np
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
%matplotlib inline

# Fijamos una semilla
np.random.seed(3)''')

st.write(
    "Creamos una función que genere nuestros datos de forma aleatoria, simulando las mediciones del experimento de un carrito, como en el laboratorio de mecánica 1"
)

st.code('''def generador_datos_simple(n_points, distance_0, measuring_time, speed, max_distance_error):
    # Esta función genera un conjunto de datos que simulan 
    # la medición de la distancia de un carrito en un riel de aire
    # en la ausencia de una fuerza sobre el carrito.
    # Se propone un error en la medición de la distancia
    """
      n_points: number of point that will be generated, integer
      distance_0 : initial distantce (at time zero) 
      measuring_time: the time inteval used for the measurement
      speed : carś speed
      max_distance_error: Maximum error measuring distance
      
    """
    
    # n_points es el número de puntos que serán generados
    
    x = np.random.random(n_points) * measuring_time
     
    # x es arreglo con n_points numeros aleatorios entre 0.0 y measuring_time
    
    error = np.random.randn(n_points) * max_distance_error 
    
    # error es un error generado aleatoriamente con un valor maximo max_distance_error

    y = distance_0 + speed*x + error 
        
    return x, y''')
st.write("generamos los datos ")
st.code('''# Generacción de las muestras (xi,yi)
n_points = 1000
distance_0 = 100.0
measure_time = 100.0
speed = 20.0
max_distance_error = 100

x, y = generador_datos_simple(n_points, distance_0, measure_time, speed, max_distance_error)''')

st.sidebar.write('Generacion datos')
n_points = st.sidebar.number_input("Numero de datos",min_value=1,value=1000,format="%i")
distance_0 = st.sidebar.number_input("Distacia 0",value=100.0,format="%f")
measure_time = st.sidebar.number_input("Medidor de tiempo",value=100.0,format="%f")
speed = st.sidebar.number_input("Velocidad",value=20.0,format="%f")
max_distance_error = st.sidebar.number_input("Maxima distancia de error",value=100.0,format="%f")

x, y = generador_datos_simple(n_points, distance_0, measure_time, speed, max_distance_error)

fig, ax = plt.subplots()
ax.set_title('Datos',fontdict = {'fontsize':18})
ax.set_ylabel('Y(cm)',fontdict = {'fontsize':16})
ax.set_xlabel('X(seg)',fontdict = {'fontsize':16})
ax.scatter(x,y,s=6)

st.pyplot(fig)

st.write("# Buscando la correlación entre las muestras: ")
st.write(r'''Se tiene un conjunto de muestras (puntos) $(x_i, y_i)$, y se busca encontrar una función $F$ que describa una posible correlación entre ellos. $X$, con valores $x_i$ (en el presente caso el tiempo) es una variable independiente, mientras que $Y$, con los valores $y_i$ (la distancia en el presente caso) depende de $X$.

Para encontrar la correlación entre las muestras, proponemos un conjunto de funciones definidas mediante la siguiente relación lineal:

$$ F(X, W, b) = b + W X $$

en donde los parámetros $b$ y $W$ son variables que deben ser actualizadas hasta encontrar los valores que definan la función que mejor describa la correlación entre $X$ y $Y$. $$ $$
Vemos que esta relación funcional es derivable respecto a todas sus variables, $X, W, b$.
La letra $W$ se emplea como abreviación de la palabra en ingles "weight", porque se relaciona con la importancia que tiene la variable $X$ en el valor de la función $F$. La letra b es la abreviación de la palabra "bias" en ingles, $Y$ se refiere a la referencia respecto a cero de la función $F$.

Para encontrar la función que describe la correlación entre los puntos, es necesario generar una métrica para describir qué tanto se acerca cada una de las funciones específicas (con valores definidos de $W$ y de $b$) a esta descripción.

La métrica que se propone es la siguiente: $$ $$

Para cada muestra $(x_i, y_i)$ se evalua $F(x_i,W,b)$ y se compara con el correspondiente valor $y_i$, la diferencia entre estos valores se eleva al cuadrado. $$ $$
   $$ (F(x_i,w,b)-y_i)^{2}$$


Finalmente se calcula el promedio de este valor sobre todas las muestras, el cual definimos como error cuadrático medio (MSE, por sus siglas en ingles, Mean Squared Error). 
Si m es el número de muestras, el MSE queda como:

$$MSE = \dfrac {1}{m}∑_{i=1}^{m}(F(x_i,w,b)-y_i)^{2} $$
''')

st.write('''#### En el siguiente código se implementa la generación del error cuadrático medio dada una función específica definida por los pesos iniciales $W$= weight_0, y $b$ = bias_0.''')

st.code('''#Inicializando los valiores de la funcion
weight_0 = 10.0
bias_0 = 100.0''')

st.sidebar.write("### Iniciando valores funcion")

weight_0 = st.sidebar.number_input("Weight",value=10.0,format="%f")
bias_0 = st.sidebar.number_input("Bias",value=100.0,format="%f")

st.write('''#### Se grafica la correspondiente función $F(X,W,b)$, junto con los puntos que representan a las muestras''')

st.code('''

def plot_x_y_y__(x, y, y_, points=True):
    # Function for plotting (x,y) and (x,y_)
    
    plt.figure(figsize=(13,8))
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.ylabel('Y(cm)', fontsize=16)
    plt.xlabel('X(seg)', fontsize=16)

    #Plotting function
    plt.plot(x, y_, color='green', lw=4, label='F(X, W, b)')
    plt.legend()

    #Plotting samples
    plt.scatter(x, y)

    plt.show()
    
''')

st.write('''#### Se grafica la correspondiente función f para los valores $x_i$, así como las muestras $(x_i, y_i)$
''')

y_ = weight_0*x + bias_0

# Using the function f, the residuos is calculated by comparing the calculated and measured values

residuo = np.mean((y-y_)**2)

st.code('''#The following arrays are generated for plotting the Function F(x, weight_0, bias_0)

y_ = weight_0*x + bias_0

# Using the function F, the residuos is calculated by comparing the calculated and measured values

residuo = np.mean((y-y_)**2)

print('residuo: {0:10.2f}'.format( residuo))

plot_x_y_y__(x, y, y_, points=True)''')

st.write('$residuo: {0:10.2f}'.format( residuo))

plot_x_y_y__(x, y, y_, points=True)

st.write('De la gráficas se observa que la función está lejos de describir la correlación entre los punto')

st.write('''Los valores del peso $W$ y el bias $b$ se actualizan iterativamente a prueba y error. Para ello se hacen los cambios de acuerdo a la sugerencia de la observado en la gráfica.''')

st.code('''
def update_weights_biases(x, y, weight, bias, delta_weight, delta_bias):
    
    weight = weight + delta_weight
    
    bias = bias + delta_bias

    #The following date are for constructing the F(x,weight, bias)
    
    y_ = weight*x + bias
       
    residuo = np.mean((y - y_)**2)
    
    #print('residuo: {:10.2f}'.format(residuo))   
        
    return weight, bias, y_, residuo''')

st.code('''weight = weight_0
bias = bias_0
delta_w = 1.0
delta_b = -10.0

weight, bias, y_, residuo = update_weights_biases(x, y , weight, bias, delta_w, delta_b)
plot_x_y_y__(x, y, y_, points=True)''')

weight = weight_0
bias = bias_0
st.sidebar.write('### iniciando deltas')
delta_w = st.sidebar.number_input("delta_w",value=1.0,format="%f")
delta_b = st.sidebar.number_input("delta_b",value=-10.0,format="%f")

weight, bias, y_, residuo = update_weights_biases(x, y , weight, bias, delta_w, delta_b)
plot_x_y_y__(x, y, y_, points=True)



st.write('El cambio es muy pequeño, por ello el peso se actualiza con delta_weight = 2.0 y el bias con delta_bias= -100.0. ')

weight = weight
bias = bias
delta_w = 6.0
delta_b = -100.0

weight, bias, y_, residuo = update_weights_biases(x, y, weight, bias, delta_w, delta_b)
plot_x_y_y__(x, y, y_, points=True) 

st.write('Aun se puede mejorar disminuyendo el peso (pendiente) y aumentando el bias (ordenada en el origen):')

st.code('''weight = weight
bias = bias
delta_w = 6.0
delta_b = -100.0

weight, bias, y_, residuo = update_weights_biases(x, y, weight, bias, delta_w, delta_b)
plot_x_y_y__(x, y, y_, points=True) ''')



weight = weight
bias = bias
delta_w = 1.5
delta_b = 50.0
weight, bias, y_, residuo = update_weights_biases(x,y, weight, bias, delta_w, delta_b)
plot_x_y_y__(x, y, y_, points=True) 

st.code('''weight = weight
bias = bias
delta_w = 1.5
delta_b = 50.0

weight, bias, y_, residuo = update_weights_biases(x,y, weight, bias, delta_w, delta_b)
plot_x_y_y__(x, y, y_, points=True) ''')

st.write(r'Se puede mejorar el resultado. Para ello se observa el cambio en el error cuadrático medio con los cambios en $W$ y en $b$.')

st.write(r'''Busquemos encontrar la función que defina la correlación generando un conjunto de funciones y calcular los correspondientes residuos.

Para ello emplearemos la función **update_weight_bias()**, la cual actualiza los parametros $W$ y $b$.''')

st.code('''update_outputs = []

weight = weight_0
bias = -3.0
delta_w = 0.2 
delta_b = 0.0
iterations = 100

for i in range(iterations):
    
    weight, bias, y_, residuo = update_weights_biases(x, y, weight, bias, delta_w, delta_b)
    
    update_outputs.append([weight, bias, y_, residuo])
    
    if i % 10 == 0 :
        print('weight: {0:8.2f}    bias: {1:5.2f}   residuo: {2:10.2f}'.format(weight,bias, residuo))''')

update_outputs = []

weight = weight_0
bias = -3.0
delta_w = 0.2 
delta_b = 0.0
iterations = 120

for i in range(iterations):
    
    weight, bias, y_, residuo = update_weights_biases(x, y, weight, bias, delta_w, delta_b)
    
    update_outputs.append([weight, bias, y_, residuo])
    
    if i % 10 == 0 :
        st.write('$weight: {0:8.2f}    bias: {1:5.2f}   residuo: {2:10.2f}'.format(weight,bias, residuo))

update_outputs_weight = update_outputs
st.code('update_outputs_weight = update_outputs')
st.code('''plt.figure(figsize=(13,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=12)
plt.ylabel('Y', fontsize=16)
plt.xlabel('X', fontsize=16)

for i in range(0,50,3):

    plt.plot(x, update_outputs[i][0]*x + update_outputs[i][1], label='v' + str(i), lw=3)
    plt.legend()

plt.scatter(x, y)

plt.show()''')

fig, ax = plt.subplots()
ax.set_ylabel('Y',fontdict = {'fontsize':16})
ax.set_xlabel('X',fontdict = {'fontsize':16})
ax.legend(fontsize=6)
ax.scatter(x,y,s=6)

for i in range(0,50,3):
    ax.plot(x, update_outputs[i][0]*x + update_outputs[i][1], label='v' + str(i), lw=3)
    ax.legend(fontsize=6)
ax.scatter(x, y,s=6)

st.pyplot(fig)

st.write('''A continuación se grafica el residuo obtenido para cada función al variar el parámetro W.
 
 El valor óptimo del parámetro es aquel que minimiza al residuo.''')

st.code('''plt.figure(figsize=(13,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.ylabel('Residuo', fontsize=18)
plt.xlabel('W', fontsize=16)

weights = []

residuos = []

for update_output in update_outputs:
    
    weights.append(update_output[0])
    
    residuos.append(update_output[3])
        
plt.scatter(weights, residuos)


plt.show()''')

fig, ax = plt.subplots()
ax.set_ylabel('Residuo',fontdict = {'fontsize':16})
ax.set_xlabel('W',fontdict = {'fontsize':16})
ax.legend(fontsize=6)

weights = []
residuos = []

for update_output in update_outputs:
    weights.append(update_output[0])
    residuos.append(update_output[3])
        
ax.scatter(weights, residuos,s=6)
st.pyplot(fig)

st.write('''Como se observa en la grafica anterior, el mínimo del parámetro $W$ esta cerca de $W$= 22.''')

st.code('''index_min = np.argmin(residuos)   # returns the index for the minimum value in the list

print('El residuo (MSE) mas pequeño tiene el valor: {0:10.2f}'.format(update_outputs[index_min][3]))

w_opt = update_outputs[index_min][0]
b_opt = update_outputs[index_min][1]

print('Los valores optimos de los parámetros son W = {0:5.2f} y b = {1:5.2f} ' \
      .format(w_opt, b_opt))''')

index_min = np.argmin(residuos)   # returns the index for the minimum value in the list

st.write('$El residuo mas pequeño tiene el valor: {0:10.2f}'.format(update_outputs[index_min][3]))

w_opt = update_outputs[index_min][0]
b_opt = update_outputs[index_min][1]

st.write('$Los valores optimos de los parámetros son W = {0:5.2f} y b = {1:5.2f} ' \
      .format(w_opt, b_opt))

st.write('''Se grafica la recta con los valores óptimos encontrados w_opt and b_opt :''')

st.code('''y_ = w_opt*x + b_opt

plot_x_y_y__(x, y, y_, points=True)''')

y_ = w_opt*x + b_opt

plot_x_y_y__(x, y, y_, points=True)

st.write('Se genera la derivada parcial del residuo respecto a $W$ para cada valor de $W$ y se grafica.')

st.code('''residuos_array = np.asarray(residuos)
residuos_grad = np.gradient(residuos_array)
#residuos_grad''')

st.code('''plt.figure(figsize=(13,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.ylabel('Derivada del MSE', fontsize=16)
plt.xlabel('W', fontsize=16)
plt.grid(True)
plt.scatter(weights, residuos_grad)

plt.show()''')

residuos_array = np.asarray(residuos)
residuos_grad = np.gradient(residuos_array)

fig, ax = plt.subplots()
ax.set_ylabel('Derivada del Residuo (MSE)',fontdict = {'fontsize':16})
ax.set_xlabel('W',fontdict = {'fontsize':16})
ax.legend(fontsize=6)
ax.grid(alpha = 0.2)
        
ax.scatter(weights, residuos_grad,s=6)
st.pyplot(fig)

st.write('''
En este caso, como el residuo es una parábola, su derivada esta dada por la recta. 
    
$$ \dfrac{∂Residuo}{∂{W}} = -\dfrac {2}{m} ∑_{i=1}^{m}(y_i - b -Wx_i)x_i= -\dfrac {2}{m} ∑_{i=1}^{m}(y_i x_i- b x_i-Wx_i^{2})$$

$$ $$
En general para cualquier otro problema, el residuo tendrá una forma muy diferente al de una parábola''')

st.write('Se generan un conjunto de funciones dejando $W$ fija y variando $b$.')

st.code('''update_outputs = []

weight =  w_opt
bias = b_opt
delta_w = 0.0
delta_b = 1.0
iterations = 100

for i in range(iterations):
    
    weight, bias, y_, residuo = update_weights_biases(x, y, weight, bias, delta_w, delta_b)
    
    update_outputs.append([weight, bias, y_, residuo])
    
    if i % 10 == 0 :
        print('weight: {0:5.2f},   bias: {1:5.2f},   residuo: {2:8.2f}'.format(weight,bias,residuo))''')

update_outputs = []
weight =  w_opt
bias = b_opt
delta_w = 0.0
delta_b = 1.0
iterations = 100
for i in range(iterations):
    weight, bias, y_, residuo = update_weights_biases(x, y, weight, bias, delta_w, delta_b)
    update_outputs.append([weight, bias, y_, residuo])
    if i % 10 == 0 :
        st.write('$weight: {0:5.2f},   bias: {1:5.2f},   residuo: {2:8.2f}'.format(weight,bias,residuo))
        
st.code('''plt.figure(figsize=(13,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=12)
plt.ylabel('Y', fontsize=16)
plt.xlabel('X', fontsize=16)

for i in range(0,50,3):

    plt.plot(x, w_opt*x + update_outputs[i][1], label='v' + str(i), lw=3)
    plt.legend()

plt.scatter(x, y)

plt.show()''')

fig, ax = plt.subplots()
ax.set_ylabel('Y',fontdict = {'fontsize':16})
ax.set_xlabel('X',fontdict = {'fontsize':16})
ax.legend(fontsize=6)
ax.scatter(x,y,s=6)

for i in range(0,50,3):
    ax.plot(x, w_opt*x + update_outputs[i][1], label='v' + str(i), lw=3)
    ax.legend(fontsize=6)
ax.scatter(x, y,s=6)

st.pyplot(fig)

st.write('''A continuación se grafica el residuo en función de $b$.
 
 El valor óptimo de $b$ es aquel que minimiza al residuo. ''')

st.code('''plt.figure(figsize=(13,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.ylabel('Residuo', fontsize=16)
plt.xlabel('b', fontsize=16)

bias = []

residuos = []

for update_output in update_outputs:
    
    bias.append(update_output[1])
    
    residuos.append(update_output[3])
    
plt.scatter(bias, residuos)

plt.show()''')

fig, ax = plt.subplots()
ax.set_ylabel('Residuo',fontdict = {'fontsize':16})
ax.set_xlabel('b',fontdict = {'fontsize':16})
ax.legend(fontsize=6)

bias = []
residuos = []

for update_output in update_outputs:
    bias.append(update_output[1])
    residuos.append(update_output[3])

ax.scatter(bias, residuos,s=6)
st.pyplot(fig)

st.write('Como se observa en la grafica el mínimo del residuo corresponde a un valor de $b$ cercano a 20.')

st.code('''index_min = np.argmin(residuos)   # returns the index for the minimum value in the list
residuo_min = update_outputs[index_min][3]

print('El residuo mas pequeño tiene el valor: {0:10.2f}'.format(residuo_min))

w_opt = update_outputs[index_min][0]
b_opt = update_outputs[index_min][1]

print('Los valores optimos de los parámetros son W = {0:5.2f} y b = {1:5.2f} ' \
      .format(w_opt, b_opt))''')

index_min = np.argmin(residuos)   # returns the index for the minimum value in the list
residuo_min = update_outputs[index_min][3]

st.write('$El residuo mas pequeño tiene el valor: {0:10.2f}'.format(residuo_min))

w_opt = update_outputs[index_min][0]
b_opt = update_outputs[index_min][1]

st.write('$Los valores optimos de los parámetros son W = {0:5.2f} y b = {1:5.2f} ' \
      .format(w_opt, b_opt))

st.write('Se grafica la recta empleando los valores óptimos $W$, w_opt, y $b$, b_opt.')

st.code('''y_ = w_opt*x + b_opt
y_hand = y_

plot_x_y_y__(x, y, y_, points=True)''')

y_ = w_opt*x + b_opt
y_hand = y_

plot_x_y_y__(x, y, y_, points=True)

st.write('Se genera la derivada parcial del residuo respecto a $W$ para cada valor de $W$ y se grafica.')

st.code('''residuos_array = np.asarray(residuos)
residuos_grad = np.gradient(residuos_array)
#residuos_grad
    
plt.figure(figsize=(13,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.ylabel('Derivada del Residuo', fontsize=18)
plt.xlabel('b', fontsize=18)

bias = []

residuos = []

for update_output in update_outputs:
    
    bias.append(update_output[1])
    
    residuos.append(update_output[3])

plt.grid(True)
plt.scatter(bias, residuos_grad)

plt.show()''')

residuos_array = np.asarray(residuos)

residuos_grad = np.gradient(residuos_array)

fig, ax = plt.subplots()
ax.set_ylabel('Derivada del Residuo',fontdict = {'fontsize':16})
ax.set_xlabel('b',fontdict = {'fontsize':16})
ax.legend(fontsize=6)

ax.scatter(bias, residuos_grad,s=6)
st.pyplot(fig)

st.write('En este caso, como el residuo es una parábola, su derivada esta dada por la recta ')
    
st.latex(r'\dfrac{∂Residuo}{∂{b}} = -\dfrac {2}{m} ∑_{i=1}^{m}(y_i - b -Wx_i)')

st.write('En general para cualquier otro problema, el residuo tendrá una forma muy diferente al de una parábola')

st.write('## Métodos alternativos para optimizar los parámetros $W$ y $b$')

st.write('### Método simple de mínimos cuadrados, de acuerdo a Gauss')

st.write(r'''El método de mínimos cuadrados nos permite encontrar una función que describe la correlación que tienen un conjunto m de puntos $(x_i, y_i)$, en donde $x_i$ son los valores que toma la variable $X$ y $y_i$ son los valores que toma la variable $Y$.


El objetivo del algoritmo es obtener la relación entre la variable independiente $X$, y la variable dependiente $Y$.

En el presente caso, se propone una función lineal para describir la correlación:''')

st.latex(r'F(X,W,b) = b + W X')
 
st.write('''
Los parámetros variables $b$  y $W$ definen a la función $F$. 

Con el método simple de mínimos cuadrados se hallan los valores óptimos w_opt y b_opt de estas variables.''')

st.write('La diferencia entre el valor real $y_i$ de $Y$ y el valor estimado $F(x_i, W, b)$ se denomina el residuo ($r_i$):')
st.latex(r'r_i = y_i - F(x_i, W, b) ')
st.write('El objetivo del método de mínimos cuadrados es minimizar la suma del cuadrado de estos  residuos, es decir, encontrar el mínimo de la función Residuo:')
st.latex(r'Residuo = \dfrac {1}{m} ∑_{i=1}^{m}(y_i - b -Wx_i)^{2}')
st.write(r'''El mínimo de esta función, se encuentra en aquellos valores de $W$ ($W_{opt}$) y $b$ ($b_{opt}$) en donde la derivada de la función residuo es igual a zero. 
    
    
Para encontrar este mínimo, primero calculamos la derivada parcial de la función respecto a cada uno de los parámetros $W$ y $b$, después, cada derivada la igualamos a cero:''')
st.latex(r'\dfrac{∂Residuo}{∂{W}} =0⟶ W_{opt}=\dfrac{∑_{i=1}^{n}(x_i–\bar x)(y_i–\bar y)}{∑_{i=1}^{m}(x_i–\bar x)^2}')
st.latex(r' \dfrac{∂Residuo}{∂{b}}=0⟶ b_{opt}=y – W_{opt} x ')
st.write(r'en donde $\bar x$ y $\bar y$ son los valores promedio de los valores de las variables $X$ y $Y$.')

st.code('''def mean_square_error(x, y):
    
    # 1) Se obtiene el promedio de los valores x_i y y_i

    mean_x = np.mean(x)

    mean_y = np.mean(y)

    # 2) se calcula (x_i-x) y (y_i-y), con x e y los promedios de x_i e y_i, respectivamente.

    x_i = []

    for i in x:

        x_i.append(np.squeeze(i) - mean_x)

    y_i = []

    for i in y:

        y_i.append(np.squeeze(i) - mean_y)

    # 3) se calcula (x_i-x)*(x_i-x)

    x_i2 = np.power(x_i, 2)

    # 4) se hacen las sumas correspondientes

    xy_sum = 0

    for i in range(len(x_i)):

        xy_sum += x_i[i]*y_i[i]

    x2_sum = 0

    for i in range(len(x_i)):

        x2_sum += x_i2[i]

    # Se definen a w y b 

    w = xy_sum/x2_sum

    b = mean_y - w*mean_x
       
    return w, b
''')
def mean_square_error(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    x_i = []
    for i in x:
        x_i.append(np.squeeze(i) - mean_x)
    y_i = []
    for i in y:
        y_i.append(np.squeeze(i) - mean_y)
    x_i2 = np.power(x_i, 2)
    xy_sum = 0
    for i in range(len(x_i)):
        xy_sum += x_i[i]*y_i[i]
    x2_sum = 0
    for i in range(len(x_i)):
        x2_sum += x_i2[i]
    w = xy_sum/x2_sum
    b = mean_y - w*mean_x
    return w, b
st.code('''
w_opt_mse, b_opt_mse = mean_square_error(x, y)

print(" w_opt = {0:5.2f}, b_opt = {1:5.2f}".format(w_opt_mse,b_opt_mse))''')


w_opt_mse, b_opt_mse = mean_square_error(x, y)
st.write("$ w_opt = {0:5.2f}, b_opt = {1:5.2f}".format(w_opt_mse,b_opt_mse))

st.code('''y_ = w_opt_mse*x + b_opt_mse

plot_x_y_y__(x, y, y_, points=True)''')

y_ = w_opt_mse*x + b_opt_mse

plot_x_y_y__(x, y, y_, points=True)

st.code('''MSE = np.mean((y-w_opt*x-b_opt)**2)
print('MSE: {0:10.2f}'.format(MSE))''')

MSE = np.mean((y-w_opt*x-b_opt)**2)
st.write('$MSE: {0:10.2f}'.format(MSE))

st.code('''y_MSE = x*w_opt + b_opt

plt.figure(figsize=(13,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.xlabel('X', fontsize=16)


plt.plot(x, y_hand, color='magenta', label='hand_MSE-residuo', lw=6)

plt.plot(x, y_MSE, color='blue', label = 'Gauss_MSE-residuo', lw=3)

plt.legend()

plt.scatter(x, y)

plt.show()
''')


y_MSE = x*w_opt + b_opt
fig, ax = plt.subplots()
ax.set_ylabel('Y',fontdict = {'fontsize':16})
ax.set_xlabel('X',fontdict = {'fontsize':16})
ax.plot(x, y_hand, color='magenta', label='hand_MSE-residuo', lw=6)
ax.legend(fontsize=6)
ax.plot(x, y_MSE, color='blue', label = 'Gauss_MSE-residuo', lw=3)
ax.legend(fontsize=6)
ax.scatter(x, y,s=6)
st.pyplot(fig)

st.write('''Como vimos anteriormente, la función $F(X,W,b)$ que describe la correlacion entre las variables $X$ y $Y$, se obtiene definiendo la métrica descrita por la función $Residuo$.
Cuando esta función Residuo tiene su mínimo (el error mínimo) su derivada es cero. Es por ello, que podriamos emplear el método de Newton-Raphson para encontrar este cero.''')

st.write('''### Método de Newton-Raphson para encontrar el cero de una función. 
    
### Este método emplea la función y su derivada.''')

st.write('''
Otra alternativa para encontrar los valores de W y b para los cuales el residuo es mínimo es emplear un método iterativo desarrollado por Isaac Newton y Joseph Raphson en el siglo XVII para encontrar los ceros de una función. Hoy en día, este método es conocido como el método de Newton-Raphson.
   
Para emplear este método en nuestro caso se tendría que obtener la derivada de la función Residuo. Los valores de $W$ y $b$ se actualizarían empleando la siguiente relación:  ''')
st.latex(r'W_{new} = W_{actual} - \dfrac {Residuo} {\dfrac {∂Residuo}{∂{W}}}')
st.latex(r'b_{new} = b_{actual} - \dfrac {Residuo} {\dfrac {∂Residuo}{∂{b}}}')
st.write('''El código que generaremos para emplear este método requiere un valor inicial de la variable que define la función, así como una epsilon para estimar si el error ya se puede considerar como cero. El método tambien toma en cuenta el número máximo de iteraciones que se deben realizar para encontrar este cero.''')
st.write('''Para introducir este método emplearemos la función $f(x)$ que depende de $x$, y de la cual queremos obtener sus ceros. La expresión correspondiente queda como:''')
st.latex(r'x_{new} = x_{actual} - \dfrac {f(x)} {\dfrac {df(x)}{d{x}}}')

file_ = open("pages/images/NewtonIteration_Ani.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<center> <img width="90%" src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
    unsafe_allow_html=True,
) 
st.markdown('(By Ralf Pfeifer - de:Image:NewtonIteration Ani.gif, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=2268473)')

st.markdown('A continuación definimos la función newton_raphson para implementar este método')


st.code('''def newton_raphson(f, Df,x0,epsilon,max_iter):
    #Approximate solution of f(x)=0 by Newton's method.

    #Parameters
    #----------
    #f : function
        
    #Df : Derivative of f(x).
    
    #x0 : Initial guess for finding the root of f(x).
    
    #epsilon :Stopping criteria: the iteration ends when abs(f(x)) < epsilon.
    
    #max_iter : Maximum number of iterations of Newton's method.

    #Returns
    #-------
    #xn : number
        #Implement Newton's method: compute the linear approximation
        #of f(x) at xn and find x intercept by the formula
            #x = xn - f(xn)/Df(xn)
        #Continue until abs(f(xn)) < epsilon and return xn.
        #If Df(xn) == 0, return None. If the number of iterations
        #exceeds max_iter, then return None.
    
    aprox_root = [x0]
    
    xn = x0
      
    # xn es la aproximación de la raíz de f. Inicialmente xn =x0 con x0 la primera aproximación
    
    for n in range(0,max_iter):
        
        fxn = f(xn)
       
        #print("xn = ", xn, "aprox_root = ", aprox_root)
        
        if abs(fxn) < epsilon:
            
            print("x = ", xn, ", f(x) = ", fxn, ", df(x)/dx = ", Dfxn)
            print('The solution is found after',n,'iterations.')
                        
            return xn, aprox_root
        
        Dfxn = Df(xn)
        
        if Dfxn == 0:
            
            print('Zero derivative. No solution found.')
            
            return None
        
        print("x = ", xn, ", f(x) = ", fxn, ", df(x)/dx = ", Dfxn)
        
        xn = xn - fxn/Dfxn
        
        aprox_root.append(xn)
        
    print('Exceeded maximum iterations. No solution found.')
    
    return None''')

def newton_raphson(f, Df,x0,epsilon,max_iter):
    aprox_root = [x0]
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            st.write("$x = ", xn, ", f(x) = ", fxn, ", df(x)/dx = ", Dfxn)
            st.write('$The solution is found after',n,'iterations.')      
            return xn, aprox_root
        Dfxn = Df(xn)
        if Dfxn == 0:
            st.write('$Zero derivative. No solution found.')
            return None
        st.write("x = ", xn, ", f(x) = ", fxn, ", df(x)/dx = ", Dfxn)
        xn = xn - fxn/Dfxn
        aprox_root.append(xn)
    st.write('$Exceeded maximum iterations. No solution found.')
    return None

st.write('Se prueba este método con una función simple')

st.code('''#p = lambda x: x**3-27

def p(x):
    
    return x**3-10''')

def p(x):    
    return x**3-10

st.code('''x_ = np.arange(-10, 10, 0.2)
print (x_.shape)
sigma_samples = 5
y_ = p(x_)
print(y_[:5])

plt.figure(figsize=(13,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.xlabel('X', fontsize=16)


plt.grid(True)

plt.plot(x_, y_, "o", ms=5, alpha= 0.5, color='r')

plt.show()

#print(y_)''')


x_ = np.arange(-10, 10, 0.2)
st.write(x_.shape)
sigma_samples = 5
y_ = p(x_)
st.write(y_[:5])

fig, ax = plt.subplots()
ax.set_ylabel('Y',fontdict = {'fontsize':16})
ax.set_xlabel('X',fontdict = {'fontsize':16})
ax.grid(True)
ax.plot(x_, y_, "o", ms=5, alpha= 0.5, color='r')
st.pyplot(fig)

st.write('Calculemos la derivada de la función')

st.code('''#Dp = lambda x: 3*x**2 

def Dp(x):
    
    return 3*x**2
''') 

def Dp(x):
    return 3*x**2

st.write('Para visualizar el método dibujemos la pendiente en cada punto en donde esta se calcula durante la búsqueda del cero.')

st.code('''def tangent_line(f, Df, x_0, a, b):
        
    x = np.linspace(a,b)
    
    y = f(x) 
    
    y_0 = f(x_0)
    
    y_tan = Df(x_0) * (x - x_0) + y_0 
    
    plt.figure(figsize=(13,8))
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.xlabel('X', fontsize=16)



    #plt.plot(x,y,'r-')
    plt.plot(x_, y_, "o", ms=5, alpha= 0.5, color='r')
    
    plt.plot(x,y_tan,'b-')
    
    plt.xlabel('x') 
    
    plt.ylabel('y') 
    
    plt.grid(True)
    
    plt.title('Plot of the function and its tangent at x') 
    
    plt.show()  ''')

def tangent_line(f, Df, x_0, a, b):
    x = np.linspace(a,b)
    y_0 = f(x_0)  
    y_tan = Df(x_0) * (x - x_0) + y_0   
    fig, ax = plt.subplots()
    ax.set_ylabel('Y',fontdict = {'fontsize':16})
    ax.set_xlabel('X',fontdict = {'fontsize':16})
    ax.grid(True)
    ax.plot(x_, y_, "o", ms=5, alpha= 0.5, color='r')   
    ax.plot(x,y_tan,'b-')
    ax.grid(True)
    ax.set_title('Plot of the function and its tangent at x') 
    st.pyplot(fig)

st.code('''
for i in newton_raphson(p,Dp,7.5,0.001,50)[1]:

    tangent_line(p, Dp, i, -10, 10)''')

for i in newton_raphson(p,Dp,7.5,0.001,50)[1]:
    tangent_line(p, Dp, i, -10, 10)
    
st.markdown(r'''Para aplicar este método en nuestro caso, es necesario implementarlo para dos variables ($W$ y $b$). Sin embargo, en nuestro caso el gradiente del Residuo varía linealmente con cada una de estas variables. Su pendiente es entonces constante.

El método en general ha sido implementado para problemas con multivariables (número de variables mayor a 2)

(Descarga el contenido y se encuentra en el archivo **Newton-Raphson-multivariate.pdf**)


Otra alternativa para encontrar los valores de $W$ y $b$ para los cuales el residuo es mínimo, es emplear un método iterativo desarrollado por Cauchy en el siglo XIX. Hoy en día este método de optimización es conocido como el método de gradiente descendente.
    
(Descarga el contenido y se encuentra en el archivo **Cauchy_gradient-descent.pdf**)
    
En este caso durante la optimización, los valores de los parámetros $W$ y $b$ cambian su valor disminuyéndolo con el respectivo valor del gradiente del residuo multiplicado par el factor $ \alpha $ (> 0).''')

st.latex(r'W_{new} = W_{actual} - \alpha \dfrac{∂Residuo}{∂{W}}')
st.latex(r'b_{new} = b_{actual} - \alpha \dfrac{∂Residuo}{∂{b}}')

st.write('### Posible idea origen de la propuesta de Cauchy')
st.write('Busquemos el mínimo de una función. Por ejemplo el error que obtenemos cuando variamos los pesos para la distribución que datos que estamos analizando')

st.code('''plt.figure(figsize=(13,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.ylabel('MSE', fontsize=18)
plt.xlabel('W', fontsize=16)

weights = []

mse = []

for update_output in update_outputs_weight:
    
    weights.append(update_output[0])
    
    mse.append(update_output[3])
    
plt.scatter(weights, mse)


plt.show()''')

fig, ax = plt.subplots()
ax.set_ylabel('MSE',fontdict = {'fontsize':16})
ax.set_xlabel('W',fontdict = {'fontsize':16})
ax.legend(fontsize=6)

weights = []
mse = []
for update_output in update_outputs_weight:
    weights.append(update_output[0])
    mse.append(update_output[3])
    
ax.scatter(weights, mse,s=6)

st.pyplot(fig)

st.code('''bias_0 = 3.0
weight_0 = update_outputs_weight[110][0]
mse_0 = update_outputs_weight[110][-1]
print(weight_0, mse_0)''')

bias_0 = 3.0
weight_0 = update_outputs_weight[110][0]
mse_0 = update_outputs_weight[110][-1]
st.write("$",weight_0, mse_0)


st.write('Empezemos con el valor de Weight_0 = 32.2')

st.code('''plt.figure(figsize=(13,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.ylabel('MSE', fontsize=18)
plt.xlabel('W', fontsize=16)

weights = []

mse = []

for update_output in update_outputs_weight:
    
    weights.append(update_output[0])
    
    mse.append(update_output[3])
    
plt.scatter(weights, mse)
plt.scatter(weight_0, mse_0, color='red', linewidths=10)

plt.show()
        ''')


fig, ax = plt.subplots()
ax.set_ylabel('MSE',fontdict = {'fontsize':16})
ax.set_xlabel('W',fontdict = {'fontsize':16})
ax.legend(fontsize=6)

weights = []
mse = []
for update_output in update_outputs_weight:
    weights.append(update_output[0])
    mse.append(update_output[3])
    
ax.scatter(weights, mse,s=6)
ax.scatter(weight_0, mse_0, color='red', linewidths=10,s=6)
st.pyplot(fig)

st.code('''mse_grad = np.gradient(np.asarray(mse))
print(mse_grad[110])''')

mse_grad = np.gradient(np.asarray(mse))
st.write("$",mse_grad[110])

mse_grad_weight_0 = mse_grad[110]
weight_1 = weight_0 - mse_grad_weight_0
bias_1 = bias_0

st.code('''mse_grad_weight_0 = mse_grad[110]
weight_1 = weight_0 - mse_grad_weight_0
bias_1 = bias_0
print(weight_1)''')

st.write(weight_1)

st.code('''alpha = 0.0001
weight_1 = weight_0 - alpha * mse_grad_weight_0
print(weight_1)''')

alpha = 0.0001
weight_1 = weight_0 - alpha * mse_grad_weight_0
st.write("$",weight_1)

def calcule_error(x, y, weight, bias):
    #The following date are for constructing the F(x,weight, bias)
    y_ = weight*x + bias
    mse = np.mean((y - y_)**2)    
    #print('residuo: {:10.2f}'.format(residuo))   
    return mse

st.code('''def calcule_error(x, y, weight, bias):
#The following date are for constructing the F(x,weight, bias)
y_ = weight*x + bias
mse = np.mean((y - y_)**2)    
#print('residuo: {:10.2f}'.format(residuo))   
return mse''')

mse_1 = calcule_error(x,y,weight_1,bias_1)
weights_interest = []
weights_interest.append(weight_0)
mses_interest = []
mses_interest.append(mse_0)
weight_start = weight_0
bias_start = bias_0
for itera in range(1,6):
    weight_new = weight_start - alpha * mse_grad_weight_0 
    bias_new = bias_start
    mse_new = calcule_error(x, y, weight_new, bias_new)
    weights_interest.append(weight_new)
    mses_interest.append(mse_new)      
    weight_start = weight_new
    bias_start = bias_new
    
st.code('''mse_1 = calcule_error(x,y,weight_1,bias_1)
weights_interest = []
weights_interest.append(weight_0)

mses_interest = []
mses_interest.append(mse_0)

weight_start = weight_0
bias_start = bias_0

for itera in range(1,6):
    
    weight_new = weight_start - alpha * mse_grad_weight_0   #The gradient is fixed, is should change for each weight
    bias_new = bias_start
    
    mse_new = calcule_error(x, y, weight_new, bias_new)
    
    weights_interest.append(weight_new)
    mses_interest.append(mse_new) 
        
    weight_start = weight_new
    bias_start = bias_new
    
    print(weights_interest[0], mses_interest[0])''')

st.write("$",weights_interest[0], mses_interest[0])

weights = []
mse = []
for update_output in update_outputs_weight:    
    weights.append(update_output[0])
    mse.append(update_output[3])

st.code('''weights = []

mse = []

for update_output in update_outputs_weight:
    
    weights.append(update_output[0])
    
    mse.append(update_output[3])
''')

fig, axs = plt.subplots(2, 3, sharey=True)

axs[0,0].set_xlabel('W', size=6)
axs[0,0].set_ylabel('MSE', size=6)
axs[0,0].xaxis.set_tick_params(labelsize=6)
axs[0,0].yaxis.set_tick_params(labelsize=6)

axs[0,1].set_xlabel('W', size=6)
axs[0,1].set_ylabel('MSE', size=6)
axs[0,1].xaxis.set_tick_params(labelsize=6)
axs[0,1].yaxis.set_tick_params(labelsize=6)

axs[0,2].set_xlabel('W', size=6)
axs[0,2].set_ylabel('MSE', size=6)
axs[0,2].xaxis.set_tick_params(labelsize=6)
axs[0,2].yaxis.set_tick_params(labelsize=6)

axs[1,0].set_xlabel('W', size=6)
axs[1,0].set_ylabel('MSE', size=6)
axs[1,0].xaxis.set_tick_params(labelsize=6)
axs[1,0].yaxis.set_tick_params(labelsize=6)

axs[1,1].set_xlabel('W', size=6)
axs[1,1].set_ylabel('MSE', size=6)
axs[1,1].xaxis.set_tick_params(labelsize=6)
axs[1,1].yaxis.set_tick_params(labelsize=6)

axs[1,2].set_xlabel('W', size=6)
axs[1,2].set_ylabel('MSE', size=6)
axs[1,2].xaxis.set_tick_params(labelsize=6)
axs[1,2].yaxis.set_tick_params(labelsize=6)

axs[0,0].set_title('Weight = '+ "{:.3f}".format(weights_interest[0]), size=6)
axs[0,0].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[0,0].scatter(weights_interest[0], mses_interest[0], color='red', linewidths=8,s=6)

axs[0,1].set_title('Weight = '+ "{:.3f}".format(weights_interest[1]), size=6)
axs[0,1].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[0,1].scatter(weights_interest[1], mses_interest[1], color='red', linewidths=8,s=6)

axs[0,2].set_title('Weight = '+ "{:.3f}".format(weights_interest[2]), size=6)
axs[0,2].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[0,2].scatter(weights_interest[2], mses_interest[2], color='red', linewidths=8,s=6)

axs[1,0].set_title('Weight = '+ "{:.3f}".format(weights_interest[3]), size=6)
axs[1,0].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[1,0].scatter(weights_interest[3], mses_interest[3], color='red', linewidths=8,s=6)

axs[1,1].set_title('Weight = '+ "{:.3f}".format(weights_interest[4]), size=6)
axs[1,1].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[1,1].scatter(weights_interest[4], mses_interest[4], color='red', linewidths=8,s=6)

axs[1,2].set_title('Weight = '+ "{:.3f}".format(weights_interest[5]), size=6)
axs[1,2].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[1,2].scatter(weights_interest[5], mses_interest[5], color='red', linewidths=8,s=6)

st.pyplot(fig)

st.code('''weight_0 = update_outputs_weight[15][0]
mse_0 = update_outputs_weight[15][-1]
print(weight_0, mse_0)''')

weight_0 = update_outputs_weight[15][0]
mse_0 = update_outputs_weight[15][-1]
st.write("$",weight_0, mse_0)

st.code('''mse_grad_weight_0 = mse_grad[15]
print(mse_grad_weight_0)''')

mse_grad_weight_0 = mse_grad[15]
st.write("$",mse_grad_weight_0)

st.code('''weights_interest = []
weights_interest.append(weight_0)

mses_interest = []
mses_interest.append(mse_0)

weight_start = weight_0
bias_start = bias_0

for itera in range(1,6):
    
    weight_new = weight_start - alpha * mse_grad_weight_0   #The gradient is fixed, is should change for each weight
    bias_new = bias_start
    
    mse_new = calcule_error(x, y, weight_new, bias_new)
    
    weights_interest.append(weight_new)
    mses_interest.append(mse_new) 
        
    weight_start = weight_new
    bias_start = bias_new
    ''')

weights_interest = []
weights_interest.append(weight_0)
mses_interest = []
mses_interest.append(mse_0)
weight_start = weight_0
bias_start = bias_0
for itera in range(1,6):
    weight_new = weight_start - alpha * mse_grad_weight_0  
    bias_new = bias_start
    mse_new = calcule_error(x, y, weight_new, bias_new)
    weights_interest.append(weight_new)
    mses_interest.append(mse_new) 
    weight_start = weight_new
    bias_start = bias_new
 
st.code('''weights = []

mse = []

for update_output in update_outputs_weight:
    
    weights.append(update_output[0])
    
    mse.append(update_output[3])''')

weights = []
mse = []
for update_output in update_outputs_weight:
    weights.append(update_output[0])
    mse.append(update_output[3])

st.code('''plt.figure(figsize=(15,8)) 

ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)
ax4 = plt.subplot(2,3,4)
ax5 = plt.subplot(2,3,5)
ax6 = plt.subplot(2,3,6)

ax1.set_xlabel('W', size=12)
ax1.set_ylabel('MSE', size=12)

ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax4.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax5.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax6.ticklabel_format(axis="y", style="sci", scilimits=(0,0))


ax1.set_title('Weight = '+ "{:.3f}".format(weights_interest[0]))
ax1.scatter(weights, mse, color='blue', linewidths=0.5)
ax1.scatter(weights_interest[0], mses_interest[0], color='red', linewidths=8)

ax2.set_title('Weight = '+ "{:.3f}".format(weights_interest[1]))
ax2.scatter(weights, mse, color='blue', linewidths=0.5)
ax2.scatter(weights_interest[1], mses_interest[1], color='red', linewidths=8)

ax3.set_title('Weight = '+ "{:.3f}".format(weights_interest[2]))
ax3.scatter(weights, mse, color='blue', linewidths=0.5)
ax3.scatter(weights_interest[2], mses_interest[2], color='red', linewidths=8)

ax4.set_title('Weight = '+ "{:.3f}".format(weights_interest[3]))
ax4.scatter(weights, mse, color='blue', linewidths=0.5)
ax4.scatter(weights_interest[3], mses_interest[3], color='red', linewidths=8)

ax5.set_title('Weight = '+ "{:.3f}".format(weights_interest[4]))
ax5.scatter(weights, mse, color='blue', linewidths=0.5)
ax5.scatter(weights_interest[4], mses_interest[4], color='red', linewidths=8)

ax6.set_title('Weight = '+ "{:.3f}".format(weights_interest[5]))
ax6.scatter(weights, mse, color='blue', linewidths=0.5)
ax6.scatter(weights_interest[5], mses_interest[5], color='red', linewidths=8)

plt.show()''')

fig, axs = plt.subplots(2, 3, sharey=True)

axs[0,0].set_xlabel('W', size=6)
axs[0,0].set_ylabel('MSE', size=6)
axs[0,0].xaxis.set_tick_params(labelsize=6)
axs[0,0].yaxis.set_tick_params(labelsize=6)

axs[0,1].set_xlabel('W', size=6)
axs[0,1].set_ylabel('MSE', size=6)
axs[0,1].xaxis.set_tick_params(labelsize=6)
axs[0,1].yaxis.set_tick_params(labelsize=6)

axs[0,2].set_xlabel('W', size=6)
axs[0,2].set_ylabel('MSE', size=6)
axs[0,2].xaxis.set_tick_params(labelsize=6)
axs[0,2].yaxis.set_tick_params(labelsize=6)

axs[1,0].set_xlabel('W', size=6)
axs[1,0].set_ylabel('MSE', size=6)
axs[1,0].xaxis.set_tick_params(labelsize=6)
axs[1,0].yaxis.set_tick_params(labelsize=6)

axs[1,1].set_xlabel('W', size=6)
axs[1,1].set_ylabel('MSE', size=6)
axs[1,1].xaxis.set_tick_params(labelsize=6)
axs[1,1].yaxis.set_tick_params(labelsize=6)

axs[1,2].set_xlabel('W', size=6)
axs[1,2].set_ylabel('MSE', size=6)
axs[1,2].xaxis.set_tick_params(labelsize=6)
axs[1,2].yaxis.set_tick_params(labelsize=6)

axs[0,0].set_title('Weight = '+ "{:.3f}".format(weights_interest[0]), size=6)
axs[0,0].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[0,0].scatter(weights_interest[0], mses_interest[0], color='red', linewidths=8,s=6)

axs[0,1].set_title('Weight = '+ "{:.3f}".format(weights_interest[1]), size=6)
axs[0,1].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[0,1].scatter(weights_interest[1], mses_interest[1], color='red', linewidths=8,s=6)

axs[0,2].set_title('Weight = '+ "{:.3f}".format(weights_interest[2]), size=6)
axs[0,2].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[0,2].scatter(weights_interest[2], mses_interest[2], color='red', linewidths=8,s=6)

axs[1,0].set_title('Weight = '+ "{:.3f}".format(weights_interest[3]), size=6)
axs[1,0].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[1,0].scatter(weights_interest[3], mses_interest[3], color='red', linewidths=8,s=6)

axs[1,1].set_title('Weight = '+ "{:.3f}".format(weights_interest[4]), size=6)
axs[1,1].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[1,1].scatter(weights_interest[4], mses_interest[4], color='red', linewidths=8,s=6)

axs[1,2].set_title('Weight = '+ "{:.3f}".format(weights_interest[5]), size=6)
axs[1,2].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[1,2].scatter(weights_interest[5], mses_interest[5], color='red', linewidths=8,s=6)

st.pyplot(fig)

st.write('Busquemos el máximi de una función. Por ejemplo el el negativo del error que obtenemos cuando variamos los pesos para la distribución que datos que estamos analizando')

st.code('''neg_mse = []
for mse_e in mse:
    n_mse = -1.0 * mse_e
    neg_mse.append(n_mse)''')
neg_mse = []
for mse_e in mse:
    n_mse = -1.0 * mse_e
    neg_mse.append(n_mse)

fig, ax = plt.subplots()
ax.set_ylabel('MSE',fontdict = {'fontsize':16})
ax.set_xlabel('W',fontdict = {'fontsize':16})
ax.legend(fontsize=6)

weights = []
mse = []
for update_output in update_outputs_weight:
    weights.append(update_output[0])
    mse.append(update_output[3])
    
ax.scatter(weights, neg_mse,s=6)

st.pyplot(fig)

st.code('''weight_to_max_0 = update_outputs_weight[110][0]
mse_to_max_0 = update_outputs_weight[110][-1]
mse_to_max_0 = -1.0 * mse_to_max_0
print(weight_to_max_0, mse_to_max_0)''')

weight_to_max_0 = update_outputs_weight[110][0]
mse_to_max_0 = update_outputs_weight[110][-1]
mse_to_max_0 = -1.0 * mse_to_max_0
st.write("$",weight_to_max_0, mse_to_max_0)

st.code('''mse_grad_to_max = np.gradient(np.asarray(neg_mse))
print(mse_grad_to_max[110])''')
mse_grad_to_max = np.gradient(np.asarray(neg_mse))
st.write(mse_grad_to_max[110])

st.code('''weights_to_max_interest = []
weights_to_max_interest.append(weight_to_max_0)

mses_to_max_interest = []
mses_to_max_interest.append(mse_to_max_0)

weight_to_max_start = weight_to_max_0
bias_to_max_start = bias_0

for itera in range(1,6):
    
    weight_to_max_new = weight_to_max_start + alpha * mse_grad_to_max[110]  #The gradient is fixed, is should change for each weight
    bias_to_max_new = bias_to_max_start
    
    mse_to_max_new = calcule_error(x, y, weight_to_max_new, bias_to_max_new)
    mse_to_max_new = -1.0 * mse_to_max_new
    
    weights_to_max_interest.append(weight_to_max_new)
    mses_to_max_interest.append(mse_to_max_new) 
        
    weight_to_max_start = weight_to_max_new
    bias_to_max_start = bias_to_max_new
    ''')

weights_to_max_interest = []
weights_to_max_interest.append(weight_to_max_0)
mses_to_max_interest = []
mses_to_max_interest.append(mse_to_max_0)
weight_to_max_start = weight_to_max_0
bias_to_max_start = bias_0
for itera in range(1,6):
    weight_to_max_new = weight_to_max_start + alpha * mse_grad_to_max[110] 
    bias_to_max_new = bias_to_max_start
    mse_to_max_new = calcule_error(x, y, weight_to_max_new, bias_to_max_new)
    mse_to_max_new = -1.0 * mse_to_max_new
    weights_to_max_interest.append(weight_to_max_new)
    mses_to_max_interest.append(mse_to_max_new) 
    weight_to_max_start = weight_to_max_new
    bias_to_max_start = bias_to_max_new
    

st.code('''plt.figure(figsize=(15,8)) 

ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)
ax4 = plt.subplot(2,3,4)
ax5 = plt.subplot(2,3,5)
ax6 = plt.subplot(2,3,6)


#ax1.scatter(weights, neg_mse)
ax1.set_xlabel('W', size=12)
ax1.set_ylabel('MSE', size=12)

ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax4.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax5.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax6.ticklabel_format(axis="y", style="sci", scilimits=(0,0))


ax1.set_title('Weight = '+ "{:.3f}".format(weights_to_max_interest[0]))
ax1.scatter(weights, neg_mse, color='blue', linewidths=0.5,)
ax1.scatter(weights_to_max_interest[0], mses_to_max_interest[0], color='red', linewidths=8)

ax2.set_title('Weight = '+ "{:.3f}".format(weights_to_max_interest[1]))
ax2.scatter(weights, neg_mse, color='blue', linewidths=0.5)
ax2.scatter(weights_to_max_interest[1], mses_to_max_interest[1], color='red', linewidths=8)

ax3.set_title('Weight = '+ "{:.3f}".format(weights_to_max_interest[2]))
ax3.scatter(weights, neg_mse, color='blue', linewidths=0.5)
ax3.scatter(weights_to_max_interest[2], mses_to_max_interest[2], color='red', linewidths=8)

ax4.set_title('Weight = '+ "{:.3f}".format(weights_to_max_interest[3]))
ax4.scatter(weights, neg_mse, color='blue', linewidths=0.5)
ax4.scatter(weights_to_max_interest[3], mses_to_max_interest[3], color='red', linewidths=8)

ax5.set_title('Weight = '+ "{:.3f}".format(weights_to_max_interest[4]))
ax5.scatter(weights, neg_mse, color='blue', linewidths=0.5)
ax5.scatter(weights_to_max_interest[4], mses_to_max_interest[4], color='red', linewidths=8)

ax6.set_title('Weight = '+ "{:.3f}".format(weights_to_max_interest[5]))
ax6.scatter(weights, neg_mse, color='blue', linewidths=0.5)
ax6.scatter(weights_to_max_interest[5], mses_to_max_interest[5], color='red', linewidths=8)

plt.show()''')

fig, axs = plt.subplots(2, 3, sharey=True)

axs[0,0].set_xlabel('W', size=6)
axs[0,0].set_ylabel('MSE', size=6)
axs[0,0].xaxis.set_tick_params(labelsize=6)
axs[0,0].yaxis.set_tick_params(labelsize=6)

axs[0,1].set_xlabel('W', size=6)
axs[0,1].set_ylabel('MSE', size=6)
axs[0,1].xaxis.set_tick_params(labelsize=6)
axs[0,1].yaxis.set_tick_params(labelsize=6)

axs[0,2].set_xlabel('W', size=6)
axs[0,2].set_ylabel('MSE', size=6)
axs[0,2].xaxis.set_tick_params(labelsize=6)
axs[0,2].yaxis.set_tick_params(labelsize=6)

axs[1,0].set_xlabel('W', size=6)
axs[1,0].set_ylabel('MSE', size=6)
axs[1,0].xaxis.set_tick_params(labelsize=6)
axs[1,0].yaxis.set_tick_params(labelsize=6)

axs[1,1].set_xlabel('W', size=6)
axs[1,1].set_ylabel('MSE', size=6)
axs[1,1].xaxis.set_tick_params(labelsize=6)
axs[1,1].yaxis.set_tick_params(labelsize=6)

axs[1,2].set_xlabel('W', size=6)
axs[1,2].set_ylabel('MSE', size=6)
axs[1,2].xaxis.set_tick_params(labelsize=6)
axs[1,2].yaxis.set_tick_params(labelsize=6)

axs[0,0].set_title('Weight = '+ "{:.3f}".format(weights_interest[0]), size=6)
axs[0,0].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[0,0].scatter(weights_interest[0], mses_interest[0], color='red', linewidths=8,s=6)

axs[0,1].set_title('Weight = '+ "{:.3f}".format(weights_interest[1]), size=6)
axs[0,1].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[0,1].scatter(weights_interest[1], mses_interest[1], color='red', linewidths=8,s=6)

axs[0,2].set_title('Weight = '+ "{:.3f}".format(weights_interest[2]), size=6)
axs[0,2].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[0,2].scatter(weights_interest[2], mses_interest[2], color='red', linewidths=8,s=6)

axs[1,0].set_title('Weight = '+ "{:.3f}".format(weights_interest[3]), size=6)
axs[1,0].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[1,0].scatter(weights_interest[3], mses_interest[3], color='red', linewidths=8,s=6)

axs[1,1].set_title('Weight = '+ "{:.3f}".format(weights_interest[4]), size=6)
axs[1,1].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[1,1].scatter(weights_interest[4], mses_interest[4], color='red', linewidths=8,s=6)

axs[1,2].set_title('Weight = '+ "{:.3f}".format(weights_interest[5]), size=6)
axs[1,2].scatter(weights, mse, color='blue', linewidths=0.5,s=6)
axs[1,2].scatter(weights_interest[5], mses_interest[5], color='red', linewidths=8,s=6)

st.pyplot(fig)

st.write('''<font size=5 color='red'>
<center> Tarea </center>''',unsafe_allow_html = True,)

st.code('''# Esta función genera un conjunto de datos que simulan 
# la medición de la distancia de un carrito en un riel de aire
# en la ausencia de una fuerza sobre el carrito.
# Se propone un error en la medición de la distancia

def generador_datos_simple(n_points, distance_0, measuring_time, speed, acelera, max_distance_error):
    
    """
      n_points: number of point that will be generated, integer
      distance_0 : initial distantce (at time zero) 
      measuring_time: the time inteval used for the measurement
      speed : carś speed
      max_distance_error: Maximum error measuring distance
      
    """
    
    # n_points es el número de puntos que serán generados
    
    x = np.random.random(n_points) * measuring_time
     
    # x es arreglo con m numeros aleatorios entre 0.0 y measuring_time
    
    error = np.random.randn(n_points) * max_distance_error 
    
    # error es un error generado aleatoriamente con un valor maximo max_distance_error

    y = distance_0 + speed*x + acelera*x**2 + error 
        
    return x, y
''')


def generador_datos_simple(n_points, distance_0, measuring_time, speed, acelera, max_distance_error):
    x = np.random.random(n_points) * measuring_time
    error = np.random.randn(n_points) * max_distance_error 
    y = distance_0 + speed*x + acelera*x**2 + error 
    return x, y

st.code('''# Generacción de las muestras (xi,yi)
n_points = 1000
distance_0 = 100000.0
measure_time = 100.0
speed = 20.0
acelera= 100.0
max_distance_error = 20000

x, y = generador_datos_simple(n_points, distance_0, measure_time, speed, acelera, max_distance_error)

print("x type", type(x), "x shape", x.shape)
print("y type", type(y), "y shape", y.shape)''')

# Generacción de las muestras (xi,yi)
n_points = 1000
distance_0 = 100000.0
measure_time = 100.0
speed = 20.0
acelera= 100.0
max_distance_error = 20000

x, y = generador_datos_simple(n_points, distance_0, measure_time, speed, acelera, max_distance_error)

st.write("$x type", type(x), "x shape", x.shape)
st.write("$y type", type(y), "y shape", y.shape)


fig, ax = plt.subplots()
ax.set_ylabel('Y(cm)',fontdict = {'fontsize':16})
ax.set_xlabel('X(seg)',fontdict = {'fontsize':16})
ax.legend(fontsize=6)
ax.scatter(x,y,s=6)
st.pyplot(fig)