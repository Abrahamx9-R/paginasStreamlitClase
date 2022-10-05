
import time
import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code

import numpy as np
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from traitlets import Int
np.random.seed(3)

st.set_page_config(
    page_title="Clase 2",
    page_icon="🏎️",
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
with open("pages/notebooks/clase_2.zip", "rb") as fp:
    btn = st.download_button(
        label="Descarga notebook",
        data=fp,
        file_name='clase_2.zip',
        mime='paginas/notebooks/',
    )

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
st.write('''Se tiene un conjunto de muestras (puntos) $(x_i, y_i)$, y se busca encontrar una función $F$ que describa una posible correlación entre ellos. $X$, con valores $x_i$ (en el presente caso el tiempo) es una variable independiente, mientras que $Y$, con los valores $y_i$ (la distancia en el presente caso) depende de $X$.

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

print('residuo: {0:10.2f}'.format( residuo))

plot_x_y_y__(x, y, y_, points=True)

st.write('De la gráficas se observa que la función está lejos de describir la correlación entre los punto')

st.write('''Los valores del peso $W$ y el bias $b$ se actualizan iterativamente a prueba y error. Para ello

se hacen los cambios de acuerdo a la sugerencia de la observado en la gráfica.''')

st.code('''
def update_weights_biases(x, y, weight, bias, delta_weight, delta_bias):
    
    weight = weight + delta_weight
    
    bias = bias + delta_bias

    #The following date are for constructing the F(x,weight, bias)
    
    y_ = weight*x + bias
       
    residuo = np.mean((y - y_)**2)
    
    #print('residuo: {:10.2f}'.format(residuo))   
        
    return weight, bias, y_, residuo''')

weight = weight_0
bias = bias_0
st.sidebar.write('### iniciando deltas')
delta_w = st.sidebar.number_input("delta_w",value=1.0,format="%f")
delta_b = st.sidebar.number_input("delta_b",value=-10.0,format="%f")

weight, bias, y_, residuo = update_weights_biases(x, y , weight, bias, delta_w, delta_b)
plot_x_y_y__(x, y, y_, points=True)

st.code('''weight = weight_0
bias = bias_0
delta_w = 1.0
delta_b = -10.0

weight, bias, y_, residuo = update_weights_biases(x, y , weight, bias, delta_w, delta_b)
plot_x_y_y__(x, y, y_, points=True)''')

st.write('El cambio es muy pequeño, por ello el peso se actualiza con delta_weight = 2.0 y el bias con delta_bias= -100.0. ')

weight = weight
bias = bias
delta_w = 6.0
delta_b = -100.0

weight, bias, y_, residuo = update_weights_biases(x, y, weight, bias, delta_w, delta_b)
plot_x_y_y__(x, y, y_, points=True) 
st.code('''weight = weight
bias = bias
delta_w = 6.0
delta_b = -100.0

weight, bias, y_, residuo = update_weights_biases(x, y, weight, bias, delta_w, delta_b)
plot_x_y_y__(x, y, y_, points=True) ''')

st.write('Aun se puede mejorar disminuyendo el peso (pendiente) y aumentando el bias (ordenada en el origen):')

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

st.write('''Busquemos encontrar la función que defina la correlación generando un conjunto de funciones y calcular los correspondientes residuos.

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
iterations = 100

for i in range(iterations):
    
    weight, bias, y_, residuo = update_weights_biases(x, y, weight, bias, delta_w, delta_b)
    
    update_outputs.append([weight, bias, y_, residuo])
    
    if i % 10 == 0 :
        st.write('$weight: {0:8.2f}    bias: {1:5.2f}   residuo: {2:10.2f}'.format(weight,bias, residuo))


fig, ax = plt.subplots()
ax.set_title('Datos',fontdict = {'fontsize':18})
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
ax.set_title('Datos',fontdict = {'fontsize':18})
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

print('El residuo mas pequeño tiene el valor: {0:10.2f}'.format(update_outputs[index_min][3]))

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

residuos_array = np.asarray(residuos)
residuos_grad = np.gradient(residuos_array)

fig, ax = plt.subplots()
ax.set_ylabel('Derivada del Residuo',fontdict = {'fontsize':16})
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

st.write('''El método de mínimos cuadrados nos permite encontrar una función que describe la correlación que tienen un conjunto m de puntos $(x_i, y_i)$, en donde $x_i$ son los valores que toma la variable $X$ y $y_i$ son los valores que toma la variable $Y$.


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
st.write('''El mínimo de esta función, se encuentra en aquellos valores de $W$ ($W_{opt}$) y $b$ ($b_{opt}$) en donde la derivada de la función residuo es igual a zero. 
    
    
Para encontrar este mínimo, primero calculamos la derivada parcial de la función respecto a cada uno de los parámetros $W$ y $b$, después, cada derivada la igualamos a cero:''')
st.latex(r'\dfrac{∂Residuo}{∂{W}} =0⟶ W_{opt}=\dfrac{∑_{i=1}^{n}(x_i–\bar x)(y_i–\bar y)}{∑_{i=1}^{m}(x_i–\bar x)^2}')
st.latex(r' \dfrac{∂Residuo}{∂{b}}=0⟶ b_{opt}=y – W_{opt} x ')
st.write('en donde $\bar x$ y $\bar y$ son los valores promedio de los valores de las variables $X$ y $Y$.')

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