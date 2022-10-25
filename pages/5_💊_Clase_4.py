import streamlit as st
import json
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

from utils import show_image_local, show_text_color

ruta = "pages/images/imagesClase4/"
st.set_page_config(
    page_title="Clase 4",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://www.extremelycoolapp.com/help',
        'Report a Bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title('Clase 4 ')

def load_lottieurl(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

lottie_hello = load_lottieurl("pages/images/healtcare.json")

st_lottie(lottie_hello,speed = 1, reverse=False,loop=True,quality="low",height=800,width=None,key=None,)

st.markdown('<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/61b15243fa124009b0d3cc8521f3fc50" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>',unsafe_allow_html=True)

show_text_color('Problema a resolver')

show_image_local(ruta+'Problema.jpg')

show_text_color('Mortalidad por diabetes')

show_image_local(ruta+'Diabetes.png')

show_text_color('Información sobre el problema a resolver')

st.markdown('''Evolución de la enfermedad de pacientes con Diabetes Mellitus despues de un año.
    
En el presente trabajo, la diabetes la caracterizamos con los siguientes diez rasgos: edad, sexo, índice de masa corporal, presión arterial promedio y las seis mediciones de suero sanguíneo siguientes:''')

st.markdown('''- Colesterol Total 
- Baja densidad de lipoproteinas
- Alta densidad de lipoproteinas
- Triglicéridos
- Concentración de Lamorigina
- Glucosa''')

show_text_color('Cuantificación de esta información')

st.markdown(r'''Se tienen información de 442 pacientes (m = 442, m es el número de muestras). La respuesta de interés, $Y$, es una medida cuantitativa del progreso de la enfermedad un año después del inicio del estudio. Los valores de $Y$ varían entre 25 y 346

Fuente de la información: [diabetes data](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)    Artículo original: en la literatura anexada en el archivo zip con el nombre Least-Angle-Regression_2004''')


st.code('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)''')

st.code('''# Los datos se encuentran el el archivo diabetes.csv. Estos se cargan en el objeto df del tipo DataFrame

df = pd.read_csv('diabetes.csv', sep ='\t')

# el objeto df contiene los 10 rasgos relevantes de los pacientes diabéticos,
# así como el progreso, Y , de la enfermedad un año después de comenzado el estudio. ''')

df = pd.read_csv('pages/datos/diabetes.csv', sep ='\t')

st.table(df.head())

st.markdown('''Las abreviaciones tienen el siguiente significado:
- AGE = Age
- SEX = Sex
- BMI = Body Mass Index (BMI)
- BP = Mean Arterial Pressure (MAP)
- S1 = Total Cholesterol (TC)
- S2 = Low Density lipoproteins (LDL)
- S3 = High Density lipoproteins (HDL)
- S4 = Triglyceride (TG, TCH)
- S5 = Serum Concentration of Lamorigine (LTG)
- S6 = Glucose (GLU)
- Y = Quantitative Measure of Diabetes Mellitus Disease Progression (QMDMDP) one year after the baseline.''')

st.code('''# El método "describe()" del DataFrame df genera una tabla con informacion estadistica
# de cada uno de los rasgos y del objetivo.

df.describe()
''')

st.table(df.describe())

st.markdown('## Se crean los histogramas para cada uno de los rasgos que caracteriza a los pacientes con diabetes:')

st.code('''plt.figure(figsize=(20,8)) 

ax1 = plt.subplot(2,4,1)
ax2 = plt.subplot(2,4,2)
ax3 = plt.subplot(2,4,3)
ax4 = plt.subplot(2,4,4)

ax1.hist(df.AGE, bins=30, color='green',edgecolor='purple', alpha=0.5)
ax1.set_xlabel('Age (years)', size=15)
ax1.set_ylabel('Frequency', size=15)

ax2.hist(df.SEX, bins=30, color='orange',edgecolor='purple', alpha=0.5)
ax2.set_xlabel('Sex', size=15)

ax3.hist(df.BMI, bins=30, color='red',edgecolor='purple', alpha=0.5)
ax3.set_xlabel('Body_mass_index', size=15)

ax4.hist(df.BP, bins=30, color='blue',edgecolor='purple', alpha=0.5)
ax4.set_xlabel('Mean_Arterial_Pressure', size=15);''')

fig = plt.figure(figsize=(20,8)) 
ax1 = plt.subplot(2,4,1)
ax2 = plt.subplot(2,4,2)
ax3 = plt.subplot(2,4,3)
ax4 = plt.subplot(2,4,4)
ax1.hist(df.AGE, bins=30, color='green',edgecolor='purple', alpha=0.5)
ax1.set_xlabel('Age (years)', size=15)
ax1.set_ylabel('Frequency', size=15)
ax2.hist(df.SEX, bins=30, color='orange',edgecolor='purple', alpha=0.5)
ax2.set_xlabel('Sex', size=15)
ax3.hist(df.BMI, bins=30, color='red',edgecolor='purple', alpha=0.5)
ax3.set_xlabel('Body_mass_index', size=15)
ax4.hist(df.BP, bins=30, color='blue',edgecolor='purple', alpha=0.5)
ax4.set_xlabel('Mean_Arterial_Pressure', size=15);
st.pyplot(fig)

st.code('''plt.figure(figsize=(20,8)) 

ax1 = plt.subplot(2,4,1)
ax2 = plt.subplot(2,4,2)
ax3 = plt.subplot(2,4,3)
ax4 = plt.subplot(2,4,4)

ax1.hist(df.S1, bins=30, color='green',edgecolor='purple', alpha=0.5)
ax1.set_xlabel('Total Cholesterol', size=15)
ax1.set_ylabel('Frequency', size=15)

ax2.hist(df.S2, bins=30, color='orange',edgecolor='purple', alpha=0.5)
ax2.set_xlabel('Low Density lipoproteins', size=15)

ax3.hist(df.S3, bins=30, color='red',edgecolor='purple', alpha=0.5)
ax3.set_xlabel('High Density lipoproteins', size=15)

ax4.hist(df.S4, bins=30, color='blue',edgecolor='purple', alpha=0.5)
ax4.set_xlabel('Triglyceride', size=15);''')

fig = plt.figure(figsize=(20,8)) 

ax1 = plt.subplot(2,4,1)
ax2 = plt.subplot(2,4,2)
ax3 = plt.subplot(2,4,3)
ax4 = plt.subplot(2,4,4)

ax1.hist(df.S1, bins=30, color='green',edgecolor='purple', alpha=0.5)
ax1.set_xlabel('Total Cholesterol', size=15)
ax1.set_ylabel('Frequency', size=15)

ax2.hist(df.S2, bins=30, color='orange',edgecolor='purple', alpha=0.5)
ax2.set_xlabel('Low Density lipoproteins', size=15)

ax3.hist(df.S3, bins=30, color='red',edgecolor='purple', alpha=0.5)
ax3.set_xlabel('High Density lipoproteins', size=15)

ax4.hist(df.S4, bins=30, color='blue',edgecolor='purple', alpha=0.5)
ax4.set_xlabel('Triglyceride', size=15);

st.pyplot(fig)

st.code('''plt.figure(figsize=(15,8)) 

ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)

ax1.hist(df.S5, bins=30, color='green',edgecolor='purple', alpha=0.5)
ax1.set_xlabel('Serum Concentration of Lamorigine', size=15)
ax1.set_ylabel('Frequency', size=15)

ax2.hist(df.S6, bins=30, color='orange',edgecolor='purple', alpha=0.5)
ax2.set_xlabel('Glucose', size=15)

ax3.hist(df.Y, bins=30, color='purple',edgecolor='black', alpha=0.5)
ax3.set_xlabel('Y(Diabetes Mellitus Disease Progression)', size=15)
''')

fig = plt.figure(figsize=(15,8)) 
 
ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)

ax1.hist(df.S5, bins=30, color='green',edgecolor='purple', alpha=0.5)
ax1.set_xlabel('Serum Concentration of Lamorigine', size=15)
ax1.set_ylabel('Frequency', size=15)

ax2.hist(df.S6, bins=30, color='orange',edgecolor='purple', alpha=0.5)
ax2.set_xlabel('Glucose', size=15)

ax3.hist(df.Y, bins=30, color='purple',edgecolor='black', alpha=0.5)
ax3.set_xlabel('Y(Diabetes Mellitus Disease Progression)', size=15)

st.pyplot(fig)

show_text_color('<center> El sistema de aprendizaje más simple </center>',size=10)

show_text_color('Proponemos que $Y$ depende linealmente con cada una de las variables $X$')

st.markdown(r'''Los pasos principales para construir el sistema son:

1. Definir un modelo proponiendo un algoritmo que relacione linealmente las variables X y Y.
2. Inicializar los parámetros que determinan el modelo, $w, b$ (pesos y biases).
3. Realizar lo siguiente en cada época:
     - Calcular el costo ($J$) o pérdida (loss, $L$).
     - Calcular el gradiente del costo respecto a los parámetros del modelo.
     - Actualizar los pesos y los biases empleando el algoritmo propuesto para la optimización. Por ejemplo, gradiente descendente. ''')

show_text_color('Algoritmo que relaciona las variables X y Y')

st.markdown(r'''    
Si se tienen $m$ muestras.
    
Para una muestra $j\in m$, con sus valores de la variable $X$, $X^{(j)}$ = $(x_{1}^{(j)}, x_{2}^{(j)}, x_{3}^{(j)},...,x_{N}^{(j)} )$, se genera la variable $Z^{(j)}$ mediante la siguiente relación:''')

st.latex(r'Z^{(j)} = w^T X^{(j)} + b = \sum_{i=1}^N w_{i} x_{i}^{(j)} + b')

st.markdown('''$Z^{(j)}$ es la suma pesada las variables $X^{(j)}$ con los pesos $w_i$, mas el bias b. Posteriormente, esta sumatoria es transformada ("activada") con una función, para generar la variable de salida $a^{(j)}$. 
    
Dado que inicialmente planteamos que en el sistema de aprendizaje $Y$ varía linealmente con cada una de las variables $X$, la función de transformación es la identidad $f(Z)=Z$.''')

st.latex(r'F(W,X^{(j)}) = a^{(j)} = f(Z^{(j)})=Z^{(j)}')

st.markdown(r'''$W$ representa tanto a los pesos $w_i$ como al bias $b$.
Para realizar el aprendizaje, se genera una métrica para el error, la cual queda definida por la función de perdida (*loss*), tambien llamada función de costo (*cost*) $J$. Esta función se obtiene realizando el promedio, sobre todas las $m$ muestras, del cuadrado de la diferencia entre el valor ($y^{(j)}$) de la muestra $j$ y el valor de la función $F(X^{(j)})$. ''')

st.latex(r'J = \frac{1}{m} \sum_{j=1}^m (y^{(j)} -F(W,X^{(j)}))^2')

show_text_color('Al inicio del proceso, para quitar cuarquier posible correlación entre las muestras, o sesgo en su generación, estas se reordenan al azar.')

st.code('df = df.sample(frac=1)')

df = df.sample(frac=1)

st.code('print(df.shape)')

st.write('$',df.shape)

st.code('print(len(df.values[:,:]))')

st.write('$',len(df.values[:,:]))

show_text_color('División de las muestras para aprender y para hacer predicciones')

st.markdown('Se dividen la muestras originales en 2 conjuntos: 90 % para el entrenamiento y 10 % para hacer inferencias (predicciones) con el sistema de aprendizaje.')

st.code('''test_ratio = 0.1

train_ratio = int((1.0-test_ratio)*len(df.values[:,:]))

df_train = df.iloc[0:train_ratio,:]
df_test  = df.iloc[train_ratio:,:]
''')

test_ratio = 0.1
train_ratio = int((1.0-test_ratio)*len(df.values[:,:]))
df_train = df.iloc[0:train_ratio,:]
df_test  = df.iloc[train_ratio:,:]

st.code('''print(df_train.shape)
print(df_test.shape)''')

st.write('$',df_train.shape)
st.write("$",df_test.shape)

show_text_color('Dada una distribución, podemos calcular su valor promedio $\mu$ y su varianza $\sigma$')

show_image_local(ruta + 'grafica.png')

st.markdown('Para trabajar con los modelos de aprendizaje,es adecuado que todas las variables tengan el mismo orden de magnitud. Por ello, se normalizan sus valores en las muestras que se emplearán en el entrenamiento, tanto las variables que describen los rasgos (X) como las variables objetivo (Y):')

st.latex('x_{i,norm} = \dfrac{x_{i}-\mu}{\sigma}')

st.latex('y_{i,norm} = \dfrac{y_{i}-\mu}{\sigma}')

st.code('''df_train_norm = (df_train - df_train.mean()) / df_train.std()
df_train_norm.head()''')

df_train_norm = (df_train - df_train.mean()) / df_train.std()
st.table(df_train_norm.head())

show_text_color('Nota importante: La normalización de las muestras de prueba se realiza con los valores de $\mu$ y $\sigma$ obtenidos con las muestras empleadas para el entrenamiento')

st.code('''df_test_norm = (df_test - df_train.mean()) / df_train.std()
df_test_norm.head()''')

df_test_norm = (df_test - df_train.mean()) / df_train.std()
st.table(df_test_norm.head())

st.markdown('Histogramas de las variables que se emplearán en el entrenamiento:')

st.code('''plt.figure(figsize=(20,8)) 

ax1 = plt.subplot(2,4,1)
ax2 = plt.subplot(2,4,2)
ax3 = plt.subplot(2,4,3)
ax4 = plt.subplot(2,4,4)

ax1.hist(df_train_norm.AGE, bins=30, color='green',edgecolor='purple', alpha=0.5)
ax1.set_xlabel('x1(Age)', size=15)
ax1.set_ylabel('Frequency', size=15)

ax2.hist(df_train_norm.SEX, bins=30, color='orange',edgecolor='purple', alpha=0.5)
ax2.set_xlabel('x2(Sex)', size=15)

ax3.hist(df_train_norm.BMI, bins=30, color='red',edgecolor='purple', alpha=0.5)
ax3.set_xlabel('x3(Body_mass_index)', size=15)

ax4.hist(df_train_norm.BP, bins=30, color='blue',edgecolor='purple', alpha=0.5)
ax4.set_xlabel('x4(Mean_Arterial_Pressure)', size=15);''')

fig = plt.figure(figsize=(20,8)) 

ax1 = plt.subplot(2,4,1)
ax2 = plt.subplot(2,4,2)
ax3 = plt.subplot(2,4,3)
ax4 = plt.subplot(2,4,4)

ax1.hist(df_train_norm.AGE, bins=30, color='green',edgecolor='purple', alpha=0.5)
ax1.set_xlabel('x1(Age)', size=15)
ax1.set_ylabel('Frequency', size=15)

ax2.hist(df_train_norm.SEX, bins=30, color='orange',edgecolor='purple', alpha=0.5)
ax2.set_xlabel('x2(Sex)', size=15)

ax3.hist(df_train_norm.BMI, bins=30, color='red',edgecolor='purple', alpha=0.5)
ax3.set_xlabel('x3(Body_mass_index)', size=15)

ax4.hist(df_train_norm.BP, bins=30, color='blue',edgecolor='purple', alpha=0.5)
ax4.set_xlabel('x4(Mean_Arterial_Pressure)', size=15);

st.pyplot(fig)

st.code('''plt.figure(figsize=(20,8)) 

ax1 = plt.subplot(2,4,1)
ax2 = plt.subplot(2,4,2)
ax3 = plt.subplot(2,4,3)
ax4 = plt.subplot(2,4,4)

ax1.hist(df_train_norm.S1, bins=30, color='green',edgecolor='purple', alpha=0.5)
ax1.set_xlabel('x5(Total Cholesterol)', size=15)
ax1.set_ylabel('Frequency', size=15)

ax2.hist(df_train_norm.S2, bins=30, color='orange',edgecolor='purple', alpha=0.5)
ax2.set_xlabel('x6(Low Density lipoproteins)', size=15)

ax3.hist(df_train_norm.S3, bins=30, color='red',edgecolor='purple', alpha=0.5)
ax3.set_xlabel('x7(High Density lipoproteins)', size=15)

ax4.hist(df_train_norm.S4, bins=30, color='blue',edgecolor='purple', alpha=0.5)
ax4.set_xlabel('x8(Triglyceride)', size=15);''')

fig = plt.figure(figsize=(20,8)) 

ax1 = plt.subplot(2,4,1)
ax2 = plt.subplot(2,4,2)
ax3 = plt.subplot(2,4,3)
ax4 = plt.subplot(2,4,4)

ax1.hist(df_train_norm.S1, bins=30, color='green',edgecolor='purple', alpha=0.5)
ax1.set_xlabel('x5(Total Cholesterol)', size=15)
ax1.set_ylabel('Frequency', size=15)

ax2.hist(df_train_norm.S2, bins=30, color='orange',edgecolor='purple', alpha=0.5)
ax2.set_xlabel('x6(Low Density lipoproteins)', size=15)

ax3.hist(df_train_norm.S3, bins=30, color='red',edgecolor='purple', alpha=0.5)
ax3.set_xlabel('x7(High Density lipoproteins)', size=15)

ax4.hist(df_train_norm.S4, bins=30, color='blue',edgecolor='purple', alpha=0.5)
ax4.set_xlabel('x8(Triglyceride)', size=15);

st.pyplot(fig)

st.code('''plt.figure(figsize=(20,8)) 

ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)

ax1.hist(df_train_norm.S5, bins=30, color='green',edgecolor='purple', alpha=0.5)
ax1.set_xlabel('x9(Serum Concentration of Lamorigine)', size=15)
ax1.set_ylabel('Frequency', size=15)

ax2.hist(df_train_norm.S6, bins=30, color='orange',edgecolor='purple', alpha=0.5)
ax2.set_xlabel('x10(Glucose)', size=15)

ax3.hist(df_train_norm.Y, bins=30, color='purple',edgecolor='black', alpha=0.5)
ax3.set_xlabel('Y(Diabetes Mellitus Disease Progression)', size=15)
''')

fig = plt.figure(figsize=(20,8)) 

ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)

ax1.hist(df_train_norm.S5, bins=30, color='green',edgecolor='purple', alpha=0.5)
ax1.set_xlabel('x9(Serum Concentration of Lamorigine)', size=15)
ax1.set_ylabel('Frequency', size=15)

ax2.hist(df_train_norm.S6, bins=30, color='orange',edgecolor='purple', alpha=0.5)
ax2.set_xlabel('x10(Glucose)', size=15)

ax3.hist(df_train_norm.Y, bins=30, color='purple',edgecolor='black', alpha=0.5)
ax3.set_xlabel('Y(Diabetes Mellitus Disease Progression)', size=15)

st.pyplot(fig)

st.markdown('Los valores de las variables X e Y se extraen de las columnas del DataFrame.')

st.code('''x_train = df_train_norm.values[:,:-1]
y_train = df_train_norm.values[:,-1:]
print(type(x_train), type(y_train))
print(x_train.shape)
print(y_train.shape)''')

x_train = df_train_norm.values[:,:-1]
y_train = df_train_norm.values[:,-1:]
st.write("$",type(x_train), type(y_train))
st.write("$",x_train.shape)
st.write("$",y_train.shape)

st.code('''x_test = df_test_norm.values[:,:-1]
y_test = df_test_norm.values[:,-1:]
print(type(x_test), type(y_test))
print(x_test.shape)
print(y_test.shape)''')

x_test = df_test_norm.values[:,:-1]
y_test = df_test_norm.values[:,-1:]
st.write("$",type(x_test), type(y_test))
st.write("$",x_test.shape)
st.write("$",y_test.shape)

st.code('''train_x = x_train.T
test_x = x_test.T

train_y = y_train
test_y =  y_test''')

train_x = x_train.T
test_x = x_test.T
train_y = y_train
test_y =  y_test

st.code('''print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)''')

st.write("$",train_x.shape)
st.write("$",train_y.shape)
st.write("$",test_x.shape)
st.write("$",test_y.shape)

st.code('train_x[:,1:3]')

st.write("$",train_x[:,1:3])

show_text_color('1. Se inicializan los parámetros $W$ de la función, $F(W,X)$, que define la relación entre X y Y.')

st.markdown('Debido a que las variables $X$ y Y fueron normalizadas a distribuciones con el ancho de una un deviación estándar, los valores de los parámetros $w_i$ se inicializan con valores pequeños, mientras que el bias $b$ se inicializa con cero.')

st.code('''def initialize_params(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = np.reshape(np.random.uniform(-0.1, 0.1, dim), (dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b''')

def initialize_params(dim):
    w = np.reshape(np.random.uniform(-0.1, 0.1, dim), (dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

st.code('''#Testing the function initialize_params (dim)

dim = train_x.shape[0]
w, b = initialize_params(dim)
print ("w = " + str(w))
print ("b = " + str(b))''')

dim = train_x.shape[0]
w, b = initialize_params(dim)
st.write("$w = " + str(w))
st.write("$b = " + str(b))

show_text_color('2. Cálculo de la función de costo y sus derivadas parciales respecto a sus parámetros')

st.markdown(r'''Una vez inicializados los pesos y el bias, se calcula la función de costo, y sus derivadas parciales respecto a cada uno de los pesos $w_{i}$ y el bias $b$. Estas derivadas se emplean para proponer nuevos valores tanto para los pesos como para el bias. 

La funcion *propagate( )* calcula la función de costo y su gradiente:

- Se tiene X 
- Se calcula ''')
st.latex('A = (w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})')
st.markdown('''- Se calcula la función de costo:''')
st.latex(r'J = \frac{1}{m}\sum_{j=1}^m(y^{(j)} -F(W,X^{(j)}))^2')
st.markdown(r'''Las derivadas de la funcion de costo respecto a los pesos $ w_k$ y el bias $ b$ son:''')

st.latex(r'\frac{\partial J}{\partial w_i} = \frac{1}{m}X(A-Y)^T')

st.latex(r'\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{j=1}^m (F(W,X^{(j)})-y^{(j)}))')

st.code('''#Definición de la función identidad.

def identity(z):
    """
    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- z
    """
    s = 1.0 * z
    
    return s''')

def identity(z):
    s = 1.0 * z
    return s

st.code('''def propagate(w, b, X, Y,X_val,Y_val):
    
    m = X.shape[1]
    m_val = X_val.shape[1]
    
    A = identity(np.dot(w.T, X)+b)
    A_val = identity(np.dot(w.T,X_val)+b)
    
    cost = (1/m)*np.sum((Y-A)**2)
    cost_val = (1/m_val)*np.sum((Y_val-A_val)**2)
    
    dw = (1/m)*np.dot(X, (A-Y).T)
    
    db = (1/m)*np.sum(A-Y)
    

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    cost_val = np.squeeze(cost_val)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost, cost_val''')

def propagate(w, b, X, Y,X_val,Y_val):
    m = X.shape[1]
    m_val = X_val.shape[1]
    A = identity(np.dot(w.T, X)+b)
    A_val = identity(np.dot(w.T,X_val)+b)
    cost = (1/m)*np.sum((Y-A)**2)
    cost_val = (1/m_val)*np.sum((Y_val-A_val)**2)
    dw = (1/m)*np.dot(X, (A-Y).T)
    db = (1/m)*np.sum(A-Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    cost_val = np.squeeze(cost_val)
    assert(cost.shape == ())
    grads = {"dw": dw,
             "db": db}   
    return grads, cost, cost_val

show_text_color('3. Actualizacion de los pesos y el bias')

st.markdown('# Para monitorear el entrenamiento, las muestras para el entrenamiento se dividen en dos grupos:')

st.markdown(r'''La función *optimize(  )*  actualiza los pesos $w$ y el bias $b$ usando el método de gradiente descendente.

Se calcula la función de costo para las muestras empleadas en el entrenamiento y para las asignadas a la validación.
    
Los pesos $ w $ y el bias $ b $ son modificados en cada época hasta que la función de costo $ J $ llega a su valor mínimo. 

Los pesos $ w_{k} $ y el bias $b$ se actualizan mediante la siguientes relaciones: ''')

st.latex(r'w_{k} := w_{k} - \alpha \frac{\partial J}{\partial w_{k}}')

st.latex(r'b := b - \alpha \frac{\partial J}{\partial b}')

st.code('''def optimize(w, b, X, Y, X_val, Y_val, epochs, learning_rate, print_cost = False):
    
    costs = []
    costs_val = []
    
    for i in range(epochs):
        
        
        grads, cost, cost_val = propagate(w, b, X, Y,X_val,Y_val)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w-learning_rate*dw
        b = b-learning_rate*db
        
        if i % 100 == 0:
            costs.append(cost)
            costs_val.append(cost_val)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost and Cost_val after epoch %i: %f %f" %(i, cost, cost_val))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs, costs_val''')

def optimize(w, b, X, Y, X_val, Y_val, epochs, learning_rate, print_cost = False):
    costs = []
    costs_val = []
    for i in range(epochs):
        grads, cost, cost_val = propagate(w, b, X, Y,X_val,Y_val)
        dw = grads["dw"]
        db = grads["db"]
        w = w-learning_rate*dw
        b = b-learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
            costs_val.append(cost_val)
        if print_cost and i % 100 == 0:
            st.write("$Cost and Cost_val after epoch %i: %f %f" %(i, cost, cost_val))
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs, costs_val

show_text_color('4. Calculo de inferencias (predicciones)')

st.markdown(r'''Una vez que se encuentran los valores de los pesos $w$ y del bias $b$ que minimizan la función de costo, la función, $F(W,X)$ que se genera con estos valores se emplea para inferir valores de Y asociados a las muestras de prueba, los cuales se comparan con los valores Y conocidos de estas muestras.

La función `predic()` calcula $F(W,X)=w ^ T X + b$ empleando los valores de $w_{opt}$ y $b_{opt}$ para los cuales el costo es mínimo.
''')

st.code('''def predict(w, b, X):
    
    w = w.reshape(X.shape[0], 1)
    
    A = identity(np.dot(w.T, X)+b)
    
    return A''')
def predict(w, b, X):
    w = w.reshape(X.shape[0], 1)
    A = identity(np.dot(w.T, X)+b)
    return A

st.markdown('Para calcular la precision del modelo, a manera de ejemplo, se usa el error cuadático medio, MSE: ')

st.latex('100-MSE*100')

st.markdown('es decir')

st.latex('100-(\dfrac{1}{m_{test}}\sum_{i}^{m_{test}} (y^{(i)}-a^{(i)})^2)*100')

show_text_color('5. Sistema de aprendizaje: se juntan los pasos 1, 2,  y 3')

st.code('''def model(X_train, Y_train, val_ratio, epochs = 2000, learning_rate = 0.5, print_cost = False):

    
    train_ratio = int((1-val_ratio)*X_train.shape[1])
    X_val = X_train[:,train_ratio:]
    Y_val = Y_train[:,train_ratio:]

    X_train = X_train[:,:train_ratio]
    Y_train = Y_train[:,:train_ratio]
    
    print("Train",X_train.shape,Y_train.shape)
    print("val",X_val.shape,Y_val.shape)
    
    # 1. inicializacion de parametros
    w, b = initialize_params(X_train.shape[0])

    # 2. y 3. Actualizacion de parametros
    parameters, grads, costs, costs_val = optimize(w, b, X_train, Y_train, X_val, Y_val, epochs, learning_rate, print_cost = print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    # 4. Predicciones
    Y_prediction_val = predict(w, b, X_val)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.power(Y_prediction_train - Y_train, 2)) * 100))
    print("val accuracy: {} %".format(100 - np.mean(np.power(Y_prediction_val - Y_val, 2)) * 100))

    
    d = {"costs": costs, "costs_val": costs_val,
         "Y_prediction_val": Y_prediction_val, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "epochs": epochs}
    
    return d      ''')

def model(X_train, Y_train, val_ratio, epochs = 2000, learning_rate = 0.5, print_cost = False):
    train_ratio = int((1-val_ratio)*X_train.shape[1])
    X_val = X_train[:,train_ratio:]
    Y_val = Y_train[:,train_ratio:]
    X_train = X_train[:,:train_ratio]
    Y_train = Y_train[:,:train_ratio]
    st.write("$Train",X_train.shape,Y_train.shape)
    st.write("$val",X_val.shape,Y_val.shape)
    w, b = initialize_params(X_train.shape[0])
    parameters, grads, costs, costs_val = optimize(w, b, X_train, Y_train, X_val, Y_val, epochs, learning_rate, print_cost = print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_val = predict(w, b, X_val)
    Y_prediction_train = predict(w, b, X_train)
    st.write("$train accuracy: {} %".format(100 - np.mean(np.power(Y_prediction_train - Y_train, 2)) * 100))
    st.write("$val accuracy: {} %".format(100 - np.mean(np.power(Y_prediction_val - Y_val, 2)) * 100))
    d = {"costs": costs, "costs_val": costs_val,
         "Y_prediction_val": Y_prediction_val, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "epochs": epochs}
    return d  

st.code('print(train_x.shape[1])')

st.write("$",train_x.shape[1])

st.code('''epochs = 2000
learning_rate = 0.005
val_ratio = 0.1
d = model(train_x, train_y.T, val_ratio=val_ratio, epochs = epochs, 
          learning_rate = learning_rate, print_cost = True)''')

st.sidebar.write('aprendizaje')
epochs = st.sidebar.number_input("epochs",min_value=0,value=2000,format="%i")
learning_rate = st.sidebar.number_input("learning rate",min_value=0.00001,value=0.005,format="%f")
val_ratio = st.sidebar.number_input("val ratio",min_value=0.0,value=0.1,format="%f")

d = model(train_x, train_y.T, val_ratio=val_ratio, epochs = epochs, 
          learning_rate = learning_rate, print_cost = True)

st.code('''costs = np.squeeze(d['costs'])
costs_val = np.squeeze(d['costs_val'])

plt.figure(figsize=(10,8)) 

plt.plot(costs)
plt.plot(costs_val)
plt.ylabel('cost', size=16)
plt.xlabel('epochs (x100)', size=16)
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()''')

costs = np.squeeze(d['costs'])
costs_val = np.squeeze(d['costs_val'])

fig = plt.figure(figsize=(10,8)) 

plt.plot(costs)
plt.plot(costs_val)
plt.ylabel('cost', size=16)
plt.xlabel('epochs (x100)', size=16)
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.legend(['Train', 'Val'], loc='upper right')
st.pyplot(fig)

st.write('''redictions_test = predict(d["w"], d["b"], train_x)
print(np.mean(np.power((predictions_test-test_y), 2)))


print("test accuracy: {} %".format(100 - np.mean(np.power(predictions_test-test_y, 2)) * 100))''')

st.code('''learning_rates = [0.01, 0.005, 0.001]

plt.figure(figsize=(10,8)) 
val_ratio = 0.1
models = {}
for i in learning_rates:
    print ("learning rate is: ", i)
    models[str(i)] = model(train_x, train_y.T, val_ratio=val_ratio, epochs = 2000, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
    plt.plot(np.squeeze(models[str(i)]["costs_val"]), label= str(models[str(i)]["learning_rate"]))
    
plt.ylabel('cost')
plt.xlabel('epocs (x100)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()''')

learning_rates = [0.01, 0.005, 0.001]
fig = plt.figure(figsize=(10,8)) 
val_ratio = 0.1
models = {}
for i in learning_rates:
    st.write("$learning rate is: ", i)
    models[str(i)] = model(train_x, train_y.T, val_ratio=val_ratio, epochs = 2000, learning_rate = i, print_cost = False)
    st.write('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
    plt.plot(np.squeeze(models[str(i)]["costs_val"]), label= str(models[str(i)]["learning_rate"]))
    
plt.ylabel('cost')
plt.xlabel('epocs (x100)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
st.pyplot(fig)

show_text_color('<center> Modificamos el sistema de aprendizaje más simple </center>',size=10)

show_text_color('Proponemos que $Y$ no depende linealmente con cada una de las variables $X$, sino ligeramente no lineal')

st.markdown('Para nuestro nuevo sistema de aprendizaje, la función $F(W,X)$ ya no varía linealmente con X. La relación (2) de las siguientes ecuaciones ya no es válida:')

st.latex(r'Z^{(j)} = w^T X^{(j)} + b = \sum_{i=1}^N w_{i} x_{i}^{(j)} + b ')

st.latex(r'F(W,X^{(j)}) = a^{(j)} = f(Z^{(j)})=Z^{(j)}')

st.markdown(r'''Para tomar en cuenta la no linealidad, la función $f(z)=z$ se cambia por una función ligeramente no lineal alrededor de cero. Esto se logra empleando en su lugar una función del tipo sigmoid, por ejemplo $f(z)=tanh(z)$. Con este cambio, la ecuación (2) se transforma entonces en:''')

st.latex(r'F(W,X^{(j)}) = a^{(j)} = f(Z^{(j)}) = tanh(Z^{(j)})')

st.code('''def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))''')

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

st.code('''def atanh(x):
    return 1.7159*tanh(2*x/3)''')
def atanh(x):
    return 1.7159*tanh(2*x/3)

st.code('print(atanh(-1.0), atanh(1.0))')
st.write("$",atanh(-1.0), atanh(1.0))

st.code('''#The following arrays are generated for plotting the Function F(x, weight_0, bias_0)
x_ = np.arange(-2, 2.0, 0.1)
y_ = atanh(x_)
#Samples and function F are plotted
plt.figure(figsize=(12,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.xlabel('X', fontsize=16)
plt.grid(True)
plt.title('Sigmoid-type = 1.7159*tanh((2/3*x)', size=20)

#Plotting function
plt.plot(x_, y_, color='green', lw=4)

plt.show()''')

x_ = np.arange(-2, 2.0, 0.1)
y_ = atanh(x_)
fig = plt.figure(figsize=(12,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.xlabel('X', fontsize=16)
plt.grid(True)
plt.title('Sigmoid-type = 1.7159*tanh((2/3*x)', size=20)
plt.plot(x_, y_, color='green', lw=4)
st.pyplot(fig)

show_text_color('Sistema de aprendizaje que relaciona las variables X y Y')

st.markdown(r'''Se tienen $m$ muestras. 
    
Para una muestra $j \in m$ $X^{(j)}$:''')

st.latex(r'Z^{(j)} = w^T X^{(j)} + b ')

st.markdown(r'$Z^{(j)}$ es la combinacion lineal entre los pesos ($w$) y la muestra $X^{(j)}$ + el bias b. Posteriormente esta sumatoria es transformada, para generar el valor de salida $a^{(j)}$, mediante una función. En el presente caso la función es del tipo sigmoide (*atanh*), porque la relación es ligeramente no-lineal.')

st.latex(r'a^{(j)} = atanh(Z^{(j)})')

st.markdown(r'La funcion de perdida (*loss*), tambien llamada costo *cost*, $\textbf {J}$, es el promedio, sobre todas las muestras, de la diferencia al cuadrado entre el valor medido ($Y^{(j)}$) y el valor que predice la funcion de activacion ($a^{(j)}$).  ')

st.latex(r'J = \frac{1}{m} \sum_{j=1}^m (a^{(j)} - Y^{(j)})^2')

st.markdown('''Para mas informacion en la literatura del archivo zip el archivo Efficient-backprop_Lecun_1998.pdf)
            
Con ello, la varianza es cercana a 1, el valor de la función es ligeramente no lineal entre -1 y +1, y la derivada es máxima para $Z^{(j)} = 1$ 

Con esto, la función de costo, $J$ está dada por:''')
st.latex(r'J = \dfrac{1}{m}\sum_{i=0}^{m}(1.7159*tanh(2*Z^{(j)}/3)-Y^{(j)})^2')

show_text_color('1. Se inicializan los parámetros del sistema de aprendizaje que definen la relación entre $X$ y $Y$')

st.code('''def initialize_params_1(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = np.reshape(np.random.uniform(-0.1, 0.1, dim), (dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b''')

def initialize_params_1(dim):
    w = np.reshape(np.random.uniform(-0.1, 0.1, dim), (dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

st.code('''def atanh(x):
    return 1.7159*tanh(2*x/3)''')
def atanh(x):
    return 1.7159*tanh(2*x/3)
st.code('''def d_atanh(z):
    return 1.7159*(1-np.square(atanh(z)))*(2/3)''')

def d_atanh(z):
    return 1.7159*(1-np.square(atanh(z)))*(2/3)

st.code('''
dim = train_x.shape[0]
w, b = initialize_params(dim)
print ("w = " + str(w))
print ("b = " + str(b))''')


dim = train_x.shape[0]
w, b = initialize_params(dim)
st.write("$w = " + str(w))
st.write("$b = " + str(b))

st.code('''def propagate_1(w, b, X, Y, X_val, Y_val):
    
    m = X.shape[1]
    m_val = X.shape[1]
    
    A = atanh(np.dot(w.T, X)+b)
    A_val = atanh(np.dot(w.T,X_val)+b)
    
    cost = (1/m)*np.sum((Y-A)**2)
    cost_val = (1/m_val)*np.sum((Y_val-A_val)**2)
    
    dA = A-Y
    dZ = d_atanh(np.dot(w.T, X)+b)
    
    dw = (1/m)*np.dot(dA, (dZ*X).T).T    
    db = (1/m)*np.sum((A-Y)*(1-np.power(A, 2)), axis=1, keepdims=True)  
    
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost, cost_val''')

def propagate_1(w, b, X, Y, X_val, Y_val):
    m = X.shape[1]
    m_val = X.shape[1]
    A = atanh(np.dot(w.T, X)+b)
    A_val = atanh(np.dot(w.T,X_val)+b)
    cost = (1/m)*np.sum((Y-A)**2)
    cost_val = (1/m_val)*np.sum((Y_val-A_val)**2)
    dA = A-Y
    dZ = d_atanh(np.dot(w.T, X)+b)
    dw = (1/m)*np.dot(dA, (dZ*X).T).T    
    db = (1/m)*np.sum((A-Y)*(1-np.power(A, 2)), axis=1, keepdims=True)  
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost, cost_val

st.code('''x1_val = np.array([[1.,2.,-2.],[2.,4.,-3.2]])
y1_val = np.array([[1,0,2]])

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost, cost_val = propagate_1(w, b, X, Y,x1_val, y1_val)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))''')

x1_val = np.array([[1.,2.,-2.],[2.,4.,-3.2]])
y1_val = np.array([[1,0,2]])

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost, cost_val = propagate_1(w, b, X, Y,x1_val, y1_val)
st.write("$dw = " + str(grads["dw"]))
st.write("$db = " + str(grads["db"]))
st.write("$cost = " + str(cost))

st.markdown(r'''    
La función *optimize_1(  )*  actualiza los pesos $w$ y el bias $b$ usando el método de gradiente descendente.

Se calcula la función de costo para las muestras empleadas en el entrenamiento y para las asignadas a la validación.
    
Los pesos $ w $ y el bias $ b $ son modificados en cada época hasta que la función de costo $ J $ llega a su valor mínimo. 

Los pesos $ w_{i} $ y el bias $b$ se actualizan mediante la siguientes relaciones: ''')

st.latex(r'w_{i} := w_{i} - \alpha \frac{\partial J}{\partial w_{i}}')

st.latex(r'b := b - \alpha \frac{\partial J}{\partial b}')

st.markdown(r'$ \alpha $ es el hyperparámetro que define la relación de aprendizaje.')

st.code('''def optimize_1(w, b, X, Y, X_val, Y_val, epochs, learning_rate, print_cost = False):
    
    costs = []
    costs_val = []
    
    for i in range(epochs):
        
        grads, cost, cost_val = propagate_1(w, b, X, Y, X_val, Y_val)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update parameters
        w = w-learning_rate*dw
        b = b-learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            costs_val.append(cost_val)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost and Cost_val after iteration %i: %f %f" %(i, cost, cost_val))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs, costs_val''')

def optimize_1(w, b, X, Y, X_val, Y_val, epochs, learning_rate, print_cost = False):
    costs = []
    costs_val = []
    for i in range(epochs):
        grads, cost, cost_val = propagate_1(w, b, X, Y, X_val, Y_val)
        dw = grads["dw"]
        db = grads["db"]
        w = w-learning_rate*dw
        b = b-learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
            costs_val.append(cost_val)
        if print_cost and i % 100 == 0:
            st.write("$Cost and Cost_val after iteration %i: %f %f" %(i, cost, cost_val))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs, costs_val

st.code('''epochs = 100
params, grads, costs, costs_val = optimize_1(w, b, X, Y, x1_val, y1_val, epochs=epochs, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))''')

epochs = 100
params, grads, costs, costs_val = optimize_1(w, b, X, Y, x1_val, y1_val, epochs=epochs, learning_rate = 0.009, print_cost = False)
st.write("$w = " + str(params["w"]))
st.write("$b = " + str(params["b"]))
st.write("$dw = " + str(grads["dw"]))
st.write("$db = " + str(grads["db"]))


st.markdown(r'''Una vez que se encuentran los valores de los pesos 𝑤 y del bias 𝑏 que minimizan la función de costo, la función, $𝐹(W,𝑋)$ que se genera con estos valores se emplea para inferir valores de Y asociados a las muestras de prueba, los cuales se comparan con los valores Y conocidos de estas muestras.
$$ $$

    
La función *predic()_1* calcula $𝐹(W,𝑋)$ empleando los valores de 𝑤 y 𝑏
para los cuales el costo es mínimo.''')

st.code('''def predict_1(w, b, X):
    
    #m = X.shape[1]
    #Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    #A = sigmoid(np.dot(w.T, X)+b)
    A = atanh(np.dot(w.T, X)+b)
    
    return A''')

def predict_1(w, b, X):
    w = w.reshape(X.shape[0], 1)
    A = atanh(np.dot(w.T, X)+b)
    return A

st.markdown('Para estructurar el sistema de aprendizaje, se juntan todos las funciones implementadas en las partes anteriores, en el orden correcto.')

st.code('''def model_1(X_train, Y_train, val_ratio, epochs = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    epochs -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    train_ratio = int((1-val_ratio)*X_train.shape[1])
    X_val = X_train[:,train_ratio:]
    Y_val = Y_train[:,train_ratio:]

    X_train = X_train[:,:train_ratio]
    Y_train = Y_train[:,:train_ratio]
    
    print("Train",X_train.shape,Y_train.shape)
    print("val",X_val.shape,Y_val.shape)

    
    # initialize parameters
    w, b = initialize_params_1(X_train.shape[0])

    # Gradient descent 
    parameters, grads, costs, costs_val = optimize_1(w, b, X_train, Y_train, X_val, Y_val, epochs, learning_rate, print_cost = print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    print(w.shape)
    
    # Predict test/train set examples
    Y_prediction_val = predict_1(w, b, X_val)
    Y_prediction_train = predict_1(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.power(Y_prediction_train - Y_train, 2)) * 100))
    print("val accuracy: {} %".format(100 - np.mean(np.power(Y_prediction_val - Y_val, 2)) * 100))

    
    d = {"costs": costs,"costs_val": costs_val,
         "Y_prediction_val": Y_prediction_val, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": epochs}
    
    return d''')


def model_1(X_train, Y_train, val_ratio, epochs = 2000, learning_rate = 0.5, print_cost = False):
    train_ratio = int((1-val_ratio)*X_train.shape[1])
    X_val = X_train[:,train_ratio:]
    Y_val = Y_train[:,train_ratio:]
    X_train = X_train[:,:train_ratio]
    Y_train = Y_train[:,:train_ratio]
    st.write("$Train",X_train.shape,Y_train.shape)
    st.write("$val",X_val.shape,Y_val.shape)
    w, b = initialize_params_1(X_train.shape[0])
    parameters, grads, costs, costs_val = optimize_1(w, b, X_train, Y_train, X_val, Y_val, epochs, learning_rate, print_cost = print_cost)
    w = parameters["w"]
    b = parameters["b"]
    st.write("$",w.shape)
    Y_prediction_val = predict_1(w, b, X_val)
    Y_prediction_train = predict_1(w, b, X_train)
    st.write("$train accuracy: {} %".format(100 - np.mean(np.power(Y_prediction_train - Y_train, 2)) * 100))
    st.write("$val accuracy: {} %".format(100 - np.mean(np.power(Y_prediction_val - Y_val, 2)) * 100))
    d = {"costs": costs,"costs_val": costs_val,
         "Y_prediction_val": Y_prediction_val, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": epochs}
    return d

st.code('''epochs = 1000
learning_rate = 0.005
val_ratio = 0.1

d = model_1(train_x, train_y.T, val_ratio=val_ratio, epochs = epochs, 
          learning_rate = learning_rate, print_cost = True)''')
epochs = 1000
learning_rate = 0.005
val_ratio = 0.1
d = model_1(train_x, train_y.T, val_ratio=val_ratio, epochs = epochs, 
          learning_rate = learning_rate, print_cost = True)

st.code('''costs = np.squeeze(d['costs'])
costs_val = np.squeeze(d['costs_val'])
plt.figure(figsize=(10,8)) 

plt.plot(costs, color='red')
plt.plot(costs_val, color='green')
plt.ylabel('cost', size=16)
plt.xlabel('epoch x 100', size=16)
plt.title("Learning rate =" + str(d["learning_rate"]), size=16)
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()''')

costs = np.squeeze(d['costs'])
costs_val = np.squeeze(d['costs_val'])
fig = plt.figure(figsize=(10,8)) 

plt.plot(costs, color='red')
plt.plot(costs_val, color='green')
plt.ylabel('cost', size=16)
plt.xlabel('epoch x 100', size=16)
plt.title("Learning rate =" + str(d["learning_rate"]), size=16)
plt.legend(['Train', 'Val'], loc='upper right')
st.pyplot(fig)

st.code('''plt.figure(figsize=(10,8))

learning_rates = [0.005, 0.003, 0.001]
val_ratio = 0.1
models = {}

for i in learning_rates:
    print ("learning rate is: ", i)
    models[str(i)] = model_1(train_x, train_y.T, val_ratio=val_ratio, epochs = 1000, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
    plt.plot(np.squeeze(models[str(i)]["costs_val"]), label= str(models[str(i)]["learning_rate"]))
plt.ylabel('cost')
plt.xlabel('epochx100')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()''')

fig = plt.figure(figsize=(10,8))

learning_rates = [0.005, 0.003, 0.001]
val_ratio = 0.1
models = {}

for i in learning_rates:
    st.write("$learning rate is: ", i)
    models[str(i)] = model_1(train_x, train_y.T, val_ratio=val_ratio, epochs = 1000, learning_rate = i, print_cost = False)
    st.write('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
    plt.plot(np.squeeze(models[str(i)]["costs_val"]), label= str(models[str(i)]["learning_rate"]))
plt.ylabel('cost')
plt.xlabel('epochx100')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
st.pyplot(fig)

show_text_color('<center> Artificial Neural Networks </center>')

st.markdown('Inspirandose en modelos que intentan describir las conecciones entre las neuronas en nuestro cerebro, se propusieron (y se siguen proponiendo) modelos de redes neuronales para generar sistemas de aprendizaje. Se les conoce con el nombre de redes neuronales artificiales, o simplemente como redes neuronales.')

show_text_color('Implementación de una red neuronal del tipo "Full Feed-forward (FFF)".')

st.markdown('''
<p>En nuestro primer modelo, la red neuronal tiene 3 capas: la capa de entrada, la capa de salida y una capa interior (en la literatura, a esta capa interior se le da el nombre de capa oculta). </p>
<p>El objetivo del modelo es encontrar una función que describa la evolución de la Diabetes Mellitus en una año, a partir de su linea base, con los rasgos de la persona que se consideran importantes para su evolución. 

Los rasgos propuestos son: edad, sexo, índice de masa corporal, presión arterial promedio y las seis mediciones de suero sanguíneo descritas al inicio de esta presentación: el colesterol total, la densidad baja de lipoproteinas, la densidad alta de lipoproteinas, los trigliceridos, la concentración de lamorigina y la glucosa</p>
<p>Esta función se genera mediante una red de neuronas artificiales. Se entiende como neurona un modelo matemático simple de una neurona biológica.</p>''',unsafe_allow_html=True)

show_text_color('Adecuando lo datos de alimentación al sistema de aprendizaje')

st.markdown('Antes de inciar el desarrollo del sistema, transformamos los datos de entrada para que sean compatibles con el modelo que desarrollaremos. El fomato de entrada de las variables X y Y, tanto para el entrenamiento como para la prueba es un poco diferente al empleado en los sistemas de aprendizaje anteriores.')


st.code('''print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)''')

st.write("$",train_x.shape)
st.write("$",train_y.shape)
st.write("$",test_x.shape)
st.write("$",test_y.shape)

st.code('''train_y = train_y.T
test_y = test_y.T''')

train_y = train_y.T
test_y = test_y.T

st.code('''print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)''')

st.write("$",train_x.shape)
st.write("$",train_y.shape)
st.write("$",test_x.shape)
st.write("$",test_y.shape)

st.write('''1.1 Definición de la arquitectura de la red neuronal.

Se emplea indistintamente la palabra neurona o nodo para referirse al modelo matematico de la neurona.

El número de nodos en la capa de entrada depende del numero de rasgos del sistema que definen la variable objetivo, la evolución de la diabetes en un año. En el presente caso el número de rasgos es diez.

El número de nodos en la capa de salida depende del tipo de problema. En el presente caso, se tiene una sola neurona, cuya salida nos da un número que cuantifica la evolución de la diabetes en un año.

En el presente modelo, sólo tenemos una capa interna, el número de nodos en ella es variable. Se hacen pruebas con diferentes números y se adopta el que genere los mejores resultados.

La función layer_sizes() genera la arquitectura de la red neuronal partiendo de los datos con que se van a alimentar a la red.
''')

st.code('''def layer_sizes(X, Y, n_h):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    
    n_x = X.shape[0] 
    
    n_h = n_h
    
    n_y = Y.shape[0]
    
    return (n_x, n_h, n_y)''')

def layer_sizes(X, Y, n_h):
    n_x = X.shape[0] 
    n_h = n_h
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

st.code('''n_h = 4
n_x, n_h, n_y = layer_sizes(train_x, train_y, n_h = n_h)''')

n_h = 4
n_x, n_h, n_y = layer_sizes(train_x, train_y, n_h = n_h)

st.code('print(n_x, n_h, n_y)')

st.write("$",n_x, n_h, n_y)
show_text_color('Network Visualization  ')

st.markdown('''Emplearemos la biblioteca en python Networkx, la cual permite la creación, manipulación y estudio de la arquitecura, dinámica y funciones complejas de redes.   
    
[NetworkX](https://networkx.github.io/)''')

st.code('''import networkx as nx

class Network(object):
    
    def  __init__ (self,sizes):
        self.num_layers = len(sizes)
        print("It has", self.num_layers, "layers,")
        self.sizes = sizes
        print("with the following number of nodes per layer",self.sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, x_of_sample):
        """Return the output of the network F(x_of_sample) """        
        for b, w in zip(self.biases, self.weights):
            x_of_sample = sigmoid(np.dot(w, x_of_sample)+b)
        return x_of_sample
    
    def graph(self,sizes):
        a=[]
        ps={}
        Q = nx.Graph()
        for i in range(len(sizes)):
            Qi=nx.Graph()    
            n=sizes[i]
            nodos=np.arange(n)
            Qi.add_nodes_from(nodos)
            l_i=Qi.nodes
            Q = nx.union(Q, Qi, rename = (None, 'Q%i-'%i))
            if len(l_i)==1:
                ps['Q%i-0'%i]=[i/(len(sizes)), 1/2]
            else:
                for j in range(len(l_i)+1):
                    ps['Q%i-%i'%(i,j)]=[i/(len(sizes)),(1/(len(l_i)*len(l_i)))+(j/(len(l_i)))]
            a.insert(i,Qi)
        for i in range(len(a)-1):
            for j in range(len(a[i])):
                for k in range(len(a[i+1])):
                    Q.add_edge('Q%i-%i' %(i,j),'Q%i-%i' %(i+1,k))
        nx.draw(Q, pos = ps)''')

import networkx as nx

class Network(object):
    
    def  __init__ (self,sizes):
        self.num_layers = len(sizes)
        print("It has", self.num_layers, "layers,")
        self.sizes = sizes
        print("with the following number of nodes per layer",self.sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, x_of_sample):
        """Return the output of the network F(x_of_sample) """        
        for b, w in zip(self.biases, self.weights):
            x_of_sample = sigmoid(np.dot(w, x_of_sample)+b)
        return x_of_sample
    
    def graph(self,sizes):
        fig, ax = plt.subplots()
        a=[]
        ps={}
        Q = nx.Graph()
        for i in range(len(sizes)):
            Qi=nx.Graph()    
            n=sizes[i]
            nodos=np.arange(n)
            Qi.add_nodes_from(nodos)
            l_i=Qi.nodes
            Q = nx.union(Q, Qi, rename = (None, 'Q%i-'%i))
            if len(l_i)==1:
                ps['Q%i-0'%i]=[i/(len(sizes)), 1/2]
            else:
                for j in range(len(l_i)+1):
                    ps['Q%i-%i'%(i,j)]=[i/(len(sizes)),(1/(len(l_i)*len(l_i)))+(j/(len(l_i)))]
            a.insert(i,Qi)
        for i in range(len(a)-1):
            for j in range(len(a[i])):
                for k in range(len(a[i+1])):
                    Q.add_edge('Q%i-%i' %(i,j),'Q%i-%i' %(i+1,k))
        nx.draw(Q, pos = ps)
        st.pyplot(fig)

st.code('''layers = [n_x, n_h, n_y]
net = Network(layers)
net.graph(layers)''')

layers = [n_x, n_h, n_y]
net = Network(layers)
net.graph(layers)

show_text_color(' 1. Inicializacion de los pesos y el bias.')

st.markdown(f'''La funcion initialize_parameters() inicializa a los pesos $W$ y el bias $b$. 

Dado que se tiene un conjunto de variables independientes, se debe definir un peso para cada variable, esto para una sola neurona de la siguiente capa. 

Entonces $W_1$ ahora es una matriz de tamaño $(n_h, n_x)$, en donde $n_h$ es el numero de nodos en la capa intera y $n_x$ es el numero de nodos en la capa de entrada, es decir, es el numero de variables independientes (rasgos).

Para cada neurona en la capa interna hay un bias, por lo que ahora $b_1$ es un vector de tamaño $(n_h, 1)$. 

En general para cada par de capas consecutivas debe haber un $W$ y un $b$.

Generalizando:

$W_i$ y $b_i$ son los parametros a definir entre la capa $i$ y la capa $i+1$. Si la capa $i$ tiene $n_i$ neuronas y la capa $i+1$ tiene $n_{i+1}$ neuronas, entonces las dimensiones de $W_{i}$ son $(n_{i+1}, n_i)$ y las de $b_i$ son $(n_{i+1}, 1)$.''')

st.code('''def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) 
    
    W1 = np.reshape(np.random.uniform(-0.1, 0.1, n_h*n_x), (n_h, n_x))
        
    b1 = np.zeros((n_h, 1))
    
    W2 = np.reshape(np.random.uniform(-0.1, 0.1, n_y*n_h), (n_y, n_h))
        
    b2 = np.zeros((n_y, 1))
    
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters''')

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) 
    
    W1 = np.reshape(np.random.uniform(-0.1, 0.1, n_h*n_x), (n_h, n_x))
        
    b1 = np.zeros((n_h, 1))
    
    W2 = np.reshape(np.random.uniform(-0.1, 0.1, n_y*n_h), (n_y, n_h))
        
    b2 = np.zeros((n_y, 1))
    
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

st.code('initialize_parameters(n_x, n_h, n_y)')

st.write(initialize_parameters(n_x, n_h, n_y))

show_text_color('2. Conección entre las neuronas de capas contiguas.')

st.markdown('''La funcion *propagate()* realiza la combinacion lineal entre los valores de salida de los nodos de una capa con los pesos y bias definidos entre esa capa y la siguiente. 

La función de activación que se aplica a esta sumatoria, es para considerar efectos no lineales.


___

Funciones de activación disponibles en la presente notebook: ''')

st.code('''#Función para considerar los efectos no lineales.
#En el presente caso se considera un modelo completamente lineal.
#Por ello la función es la identidad.

def identity(z):
    """
    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- z
    """
    s = 1.0 * z
    
    return s''')

def identity(z):
    s = 1.0 * z
    return s

st.code('''def identity_derivative(z):
    return 1.0 * z * (1/z)''')
def identity_derivative(z):
    return 1.0 * z * (1/z)
st.code('''def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))''')
def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

st.code('''def tanh_derivative(z):
    return 1-np.power(tanh(z), 2)''')

def tanh_derivative(z):
    return 1-np.power(tanh(z), 2)

st.code('''def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1/(1+np.exp(-z))
    
    return s''')
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s
st.code('''def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))''')

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

st.code('''def propagate(X, Y, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Zi es la combinacion lineal entre x y w
    # Ai es la aplicacion de una funcion de activacion a Zi
    
    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2
    
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    m = Y.shape[1] # number of samples

    cost = (1/m)*np.sum((Y-A2)**2)
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Calculo de derivadas
    
    dZ2 = 2*(A2-Y)
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2)*tanh_derivative(A1)
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis = 1, keepdims = True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    
    return A2, cache, cost, grads''')

def propagate(X, Y, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2
    assert(A2.shape == (1, X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    m = Y.shape[1] 
    cost = (1/m)*np.sum((Y-A2)**2)
    cost = np.squeeze(cost)   
    assert(isinstance(cost, float))
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]   
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = 2*(A2-Y)
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2)*tanh_derivative(A1)
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis = 1, keepdims = True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return A2, cache, cost, grads

st.code('''def validation(X, Y, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    Y -- output data of size (n_y, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    cost -- the value of cost
    grads -- a dictionary contains derivatives to update parameters
    """
    # Regresa cada parametro del diccionario "parameters"
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Pasos 1 y 2:
    
    # Zi es la combinacion lineal entre x y w
    # Ai es la aplicacion de una funcion de activacion a Zi:
    
    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2
    
    # se verifican las dimensiones de A2:
    
    assert(A2.shape == (1, X.shape[1]))

    # Paso 3:
    
    # numero de muestras:
    
    m = Y.shape[1] 
    
    # se calcula el costo:

    cost = (1/m)*np.sum((Y-A2)**2)
    
    # Asegura que cost sea un escalar:
    
    cost = np.squeeze(cost)      
                                
    assert(isinstance(cost, float))  
    
    return cost''')

def validation(X, Y, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2
    assert(A2.shape == (1, X.shape[1]))
    m = Y.shape[1] 
    cost = (1/m)*np.sum((Y-A2)**2)
    cost = np.squeeze(cost)                         
    assert(isinstance(cost, float))     
    return cost

show_text_color('3. Cálculo de la función de costo durante la optimización de los parámetros que definen al modelo.')

st.markdown(r'''Recordemos que la funcion de costo, $J$, nos permite saber qué tan bien se esta ajustando el modelo a la variable objetivo de las muestras. 

Para ello se buscan los parámetros que minimizen a esta función. 

En el presente caso, la función de costo está definida por la relación siguiente: ''')

st.latex(r'J = \dfrac{1}{m} \sum_{i=1}^m(a_i - y_i)^2')

st.code('''def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[1] # number of samples

    cost = (1/m)*np.sum((Y-A2)**2)
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost''')

def compute_cost(A2, Y, parameters):
    m = Y.shape[1] 
    cost = (1/m)*np.sum((Y-A2)**2)
    cost = np.squeeze(cost)   
    assert(isinstance(cost, float))
    return cost

st.markdown('Para encontrar a los valores optimos de los parametros, estos se acualizan en cada época empleando el algoritmo de gradiente descendente. El cual esta definido por la siguientes relaciones:')

st.latex(r'\omega_{k-new} = \omega_k - \alpha \dfrac{\partial J(\omega, b)}{\partial \omega_k}')

st.latex(r'b_{l-new} = b_l - \alpha \dfrac{\partial J(\omega, b)}{\partial b_l}')

st.markdown(r'Es por ello necesario calcular las derivadas del costo respecto a cada uno de los parametros que definen al sistema de aprendizaje. $\alpha$ es la relación de aprendizaje.')

st.code('''def calculation_of_derivatives(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Calculo de derivadas
    
    dZ2 = 2*(A2-Y)
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1, 2))
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis = 1, keepdims = True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads''')

def calculation_of_derivatives(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]     
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = 2*(A2-Y)
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1, 2))
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis = 1, keepdims = True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

show_text_color('4. Optimizacion de los pesos y los bias.')

st.code('''def optimize(parameters, grads, learning_rate = 0.1):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Update rule for each parameter
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters''')

def optimize(parameters, grads, learning_rate = 0.1):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

show_text_color('5. Las predicciones se realizan con los parametros óptimos encontrados en el entrenamiento.')

st.code('''def predict(parameters, X, Y):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    predictions =  []
    A2, cache, cost, grads = propagate(X, Y, parameters)
    predictions = identity(A2) 
    
    return predictions''')

def predict(parameters, X, Y):
    predictions =  []
    A2, cache, cost, grads = propagate(X, Y, parameters)
    predictions = identity(A2) 
    return predictions

show_text_color('Las funciones anteriores se integran para generar, entrenar y validar la red neuronal.')

st.code('''def nn_model(X, Y, val_ratio, n_h, epochs, alpha, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    train_ratio = int((1-val_ratio)*X.shape[1])
    X_val = X[:,train_ratio:]
    Y_val = Y[:,train_ratio:]

    X = X[:,:train_ratio]
    Y = Y[:,:train_ratio]
    
    print("Train",X.shape,Y.shape)
    print("val",X_val.shape,Y_val.shape)
    
    np.random.seed(3)
    
    n_x, n_h, n_y = layer_sizes(X, Y, n_h = n_h)
        
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"] 
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)
    
    costs=[]
    costs_val = []
    params = []
    

    for i in range(0, epochs):
         
        A2, cache, cost, grads = propagate(X, Y, parameters)
        
        cost_val = validation(X_val, Y_val, parameters)
 
        parameters = optimize(parameters, grads, alpha)
        
        params.append(parameters)
        
        costs.append(cost)
        
        costs_val.append(cost_val)
                
        # Print the cost every 1000 iterations
        if print_cost and i % 100 == 0:
            print ("Cost and Cost_val in epoch %i: %f %f" %(i, cost, cost_val))
            
    return parameters, costs, params, costs_val''')

def nn_model(X, Y, val_ratio, n_h, epochs, alpha, print_cost=False):
    train_ratio = int((1-val_ratio)*X.shape[1])
    X_val = X[:,train_ratio:]
    Y_val = Y[:,train_ratio:]
    X = X[:,:train_ratio]
    Y = Y[:,:train_ratio]
    st.write("$Train",X.shape,Y.shape)
    st.write("$val",X_val.shape,Y_val.shape)
    np.random.seed(3)
    n_x, n_h, n_y = layer_sizes(X, Y, n_h = n_h)
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"] 
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    costs=[]
    costs_val = []
    params = []
    for i in range(0, epochs):         
        A2, cache, cost, grads = propagate(X, Y, parameters)        
        cost_val = validation(X_val, Y_val, parameters)
        parameters = optimize(parameters, grads, alpha)
        params.append(parameters)
        costs.append(cost)
        costs_val.append(cost_val)
        if print_cost and i % 100 == 0:
            st.write("$Cost and Cost_val in epoch %i: %f %f" %(i, cost, cost_val))
    return parameters, costs, params, costs_val

show_text_color('Entrenamiento:')

st.markdown('''Para monitorear el entrenamiento, las muestras para el entrenamiento se dividen en dos grupos:
El (1.0 - val_ratio) (90% en el presente caso) de ellas se emplean para realizar el entrenamiento y el (val_ratio) (el 10% en el presente caso) restante para evaluar, "validar", la calidad del entrenamiento.''')

st.code('''epochs = 2000
learning_rate = 0.008
val_ratio = 0.1
n_h = 4

opt_parameters, costs, params, costs_val = nn_model(train_x, train_y, val_ratio=val_ratio, n_h = n_h, epochs = epochs, alpha=learning_rate, print_cost=True)
''')

epochs = 2000
learning_rate = 0.008
val_ratio = 0.1
n_h = 4

opt_parameters, costs, params, costs_val = nn_model(train_x, train_y, val_ratio=val_ratio, n_h = n_h, epochs = epochs, alpha=learning_rate, print_cost=True)

st.code('''costs = np.squeeze(costs)

plt.figure(figsize=(10,8)) 
plt.plot(costs, color='red')
plt.plot(costs_val, color='green')

plt.ylabel('Cost', size=16)
plt.xlabel('epochs', size=16)
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()''')

costs = np.squeeze(costs)

fig = plt.figure(figsize=(10,8)) 
plt.plot(costs, color='red')
plt.plot(costs_val, color='green')

plt.ylabel('Cost', size=16)
plt.xlabel('epochs', size=16)
plt.legend(['Train', 'Val'], loc='upper right')
st.pyplot(fig)

st.write(''' ## Precisión:

Para calcular la precisión del modelo se usa el error cuadrático medio, MSE:''')

st.latex(f'100-MSE*100')
st.markdown('es decir')
st.latex('100-(\dfrac{1}{m_{test}}\sum_{i=1}^{m_{test}} (y_{i}-a_{i})^2)*100')

st.code('''predictions_test = predict(opt_parameters, test_x, test_y)

print("test accuracy: {} %".format(100 - np.mean(np.power(predictions_test-test_y, 2)) * 100))
''')

predictions_test = predict(opt_parameters, test_x, test_y)

st.write("$test accuracy: {} %".format(100 - np.mean(np.power(predictions_test-test_y, 2)) * 100))


