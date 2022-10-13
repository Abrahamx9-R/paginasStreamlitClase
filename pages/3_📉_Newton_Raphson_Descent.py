import streamlit as st
import json
from streamlit_lottie import st_lottie
import base64
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import numpy as np


st.set_page_config(
    page_title="Método de Newton-Raphson",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://www.extremelycoolapp.com/help',
        'Report a Bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title('Método de Newton-Raphson')

def load_lottieurl(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

lottie_hello = load_lottieurl("pages/images/growbisnes.json")

st_lottie(lottie_hello,speed = 1, reverse=False,loop=True,quality="low",height=600,width=None,key=None,)

st.markdown(r'''El método de Newton-Rapshon se usa para encontrar las raíces de una función real. 

Se parte de la derivada de la función $f$ y una suposición inicial $x_0$ para una raíz de $f$. Si la función satisface suposiciones suficientes y la suposición inicial es cercana, entonces se mejora la aproximación de la raíz con:''')

st.latex(r" x_1 = x_0 - \dfrac{f(x_0)}{f´(x_0)}")

st.markdown(r''' $(x_1, 0)$ es la intersección del eje $x$ y la tangente de la gráfica de $f$ en $(x_0, f(x_0))$.
[Newton´s method](https://en.wikipedia.org/wiki/Newton%27s_method)''')

file_ = open("pages/images/NewtonIteration_Ani.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<center> <img width="90%" src="data:image/gif;base64,{data_url}" alt="cat gif"> </center>',
    unsafe_allow_html=True,
) 
st.markdown('(By Ralf Pfeifer - de:Image:NewtonIteration Ani.gif, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=2268473)')

st.code('''def newton(f,Df,x0,epsilon,max_iter):
    # Approximate solution of f(x)=0 by Newton's method.

    # Parameters
    # ----------
    # f : function
        
    # Df : Derivative of f(x).
    
    # x0 : Initial guess for a solution f(x)=0.
    
    # epsilon :Stopping criteria is abs(f(x)) < epsilon.
    
    # max_iter : Maximum number of iterations of Newton's method.

    # Returns
    # -------
    # xn : number
    #     Implement Newton's method: compute the linear approximation
    #     of f(x) at xn and find x intercept by the formula
    #         x = xn - f(xn)/Df(xn)
    #     Continue until abs(f(xn)) < epsilon and return xn.
    #     If Df(xn) == 0, return None. If the number of iterations
    #     exceeds max_iter, then return None.

    aprox_root = [x0]
    
    xn = x0
      
    # xn es la aproximación de la raíz de f. Inicialmente xn =x0 con x0 la primera aproximación
    
    for n in range(0,max_iter):
        
        fxn = f(xn)
        
        if abs(fxn) < epsilon:
            
            print('Found solution after',n,'iterations.')
            
            return xn, aprox_root
        
        Dfxn = Df(xn)
        
        if Dfxn == 0:
            
            print('Zero derivative. No solution found.')
            
            return None
        
        xn = xn - fxn/Dfxn
        
        aprox_root.append(xn)
        
    print('Exceeded maximum iterations. No solution found.')
    
    return None''')
    
def newton(f,Df,x0,epsilon,max_iter):    
    aprox_root = [x0]  
    xn = x0
    for n in range(0,max_iter):   
        fxn = f(xn)        
        if abs(fxn) < epsilon:            
            st.write('$Found solution after',n,'iterations.')            
            return xn, aprox_root        
        Dfxn = Df(xn)        
        if Dfxn == 0:            
            st.write('Zero derivative. No solution found.')            
            return None        
        xn = xn - fxn/Dfxn        
        aprox_root.append(xn)        
    st.write('$Exceeded maximum iterations. No solution found.')    
    return None

st.code('''p = lambda x: (x-1)**2 - 1
Dp = lambda x: 2*(x-1)

newton(p, Dp,10,1e-10,10)''')

p = lambda x: (x-1)**2 - 1
Dp = lambda x: 2*(x-1)
newton(p, Dp,10,1e-10,10) 

st.code(''' def tangent_line(f,Df, x_0,a,b):
    x = np.linspace(a,b)
    y = f(x) 
    y_0 = f(x_0)
    y_tan = Dp(x_0) * (x - x_0) + y_0 
   
  #plotting
    plt.plot(x,y,'r-')
    plt.plot(x,y_tan,'b-')
    plt.axis([a,b,a,b])
    plt.xlabel('x')     
    plt.ylabel('y')  
    plt.grid(True)
    plt.title('Plot of a function with tangent line') 
    plt.show()  ''')

def tangent_line(f,Df, x_0,a,b):
    x = np.linspace(a,b)
    y = f(x) 
    y_0 = f(x_0)
    y_tan = Dp(x_0) * (x - x_0) + y_0     
    fig, ax = plt.subplots()
    ax.set_ylabel('Y',fontdict = {'fontsize':16})
    ax.set_xlabel('X',fontdict = {'fontsize':16})
    ax.set_title('Plot of a function with tangent line')
    ax.grid(visible=True)
    ax.set_ylim([a,b])
    ax.set_xlim([a,b])
    ax.plot(x,y,'r-')
    ax.plot(x,y_tan,'b-')
    ax.legend(fontsize=6)
    st.pyplot(fig)

st.code('''p = lambda x: (x-1)**2 - 1
Dp = lambda x: 2*(x-1)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

for i in newton(p, Dp,13,1e-10,10)[1]:
    
    print(i)

    tangent_line(p, Dp, i, -5, 5)''')


p = lambda x: (x-1)**2 - 1
Dp = lambda x: 2*(x-1)

for i in newton(p, Dp,13,1e-10,10)[1]:
    st.write("$",i)
    tangent_line(p, Dp, i, -5, 5)

st.title('Gradient Descent')

st.markdown(f'''Es un algoritmo de optimizacion para encontrar el minimo local de una funcion diferenciable.

Para encontrar un mínimo local de una función usando el descenso de gradiente, se toman pasos proporcionales al negativo del gradiente (o gradiente aproximado) de la función en el punto actual.

[Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)''')

st.code('''f = lambda x: x**3 - 3*x**2 + 7
x = np.linspace(-1,3,500)
plt.plot(x, f(x))
plt.show()''')
f = lambda x: x**3 - 3*x**2 + 7
x = np.linspace(-1,3,500)
fig, ax = plt.subplots()
ax.plot(x, f(x))
st.pyplot(fig)

st.code('''df = lambda x: 3*x**2 - 6*x
x = np.linspace(-1,3,500)
plt.plot(x, df(x))
plt.show()''')
df = lambda x: 3*x**2 - 6*x
x = np.linspace(-1,3,500)
fig, ax = plt.subplots()
ax.plot(x, df(x))
st.pyplot(fig)

st.code('''next_x = 6  # We start the search at x=6
gamma = 0.01  # Step size multiplier
precision = 0.00001  # Desired precision of result
max_iters = 10000  # Maximum number of iterations

xs = [next_x]

for _i in range(max_iters):
    
    current_x = next_x
    
    next_x = current_x - gamma * df(current_x)
    
    xs.append(next_x)

    step = next_x - current_x
    
    if abs(step) <= precision:
        
        break
        
print("Minimum at {0:5.2f}".format(next_x))

# The output for the above will be something like
# "Minimum at 2.2499646074278457"''')

next_x = st.sidebar.number_input("X donde inicia la busqueda (next_x)",value=6.0,format="%f")
gamma = st.sidebar.number_input("Paso de busqueda (gamma)",value=0.01,format="%f")
precision = st.sidebar.number_input("Presicion del resultado (precision)",value=0.00001,format="%f")
max_iters = st.sidebar.number_input("Maximo numero de iteraciones (max_inters)",value=10000,format="%i")

xs = [next_x]
for _i in range(max_iters):    
    current_x = next_x    
    next_x = current_x - gamma * df(current_x)    
    xs.append(next_x)
    step = next_x - current_x    
    if abs(step) <= precision:        
        break        
st.write("$Minimum at {0:5.2f}".format(next_x))

st.code('''plt.plot(x, f(x), 'red')
xs = np.asarray(xs[4:17])
plt.scatter(xs, f(xs))
#plt.xlim([0, 1])

plt.show()''')


xs = np.asarray(xs[4:17])
fig, ax = plt.subplots()
#plt.xlim([0, 1])\
ax.plot(x, f(x), 'red')
ax.scatter(xs, f(xs),s=6)
st.pyplot(fig)

st.markdown('''
[Implement Gradient Descent in Python: towards data science](https://towardsdatascience.com/implement-gradient-descent-in-python-9b93ed7108d1)

[Implementation of Gradient Descent in Python: medium](https://medium.com/coinmonks/implementation-of-gradient-descent-in-python-a43f160ec521)

[Linear Regression using Gradient Descent](https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931)

[Video Ng](https://www.youtube.com/watch?v=TuttBDdbls8&t=122s)''')

st.markdown('''___

# Ejemplo

## Ajuste de un pico de difracción de rayos X con una Gaussiana usando el metodo de Newton-Raphson:''')

st.code('''X = []
Y = []

fopen = open('diffraction-pattern.xy')

fread = fopen.readlines()

for j in range(len(fread)):
    
    sample = fread[j].split()

    X.append(float(sample[0]))
    Y.append(float(sample[1]))

X_array = np.asarray(X)
Y_array = np.asarray(Y)''')
X = []
Y = []
fopen = open('pages/datos/diffraction-pattern.xy')
fread = fopen.readlines()
for j in range(len(fread)):
    sample = fread[j].split()
    X.append(float(sample[0]))
    Y.append(float(sample[1]))
X_array = np.asarray(X)
Y_array = np.asarray(Y)

st.code('''import matplotlib.pyplot as plt
plt.figure(figsize=(13,8))

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=16)

plt.plot(X_array, Y_array, '.', color='red', markersize=3)
plt.ylabel('intensity', fontsize=16)
plt.xlabel('2*theta', fontsize=16)

plt.title('X-ray Diffraction Pattern', fontsize=16)
plt.show()''')

fig, ax = plt.subplots()
ax.set_ylabel('intensity')
ax.set_xlabel(r'$2*\theta$')
ax.plot(X_array, Y_array, '.', color='red', markersize=3)
ax.set_title('X-ray Diffraction Pattern', fontsize=16)
st.pyplot(fig)

st.code('''import matplotlib.pyplot as plt

x_ = X_array[530:730]
y_ = Y_array[530:730]

plt.figure(figsize=(13,8))

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=16)

plt.plot(x_, y_, '.', color='red', markersize=6)
plt.ylabel('intensity', fontsize=16)
plt.xlabel('2*theta', fontsize=16)
 
plt.title('XRDP')
plt.show()''')

x_ = X_array[530:730]
y_ = Y_array[530:730]
fig, ax = plt.subplots()
ax.set_ylabel('intensity')
ax.set_xlabel(r'$2*\theta$')
ax.plot(x_, y_, '.', color='red', markersize=6)
ax.set_title('XRDP')
st.pyplot(fig)

st.write('### Ajustando a los datos una curva Gaussiana')

st.code('''import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x_ * y_) / sum(y_)
sigma = np.sqrt(sum(y_ * (x_ - mean)**2) / sum(y_))

def Gauss(x_, a, x0, sigma):
    return a * np.exp(-(x_ - x0)**2 / (2 * sigma**2))

popt,pcov = curve_fit(Gauss, x_, y_, p0=[max(y_), mean, sigma])

print(popt)
print(pcov)

plt.figure(figsize=(13,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)

plt.plot(x_, y_, 'b+:', label='data')
plt.plot(x_, Gauss(x_, *popt), 'r-', label='fit')
plt.ylabel('intensity', fontsize=16)
plt.xlabel('2*theta', fontsize=16)

plt.show()''')

mean = sum(x_ * y_) / sum(y_)
sigma = np.sqrt(sum(y_ * (x_ - mean)**2) / sum(y_))
def Gauss(x_, a, x0, sigma):
    return a * np.exp(-(x_ - x0)**2 / (2 * sigma**2))
popt,pcov = curve_fit(Gauss, x_, y_, p0=[max(y_), mean, sigma])
st.write("$",popt)
st.write("$",pcov)

fig, ax = plt.subplots()
ax.set_ylabel('intensity')
ax.set_xlabel(r'$2*\theta$')
ax.plot(x_, y_, 'b+:', label='data')
ax.plot(x_, Gauss(x_, *popt), 'r-', label='fit')

st.pyplot(fig)

st.markdown('### Ajustando a los datos una curva Lorentziana')

st.code('''import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# weighted arithmetic mean (corrected - check the section below)
mean = sum(x_ * y_) / sum(y_)
sigma = np.sqrt(sum(y_ * (x_ - mean)**2) / sum(y_))

def Lorentzian(x_, a, x0, sigma):
    
    return (a /3.14159 ) * ((sigma/2) / ((x_-x0)**2 + (sigma/2)**2))

popt,pcov = curve_fit(Lorentzian, x_, y_, p0=[max(y_), mean, sigma])

print(popt)
print(pcov)

plt.figure(figsize=(13,8))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)

plt.plot(x_, y_, 'b+:', label='data')
plt.plot(x_, Lorentzian(x_, *popt), 'r-', label='fit')
plt.ylabel('intensity', fontsize=16)
plt.xlabel('2*theta', fontsize=16)

plt.show()''')

mean = sum(x_ * y_) / sum(y_)
sigma = np.sqrt(sum(y_ * (x_ - mean)**2) / sum(y_))
def Lorentzian(x_, a, x0, sigma):
    return (a /3.14159 ) * ((sigma/2) / ((x_-x0)**2 + (sigma/2)**2))
popt,pcov = curve_fit(Lorentzian, x_, y_, p0=[max(y_), mean, sigma])

st.write(popt)
st.write(pcov)


fig, ax = plt.subplots()
ax.set_ylabel('intensity')
ax.set_xlabel(r'$2*\theta$')
ax.plot(x_, y_, 'b+:', label='data')
ax.plot(x_, Lorentzian(x_, *popt), 'r-', label='fit')

st.pyplot(fig)

st.markdown('para mas informacion [from scipy.optimize import curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)')