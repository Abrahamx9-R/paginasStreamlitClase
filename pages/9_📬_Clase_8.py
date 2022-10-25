import streamlit as st
import json
from streamlit_lottie import st_lottie
import numpy as np
import matplotlib.pyplot as plt

import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Activation, Flatten
from tensorflow.keras.models import Model
import pydot
import IPython
from tensorflow.keras.utils import plot_model


from matplotlib.pyplot import imshow, show

import pickle
import gzip

np.random.seed(1)


np.random.seed(10)

from utils import show_image_local, show_text_color

ruta = "pages/images/imagesClase8/"
st.set_page_config(
    page_title="Clase 8",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://www.extremelycoolapp.com/help',
        'Report a Bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title('Clase 8 ')

def load_lottieurl(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

lottie_hello = load_lottieurl("pages/images/cat_playing_animation.json")

st_lottie(lottie_hello,speed = 1, reverse=False,loop=True,quality="low",height=800,width=None,key=None,)

st.markdown('<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/61b15243fa124009b0d3cc8521f3fc50" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>',unsafe_allow_html=True)

show_image_local(ruta + 'Picture1.png')

show_image_local(ruta + 'From-Bokhimi.png')

show_text_color('Topic that the Machine will learn: Handwritten Digit Recognition',size=10)

show_text_color('From Learning Machines to Smart Machines')

st.markdown('Literatura recomendada archivo What-is-smart-machines.pdf')

show_text_color('Classification Predictive Modeling')

st.markdown(r"Classification predictive modeling is the task of approximating a mapping function F from input variables (X) to <font color='red' > discrete <font color='black' > target variables (Y). In statistics a variable that can take on one of a limited number of possible values is called a **categorical**  **variable**",unsafe_allow_html=True)

show_text_color('Regression Predictive Modeling')

st.markdown("Regression predictive modeling is the task of approximating a mapping function F from input variables (X) to a <font color='red' > continuous <font color='black' > target variable (Y).",unsafe_allow_html=True)

show_text_color('Classification')

st.markdown('''A classification problem requires that samples be classified into one of two or more classes.

A classification can have real-valued or discrete input variables.

A problem with two classes is often called a two-class or binary classification.

A problem with more than two classes is often called a multi-class classification.

A problem where a sample is assigned multiple classes is called a multi-label classification.''')

show_text_color('Information about the topic: Handwritten Digit Recognition using MNIST database')


show_image_local(ruta + 'image-segmentation.png')

st.markdown ('''Documentation: [Image Segmentation](https://labelbox.com/image-segmentation-overview)
[Maya Gliphs Segmentation](./Literatura/Maya_glyph_segmentation_2017.pdf)
    
[Medical Image Segmentation](./Literatura/Medical-image-segmentation-review_2021.pdf)
    
[Segmentation Review](./Literatura/Segmentation-review_2018.pdf)
    
[Segmentation using Deep Learning](./Literatura/Image.Segmentation-Using-Deep-Learning-a-survey_2020.pdf)''')

st.write('Hand written Zip code recognition en el archivo zip Back-propag-hand-written-cnn-lecun-1989.pdf')

st.write('The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits (samples) that is commonly used for training various image processing systems.')

st.code('''import numpy as np
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Activation, Flatten
from tensorflow.keras.models import Model
import pydot
import IPython
from IPython.display import SVG
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import pickle
import gzip

np.random.seed(1)
%matplotlib inline''')



st.code('''print("Numpy version", np.__version__)
print("TensorFlow version", tf.__version__)
print("Keras version", keras.__version__)
print("Pydot version", pydot.__version__)
print("Ipython version", IPython.__version__)
print("Matplotlib version", matplotlib.__version__)
print("Pickle version", pickle.format_version)
from platform import python_version
print("Python version", python_version())''')

st.write("$Numpy version", np.__version__)
st.write("$TensorFlow version", tf.__version__)
st.write("$Keras version", keras.__version__)
st.write("$Pydot version", pydot.__version__)
st.write("$Ipython version", IPython.__version__)
st.write("$Matplotlib version", matplotlib.__version__)
st.write("$Pickle version", pickle.format_version)
from platform import python_version
st.write("$Python version", python_version())
st.write('----')
show_text_color('Samples preparation')
st.markdown('''The database mnist of samples can be downloaded from the following URL: 
    
[MNIST data download](http://yann.lecun.com/exdb/mnist/)''')

st.markdown('''The samples to train and test the neuronal network are in the file 'mnist.pkl.gz'.

    gzip.open(filename, mode='rb') open the compressed binary file 'filename'.
    
   The documentation of gzip.open can be found at [gzip.open(...)](https://docs.python.org/3/library/gzip.html#gzip.open)

    pickle.load(file, encoding = 'latin1') decode the file 'file' in latin1

Documentation: [pickle.load(...)](https://docs.python.org/3/library/pickle.html#pickle.load)

The function 'load_samples()' has three samples sets as output: 

    learn_samples  # Samples for training
    val_samples    # Samples for validation
    test_samples   # Samples for testing''')

st.code('''# The database is in the working directory: mnist.pkl.gz file.
    
def load_samples():

    f = gzip.open('mnist.pkl.gz', 'rb')
    
    learn_samples, val_samples, test_samples = pickle.load(f, encoding="latin1")
    
    f.close()
    
    return (learn_samples, val_samples, test_samples)''')
    
def load_samples():
    f = gzip.open('pages/datos/mnist.pkl.gz', 'rb')
    learn_samples, val_samples, test_samples = pickle.load(f, encoding="latin1")
    f.close()
    return (learn_samples, val_samples, test_samples)

st.code('learn_samples, val_samples, test_samples = load_samples()')

learn_samples, val_samples, test_samples = load_samples()
 
st.write('Each of these sets is a tuple with two entries:')

st.code('''print("The type of train_samples: ", type(learn_samples), "with length: ", len(learn_samples) )
print("The type of val_data: ", type(val_samples), "with length: ", len(val_samples) )
print("The type of test_data: ", type(test_samples), "with length: ", len(test_samples) )''')

st.write("$The type of train_samples: ", type(learn_samples), "with length: ", len(learn_samples) )
st.write("$The type of val_data: ", type(val_samples), "with length: ", len(val_samples) )
st.write("$The type of test_data: ", type(test_samples), "with length: ", len(test_samples) )

st.write('----')
show_text_color('Analyzing the samples extracted from MNIST')

st.markdown('The first entry of a sample corresponds to the network input, the values of the pixels, which are the image features. The second entry corresponds to the target variable. It is the digit value associated to the image. It is to note that pixel values were rescaled to values between 0.0 and 1.0')

st.code('''print("features 150 to 199 of the first training sample \n \n", learn_samples[0][0][150:200])
print("\n y value of the first training sample =",learn_samples[1][0])''')

st.write("features 150 to 199 of the first training sample \n \n", learn_samples[0][0][150:200])
st.write("\n y value of the first training sample =",learn_samples[1][0])

show_text_color('Viewing one sample from the data sets')

st.markdown('''The digits in the MNIST dataset are images of 28x28 pixels. 
    
In the recovered datasets, images were represented by vectors of dimension 28x28=784. 
    
To deploy the digit image of a sample (index), its vector representation is changed to a matrix with dimensions 28x28.
    
    
This is done by using the following function:
    
plt.imshow(sets[0][index].reshape((28, 28)),cmap='gray')      #Images are in shades of gray

Documentation: [matplotlib.pyplot.imshow(...)](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow)''')

st.code('''index = 0

plt.imshow(learn_samples[0][index].reshape((28, 28)),cmap='gray')

print(learn_samples[1][index], "is the digit corresponding to the sample", index)
print("\n This is its image")''')

index = 0

fig = plt.figure(figsize=(5,5)) 

ax = plt.imshow(learn_samples[0][index].reshape((28, 28)),cmap='gray')

st.write(learn_samples[1][index], "is the digit corresponding to the sample", index)
st.write("\n This is its image")
st.pyplot(fig)

show_text_color('Separation of the samples into features (inputs) and targets:')

st.code('''x_learn = learn_samples[0]   # input (features) in the training data set
y_learn = learn_samples[1]   # target (the digit) in the training data set

x_val = val_samples[0]   # input (features) in the validation data set
y_val = val_samples[1]   # target (the digit) in the validation data set

x_test = test_samples[0]     # input (features) in the testing data set
y_test = test_samples[1]     # target (the digit) in the testing data set
''')

x_learn = learn_samples[0]  
y_learn = learn_samples[1]  

x_val = val_samples[0]  
y_val = val_samples[1]   

x_test = test_samples[0]   
y_test = test_samples[1]    

st.code('''print(type(x_learn))
print(x_learn.shape)
print(y_learn.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)''')
st.write(type(x_learn))
st.write(x_learn.shape)
st.write(y_learn.shape)
st.write(x_val.shape)
st.write(y_val.shape)
st.write(x_test.shape)
st.write(y_test.shape)

st.code('y_learn')
st.write(y_learn)

show_text_color('One-hot encoding of the target variable Y')

st.markdown('''The target value can have one of ten elements (classes), the digits (0, 1, 2, 3, 4, 5, 6, 7, 8, 9). 

The train_y and test_y sets are arrangements in which each entry contains a digit. Each digit is represented as a integer of 64 bits.
    
We change this representation to a vectorial one following One-hot encoding 
[One-hot encoding](https://en.wikipedia.org/wiki/One-hot).
    
In the One-Hot encoding, a digit is represented with a vector having dimension 10 (because we have 10 classes) with 1.0 in the vector index corresponding to the digit and 0.0 elsewhere in the vector. ''')

st.markdown('''Digit |     One-hot representation 
--- | --- 
 0  | [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 1  | [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 2  | [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 3  | [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 4  | [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 5  | [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 6  | [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 7  | [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 8  | [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 9  | [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]''')

show_text_color('Demo using numpy.eye')

st.code('np.eye(10)')
st.write(np.eye(10))

st.code('np.eye(10)[0]')
st.write(np.eye(10)[0])

st.code('np.eye(10)[1]')
st.write(np.eye(10)[1])

st.code('y_learn[0:5]')
st.write(y_learn[0:5])

st.write('np.eye(10)[train_y[0:5].reshape(-1)]')
st.code('np.eye(10)[y_learn[0:5]]')
st.write(np.eye(10)[y_learn[0:5]])

show_text_color('End of demo using numpy.eye')

st.code('''learn_y = np.eye(10)[y_learn]

val_y = np.eye(10)[y_val]

test_y = np.eye(10)[y_test]''')

learn_y = np.eye(10)[y_learn]

val_y = np.eye(10)[y_val]

test_y = np.eye(10)[y_test]

st.code('''print("Y: Digit representation for the first learning sample \n", y_learn[0])
print("Y: One-hot representation for the first leaning sample \n",learn_y[0])''')

st.write("$Y: Digit representation for the first learning sample \n", y_learn[0])
st.write("$Y: One-hot representation for the first leaning sample \n",learn_y[0])

st.markdown('''For convenience, the dimensions of the input sets will be changed to the format:

(number of samples, image width, image length).''')

st.code('''learn_x = x_learn.reshape(50000, 28, 28)
val_x  = x_val.reshape(10000, 28, 28)
test_x = x_test.reshape(10000, 28, 28)''')

learn_x = x_learn.reshape(50000, 28, 28)
val_x  = x_val.reshape(10000, 28, 28)
test_x = x_test.reshape(10000, 28, 28)

st.markdown('In summary, the learning and test sample sets are based on the following dimensions:')

st.code('''print ("number of learning samples = " + str(learn_x.shape[0]))
print ("number of validation samples = " + str(val_x.shape[0]))
print ("number of test samples = " + str(test_x.shape[0]))
print ("learn_x shape: " + str(learn_x.shape))
print ("learn_y shape: " + str(learn_y.shape))

print ("val_x shape: " + str(val_x.shape))
print ("val_y shape: " + str(val_y.shape))

print ("test_x shape: " + str(test_x.shape))
print ("test_y shape: " + str(test_y.shape))''')

st.write("$number of learning samples = " + str(learn_x.shape[0]))
st.write("$number of validation samples = " + str(val_x.shape[0]))
st.write("$number of test samples = " + str(test_x.shape[0]))
st.write("$learn_x shape: " + str(learn_x.shape))
st.write("$learn_y shape: " + str(learn_y.shape))
st.write("$val_x shape: " + str(val_x.shape))
st.write("$val_y shape: " + str(val_y.shape))
st.write("$test_x shape: " + str(test_x.shape))
st.write("$test_y shape: " + str(test_y.shape))

show_text_color('Constructing the Learning Machine',size=10)

show_text_color('Model of the Machine: Full-Connected Feed-Forward Network (FF) with one hidden layer with twenty neurons. The nodes in the output layer will be activated with the softmax function')

st.markdown("## The architecture of the neural network will be shown")

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
        
st.code('''# Architecture of the neural network we want to implement in the present notebook

layers = [784,20,10]
net = Network(layers)
net.graph(layers)''')


layers = [784,20,10]
net = Network(layers)
net.graph(layers)

show_text_color('Definition of the neural network architecture')

st.markdown('## Keras has two different modes to define the architecture:')
st.markdown('''1. The sequential model. It is a sequential stack of layers.
    
2. The functional API. It is the way to go for defining complex models, such as multi-output models, directed acyclic graphs, or models with shared layers.  

In the present case, we will use this last mode for constructing the architecture of the network.
    

Documentation: [Keras Functional API](https://keras.io/getting-started/functional-api-guide/)''')

st.code('''def architecture(input_shape, num_clases):
    
    # Defining the input as a tensor with shape input_shape. 
    inputs = Input(input_shape, name='input-layer')
    
    # Flattening the input tensor of dimensions (28,28,1) to a tensor of dimensions (784, 1)
    x = Flatten()(inputs)
    
    # Defining the first hidden layer with 20 nodes and sigmoid as activation function
    x = Dense(20, kernel_initializer='uniform', bias_initializer='zeros', name='hidden-layer')(x)
    x = Activation('sigmoid')(x)
    
    # Defining the output layer with num_clases nodes 
    x = Dense(num_clases, kernel_initializer='uniform', bias_initializer='zeros')(x)
    
    # For the output layer we use the activation function 'softmax'
    outputs = Activation('softmax', name='output.layer')(x)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.    
    arch_model = Model(inputs = inputs, outputs = outputs, name='MnistModel')

    return arch_model''')

def architecture(input_shape, num_clases):
    inputs = Input(input_shape, name='input-layer')
    x = Flatten()(inputs)
    x = Dense(20, kernel_initializer='uniform', bias_initializer='zeros', name='hidden-layer')(x)
    x = Activation('sigmoid')(x)
    x = Dense(num_clases, kernel_initializer='uniform', bias_initializer='zeros')(x)
    outputs = Activation('softmax', name='output.layer')(x)   
    arch_model = Model(inputs = inputs, outputs = outputs, name='MnistModel')
    return arch_model

st.markdown('   *The softmax activation function is always used for classification when the number (K) of classes is larger than two.* ')

show_image_local(ruta + 'ecuacion.png')

st.markdown('lectura recomendada Activation functions archivo activation_functions_2018.pdf')

show_text_color('Constructing the neural network model for the Learning Machine',size=10)

st.code('''one_image = (28,28)
num_classes= 10

# Generating a model using the architecture defined for the neural network
mnist_model = architecture(one_image, num_classes)''')

one_image = (28,28)
num_classes= 10
mnist_model = architecture(one_image, num_classes)

show_text_color("Model's plot and summary")

st.markdown("The function 'plot_model()' generates a graphic with the layers and their number of input ands output weights.")
st.markdown('''Documentation: [Model visualization](https://keras.io/visualization/#training-history-visualization)''')

st.code("plot_model(mnist_model, to_file='FF_mnist_model.png', show_shapes=True, show_layer_names=True)")
plot_model(mnist_model, to_file='FF_mnist_model.png', show_shapes=True, show_layer_names=True)
## no funciona aun 

st.code('mnist_model.summary()')
mnist_model.summary(print_fn=lambda x: st.text(x))

show_text_color('Optimization method')

st.markdown('''This requires defining the optimization algorithm, the loss function and the metric.
    
In the present case we are using the algorithm of Stochastic Gradient descent with learning rate "lr", "momentum" without Nesterov acceleration".


literatura recomendada An overview of gradient descent optimization algorithms en el archivo SGD_overview_2016-17.pdf

This publication also comments some other optimization variants of this algorithm; Adagrad, Adadelta, RMStrop and Adam.''')

show_text_color('Optimizer')

st.code('''learning_rate = 0.01

optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)''')

learning_rate = 0.01

optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

show_text_color('The cost (loss) and Metric functions')

st.markdown(f'''The cost function *J* is the one defined as "categorical_crossentropy"''')
st.latex(' J = \frac{1}{m} \sum_{i=1}^m \sum_{k=0}^{K-1}(y_k^{(i)}*\log{(F_k(x^{(i)})))}')
    
st.markdown(r'''where $F_k(x^{(i)})$ is the predicted value and $y_k^{(i)}$ is the target value for the sample *i*; *K* is the number of classes and *m* is the number of samples.
    
[Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)
    
[Categorical cross entropy](https://www.deeplearningbook.org/)
    

A metric function is similar to a loss function, except that the results from evaluating a metric are not used when training the model. You may use any of the loss functions as a metric function. In the present example, we are using "accuracy" as metrics:
    
*Accuracy = Number of correct predictions / Total number of predictions made*
    

Categorical crossentropy will compare the distribution of the predictions (the activations in the output layer, one for each class) with the true distribution, where the probability of the true class is set to 1, and 0 for the other classes.

To put it in a different way, the true class is represented as an encoded vector, and the closer the model’s outputs are to that vector, the lower the loss.
    
Documentation: [keras.compile(...)](https://keras.io/models/model/#compile)''')

st.code("""loss_function = 'categorical_crossentropy'
metric_function = 'accuracy'""")

loss_function = 'categorical_crossentropy'
metric_function = 'accuracy'

show_text_color('Compiling the model')

st.code('mnist_model.compile(optimizer = optimizer, loss = loss_function, metrics = [metric_function])')
mnist_model.compile(optimizer = optimizer, loss = loss_function, metrics = [metric_function])

show_text_color('The Machine is learning')

st.markdown('''Documentation: [keras.fit(...)](https://keras.io/models/model/#fit)''')

st.code('''start_time = time.time()

num_epochs = 100

history = mnist_model.fit(x = learn_x, y = learn_y, epochs=num_epochs, batch_size = 100, validation_data=(val_x,val_y), shuffle=False, verbose=2)

end_time = time.time()
print("Time for learning: {:10.4f}s".format(end_time - start_time))''')

start_time = time.time()
num_epochs = 100
history = mnist_model.fit(x = learn_x, y = learn_y, epochs=num_epochs, batch_size = 100, \
                          validation_data=(val_x,val_y), shuffle=False, verbose=False)
end_time = time.time()
st.write("Time for learning: {:10.4f}s".format(end_time - start_time))


















