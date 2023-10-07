# VAE_with_Audio_pipeline
In this repository we combine all the blocks we made previously and combining them all to encode and decode audio Data
class Autoencoder

It respresents a deep autoencoder and its mirror components to construct encoder and decoder





COMPONENTS OF THE CLASS
----------------------------------------------------
----------------------------------------------------
----------------------------------------------------

Main componenets of the encoder class where we have
----------------------------------------------------
----------------------------------------------------
Input_shape ->(List)
Convolutions filters ->(List)
Convolutions kernels ->(List)
Convolutions strides ->(List)
Latent space ->(List)


Examples of the Inputs 
----------------------------------------------------
----------------------------------------------------
[28x28 pixels , 28x28 pixels , 1 channel(its because it looks like greyscale image) ]
[Number of filters : 2,4,8]
[These are the filters we are applying : 3x3,5x5,3x3]
[The data is being downsampled if the value increase in ascending : 1,2,2]
[Our bottle neck has 2 dimensions]



Main Objects that we want to create 
----------------------------------------------------
----------------------------------------------------

encoder = None
decoder = None
model = None





#Private attributes
------------------------------------------------------------
------------------------------------------------------------
number of convolution layers = length of convolution filters
shape before bottleneck = None -> (We will use this later so that we have an idea of the shape of the spectogram before it was compressed)
_build() -> (This method will be used to build the encoder)


FUNCTIONS TO BUILD THE COMPONENTS
----------------------------------------------------
----------------------------------------------------
----------------------------------------------------

Function of _build()
---------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------
In the build Method we have two sub methods which will be used to build our objects such as encoder and decoder which we described above



_build_encoder() 
---------------------------
---------------------------

Model
--------------------------------------------------------------------
We add the combination of Encoder input and Bottle Neck to the Model 
Encoder Input , Bottle Neck -> Model

encoder = Model(encoder_input, bottleneck, name="encoder")





_build_decoder()
-----------------
-----------------


Adding a conv layer like previous function
As we have a single channel for spectogram 
We are using the first layer which we ignored in previous process

This is for one channel
Adding this to incoming layers
Adding the final activation layer




Model
--------------------------------------------------------------------
We add the combination of Encoder input and Bottle Neck to the Model 
Decoder Input , Decoder Output -> Model

encoder = Model(encoder_input, bottleneck, name="encoder")




Function of _build_encoder() 
------------------------------------------------------------
Parts to build encoder are

# Encoder input
# Convolution Layer
# Bottle Neck


How to build
Encoder input -> Convolution Layer -> Bottle Neck


encoder_input = _add_encoder_input()
conv_layers = _add_conv_layers(encoder_input) #Convolution Layer = Encoder Input -> Convolution Layer
bottleneck =_add_bottleneck(conv_layers) # Bottleneck = Convolution Layer -> Bottleneck





Function of _add_encoder_input() 
------------------------------------------------------------
This Method will provide us with input of the encoder
returns the Input according to input shape
and we name it as encoder_input layer






Function of _add_conv_layer() -> This makes on layer which we will later on add with the encoder input
------------------------------------------------------------------------------------------------------
It adds a convolutional block to graph of layers consisting of
Conv2D { Filters , Kernel Size , Strides, padding , name }
ReLU
Batch Normalization




Function of _add_bottleneck() -> Here we will add a bottle neck to our input 
-------------------------------------------------------------------------------------------


We get the input as 
[2,7,Width=7,Height=7,Num_channels=32]
but we want 
[Width=7,Height=7,Num_channels=32] which is the shape before bottle neck


We want to flatten data to add to bottleneck which is a dense layer
Data -> Flatten -> Pass to Bottleneck


and we get the x that will later be used in the for loop where we add the convolution layers with the encoder input


Function of _add_decoder_input(): -> Here we will add decode the input
-------------------------------------------------------------------------------------------
We are converting shape of our input according to latent space dimension

Function of _add_dense_layer(): -> Here we will get a dense layer
-------------------------------------------------------------------------------------------
This function creates dense layers
[1,2,4]-> 8 : Multiplied all numbers to produce a product
# Passing the decoder input to a dense layer

Function of _add_reshape_layer(): -> Here we will reshape the layer
-------------------------------------------------------------------------------------------
We are reshaping the dense layer according to the shape before the bottle neck


Function of _add_conv_transpose_layer(x): -> for single convolution transpose of layer
-------------------------------------------------------------------------------------------

Layer nuumber = No of convolution layer - Index of the layers 



This makes the convolution transpose layer :
--------------------------------------------
we are taking Transpose of the Convolution layers
We want to take parameters as following 
Convolution filters , Covolution kernels and Convolution Strides should be go through layer index





We took the transpose of the updated layer

Then we applied the ReLU to the layer
Then we applied batch Normalization on the layer





Function of _add_conv_transpose_layers(x): -> Adds convolution transpose of the all the layer
-------------------------------------------------------------------------------------------

Adding transpose convolution block
We are passing it the x that is like the graph of all the layers 
x -> Encoder input at each layer index
we want to loop through all the convolution layers and stop at the first layer
In the for loop:
if range was : 0   ->   self._num_conv_layers [0,1,2] -> [2,1,0]
if range was : 1   ->   self._num_conv_layers [1,2]   -> [2,1]           (We want to use this format)
We took the transpose of x


And our function gives us the decoded input 


FUNCTIONS TO BUILD THE ENCODER
----------------------------------------------------
----------------------------------------------------
----------------------------------------------------

Function of _add_conv_layers() -> Here we are adding all the Layers together with the input 
-------------------------------------------------------------------------------------------
This Method will create all the with Convolution Blocks 


Inputs are 
#Encoder input     (Encoder Input -> Conv Layers)

We will create x here which is equal to encoder input 
Now we want to go layer by layer to all the steps
The loop will run until it gets all the layers in our architecture
In the loop we are:

x = on every index of Encoder Input -> adding conv layer
# adding encoder input at each layer index using _add_conv_layer and it will run until the last number of convolution Layer
# x will be like a graph of all the layers


Now This function will give us the x which is Encoder Input + Convolution Layer




FUNCTIONS TO BUILD THE DECODER
----------------------------------------------------
----------------------------------------------------
----------------------------------------------------


We are going to implement build decoder which will form our decoder
Provide the dense layer from decoder input : (Input of decoder -> Dense Layer)
Then we are reshaping the layer
Taking Convolution Transpose of Layer to transform it back
Then we are going to take decoder input
Then we are going to create a decoder Model



VAE represents a Deep Convolutional variational autoencoder architecture with
mirrored encoder and decoder components.


We are chnaging the structure of bottle neck of vanilla encoder
which was given as


For Vanilla AUTOENCODER
------------------------
------------------------


Function of bottleneck() -> Here we will add a bottle neck to our input 
-------------------------------------------------------------------------------------------


We get the input as 
[2,7,Width=7,Height=7,Num_channels=32]
but we want 
[Width=7,Height=7,Num_channels=32] which is the shape before bottle neck


We want to flatten data to add to bottleneck which is a dense layer
Data -> Flatten -> Pass to Bottleneck


and we get the x that will later be used in the for loop where we add the convolution layers with the encoder input 




For Variational AUTOENCODER
---------------------------
---------------------------



Changes done in functions 

Function of bottleneck() -> Here we will add a bottle neck to our input 
-------------------------------------------------------------------------------------------
Flatten data and add bottleneck with Guassian sampling (Dense layer).
mu is a dense layer according to its lantent dimension
Then we are trying to create log varaince
We are trying to perform computations in lambda layer where we have our sample points with our x
Whatever we passed in lambda layers then comes in the args
Epsilon is sample point from standard distribution 
#sample point = mu + Exp(log ((standard deviation)^2) / 2) * random normal distribution of random sample point





Function of reconstruction_loss() -> Here we are just checking difference between target and predicted values
--------------------------------------------------------------------------------------------------------------
Find RMSE

In reconstruction loss we take sqaure of error and we are specifying on which axis we are taking mean -> axis 1,2,3 




Function of kl_loss() -> Here we measure difference between two probability distribution
--------------------------------------------------------------------------------------------------------------
KL loss provides Symmetry and measures difference between two probability distribution
KL loss = 1/2 summation((1+log (std deviation)^2)-(mu^2)-(std deviation^2))   -> axis 1





Function of combined_loss() 
--------------------------------------------------------------------------------------------------------------
Here we are calculating all the loses
LOSS= aplha + RMSE + KL

aplha can be treated as a hyper-parameter to optimise

#Preprocessing Pipeline

# 1-Load the file
---------------
---------------


#Constructor  
--------------------
#We are passing a constructor in the loader Class that will construct the class
#It has following arguments 
# Sample rate -> we load audio file with this
# Duration -> How long audio will play (sec)
# Mono -> it is boolean and tells us if audio is mono or stereo




# Load 
------------
#Then we added a load class that will load our signal with the parameters of the constructor and it will give us the constructed signal
#This will return us the signal with sample rate , Duration and will tell if the signal is mono oe stereo







#Pad the signal if necessay
-----------------------------
-----------------------------
Padder is the Function to padding the data in array


#Constructor Class 
--------------------
Here we have mode



#This function is for left Padding 
------------------------------------
#Here we pass the array and number of missing items
#[1,2,3] -> Left padding of 2 -> [0,0,1,2,3]
#Pad using the numpy array
#Array | number of missing items -> 0 



#This function is for right Padding
-----------------------------------
#Here we pass the array and number of missing items
#[1,2,3] -> Right padding of 2 -> [1,2,3,0,0]
#Array |  0 -> number of missing items


#Extracting log spectogram using Librosa
----------------------------------------
----------------------------------------

#LogSpectogramExtractor creates a log spectogram in dB from a time series Signal

#Constructor Class 
--------------------
#Here we have following parameters
#Frame Size 
#Hop Length

#Extractor 
--------------------

#(1+frame_size/2,num_frame)  1024 -> 513 -> 512
#We take the signal then we use framesize and we decide the hoplength we want 







#Normalize Spectogram
----------------------
----------------------
#MinMax Normalization applies min max normalization to array


#Constructor Class 
--------------------
# min_value -> We take a minimum value that is mapped close to zero
# max_value -> We take a maximum value that is mapped close to one

#Normalize Class 
--------------------
#array will be squished between 0 and 1
#1st step to Normalize array -> (array-array.min())/(array.max()-array.min())
#2nd step to Normalize array -> Normalize array + (self.max - self.min)




#Denormalize Class
--------------------
#array will be squished between 0 and 1
#1st step to Normalize array -> (norm_array-self.min)/(self.max()-self.min())
#2nd step to Normalize array -> array * (original_max - original_min) + original_min






#Save the Normalized Spectogram
--------------------------------
--------------------------------


#Saver Class
--------------------
#saver is responsible to save features, and the min max values

#Constructor Class 
--------------------
#We need feature save directory and min max value directory as the constructors that make the saver

#save_feature Class 
--------------------
#We generated save path and generated the path
#save the path of the feature


#save_min_max_values Class 
----------------------------
#We generated save path and generated the path
#save the path of the feature


#save_min_max_values Class 
----------------------------
save_min_max_values








#PreprocessingPipeline processes audio files in a directory, applying the following steps to each file:
        # 1- load a file
        # 2- pad the signal (if necessary)
        # 3- extracting log spectrogram from signal
        # 4- normalise spectrogram
        # 5- save the normalised spectrogram
-----------------------------------------------------
------------------------------------------------------


#process Class 
----------------------------
#looping through filepath which is the directory containing all the file
#We are going through all the directories and subdirectories
#Now we are reconstructing the filepath root directory + file 


#_process_file Method 
----------------------------
#it is the bulk of the class
#We will load the signal
#Now we are deciding wether we want to apply padding or not
#Then we are moving to is padding necessary function
#if the signal need padding then we will apply that to the signal
#Now we are extracting the feature
#Then we will save the normalized feature
#Then we will save the path




#_is_padding_necessary Method 
----------------------------
#How are we gonna check if the signal needs padding or not 
#Duration is fixed and we know number of expected samples
#So if the length of the signal is less than expected samples then we provide results as boolean value

#_apply_padding Method 
----------------------------
#Here we will give the signal
#Number of missing samples = Number of expected samples - length of signal


#_store_min_max_value Method 
----------------------------
#This will store the dictionary of minimum and maximum value










