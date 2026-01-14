import numpy as np 
import tensorflow as tf 
import csv
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import matplotlib.pyplot as plt


def Translate_Tensorflow_MLP(file_out:str, input_names:list[str], output_names:list[str], model:tf.keras.models.Sequential, \
                             scaler_input:str="minmax", input_norm_1:list[float]=[], input_norm_2:list[float]=[], \
                             scaler_output:str="minmax", output_norm_1:list[float]=[], output_norm_2:list[float]=[]):
    # This function writes the MLP to a format which can be read by the SU2 MLP import tool
    # Inputs:
    # - file_out: output file name without extension
    # - input_names: list of strings with the variable names of the MLP input(s)
    # - output names: list of strings with the variable names of the MLP output(s)
    # - model: tensorflow.keras.model; the trained model
    # - input_norm_1: lower normalization values for the input
    # - input_norm_2: upper normalization values for the input
    # - output_norm_1: lower normalization values for the output
    # - output_norm_2: upper normalization values for the output

    # MLP config
    model_config = model.get_config()

    # Number of input variables in the model
    n_inputs = len(input_names)
    # Number of output variables in the model
    n_outputs = len(output_names)

    # Checking if number of provided input and output names are equal to those in the model
    if not n_inputs == len(input_names):
        raise Exception("Number of provided input names unequal to the number of inputs in the model")
    if not n_outputs == len(output_names):
        raise Exception("Number of provided output names unequal to the number of outputs in the model")

    if len(input_norm_2) != len(input_norm_1):
        raise Exception("Upper and lower input normalizations should have the same length")
    if len(output_norm_2) != len(output_norm_1):
        raise Exception("Upper and lower output normalizations should have the same length")

    if len(input_norm_2) > 0 and len(input_norm_1) != n_inputs:
        raise Exception("Input normalization not provided for all inputs")

    if len(output_norm_2) > 0 and len(output_norm_1) != n_outputs:
        raise Exception("Output normalization not provided for all outputs")


    # Creating output file
    fid = open(file_out+'.mlp', 'w+')
    fid.write("<header>\n\n")
    n_layers = len(model_config['layers'])

    # Writing number of neurons per layer
    fid.write('[number of layers]\n%i\n\n' % n_layers)
    fid.write('[neurons per layer]\n')
    activation_functions = []

    for iLayer in range(n_layers-1):
        layer_class = model_config['layers'][iLayer]['class_name']
        if layer_class == 'InputLayer':
            # In case of the input layer, the input shape is written instead of the number of units
            activation_functions.append('linear')
            print(model_config['layers'][iLayer]['config'])
            n_neurons = model_config['layers'][iLayer]['config']['batch_shape'][1]
        else:
            activation_functions.append(model_config['layers'][iLayer]['config']['activation'])
            n_neurons = model_config['layers'][iLayer]['config']['units']

        fid.write('%i\n' % n_neurons)
    fid.write('%i\n' % n_outputs)

    activation_functions.append('linear')

    # Writing the activation function for each layer
    fid.write('\n[activation function]\n')
    for iLayer in range(n_layers):
        fid.write(activation_functions[iLayer] + '\n')

    # Writing the input and output names
    fid.write('\n[input names]\n')
    for input in input_names:
        fid.write(input + '\n')
    
    fid.write("\n[input regularization method]\n%s\n" % scaler_input)

    if len(input_norm_1) > 0:
        fid.write('\n[input normalization]\n')
        for i in range(len(input_names)):
            fid.write('%+.16e\t%+.16e\n' % (input_norm_1[i], input_norm_2[i]))
    
    fid.write('\n[output names]\n')
    for output in output_names:
        fid.write(output+'\n')
    
    fid.write("\n[output regularization method]\n%s\n" % scaler_output)

    if len(output_norm_1) > 0:
        fid.write('\n[output normalization]\n')
        for i in range(len(output_names)):
            fid.write('%+.16e\t%+.16e\n' % (output_norm_1[i], output_norm_2[i]))

    fid.write("\n</header>\n")
    # Writing the weights of each layer
    fid.write('\n[weights per layer]\n')
    for layer in model.layers:
        fid.write('<layer>\n')
        weights = layer.get_weights()[0]
        for row in weights:
            fid.write("\t".join(f'{w:+.16e}' for w in row) + "\n")
        fid.write('</layer>\n')
    
    # Writing the biases of each layer
    fid.write('\n[biases per layer]\n')
    
    # Input layer biases are set to zero
    fid.write('%+.16e\t%+.16e\t%+.16e\n' % (0.0, 0.0, 0.0))

    for layer in model.layers:
        biases = layer.get_weights()[1]
        fid.write("\t".join([f'{b:+.16e}' for b in biases]) + "\n")

    fid.close()


# Set up test problem: two inputs (u, v), one output (y)
t = np.linspace(0, 1, 5000)
u = (1 - t) * np.cos(48*np.pi*t)
v = (1 - t) * np.sin(48*np.pi*t)

y = np.sin(2*np.pi*(u*u + v))

X_dim = np.hstack((u[:,np.newaxis], v[:,np.newaxis]))
Y_dim = y 

with open("reference_data.csv","w+") as fid:
    fid.write("u\tv\ty\n")
    csvWriter = csv.writer(fid, delimiter="\t")
    csvWriter.writerows(np.hstack((X_dim, Y_dim[:,np.newaxis])))

# Define input and output regularization methods
scaler_function_x = "standard"
scaler_function_y = "robust"

# Hidden layer architecture
hidden_layers = [10,10,10]


# Scale input and output data
if scaler_function_x == "standard":
    scaler_x = StandardScaler()
elif scaler_function_x == "robust":
    scaler_x = RobustScaler()
else:
    scaler_x = MinMaxScaler()

if scaler_function_y == "standard":
    scaler_y = StandardScaler()
elif scaler_function_y == "robust":
    scaler_y = RobustScaler()
else:
    scaler_y = MinMaxScaler()

scaler_x.fit(X_dim)
scaler_y.fit(Y_dim[:,np.newaxis])

X_norm = scaler_x.transform(X_dim)
Y_norm = scaler_y.transform(Y_dim[:,np.newaxis])

# Collect input and output normalization values
if scaler_function_x == "standard":
    input_norm_1 = scaler_x.mean_
    input_norm_2 = scaler_x.scale_ 
elif scaler_function_x == "robust":
    input_norm_1 = scaler_x.center_
    input_norm_2 = scaler_x.scale_
else:
    input_norm_1 = scaler_x.data_min_ 
    input_norm_2 = scaler_x.data_max_

if scaler_function_y == "standard":
    output_norm_1 = scaler_y.mean_
    output_norm_2 = scaler_y.scale_ 
elif scaler_function_y == "robust":
    output_norm_1 = scaler_y.center_
    output_norm_2 = scaler_y.scale_
else:
    output_norm_1 = scaler_y.data_min_ 
    output_norm_2 = scaler_y.data_max_

# Define MLP to be trained
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input([2]))
for NN in hidden_layers:
    model.add(tf.keras.layers.Dense(NN,activation='sigmoid',kernel_initializer="he_uniform"))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Set up training method
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(10**-2, decay_steps=1000,decay_rate=0.99,staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule,beta_1=0.9,beta_2=0.999,epsilon=1e-08,amsgrad=False)

# Commence training
model.compile(optimizer=opt,loss="mean_squared_error",metrics=["mape"])
history = model.fit(X_norm, Y_norm, epochs=1000,shuffle=True)

# Translate TensorFlow model to MLPCpp ASCII format
Translate_Tensorflow_MLP(file_out="MLP_test", input_names=["u","v"], output_names=["y"], model=model,\
                         scaler_input=scaler_function_x, input_norm_1=input_norm_1,input_norm_2=input_norm_2,\
                         scaler_output=scaler_function_y, output_norm_1=output_norm_1, output_norm_2=output_norm_2)

# Visualize labeled data and model predictions
Y_norm_pred = model.predict(X_norm)
Y_dim_pred = scaler_y.inverse_transform(Y_norm_pred)

# Run MLPCpp to evaluate MLP
os.system("./test_MLPCpp")

# Read predicted data from MLPCpp
Y_dim_pred_MLPCpp = np.loadtxt("predicted_data.csv",delimiter='\t',skiprows=1)[:, 2]

# Visualize reference data, predicted data from TensorFlow, and predicted data from MLPCpp model.
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(X_dim[:,0],X_dim[:,1], Y_dim, 'k.',label="Training data")
ax.plot3D(X_dim[:,0],X_dim[:,1], Y_dim_pred[:,0],'r.', label="Model prediction")
ax.plot3D(X_dim[:,0],X_dim[:,1], Y_dim_pred_MLPCpp,'g.', label="Model prediction (MLPCpp)")
ax.set_xlabel("u")
ax.set_ylabel("v")
ax.set_zlabel("y")

ax.legend()
plt.show()