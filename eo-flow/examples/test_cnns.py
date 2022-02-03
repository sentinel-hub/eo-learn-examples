import eoflow.models.tempnets_task.cnn_tempnets as cnn_tempnets
import tensorflow as tf

# Model configuration CNNLSTM
import numpy as np
import os
import tensorflow_addons as tfa
########################################################################################################################
########################################################################################################################


path = '/home/johann/Documents/Syngenta/2020/fold_5/'
x_train = np.load(os.path.join(path, 'training_x_bands.npy'))
y_train = np.load(os.path.join(path, 'training_y.npy'))
x_train = x_train.reshape(756, 27, 10)
x_val = np.load(os.path.join(path, 'val_x_bands.npy'))
x_val = x_val.reshape(190, 27, 10)
y_val = np.load(os.path.join(path, 'val_y.npy'))



# Model configuration CNN
model_cfg_cnn = {
    "learning_rate": 10e-5,
    "keep_prob" : 0.5,
    "nb_conv_filters": 16,
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 2048,
    "nb_fc_stacks": 1, #Nb FCN layers
    "kernel_size" : 1,
    "nb_conv_strides" :1,
    "kernel_initializer" : 'he_normal',
    "batch_norm": True,
    "padding": "VALID",#"VALID", CAUSAL works great?!
    "kernel_regularizer" : 1e-6,
    "final_layer" : 'Flatten',
    "loss": "huber",
    "enumerate" : True,
    "metrics": "r_square"
}


model_cnn = cnn_tempnets.TempCNNModel(model_cfg_cnn)
# Prepare the model (must be run before training)
model_cnn.prepare()

# Train the model
model_cnn.train_and_evaluate(
    train_dataset=(x_train, y_train),
    val_dataset=(x_val, y_val),
    num_epochs=500,
    iterations_per_epoch=5,
    batch_size = 8,
    model_directory='/home/johann/Documents/model'
)

model_cnn.load_weights('./')
t = model_cnn.predict(x_val)
import matplotlib.pyplot as plt
plt.scatter(t, y_val)
plt.show()
########################################################################################################################
########################################################################################################################

# Model configuration CNN
model_cfg_cnn2d = {
    "learning_rate": 10e-5,
    "keep_prob" : 0.5,
    "nb_conv_filters": 128,
    "nb_conv_stacks": 3,  # Nb Conv layers
    "nb_fc_neurons" : 2048,
    "nb_fc_stacks": 1, #Nb FCN layers
    "kernel_size" : [1,1],
    "nb_conv_strides" : [1,1],
    "kernel_initializer" : 'he_normal',
    "batch_norm": True,
    "padding": "VALID",#"VALID", CAUSAL works great?!
    "kernel_regularizer" : 1e-6,
    "final_layer" : 'Flatten',
    "loss": "huber",
    "enumerate" : True,
    "metrics": ["mse", "mae"]
}



model_cnn = cnn_tempnets.HistogramCNNModel(model_cfg_cnn2d)
# Prepare the model (must be run before training)
model_cnn.prepare()
model_cnn.build((None, 30, 32, 9))
model_cnn.summary()
output_file_name_cnnlstm, checkpoint = utils.define_callbacks(path_DL, model_cfg_cnnlstm, prefix = 'cnnlstm_')


# Train the model
model_cnn.train_and_evaluate(
    train_dataset=train_ds,
    val_dataset=val_ds,
    num_epochs=500,
    iterations_per_epoch=iterations_per_epoch,
    model_directory=os.path.join(path_DL, os.path.join(output_file_name_cnnlstm, "model")),
    save_steps=10,
    summary_steps='epoch',
    callbacks=[checkpoint]
)
