import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import pickle

import os

import tensorflow as tf

from . import Configurable


class BaseModelCustomTraining(tf.keras.Model, Configurable):
    def __init__(self, config_specs):
        tf.keras.Model.__init__(self)
        Configurable.__init__(self, config_specs)

        self.net = None
        self.init_model()

    def init_model(self):
        """ Called on __init__. Keras self initialization. Create self here if does not require the inputs shape """
        pass

    def build(self, inputs_shape):
        """ Keras method. Called once to build the self. Build the self here if the input shape is required. """
        pass

    def call(self, inputs, training=False):
        pass

    def prepare(self, optimizer=None, loss=None, metrics=None,
                epoch_loss_metric = None,
                epoch_val_metric = None,
                **kwargs):
        """ Prepares the self for training and evaluation. This method should create the
        optimizer, loss and metric functions and call the compile method of the self. The self
        should provide the defaults for the optimizer, loss and metrics, which can be overriden
        with custom arguments. """

        raise NotImplementedError

    @tf.function
    def train_step(self,
                   train_ds):
        # pb_i = Progbar(len(list(train_ds)), stateful_metrics='acc')

        for x_batch_train, y_batch_train in train_ds:  # tqdm
            with tf.GradientTape() as tape:
                y_preds = self.call(x_batch_train,
                                    training=True)
                cost = self.loss(y_batch_train, y_preds)

            grads = tape.gradient(cost, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.loss_metric.update_state(cost)
            self.metric.update_state(y_batch_train+1, y_preds+1)

    # Function to run the validation step.
    @tf.function
    def val_step(self, val_ds):
        for x, y in val_ds:
            y_preds = self.call(x, training=False)
            cost = self.loss(y, y_preds)
            self.loss_metric.update_state(cost)
            self.metric.update_state(y +1, y_preds+1)

    def fit(self,
            dataset,
            val_dataset,
            batch_size,
            num_epochs,
            model_directory,
            iterations_per_epoch=10,
            function=np.min):

        train_loss, val_loss, val_acc = ([np.inf] for i in range(3))

        x_train, y_train = dataset
        n, t, d = x_train.shape

        x_val, y_val = val_dataset
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

        _ = self(tf.zeros([n, t, d]))


        for epoch in range(num_epochs + 1):

            x_train, y_train = shuffle(x_train, y_train)
            train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

            self.train_step(train_ds)

            # End epoch
            loss_epoch = self.loss_metric.result().numpy()
            train_loss.append(loss_epoch)
            self.loss_metric.reset_states()

            if epoch % iterations_per_epoch == 0:
                self.val_step(val_ds)
                val_loss_epoch = self.loss_metric.result().numpy()
                val_acc_result = self.metric.result().numpy()
                print(
                    "Epoch {0}: Train loss {1}, Val loss {2}, Val acc {3}".format(
                        str(epoch),
                        str(loss_epoch),
                        str(round(val_loss_epoch, 4)),
                        str(round(val_acc_result, 4)),
                    ))

                if val_acc_result < function(val_acc):
                    self.save_weights(os.path.join(model_directory, 'model'))

                val_loss.append(val_loss_epoch)
                val_acc.append(val_acc_result)
                self.loss_metric.reset_states()
                self.metric.reset_states()

        # History of the training
        losses = dict(train_loss_results=train_loss,
                      val_loss_results=val_acc
                      )
        with open(os.path.join(model_directory, 'history.pickle'), 'wb') as d:
            pickle.dump(losses, d, protocol=pickle.HIGHEST_PROTOCOL)


    def train_and_evaluate(self,
                           train_dataset,
                           val_dataset,
                           num_epochs,
                           iterations_per_epoch,
                           model_directory,
                           **kwargs):

        return self.fit(train_dataset,
                        val_dataset,
                        num_epochs=num_epochs,
                        model_directory=model_directory,
                        iterations_per_epoch=iterations_per_epoch,
                        **kwargs)