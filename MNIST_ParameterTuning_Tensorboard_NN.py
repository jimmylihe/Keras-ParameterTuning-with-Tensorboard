# import libraries
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import gc
from multiprocessing import cpu_count

class generator(keras.utils.Sequence):
    def __init__(self, images, labels, batch_size):
        self.labels = labels
        self.images = images
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.images.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size:(idx+1)* self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx+1)* self.batch_size]
        return batch_x, batch_y

def train_test_model(hparams,run_name):
    """this function trains a NN based on the hparams chosen and outputs the accuracy"""
    model = Sequential()
    model.add(Dense(hparams[HP_NUM_UNITS_Ll], activation='relu', input_dim=784))
    model.add(Dropout(hparams[HP_DROPOUT_L1]))
    model.add(Dense(hparams[HP_NUM_UNITS_L2], activation='relu'))
    model.add(Dropout(hparams[HP_DROPOUT_L2]))
    model.add(Dense(10, activation='softmax'))
    adam = keras.optimizers.Adam(learning_rate=hparams[HP_OPTIMIZER_LEARNING_RATE], beta_1=0.9, beta_2=0.999, amsgrad=hparams[HP_OPTIMIZER_AMSGRAD])
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    #tb_callback = keras.callbacks.TensorBoard(log_dir="logs\\hparam_tuning\\" + str(run_name), write_graph=True, write_images=True)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    train_batch_generator = generator(X_train,y_train,hparams[HP_BATCH_SIZE])
    validation_batch_generator = generator(X_val,y_val,hparams[HP_BATCH_SIZE])
    model.fit_generator(generator=train_batch_generator,
                        steps_per_epoch=int(np.ceil(X_train.shape[0]/hparams[HP_BATCH_SIZE])),
                        epochs=100,
                        verbose=2,
                        use_multiprocessing=False,
                        workers=cpu_count(),
                        max_queue_size=32,
                        validation_data=validation_batch_generator,
                        validation_steps=int(np.ceil(X_val.shape[0]/hparams[HP_BATCH_SIZE])),
                        callbacks=[callback])
    _, accuracy = model.evaluate_generator(generator=validation_batch_generator,
                                           steps=int(np.ceil(X_val.shape[0]/hparams[HP_BATCH_SIZE])),
                                           max_queue_size=32,
                                           workers=cpu_count(),
                                           use_multiprocessing=False,
                                           verbose=0)
    #model.fit(x=X_train,y=y_train,epochs=1, batch_size=hparams[HP_BATCH_SIZE], verbose=2 ,use_multiprocessing=True)
    #_, accuracy = model.evaluate(X_val,y_val,verbose=1,use_multiprocessing=True)
    return accuracy

def run(run_dir, hparams,run_name):
    """run model and store accuracy"""
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams,run_name)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

def main():
    # Import MNIST dataset from openml
    scikit_learn_data_path = './scikit_learn_data'
    dataset = fetch_openml('mnist_784', version=1, data_home=scikit_learn_data_path)

    # Data preparation
    data_X = dataset['data']
    data_X = data_X.astype('float32')
    data_X /= 255
    data_Y = dataset['target']

    # Apply one-hot encoding to data_Y
    data_Y = keras.utils.to_categorical(data_Y)

    # Split your data into a train set (50%), validation set (20%) and a test set (30%)
    global X_train, X_test, y_train, y_test, X_val, y_val
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.5, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

    # reduce memory loads
    del dataset, data_X, data_Y

    # initialize parameters to tune
    global HP_NUM_UNITS_Ll,HP_DROPOUT_L1,HP_NUM_UNITS_L2,HP_DROPOUT_L2,HP_OPTIMIZER_LEARNING_RATE,HP_OPTIMIZER_AMSGRAD,HP_BATCH_SIZE, METRIC_ACCURACY
    HP_NUM_UNITS_Ll = hp.HParam('l1_num_units', hp.Discrete([64, 128]))
    HP_DROPOUT_L1 = hp.HParam('l1_dropout', hp.Discrete([0.1, 0.2]))
    HP_NUM_UNITS_L2 = hp.HParam('l2_num_units', hp.Discrete([128, 64]))
    HP_DROPOUT_L2 = hp.HParam('l2_dropout', hp.Discrete([0.1, 0.2]))
    HP_OPTIMIZER_LEARNING_RATE = hp.HParam('optimizer_learning_rate', hp.Discrete([0.001, 0.0001]))
    HP_OPTIMIZER_AMSGRAD = hp.HParam('optimizer_amsgrad', hp.Discrete([False, True]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 128]))
    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[
                HP_NUM_UNITS_Ll,
                HP_DROPOUT_L1,
                HP_NUM_UNITS_L2,
                HP_DROPOUT_L2,
                HP_OPTIMIZER_LEARNING_RATE,
                HP_OPTIMIZER_AMSGRAD,
                HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )

    session_num = 0
    for num_l1 in HP_NUM_UNITS_Ll.domain.values:
        for drop_l1 in HP_DROPOUT_L1.domain.values:
            for num_l2 in HP_NUM_UNITS_L2.domain.values:
                for drop_l2 in HP_DROPOUT_L2.domain.values:
                    for opti_lr in HP_OPTIMIZER_LEARNING_RATE.domain.values:
                        for opti_ams in HP_OPTIMIZER_AMSGRAD.domain.values:
                            for batchsize in HP_BATCH_SIZE.domain.values:
                                gc.collect()
                                hparams = {
                                    HP_NUM_UNITS_Ll:num_l1,
                                    HP_DROPOUT_L1:drop_l1,
                                    HP_NUM_UNITS_L2:num_l2,
                                    HP_DROPOUT_L2:drop_l2,
                                    HP_OPTIMIZER_LEARNING_RATE:opti_lr,
                                    HP_OPTIMIZER_AMSGRAD:opti_ams,
                                    HP_BATCH_SIZE:batchsize}
                                run_name = "run-%d" % session_num
                                print('--- Starting trial: %s' % run_name)
                                print({h.name: hparams[h] for h in hparams})
                                run('logs/hparam_tuning/' + run_name, hparams,run_name)
                                session_num += 1

if __name__ == '__main__':
    main()

# tensorboard --logdir=logs/hparam_tuning
