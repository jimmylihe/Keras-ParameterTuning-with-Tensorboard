# import libraries
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization
import gc
from multiprocessing import cpu_count
from itertools import product

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
        batch_x = np.array([np.reshape(batch,(28,28,1)) for batch in batch_x]) # un-flatten batch_x for CNN
        return batch_x, batch_y

def train_test_model(hparams):
    """this function trains a CNN based on the hparams chosen and outputs the val_accuracy"""
    model = Sequential()
    model.add(Conv2D(hparams[HP_NUM_UNITS_CLl],kernel_size=hparams[HP_KERNEL_SIZE_CLl],strides=hparams[HP_STRIDES_CLl],activation='relu',input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(hparams[HP_NUM_UNITS_CL2],kernel_size=hparams[HP_KERNEL_SIZE_CL2],strides=hparams[HP_STRIDES_CL2],activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=hparams[HP_POOL_SIZE_PL1]))
    model.add(Dropout(hparams[HP_DROPOUT_CL2]))

    model.add(Conv2D(hparams[HP_NUM_UNITS_CL3],kernel_size=hparams[HP_KERNEL_SIZE_CL3],strides=hparams[HP_STRIDES_CL3],activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(hparams[HP_NUM_UNITS_CL4],kernel_size=hparams[HP_KERNEL_SIZE_CL4],strides=hparams[HP_STRIDES_CL4],activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=hparams[HP_POOL_SIZE_PL2]))
    model.add(Dropout(hparams[HP_DROPOUT_CL4]))

    model.add(Flatten())
    model.add(Dense(hparams[HP_NUM_UNITS_DLl], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hparams[HP_DROPOUT_DL1]))
    model.add(Dense(hparams[HP_NUM_UNITS_DL2], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hparams[HP_DROPOUT_DL2]))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=hparams[HP_OPTIMIZER], loss='categorical_crossentropy', metrics=['accuracy'])
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
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
    return accuracy

def run(run_dir, hparams):
    """run model and store accuracy"""
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
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
    global HP_NUM_UNITS_CLl, HP_KERNEL_SIZE_CLl, HP_STRIDES_CLl, HP_NUM_UNITS_CL2, HP_KERNEL_SIZE_CL2, HP_STRIDES_CL2, HP_POOL_SIZE_PL1, HP_DROPOUT_CL2
    global HP_NUM_UNITS_CL3, HP_KERNEL_SIZE_CL3, HP_STRIDES_CL3, HP_NUM_UNITS_CL4, HP_KERNEL_SIZE_CL4, HP_STRIDES_CL4, HP_POOL_SIZE_PL2, HP_DROPOUT_CL4
    global HP_NUM_UNITS_DLl, HP_DROPOUT_DL1, HP_NUM_UNITS_DL2, HP_DROPOUT_DL2
    global HP_OPTIMIZER, HP_BATCH_SIZE, METRIC_ACCURACY
    HP_NUM_UNITS_CLl = hp.HParam('cl1_num_units', hp.Discrete([32]))
    HP_KERNEL_SIZE_CLl = hp.HParam('cl1_kernel_size', hp.Discrete([3]))
    HP_STRIDES_CLl = hp.HParam('cl1_strides', hp.Discrete([1]))
    HP_NUM_UNITS_CL2 = hp.HParam('cl2_num_units', hp.Discrete([32]))
    HP_KERNEL_SIZE_CL2 = hp.HParam('cl2_kernel_size', hp.Discrete([3]))
    HP_STRIDES_CL2 = hp.HParam('cl2_strides', hp.Discrete([1]))
    HP_POOL_SIZE_PL1 = hp.HParam('pl1_pool_size', hp.Discrete([2]))
    HP_DROPOUT_CL2 = hp.HParam('cl2_dropout', hp.Discrete([0.25]))

    HP_NUM_UNITS_CL3 = hp.HParam('cl3_num_units', hp.Discrete([64]))
    HP_KERNEL_SIZE_CL3 = hp.HParam('cl3_kernel_size', hp.Discrete([3]))
    HP_STRIDES_CL3 = hp.HParam('cl3_strides', hp.Discrete([1]))
    HP_NUM_UNITS_CL4 = hp.HParam('cl4_num_units', hp.Discrete([64]))
    HP_KERNEL_SIZE_CL4 = hp.HParam('cl4_kernel_size', hp.Discrete([3]))
    HP_STRIDES_CL4 = hp.HParam('cl4_strides', hp.Discrete([1]))
    HP_POOL_SIZE_PL2 = hp.HParam('pl2_pool_size', hp.Discrete([2]))
    HP_DROPOUT_CL4 = hp.HParam('cl4_dropout', hp.Discrete([0.25]))

    HP_NUM_UNITS_DLl = hp.HParam('dl1_num_units', hp.Discrete([512]))
    HP_DROPOUT_DL1 = hp.HParam('dl1_dropout', hp.Discrete([0.25]))
    HP_NUM_UNITS_DL2 = hp.HParam('dl2_num_units', hp.Discrete([1024]))
    HP_DROPOUT_DL2 = hp.HParam('dl2_dropout', hp.Discrete([0.5]))

    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam','rmsprop']))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 64, 128]))
    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS_CLl, HP_KERNEL_SIZE_CLl, HP_STRIDES_CLl, HP_NUM_UNITS_CL2, HP_KERNEL_SIZE_CL2, HP_STRIDES_CL2, HP_POOL_SIZE_PL1, HP_DROPOUT_CL2,
                     HP_NUM_UNITS_CL3, HP_KERNEL_SIZE_CL3, HP_STRIDES_CL3, HP_NUM_UNITS_CL4, HP_KERNEL_SIZE_CL4, HP_STRIDES_CL4, HP_POOL_SIZE_PL2, HP_DROPOUT_CL4,
                     HP_NUM_UNITS_DLl, HP_DROPOUT_DL1, HP_NUM_UNITS_DL2, HP_DROPOUT_DL2,
                     HP_OPTIMIZER, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )

    session_num = 1
    for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v in product(HP_NUM_UNITS_CLl.domain.values, HP_KERNEL_SIZE_CLl.domain.values, HP_STRIDES_CLl.domain.values, HP_NUM_UNITS_CL2.domain.values, HP_KERNEL_SIZE_CL2.domain.values, HP_STRIDES_CL2.domain.values, HP_POOL_SIZE_PL1.domain.values, HP_DROPOUT_CL2.domain.values,
                                                           HP_NUM_UNITS_CL3.domain.values, HP_KERNEL_SIZE_CL3.domain.values, HP_STRIDES_CL3.domain.values, HP_NUM_UNITS_CL4.domain.values, HP_KERNEL_SIZE_CL4.domain.values, HP_STRIDES_CL4.domain.values, HP_POOL_SIZE_PL2.domain.values, HP_DROPOUT_CL4.domain.values,
                                                           HP_NUM_UNITS_DLl.domain.values, HP_DROPOUT_DL1.domain.values, HP_NUM_UNITS_DL2.domain.values, HP_DROPOUT_DL2.domain.values,
                                                           HP_OPTIMIZER.domain.values, HP_BATCH_SIZE.domain.values):
        hparams = {HP_NUM_UNITS_CLl: a, HP_KERNEL_SIZE_CLl: b, HP_STRIDES_CLl: c, HP_NUM_UNITS_CL2: d,
                   HP_KERNEL_SIZE_CL2: e, HP_STRIDES_CL2: f, HP_POOL_SIZE_PL1: g, HP_DROPOUT_CL2: h,
                   HP_NUM_UNITS_CL3: i, HP_KERNEL_SIZE_CL3: j, HP_STRIDES_CL3: k, HP_NUM_UNITS_CL4: l,
                   HP_KERNEL_SIZE_CL4: m, HP_STRIDES_CL4: n, HP_POOL_SIZE_PL2: o, HP_DROPOUT_CL4: p,
                   HP_NUM_UNITS_DLl: q, HP_DROPOUT_DL1: r, HP_NUM_UNITS_DL2: s, HP_DROPOUT_DL2: t,
                   HP_OPTIMIZER: u, HP_BATCH_SIZE: v}
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning/' + run_name, hparams)
        session_num += 1
        gc.collect()

if __name__ == '__main__':
    main()

# tensorboard --logdir=logs/hparam_tuning