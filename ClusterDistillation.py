import numpy as np
import argparse
import pandas as pd
import warnings
import random
from numpy.random import seed
from tensorflow.random import set_seed
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.cluster import KMeans
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
warnings.filterwarnings('ignore')


tf.keras.utils.set_random_seed(10)


parser = argparse.ArgumentParser(description='A test program.')

parser.add_argument("-p", "--print_string", nargs='*')

args = parser.parse_args()

print("#################################")
print("dataset: ", args.print_string[0])
print("unlabel percentage: ", float(args.print_string[1]))



filename = args.print_string[0]

data_train = np.loadtxt("dataset/"+filename+"_TRAIN.txt")
data_test = np.loadtxt("dataset/"+filename+"_TEST.txt")



# train and test
x_train = data_train[:, 1:]
y_train = data_train[:, 0]
x_test = data_test[:, 1:]
y_test = data_test[:, 0]

# split data label and unlabel
porc_unlabel = float(args.print_string[1])

if porc_unlabel > 0:
  x_label, x_unlabel, y_label, y_unlabel = train_test_split(x_train, y_train, test_size=porc_unlabel, random_state=10, stratify=y_train)
else:
  x_label = x_train
  x_unlabel = np.array([])
  y_label = y_train
  y_unlabel = np.array([])

# one hot
enc = OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_label = enc.transform(y_label.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# unlabel with 0
y_fake_label = np.zeros(shape=(len(y_unlabel), len(np.unique(y_train))))

# concatenate label with unlabel
if x_unlabel.shape[0]>0:
  x = np.concatenate((x_label, x_unlabel))
  y = np.concatenate((y_label, y_fake_label))
else:
  x = x_label
  y = y_label

train = np.concatenate((x,y), axis=1)
print("#########################################################")
print("TRAINING")

# shuffle
np.random.shuffle(train)

# split
x = train[:, :-len(y_label[0])]
y = train[:, -len(y_label[0]):]

# reshape
x = np.reshape(x, [x.shape[0], x.shape[1], 1])
x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1], 1])
y_true = np.argmax(y_test, axis=1)


accuracy = []
recall = []
precision = []

# 10 models
for random_seed in range(10):
  tf.random.set_seed(random_seed)
  tf.config.experimental.enable_op_determinism()
  n_feature_maps = 64

  cnn_input = tf.keras.Input(shape=(x.shape[1], 1))

  # BLOCK 1

  conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(cnn_input)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)

  conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)

  conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # expand channels for the sum
  shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(cnn_input)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

  output_block_1 = keras.layers.add([shortcut_y, conv_z])
  output_block_1 = keras.layers.Activation('relu')(output_block_1)

  # BLOCK 2

  conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)

  conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)

  conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # expand channels for the sum
  shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

  output_block_2 = keras.layers.add([shortcut_y, conv_z])
  output_block_2 = keras.layers.Activation('relu')(output_block_2)

  # BLOCK 3

  conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)

  conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)

  conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # no need to expand channels because they are equal
  shortcut_y = keras.layers.BatchNormalization()(output_block_2)

  output_block_3 = keras.layers.add([shortcut_y, conv_z])
  output_block_3 = keras.layers.Activation('relu')(output_block_3)

  # FINAL

  gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)


  model_cnn = keras.models.Model(cnn_input, gap_layer)

  classifier_input = tf.keras.Input(shape=(128,))
  classifier_output = tf.keras.layers.Dense(len(np.unique(y_true)), activation="softmax")(classifier_input)
  classifier = tf.keras.Model(classifier_input, classifier_output, name="classifier")


  cnn_classif_input = tf.keras.Input(shape=(x.shape[1], 1), name="cnn")
  cnn_series = model_cnn(cnn_classif_input)
  classif_series = classifier(cnn_series)
  final_classif_original = tf.keras.Model(cnn_classif_input, classif_series, name="CnnClf")


  def my_loss(y_truee, y_pred):
      cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
      l1 = cce(y_truee, y_pred) 
      mask = tf.reduce_sum(y_truee,axis=1)
      mloss = tf.multiply(mask,l1)
      mloss = tf.reduce_sum(mloss/tf.math.maximum(tf.reduce_sum(mask),1))
      return mloss

  final_classif_original.compile(
  loss = my_loss,
  optimizer= tf.keras.optimizers.Adam(),
    metrics = ['accuracy'])

  reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

  callbacks = [reduce_lr]

  final_classif_original.fit(x,  y, verbose=2, epochs=1500, batch_size=16, callbacks=callbacks)#, validation_data=(x_test, y_test))

  y_pred = final_classif_original.predict(x_test)

  y_pred = np.argmax(y_pred, axis=1)

  acc = accuracy_score(y_true, y_pred)
  sco = precision_score(y_true, y_pred, average='macro')
  rec = recall_score(y_true, y_pred, average='macro')

  accuracy.append(acc)
  recall.append(rec)
  precision.append(sco)


  print(sco, acc, rec)


res = pd.DataFrame({"precision": precision, "accuracy": accuracy, "recall": recall})
res.to_csv("sup_Resnet_"+filename+"_"+str(porc_unlabel)+".csv")


accuracy = []
recall = []
precision = []
y_kmeans_onehot = final_classif_original.predict(x)

for random_seed in range(10):
  print("model:", random_seed)
  tf.random.set_seed(random_seed)
  tf.config.experimental.enable_op_determinism()

  n_feature_maps = 64

  cnn_input = tf.keras.Input(shape=(x.shape[1], 1))


  # BLOCK 1

  conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(cnn_input)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)

  conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)

  conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # expand channels for the sum
  shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(cnn_input)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

  output_block_1 = keras.layers.add([shortcut_y, conv_z])
  output_block_1 = keras.layers.Activation('relu')(output_block_1)

  # BLOCK 2

  conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)

  conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)

  conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # expand channels for the sum
  shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

  output_block_2 = keras.layers.add([shortcut_y, conv_z])
  output_block_2 = keras.layers.Activation('relu')(output_block_2)

  # BLOCK 3

  conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)

  conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)

  conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # no need to expand channels because they are equal
  shortcut_y = keras.layers.BatchNormalization()(output_block_2)

  output_block_3 = keras.layers.add([shortcut_y, conv_z])
  output_block_3 = keras.layers.Activation('relu')(output_block_3)

  # FINAL

  gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)


  model_cnn = keras.models.Model(cnn_input, gap_layer)



  classifier_input = tf.keras.Input(shape=(128,))
  classifier_output = tf.keras.layers.Dense(len(np.unique(y_true)), activation="softmax")(classifier_input)
  classifier = tf.keras.Model(classifier_input, classifier_output, name="classifier")

  pretext_input = tf.keras.Input(shape=(128,))
  pretext_output = tf.keras.layers.Dense(len(np.unique(y_true)), activation="softmax")(pretext_input)
  pretext_classifier = tf.keras.Model(pretext_input, pretext_output, name="pretext_classifier")

  cnn_classif_input = tf.keras.Input(shape=(x.shape[1], 1), name="cnn")
  cnn_series = model_cnn(cnn_classif_input)
  classif_series = classifier(cnn_series)
  pretext_series = pretext_classifier(cnn_series)

  final_classif = tf.keras.Model(cnn_classif_input, outputs=[classif_series,pretext_series], name="CnnClf")

  #final_classif.summary()


  def my_loss_sup(y_truee, y_pred):

    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    l1 = cce(y_truee, y_pred)
    mask = tf.reduce_sum(y_truee,axis=1)#
    mloss = tf.multiply(mask,l1)
    mloss = (1.0-porc_unlabel)*tf.reduce_mean(mloss)#/tf.math.maximum(tf.reduce_sum(mask),1))

    return mloss

  def my_loss_unsup(y_trux, y_pred):

    cce2 = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    local_y = y_trux[:,:len(np.unique(y_true))]
    local_kmeans_y = y_trux[:,len(np.unique(y_true)):]
    
    l2 = cce2(local_kmeans_y, y_pred)
    mask2 = 1.0-tf.reduce_sum(local_y,axis=1)
    mloss2 = tf.multiply(mask2,l2)
    mloss2 = porc_unlabel*tf.reduce_mean(mloss2)#/tf.math.maximum(tf.reduce_sum(mask2),1))

    return  mloss2

  final_classif.compile(
  loss = [my_loss_sup, my_loss_unsup],
  optimizer= tf.keras.optimizers.Adam())

  reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

  callbacks = [reduce_lr]

  y_segunda_loss = np.concatenate((y, y_kmeans_onehot),axis=1)

  final_classif.fit(x, [y, y_segunda_loss], verbose=2, epochs=1500, batch_size=None, callbacks=callbacks)

  
  y_pred = final_classif.predict(x_test)[0]

  y_pred = np.argmax(y_pred, axis=1)

  acc = accuracy_score(y_true, y_pred)
  sco = precision_score(y_true, y_pred, average='macro')
  rec = recall_score(y_true, y_pred, average='macro')

  accuracy.append(acc)
  recall.append(rec)
  precision.append(sco)

res = pd.DataFrame({"precision": precision, "accuracy": accuracy, "recall": recall})
res.to_csv("semisup_Distillation_"+filename+"_"+str(porc_unlabel)+".csv")


accuracy = []
recall = []
precision = []

k = 2
features = model_cnn(x)
gm = GaussianMixture(n_components=k*len(np.unique(y_true)), random_state=10).fit(features)
predict2 = gm.predict(features)
 
  
y_kmeans_onehot = to_categorical(predict2, num_classes=k*len(np.unique(y_true)))



for random_seed in range(10):
  print("model:", random_seed)
  tf.random.set_seed(random_seed)
  tf.config.experimental.enable_op_determinism()

  n_feature_maps = 64

  cnn_input = tf.keras.Input(shape=(x.shape[1], 1))


  # BLOCK 1

  conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(cnn_input)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)

  conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)

  conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # expand channels for the sum
  shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(cnn_input)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

  output_block_1 = keras.layers.add([shortcut_y, conv_z])
  output_block_1 = keras.layers.Activation('relu')(output_block_1)

  # BLOCK 2

  conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)

  conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)

  conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # expand channels for the sum
  shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

  output_block_2 = keras.layers.add([shortcut_y, conv_z])
  output_block_2 = keras.layers.Activation('relu')(output_block_2)

  # BLOCK 3

  conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)

  conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)

  conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # no need to expand channels because they are equal
  shortcut_y = keras.layers.BatchNormalization()(output_block_2)

  output_block_3 = keras.layers.add([shortcut_y, conv_z])
  output_block_3 = keras.layers.Activation('relu')(output_block_3)

  # FINAL

  gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)


  model_cnn = keras.models.Model(cnn_input, gap_layer)



  classifier_input = tf.keras.Input(shape=(128,))
  classifier_output = tf.keras.layers.Dense(len(np.unique(y_true)), activation="softmax")(classifier_input)
  classifier = tf.keras.Model(classifier_input, classifier_output, name="classifier")

  pretext_input = tf.keras.Input(shape=(128,))
  pretext_output = tf.keras.layers.Dense(k*len(np.unique(y_true)), activation="softmax")(pretext_input)
  pretext_classifier = tf.keras.Model(pretext_input, pretext_output, name="pretext_classifier")

  cnn_classif_input = tf.keras.Input(shape=(x.shape[1], 1), name="cnn")
  cnn_series = model_cnn(cnn_classif_input)
  classif_series = classifier(cnn_series)
  pretext_series = pretext_classifier(cnn_series)

  final_classif = tf.keras.Model(cnn_classif_input, outputs=[classif_series,pretext_series], name="CnnClf")



  def my_loss_sup(y_truee, y_pred):

    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    l1 = cce(y_truee, y_pred)
    mask = tf.reduce_sum(y_truee,axis=1)#
    mloss = tf.multiply(mask,l1)
    mloss = (1.0-porc_unlabel)*tf.reduce_mean(mloss)#/tf.math.maximum(tf.reduce_sum(mask),1))

    return mloss

  def my_loss_unsup(y_trux, y_pred):

    cce2 = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    local_y = y_trux[:,:len(np.unique(y_true))]
    local_kmeans_y = y_trux[:,len(np.unique(y_true)):]
    
    l2 = cce2(local_kmeans_y, y_pred)
    mask2 = 1.0-tf.reduce_sum(local_y,axis=1)
    mloss2 = tf.multiply(mask2,l2)
    mloss2 = porc_unlabel*tf.reduce_mean(mloss2)#/tf.math.maximum(tf.reduce_sum(mask2),1))

    return  mloss2

  final_classif.compile(
  loss = [my_loss_sup, my_loss_unsup],
  optimizer= tf.keras.optimizers.Adam())

  reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

  callbacks = [reduce_lr]

  y_segunda_loss = np.concatenate((y, y_kmeans_onehot),axis=1)

  final_classif.fit(x, [y, y_segunda_loss], verbose=2, epochs=1500, batch_size=None, callbacks=callbacks)

  
  y_pred = final_classif.predict(x_test)[0]

  y_pred = np.argmax(y_pred, axis=1)

  acc = accuracy_score(y_true, y_pred)
  sco = precision_score(y_true, y_pred, average='macro')
  rec = recall_score(y_true, y_pred, average='macro')

  accuracy.append(acc)
  recall.append(rec)
  precision.append(sco)
  print(sco, acc, rec)

res = pd.DataFrame({"precision": precision, "accuracy": accuracy, "recall": recall})
res.to_csv("ClusterDistillation_hard"+filename+"_"+str(porc_unlabel)+".csv")



accuracy = []
recall = []
precision = []

k = 2
features = model_cnn(x)
gm = GaussianMixture(n_components=k*len(np.unique(y_true)), random_state=10).fit(features)
predict2 = gm.predict_proba(features)
 

y_kmeans_onehot = predict2

for random_seed in range(10):
  print("model:", random_seed)
  tf.random.set_seed(random_seed)
  tf.config.experimental.enable_op_determinism()

  n_feature_maps = 64

  cnn_input = tf.keras.Input(shape=(x.shape[1], 1))


  # BLOCK 1

  conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(cnn_input)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)

  conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)

  conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # expand channels for the sum
  shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(cnn_input)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

  output_block_1 = keras.layers.add([shortcut_y, conv_z])
  output_block_1 = keras.layers.Activation('relu')(output_block_1)

  # BLOCK 2

  conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)

  conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)

  conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # expand channels for the sum
  shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

  output_block_2 = keras.layers.add([shortcut_y, conv_z])
  output_block_2 = keras.layers.Activation('relu')(output_block_2)

  # BLOCK 3

  conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)

  conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)

  conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # no need to expand channels because they are equal
  shortcut_y = keras.layers.BatchNormalization()(output_block_2)

  output_block_3 = keras.layers.add([shortcut_y, conv_z])
  output_block_3 = keras.layers.Activation('relu')(output_block_3)

  # FINAL

  gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)


  model_cnn = keras.models.Model(cnn_input, gap_layer)



  classifier_input = tf.keras.Input(shape=(128,))
  classifier_output = tf.keras.layers.Dense(len(np.unique(y_true)), activation="softmax")(classifier_input)
  classifier = tf.keras.Model(classifier_input, classifier_output, name="classifier")

  pretext_input = tf.keras.Input(shape=(128,))
  pretext_output = tf.keras.layers.Dense(k*len(np.unique(y_true)), activation="softmax")(pretext_input)
  pretext_classifier = tf.keras.Model(pretext_input, pretext_output, name="pretext_classifier")

  cnn_classif_input = tf.keras.Input(shape=(x.shape[1], 1), name="cnn")
  cnn_series = model_cnn(cnn_classif_input)
  classif_series = classifier(cnn_series)
  pretext_series = pretext_classifier(cnn_series)

  final_classif = tf.keras.Model(cnn_classif_input, outputs=[classif_series,pretext_series], name="CnnClf")



  def my_loss_sup(y_truee, y_pred):

    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    l1 = cce(y_truee, y_pred)
    mask = tf.reduce_sum(y_truee,axis=1)#
    mloss = tf.multiply(mask,l1)
    mloss = (1.0-porc_unlabel)*tf.reduce_mean(mloss)#/tf.math.maximum(tf.reduce_sum(mask),1))

    return mloss

  def my_loss_unsup(y_trux, y_pred):

    cce2 = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    local_y = y_trux[:,:len(np.unique(y_true))]
    local_kmeans_y = y_trux[:,len(np.unique(y_true)):]
    
    l2 = cce2(local_kmeans_y, y_pred)
    mask2 = 1.0-tf.reduce_sum(local_y,axis=1)
    mloss2 = tf.multiply(mask2,l2)
    mloss2 = porc_unlabel*tf.reduce_mean(mloss2)#/tf.math.maximum(tf.reduce_sum(mask2),1))


    return  mloss2

  final_classif.compile(
  loss = [my_loss_sup, my_loss_unsup],
  optimizer= tf.keras.optimizers.Adam())

  reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

  callbacks = [reduce_lr]

  y_segunda_loss = np.concatenate((y, y_kmeans_onehot),axis=1)

  final_classif.fit(x, [y, y_segunda_loss], verbose=2, epochs=1500, batch_size=None, callbacks=callbacks)

  
  y_pred = final_classif.predict(x_test)[0]

  y_pred = np.argmax(y_pred, axis=1)

  acc = accuracy_score(y_true, y_pred)
  sco = precision_score(y_true, y_pred, average='macro')
  rec = recall_score(y_true, y_pred, average='macro')

  accuracy.append(acc)
  recall.append(rec)
  precision.append(sco)

res = pd.DataFrame({"precision": precision, "accuracy": accuracy, "recall": recall})
res.to_csv("ClusterDistillation_soft"+filename+"_"+str(porc_unlabel)+".csv")


