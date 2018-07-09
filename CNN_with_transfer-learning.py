
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from scipy.io import loadmat
from keras.models import load_model

from keras.applications.inception_v3 import InceptionV3,preprocess_input


# In[ ]:


train_dir = "./train"
dev_dir = "./dev"
test_dir = "./testdata"
output_file = "./output/submission.csv"
test_file = "./input/test.csv"


# In[ ]:


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, #归一化到±1之间
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
)
dev_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, #归一化到±1之间
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
)


# In[ ]:


train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                  target_size=(299,299),#Inception V3规定大小
                                  batch_size=64,
                                  class_mode='categorical')
dev_generator = dev_datagen.flow_from_directory(directory=dev_dir,
                                target_size=(299,299),
                                batch_size=64,
                                class_mode='categorical')


# In[ ]:


base_model = InceptionV3(weights='imagenet',include_top=False)


# In[ ]:


x = base_model.output
x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
x = Dense(1024,activation='relu')(x)
predictions = Dense(10,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)


# In[ ]:


def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 17 # max_pooling_2d_2
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


setup_to_transfer_learning(model,base_model)
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=800,
                    epochs=15,
                    verbose=1,# 0 = silent, 1 = progress bar, 2 = one line per epoch.
                    validation_data=dev_generator,
                    )


# In[ ]:


setup_to_fine_tune(model,base_model)
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=800,
                    epochs=10,
                    verbose=1,# 0 = silent, 1 = progress bar, 2 = one line per epoch.
                    validation_data=dev_generator,
                    )


# In[ ]:


setup_to_fine_tune(model,base_model)
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=800,
                    epochs=10,
                    verbose=1,# 0 = silent, 1 = progress bar, 2 = one line per epoch.
                    validation_data=dev_generator,
                    )


# In[ ]:


model.save('./digits_iv3_t0.h5')


# In[ ]:


plt.plot(history.history['loss'], color='b')
plt.plot(history.history['val_loss'], color='r')
plt.show()
plt.plot(history.history['acc'], color='b')
plt.plot(history.history['val_acc'], color='r')
plt.show()


# In[ ]:


mnist_testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')
x_test = mnist_testset.astype("float32")


# In[ ]:


test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
test_generator = test_datagen.flow_from_directory(directory=test_dir,
                                  target_size=(299,299),#Inception V3规定大小
                                  batch_size=64,
                                  shuffle=False,
                                  class_mode=None)


# In[ ]:


y_hat = model.predict_generator(test_generator,verbose=1)
y_pred = np.argmax(y_hat,axis=1)


# In[ ]:


y_pred


# In[ ]:


import cv2
check = 0
fig, ax = plt.subplots(1, 1, figsize=(6,6))
ax.imshow(x_test[check].reshape(28,28),cmap='gray')
ax.set_title('28x28 data')
print(y_pred[check])
#path = "./testdata/test/test/test_"+(str(check)).zfill(7) + ".jpg"


# In[ ]:


with open(output_file, 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(len(y_pred)) :
        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))

