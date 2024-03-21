#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist


# In[2]:


(x_train , y_train) , (x_test , y_test) = fashion_mnist.load_data()


# In[3]:


x_train , x_test = x_train/255.0 , x_test/255.0


# # model

# In[4]:


model_0 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(75,activation='selu'),
    keras.layers.Dense(10,activation='softmax')
])


# In[5]:


model_0.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[6]:


model_0.summary()


# In[7]:


histori = model_0.fit(x_train , y_train , validation_split=0.15 , epochs=20)


# # initialze

# In[8]:


init_1 = keras.initializers.HeNormal

init_2 = keras.initializers.lecun_normal

init_3 = keras.initializers.GlorotNormal

# make initializer

init_4 = keras.initializers.VarianceScaling(scale=0.1 , mode='fan_avg')


model_1 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(100,activation='relu',kernel_initializer=init_1),
    keras.layers.Dense(75,activation='selu',kernel_initializer=init_2),
    keras.layers.Dense(10,activation='softmax',kernel_initializer=init_3)
])


#                              SELU > ELU > LEAKY_RELU > RELU > TANh > SIGMOID

# # Batch normiliztion

# In[11]:


model_2 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization()  ,
    keras.layers.Dense(100,kernel_initializer=init_1 , use_bias=False), # when you are doing Batch norm u can False bias
    keras.layers.BatchNormalization() ,    
    keras.layers.Activation('relu'),                                              # activatein after Batch normaliz             
    keras.layers.Dense(75,activation='selu',kernel_initializer=init_2),
    keras.layers.BatchNormalization()   ,
    keras.layers.Dense(10,activation='softmax',kernel_initializer=init_3),
])


# In[13]:


model_2.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[14]:


history = model_2.fit(x_train , y_train , validation_split=0.15 , epochs=10)


# In[16]:


model_2.summary()


# # Treansfer learning

# In[28]:


model_2.save('modell.h5')


# In[27]:


model_2 = keras.models.load_model('modell.h5')


# # make a clond from main model to  ensure

# In[31]:


model_2_clond = keras.models.clone_model(model_2)
model_2_clond.set_weights(model_2.get_weights())


# In[32]:


model = keras.models.Sequential(model_2.layers[:-1])
model.add(keras.layers.Dense(1,activation='softmax'))


# In[33]:


for layer in model.layers[:-1]:
    layer.trainable = False
    print(layer.trainable)


# In[34]:


model.summary()


# In[39]:


model.compile(optimizer='sgd' , loss='binary_crossentropy' , metrics=['accuracy'])


# In[40]:


model.fit(x_train_new , y_train_new , epochs=10 , validation_split=0.15)


# #  overfiting

# # 1- L1,2 regualize

# In[7]:


reg_1 = keras.regularizers.L1(l1=0.01)

reg_2 = keras.regularizers.L2(l2=0.02)


reg_3 = keras.regularizers.L1L2(l1=0.01,l2=0.02)



modelr = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(100,activation='relu',kernel_regularizer=reg_1),
    keras.layers.Dense(75,activation='selu',kernel_regularizer=reg_2),
    keras.layers.Dense(50,activation='selu',kernel_regularizer=reg_3),
    keras.layers.Dense(10,activation='softmax')
])


# In[11]:


modelr.compile(optimizer='sgd' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])


# In[12]:


modelr.fit(x_train , y_train , epochs=10 , validation_split=0.15)


# # 2-Dropout

# In[14]:


modeld = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(75,activation='selu'),
    keras.layers.AlphaDropout(0.2),
    keras.layers.Dense(50,activation='selu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10,activation='softmax')
])


# # Max norm

# In[ ]:


modelm = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(75,activation='selu',kernel_constraint=keras.constraints.max_norm(1.0)), 
    keras.layers.Dense(50,activation='selu'),
    keras.layers.Dense(10,activation='softmax')
])

