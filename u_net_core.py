
from tensorflow.keras.models import Model # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, MaxPooling2D, Conv2D, BatchNormalization # Импортируем стандартные слои keras
from tensorflow.keras import backend as K # Импортируем модуль backend keras'а
from tensorflow.keras.optimizers import Adam # Импортируем оптимизатор Adam
from tensorflow.keras import utils # Импортируем модуль utils библиотеки tensorflow.keras для получения OHE-представления
import matplotlib.pyplot as plt # Импортируем модуль pyplot библиотеки matplotlib для построения графиков
from tensorflow.keras.preprocessing.image import img_to_array, load_img # Импортируем модуль image для работы с изображениями
import numpy as np # Импортируем библиотеку numpy
from skimage import color 
import os # Импортируем библиотеку os для раоты с фаловой системой
def dice_coef(y_true, y_pred):
    return ( 2.* K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
def loadpic():
    images = []
    directory = '/home/maxim/datasets/dataset1/images_prepped_train'
    for filename in os.listdir(directory):
        images.append(img_to_array(load_img(os.path.join(directory,filename), target_size=(352, 480))).astype('uint8'))

    imagesTest = []
    directory = '/home/maxim/datasets/dataset1/images_prepped_test'
    for filename in os.listdir(directory):
      imagesTest.append(img_to_array(load_img(os.path.join(directory,filename), target_size=(352, 480))).astype('uint8'))

    segments = []
    directory = '/home/maxim/datasets/dataset1/annotations_prepped_train'
    for filename in os.listdir(directory):
       segments.append(img_to_array(load_img(os.path.join(directory,filename), target_size=(352, 480))).astype('uint8')[:,:,0])
    segmentsTest = []
    directory = '/home/maxim/datasets/dataset1/annotations_prepped_test'
    for filename in os.listdir(directory):
       segmentsTest.append(img_to_array(load_img(os.path.join(directory,filename), target_size=(352, 480))).astype('uint8')[:,:,0])
    return images,imagesTest,segments,segmentsTest
def oneHotAll(dset):
    return utils.to_categorical(dset, num_classes=12)

def reduceTags(dset):
  res = dset.copy()
  res[dset >= CLIP_CLASSES - 1] = CLIP_CLASSES - 1
  return res

def oneHotReduced(dset,CLIP_CLASSES):
  return utils.to_categorical(reduceTags(dset), num_classes=CLIP_CLASSES)

def getTrainandTest(Clip_class=3):
    """#Создаём обучающую выборку"""

    xTrainFull = np.array(images)
    yTrainFull = np.array(segments)[:,:,:,None]

    """# **Создаем тестовую выборку**"""

    xTestFull = np.array(imagesTest)
    yTestFull = np.array(segmentsTest)[:,:,:,None]

    """**Преобразуем картинку сегментации в OHE**"""
    yTrainRFull = oneHotReduced(yTrainFull,Clip_class)
    yTestRFull = oneHotReduced(yTestFull,Clip_class)
    xTrainFull = color.rgb2lab(xTrainFull[:])
    xTestFull = color.rgb2lab(xTestFull[:])
    return xTrainFull,yTrainRFull,yTestFull,yTestRFull

"""#Расширенная U-net"""
def unetWithMask(num_classes = 3, input_shape= (352, 480, 3)):
    img_input = Input(input_shape)  # создание слоя img_input                                      
    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input) #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)         #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    block_1_out = Activation('relu')(x)                                    #слой активации 
    block_1_out_mask = Conv2D(64, (1, 1), padding='same')(block_1_out)     #слой 2-мерной свёртки
    x = MaxPooling2D()(block_1_out)                                        #слой 2-мерного dawn sampling
    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)        #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)        #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    block_2_out = Activation('relu')(x)                                    #слой активации 
    block_2_out_mask = Conv2D(128, (1, 1), padding='same')(block_2_out)    #слой 2-мерной свёртки
    x = MaxPooling2D()(block_2_out)                                        #слой 2-мерного dawn sampling
    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)        #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)        #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации 
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)        #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    block_3_out = Activation('relu')(x)                                    #слой активации
    block_3_out_mask = Conv2D(256, (1, 1), padding='same')(block_3_out)    #слой 2-мерной свёртки
    x = MaxPooling2D()(block_3_out)                                        #слой 2-мерного dawn sampling
    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)        #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)        #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)        #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    block_4_out = Activation('relu')(x)                                    #слой активации
    block_4_out_mask = Conv2D(512, (1, 1), padding='same')(block_4_out)    #слой 2-мерной свёртки   
    x = MaxPooling2D()(block_4_out)                                        #слой 2-мерного dawn sampling
    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)        #слой 2-мерной свёртки  
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)        #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)        #слой 2-мерной свёртки 
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)    #слой 2-мерного dawn sampling
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = concatenate([x, block_4_out, block_4_out_mask])                    #слой cocatenate
    x = Conv2D(512, (3, 3), padding='same')(x)                             #слой 2-мерной свёртки 
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = Conv2D(512, (3, 3), padding='same')(x)                             #слой 2-мерной свёртки 
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)    #слой 2-мерного dawn sampling
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = concatenate([x, block_3_out, block_3_out_mask])                    #слой cocatenate
    x = Conv2D(256, (3, 3), padding='same')(x)                             #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = Conv2D(256, (3, 3), padding='same')(x)                             #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)    #слой 2-мерного dawn sampling
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = concatenate([x, block_2_out, block_2_out_mask])                    #слой cocatenate
    x = Conv2D(128, (3, 3), padding='same')(x)                             #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = Conv2D(128, (3, 3), padding='same')(x)                             #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)     #слой 2-мерного dawn sampling
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = concatenate([x, block_1_out, block_1_out_mask])                    #слой cocatenate
    x = Conv2D(64, (3, 3), padding='same')(x)                              #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = Conv2D(64, (3, 3), padding='same')(x)                              #слой 2-мерной свёртки
    x = BatchNormalization()(x)                                            #слой нормализации
    x = Activation('relu')(x)                                              #слой активации
    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)        #выходной слой 2- мерной свёртки с активацией "softmax"
    model = Model(img_input, x)                                            #создание модели                                   
    # Компилируем модель 
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    return model

if __name__ == '__main__':
    images,imagesTest,segments,segmentsTest =loadpic()
    CLIP_CLASSES = 3
    xTrainFull, yTrainRFull, yTestFull, yTestRFull = getTrainandTest(CLIP_CLASSES)
    model = unetWithMask(CLIP_CLASSES, (352, 480, 3))
    history1 = model.fit(xTrainFull, yTrainRFull, epochs=75, batch_size=10, validation_split=0.2)
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    history2 = model.fit(xTrainFull, yTrainRFull, epochs=50, batch_size=10, validation_split=0.2)

    plt.plot(history.history1['dice_coef'],
             label='Доля верных ответов на обучающем наборе')
    plt.plot(history.history1['val_dice_coef'],
             label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()

    plt.plot(history2.history['dice_coef'],
             label='Доля верных ответов на обучающем наборе')
    plt.plot(history2.history['val_dice_coef'],
             label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()

    predM3 = modelM3.predict(xTrainFull[:])
    print(predM3.shape)

    outM3 = predM3.argmax(-1)

    plt.imshow(color.lab2rgb(xTrainFull[220]))
    plt.show()

    plt.imshow(yTrainRFull[220])
    plt.show()

    plt.imshow(outM3[220])
    plt.show()

    print("ok")
