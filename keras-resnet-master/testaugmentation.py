#尝试一下从中心来chop图像

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import resnet
import os
import pandas as pd
import random
from skimage import data, exposure, img_as_float
from skimage.transform import resize
#一、初始化
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.0005, patience=20)
csv_logger = CSVLogger('imagemodel.csv')

batch_size = 10
nb_classes = 10
nb_epoch = 150
data_augmentation = True

img_rows, img_cols = 512,512
img_channels = 3

# 二、定义标签
y_train=np.zeros([2200,1])
for i in range(10):
    for j in range(220):
        y_train[i*220+j,0]=i

y_val=np.zeros([550,1])
for i in range(10):
    for j in range(55):
        y_val[i*55+j,0]=i
#三、读取训练图片
X_train=np.zeros([2200,img_rows,img_cols,3])
X_val=np.zeros([550,img_rows,img_cols,3])
#先读训练集
#第0类
for i in range(220):
    filename=r'C:\Data\train\train\HTC-1-M7\(HTC-1-M7)'+str(i+1)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_train[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第1类
for i in range(220):
    filename=r'C:\Data\train\train\iPhone-4s\(iP4s)'+str(i+1)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_train[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第2类iphone6
for i in range(220):
    filename=r'C:\Data\train\train\iPhone-6\(iP6)'+str(i+1)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_train[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第3类LG-Nexus-5x
for i in range(220):
    filename=r'C:\Data\train\train\LG-Nexus-5x\(LG5x)'+str(i+1)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_train[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第4类MOTORALA-DROID-MAXX
for i in range(220):
    filename=r'C:\Data\train\train\Motorola-Droid-Maxx\(MotoMax)'+str(i+1)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_train[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第5类MATOROLA-NEXUS-6
for i in range(220):
    filename=r'C:\Data\train\train\Motorola-Nexus-6\(MotoNex6)'+str(i+1)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_train[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第6类
for i in range(220):
    filename=r'C:\Data\train\train\Motorola-X\(MotoX)'+str(i+1)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_train[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第7类
for i in range(220):
    filename=r'C:\Data\train\train\Samsung-Galaxy-Note3\(GalaxyN3)'+str(i+1)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_train[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第8类
for i in range(220):
    filename=r'C:\Data\train\train\Samsung-Galaxy-S4\(GalaxyS4)'+str(i+1)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_train[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第9类
for i in range(220):
    filename=r'C:\Data\train\train\Sony-NEX-7\(Nex7)'+str(i+1)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_train[i,:,:,:]=sample[Left:Right,Down:Up,:]

print('X_train.shape=',X_train.shape)
#测试集
#第0类
for i in range(55):
    filename=r'C:\Data\train\train\HTC-1-M7\(HTC-1-M7)'+str(i+221)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_val[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第1类
for i in range(55):
    filename=r'C:\Data\train\train\iPhone-4s\(iP4s)'+str(i+221)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_val[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第2类iphone6
for i in range(55):
    filename=r'C:\Data\train\train\iPhone-6\(iP6)'+str(i+221)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_val[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第3类LG-Nexus-5x
for i in range(55):
    filename=r'C:\Data\train\train\LG-Nexus-5x\(LG5x)'+str(i+221)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_val[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第4类MOTORALA-DROID-MAXX
for i in range(55):
    filename=r'C:\Data\train\train\Motorola-Droid-Maxx\(MotoMax)'+str(i+221)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_val[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第5类MATOROLA-NEXUS-6
for i in range(55):
    filename=r'C:\Data\train\train\Motorola-Nexus-6\(MotoNex6)'+str(i+221)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_val[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第6类
for i in range(55):
    filename=r'C:\Data\train\train\Motorola-X\(MotoX)'+str(i+221)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_val[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第7类
for i in range(55):
    filename=r'C:\Data\train\train\Samsung-Galaxy-Note3\(GalaxyN3)'+str(i+221)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_val[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第8类
for i in range(55):
    filename=r'C:\Data\train\train\Samsung-Galaxy-S4\(GalaxyS4)'+str(i+221)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_val[i,:,:,:]=sample[Left:Right,Down:Up,:]
#第9类
for i in range(55):
    filename=r'C:\Data\train\train\Sony-NEX-7\(Nex7)'+str(i+221)+r'.jpg'
    print(filename)
    sample=mping.imread(filename)
    Left=0.5*sample.shape[0]-0.5*img_rows
    Right=0.5*sample.shape[0]+0.5*img_rows
    Down=0.5*sample.shape[1]-0.5*img_cols
    Up=0.5*sample.shape[1]+0.5*img_cols
    Left=int(Left)
    Right=int(Right)
    Down=int(Down)
    Up=int(Up)
    X_val[i,:,:,:]=sample[Left:Right,Down:Up,:]
print('X_val.shape=',X_val.shape)

# 用ONE-HOT来存标签
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)
#将像素值改成float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

#六、白化
mean_image_train = np.mean(X_train, axis=0)
mean_image_val =np.mean(X_val, axis=0)
X_train -= mean_image_train
X_val -= mean_image_val

del mean_image_train
del mean_image_val

X_train /= 128.
X_val /= 128.



#七、搭建模型
model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#八、数据增强和训练
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_val, Y_val),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=False,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=False,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_data=(X_val, Y_val),
                        epochs=nb_epoch, verbose=1, max_q_size=100,
                        callbacks=[lr_reducer, early_stopper, csv_logger])

    model.summary()

    # 四、读取测试图片
    del X_train
    del y_train
    del X_val
    del y_val
    X_test = np.zeros([2640, img_rows, img_cols, 3])
    i = 0
    for fname in sorted(os.listdir(r'C:\Data\test\test')):
        print(fname)
        testfilename = 'C:\Data\\test\\test\\' + fname
        temp = mping.imread(testfilename)
        X_test[i, :img_rows, :img_cols, :] = temp[:img_rows, :img_cols, :]
        i += 1
    X_test = X_test.astype('float32')
    mean_image_test = np.mean(X_test, axis=0)
    X_test -= mean_image_test
    del mean_image_test
    X_test /= 128.
    print('X_test.shape=',X_test.shape)
    # 九、预测

    pred=model.predict(X_test)
    print(pred.shape)
    result=np.zeros([2640,1])
    for i in range(2640):
        result[i,0]=np.argmax(pred[i,:])
    print(result.shape)
    np.savetxt(r"pred.csv", pred, delimiter=',')
    np.savetxt(r"result.csv", result, delimiter=',')

#gamma 0.8
    del X_test
    X_test = np.zeros([2640, img_rows, img_cols, 3])
    i = 0
    for fname in sorted(os.listdir(r'C:\Data\test\test')):
        print(fname)
        testfilename = 'C:\Data\\test\\test\\' + fname
        temp = mping.imread(testfilename)
        sample_gamma_corrected1 = exposure.adjust_gamma(temp, 0.8)
        X_test[i, :img_rows, :img_cols, :] = temp[:img_rows, :img_cols, :]
        i += 1
    X_test = X_test.astype('float32')
    mean_image_test = np.mean(X_test, axis=0)
    X_test -= mean_image_test
    del mean_image_test
    X_test /= 128.
    print('X_test.shape=', X_test.shape)
    # 预测

    pred1 = model.predict(X_test)
    print(pred1.shape)
    result1 = np.zeros([2640, 1])
    for i in range(2640):
        result1[i, 0] = np.argmax(pred1[i, :])
    print(result1.shape)
    np.savetxt(r"pred1.csv", pred1, delimiter=',')
    np.savetxt(r"result1.csv", result1, delimiter=',')


#gamma 1.2
    del X_test
    X_test = np.zeros([2640, img_rows, img_cols, 3])
    i = 0
    for fname in sorted(os.listdir(r'C:\Data\test\test')):
        print(fname)
        testfilename = 'C:\Data\\test\\test\\' + fname
        temp = mping.imread(testfilename)
        sample_gamma_corrected1 = exposure.adjust_gamma(temp, 1.2)
        X_test[i, :img_rows, :img_cols, :] = temp[:img_rows, :img_cols, :]
        i += 1
    X_test = X_test.astype('float32')
    mean_image_test = np.mean(X_test, axis=0)
    X_test -= mean_image_test
    del mean_image_test
    X_test /= 128.
    print('X_test.shape=', X_test.shape)
    # 九、预测

    pred2 = model.predict(X_test)
    print(pred2.shape)
    result2 = np.zeros([2640, 1])
    for i in range(2640):
        result2[i, 0] = np.argmax(pred2[i, :])
    print(result2.shape)
    np.savetxt(r"pred2.csv", pred2, delimiter=',')
    np.savetxt(r"result2.csv", result2, delimiter=',')


#resize 1.5
    del X_test
    X_test = np.zeros([2640, img_rows, img_cols, 3])
    i = 0
    for fname in sorted(os.listdir(r'C:\Data\test\test')):
        print(fname)
        testfilename = 'C:\Data\\test\\test\\' + fname
        temp = mping.imread(testfilename)
        sample_resize3 = resize(sample, (int(1.5 * sample.shape[0]), int(1.5 * sample.shape[1])), mode='reflect')
        X_test[i, :img_rows, :img_cols, :] = sample_resize3[int(0.5 * sample_resize3.shape[0] - 0.5 * img_rows):int(0.5 * sample_resize3.shape[0] + 0.5 * img_rows), int(0.5 * sample_resize3.shape[1] - 0.5 * img_rows):int(0.5 * sample_resize3.shape[1] + 0.5 * img_rows), :]
        i += 1
    X_test = X_test.astype('float32')
    mean_image_test = np.mean(X_test, axis=0)
    X_test -= mean_image_test
    del mean_image_test
    X_test /= 128.
    print('X_test.shape=', X_test.shape)
    # 九、预测

    pred3 = model.predict(X_test)
    print(pred3.shape)
    result3 = np.zeros([2640, 1])
    for i in range(2640):
        result3[i, 0] = np.argmax(pred3[i, :])
    print(result3.shape)
    np.savetxt(r"pred3.csv", pred3, delimiter=',')
    np.savetxt(r"result3.csv", result3, delimiter=',')

#resize 2.0
    del X_test
    X_test = np.zeros([2640, img_rows, img_cols, 3])
    i = 0
    for fname in sorted(os.listdir(r'C:\Data\test\test')):
        print(fname)
        testfilename = 'C:\Data\\test\\test\\' + fname
        temp = mping.imread(testfilename)
        sample_resize4 = resize(sample, (int(2.0 * sample.shape[0]), int(2.0 * sample.shape[1])), mode='reflect')
        X_test[i, :img_rows, :img_cols, :] = sample_resize4[int(0.5 * sample_resize3.shape[0] - 0.5 * img_rows):int(0.5 * sample_resize3.shape[0] + 0.5 * img_rows), int(0.5 * sample_resize3.shape[1] - 0.5 * img_rows):int(0.5 * sample_resize3.shape[1] + 0.5 * img_rows), :]
        i += 1
    X_test = X_test.astype('float32')
    mean_image_test = np.mean(X_test, axis=0)
    X_test -= mean_image_test
    del mean_image_test
    X_test /= 128.
    print('X_test.shape=', X_test.shape)
    # 九、预测

    pred4 = model.predict(X_test)
    print(pred4.shape)
    result4 = np.zeros([2640, 1])
    for i in range(2640):
        result4[i, 0] = np.argmax(pred4[i, :])
    print(result4.shape)
    np.savetxt(r"pred4.csv", pred4, delimiter=',')
    np.savetxt(r"result4.csv", result4, delimiter=',')


#集成四个结果
final_result=np.zeros([2640,10])
final_pred=np.zeros([1640,1])
for i in range(2640):
    slice=int(pred[i,0])
    final_result[i,slice] += 1
    slice1=int(pred1[i,0])
    final_result[i,slice1] += 1
    slice2=int(pred2[i,0])
    final_result[i,slice2] += 1
    slice3=int(pred3[i,0])
    final_result[i,slice3] += 1
    slice4=int(pred4[i,0])
    final_result[i,slice4] += 1

for i in range(2640):
    final_pred[i,0]=np.argmax(final_result[i,:])

np.savetxt(r"final_pred.csv", final_pred, delimiter=',')
np.savetxt(r"final_result.csv", final_result, delimiter=',')