
import csv
import cv2
import numpy as np


 


car_images=[]
steering_angles=[]

i=0      
loop=1
with open('data/driving_log.csv','r') as f:
        reader = csv.reader(f)
        
        for row in reader:

            if i>=1:
                
                steering_center = float(row[3])
                
                for times in range(loop):
                   # create adjusted steering measurements for the side camera images
                    correction = 0.13 # this is a parameter to tune
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction
                    source_path_ctr=row[0]
                    filename_ctr=source_path_ctr.split('/')[-1]
                    source_path_lft=row[1]
                    filename_lft=source_path_lft.split('/')[-1]
                    source_path_rht=row[2]
                    filename_rht=source_path_rht.split('/')[-1]
                    # read in images from center, left and right cameras
                    path = 'data/IMG/'# fill in the path to your training IMG directory
             
                    img_center = cv2.cvtColor(cv2.imread(path + filename_ctr),cv2.COLOR_BGR2RGB)
                    img_left = cv2.cvtColor(cv2.imread(path + filename_lft),cv2.COLOR_BGR2RGB)
                    img_right = cv2.cvtColor(cv2.imread(path + filename_rht),cv2.COLOR_BGR2RGB)

                    # add images and angles to data set

                    car_images.append(img_center)
                    car_images.append(img_left)
                    car_images.append(img_right)
                    steering_angles.append(steering_center)
                    steering_angles.append(steering_left)
                    steering_angles.append(steering_right)

                
               

            i+=1


augmented_images,augmented_measurements=[],[]
for image, measurement in zip(car_images,steering_angles):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from sklearn.utils import shuffle

X_train=np.array(augmented_images)
y_train=np.array(augmented_measurements)


#batch_size=32

#def train_data_generator(train,label,batch_size):
#    while True:
#        for offset in range(0, len(train), batch_size):
#                X_batch,y_batch = shuffle(train[offset:offset+batch_size],label[offset:offset+batch_size])
#        yield shuffle(X_batch, y_batch)
        
#def valid_data_generator(valid,label,batch_size):
#    while True:
#        for offset in range(0, len(valid), batch_size):
#                X_batch,y_batch = shuffle(valid[offset:offset+batch_size],label[offset:offset+batch_size])
#        yield shuffle(X_batch, y_batch)      
        



#build LeNet

model = Sequential()
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))

####model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(Convolution2D(24, 3, 3, activation='tanh',W_regularizer = l2(0.001)))
#model.add(MaxPooling2D((2, 2)))
#model.add(Convolution2D(36, 3, 3,activation='tanh',W_regularizer = l2(0.001)))
#model.add(MaxPooling2D((2, 2)))
#model.add(Convolution2D(48, 3, 3,activation='tanh',W_regularizer = l2(0.001)))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.2))
#model.add(Activation('tanh'))


#model.add(Flatten())
#model.add(Dense(120,W_regularizer = l2(0.001)))
#model.add(Activation('tanh'))
#model.add(Dropout(0.6))
#model.add(Dense(84,W_regularizer = l2(0.001)))
#model.add(Activation('tanh'))
#model.add(Dropout(0.6))
#model.add(Dense(1,W_regularizer = l2(0.001)))
#model.add(Activation('tanh'))
#model.add(Dropout(0.6))


########test NVIDIA structure
## convolutional layers
model.add(Cropping2D(cropping=((60,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))


## fully connected layers
model.add(Flatten())
model.add(Dense(100))

model.add(Dense(50))

model.add(Dense(10))

model.add(Dense(1))




model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2, shuffle=True,nb_epoch=3)

model.save('model.h5')




