import tensorflow as tf
from tf.keras.models import Model
from tf.keras.layers import Input, Conv2, BatchNormalization, Relu, MaxPooling2D, UpSampling2D

#################################### helper functions #########################################
def conv_bn_relu(inputs):
    out = Conv2D(24, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(inputs)
    out = BatchNormalization()(out)
    out = Relu()(out)
    return out

##################################### model structure #########################################
#---------------------------------------- encoder --------------------------------------------#
inputs = Input(shape=(512,512,2))

a1 = Conv2D(24, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(inputs)
a1 = BatchNormalization()(a1)
a1 = Relu()(a1)

a2 = Conv2D(24, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(a1)
a2 = BatchNormalization()(a2)
a2 = Relu()(a2)
a2 = Merge
#---------------------------------------------
b1 = MaxPooling2D((2, 2), padding='valid')(a2)

b1 = Conv2D(48, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(b1)
b1 = BatchNormalization()(b1)
b1 = Relu()(b1)

b2 = Conv2D(48, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(b1)
b2 = BatchNormalization()(b2)
b2 = Relu()(b2)
#---------------------------------------------
c1 = MaxPooling2D((2, 2), padding='valid')(b3)

c1 = Conv2D(96, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(c1)
c1 = BatchNormalization()(c1)
c1 = Relu()(c1)

c2 = Conv2D(96, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(c1)
c2 = BatchNormalization()(c2)
c2 = Relu()(c2)
#---------------------------------------------
d1 = MaxPooling2D((2, 2), padding='valid')(c3)

d1 = Conv2D(192, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(d1)
d1 = BatchNormalization()(b1)
d1 = Relu()(b1)

d2 = Conv2D(192, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(d1)
d2 = BatchNormalization()(d2)
d2 = Relu()(d2)
#---------------------------------------- decoder --------------------------------------------#
f1 = UpSampling2D((2, 2), interpolation='nearest')(d2)

f1 = Conv2D(96, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(f1)
f1 = BatchNormalization()(f1)
f1 = Relu()(f1)

f2 = Conv2D(96, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(f1)
f2 = BatchNormalization()(f2)
f2 = Relu()(f2)
#---------------------------------------------
g1 = UpSampling2D((2, 2), interpolation='nearest')(f2)

g1 = Conv2D(48, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(g1)
g1 = BatchNormalization()(g1)
g1 = Relu()(g1)

g2 = Conv2D(48, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(g1)
g2 = BatchNormalization()(g2)
g2 = Relu()(g2)
#---------------------------------------------
h1 = UpSampling2D((2, 2), interpolation='nearest')(g2)

h1 = Conv2D(24, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(h1)
h1 = BatchNormalization()(h1)
h1 = Relu()(h1)

h2 = Conv2D(24, 3, 1, "same", kernel_initializer='he_normal', bias_initializer='zeros')(h1)
h2 = BatchNormalization()(h2)
h2 = Relu()(h2)
#---------------------------------------------

