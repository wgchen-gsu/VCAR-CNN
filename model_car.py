from keras.models import Model
from keras.layers import Input, Add, Activation
from keras.layers.convolutional import Conv2D 
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf

class L0Loss:
    def __init__(self):
        self.gamma = K.variable(2.)

    def __call__(self):
        def calc_loss(y_true, y_pred):
            loss = K.pow(K.abs(y_true - y_pred) + 1e-8, self.gamma)
            return loss
        return calc_loss


class UpdateAnnealingParameter(Callback):
    def __init__(self, gamma, nb_epochs, verbose=0):
        super(UpdateAnnealingParameter, self).__init__()
        self.gamma = gamma
        self.nb_epochs = nb_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        new_gamma = 2.0 * (self.nb_epochs - epoch) / self.nb_epochs
        K.set_value(self.gamma, new_gamma)

        if self.verbose > 0:
            print('\nEpoch %05d: UpdateAnnealingParameter reducing gamma to %s.' % (epoch + 1, new_gamma))

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 255.0
    y_pred = K.clip(y_pred, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def get_model(model_name="srresnet"):
    if model_name == "srresnet":
        return get_srresnet_model()
    elif model_name == "unet":
        return get_unet_model(out_ch=3)
    else:
        raise ValueError("model_name should be 'srresnet'or 'unet'")

def res_block_a(x_in, num_filters, expansion, kernel_size):
    x = Conv2D(num_filters * expansion, kernel_size, padding='same')(x_in)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size, padding='same')(x)
    x = Add()([x_in, x])
    return x

def get_srresnet_model(input_channel_num=3, feature_dim=24, resunit_num=10):    
    scale = 4
    output_channel_num = 1
    inputs = Input(shape=(None, None, input_channel_num))    
        
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x0 = x
 
    for i in range(resunit_num):
        x = res_block_a(x, feature_dim, expansion=2, kernel_size=3)

    x = Conv2D(output_channel_num*scale**2, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    
    m = Conv2D(output_channel_num*scale**2, (3, 3), padding="same", kernel_initializer="he_normal")(x0)
    
    x = Add()([x, m])
    x = Conv2D(output_channel_num, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")(x) 
    
    model = Model(inputs=inputs, outputs=x, name="my_resnet")

    return model

def main():
    model = get_model()
    model.summary()

if __name__ == '__main__':
    main()
