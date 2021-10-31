import tensorflow.keras.layers as layers
import tensorflow as tf
import math

# Probability Distribution Function Neural Network (PDNN) layer
# it takes 2 arguments - size of input & number of outputs
# e.g. PDNN(7, 10)
# input = 7
# output = 10

class PDNN(layers.Layer):
    def __init__(self, num_outputs, PDFS):
        super(PDNN, self).__init__()
        self.num_outputs = num_outputs
        self.PDFS = PDFS

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=[self.PDFS,self.num_outputs,1],
            name="w",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=[self.PDFS,self.num_outputs,1],
            name="b",
            trainable=True,
        )
        self.m = self.add_weight(
            shape=[1,self.PDFS,self.num_outputs,1],
            name="m",
            trainable=True,
        )
    def call(self, input_tensor):   
        pi = tf.constant(math.pi)
        e = tf.constant(math.e)

        space = tf.constant([value/10 for value in range(1,1000 )], dtype=tf.float32)
        space = tf.reshape(space, [1,999])
        space = tf.tile(space, [self.num_outputs,1])
        space = tf.reshape(space, [1,self.num_outputs,999])
        space = tf.tile(space, [self.PDFS,1,1])
        
        input_tensor = tf.reshape(input_tensor, [1,self.num_outputs])
        input_tensor = tf.tile(input_tensor, [self.PDFS,1])
        input_tensor = tf.reshape(input_tensor, [self.PDFS,self.num_outputs,1])
        input_tensor = self.w*input_tensor+self.b
        
        pdf = tf.complex(tf.math.cos(input_tensor*space), tf.math.sin(input_tensor*space))

        pdf = tf.reshape(pdf, [1,self.PDFS,self.num_outputs,999])
        pdf = pdf*tf.complex(self.m, self.m)
        pdf = tf.math.reduce_sum(pdf, axis=1, keepdims=True)
        pdf = tf.reshape(pdf, [1,self.num_outputs,999])
        pdf = tf.abs(pdf)
        return pdf 