""" Custom Layers:


"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D

class AttentionMaps(tf.keras.layers.Layer):
  
  def __init__(self, dim_k, reg_value, **kwargs):
    super(AttentionMaps, self).__init__(**kwargs)

    self.dim_k = dim_k
    self.reg_value = reg_value

    self.Wv = Dense(self.dim_k, activation=None,\
                        kernel_regularizer=tf.keras.regularizers.l2(self.reg_value),\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=2))
    self.Wq = Dense(self.dim_k, activation=None,\
                        kernel_regularizer=tf.keras.regularizers.l2(self.reg_value),\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=3))

  def call(self, image_feat, ques_feat):
    """
    The main logic of this layer.
    """  

    # Affinity Matrix C
    # (QT)(Wb)V 
    C = tf.matmul(ques_feat, tf.transpose(image_feat, perm=[0,2,1])) # [b, 23, 49]
    # tanh((QT)(Wb)V)
    C = tf.keras.activations.tanh(C) 

    # (Wv)V
    WvV = self.Wv(image_feat)                             # [b, 49, dim_k]
    # (Wq)Q
    WqQ = self.Wq(ques_feat)                              # [b, 23, dim_k]

    # ((Wq)Q)C
    WqQ_C = tf.matmul(tf.transpose(WqQ, perm=[0,2,1]), C) # [b, k, 49]
    WqQ_C = tf.transpose(WqQ_C, perm =[0,2,1])            # [b, 49, k]

    # ((Wv)V)CT                                           # [b, k, 23]
    WvV_C = tf.matmul(tf.transpose(WvV, perm=[0,2,1]), tf.transpose(C, perm=[0,2,1]))  
                        
    WvV_C = tf.transpose(WvV_C, perm =[0,2,1])            # [b, 23, k]

    #---------------image attention map------------------
    # We find "Hv = tanh((Wv)V + ((Wq)Q)C)" ; H_v shape [49, k]

    H_v = WvV + WqQ_C                                     # (Wv)V + ((Wq)Q)C
    H_v = tf.keras.activations.tanh(H_v)                  # tanh((Wv)V + ((Wq)Q)C) 

    #---------------question attention map---------------
    # We find "Hq = tanh((Wq)Q + ((Wv)V)CT)" ; H_q shape [23, k]

    H_q = WqQ + WvV_C                                     # (Wq)Q + ((Wv)V)CT
    H_q = tf.keras.activations.tanh(H_q)                  # tanh((Wq)Q + ((Wv)V)CT) 
        
    return [H_v, H_q]                                     # [b, 49, k], [b, 23, k]
  
  def get_config(self):
    """
    This method collects the input shape and other information about the layer.
    """
    config = {
        'dim_k': self.dim_k,
        'reg_value': self.reg_value
    }
    base_config = super(AttentionMaps, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class ContextVector(tf.keras.layers.Layer):
 
  def __init__(self, reg_value, **kwargs):
    super(ContextVector, self).__init__(**kwargs)

    self.reg_value = reg_value

    self.w_hv = Dense(1, activation='softmax',\
                        kernel_regularizer=tf.keras.regularizers.l2(self.reg_value),\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=4))
    self.w_hq = Dense(1, activation='softmax',\
                        kernel_regularizer=tf.keras.regularizers.l2(self.reg_value),\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=5)) 
    

  def call(self, image_feat, ques_feat, H_v, H_q):
    """
    The main logic of this layer.
    """  
    # attention probabilities of each image region vn; a_v = softmax(wT_hv * H_v)
    a_v = self.w_hv(H_v)                               # [b, 49, 1]

    # attention probabilities of each word qt ;        a_q = softmax(wT_hq * H_q)
    a_q = self.w_hq(H_q)                               # [b, 23, 1]

    # context vector for image
    v = a_v * image_feat                               # [b, 49, dim_d]
    v = tf.reduce_sum(v, 1)                            # [b, dim_d]

    # context vector for question
    q = a_q * ques_feat                                # [b, 23, dim_d]
    q = tf.reduce_sum(q, 1)                            # [b, dim_d]


    return [v, q]

  def get_config(self):
    """
    This method collects the input shape and other information about the layer.
    """
    config = {
        'reg_value': self.reg_value
    }
    base_config = super(ContextVector, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class PhraseLevelFeatures(tf.keras.layers.Layer):
    
    def __init__(self, dim_d, **kwargs):
      super().__init__(**kwargs)
      
      self.dim_d = dim_d
      
      self.conv_unigram = Conv1D(dim_d, kernel_size=1, strides=1,\
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=6)) 
      self.conv_bigram =  Conv1D(dim_d, kernel_size=2, strides=1, padding='same',\
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=7)) 
      self.conv_trigram = Conv1D(dim_d, kernel_size=3, strides=1, padding='same',\
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=8)) 
    
    
    def call(self, word_feat):
      """
      The main logic of this layer.
    
      Compute the n-gram phrase embeddings (n=1,2,3)
      """
      # phrase level unigram features
      x_uni = self.conv_unigram(word_feat)                    # [b, 23, dim_d]
    
      # phrase level bigram features
      x_bi  = self.conv_bigram(word_feat)                     # [b, 23, dim_d]
    
      # phrase level trigram features
      x_tri = self.conv_trigram(word_feat)                    # [b, 23, dim_d]
    
      # Concat
      x = tf.concat([tf.expand_dims(x_uni, -1),\
                      tf.expand_dims(x_bi, -1),\
                      tf.expand_dims(x_tri, -1)], -1)         # [b, 23, dim_d, 3]
    
      # https://stackoverflow.com/a/36853403
      # Max-pool across n-gram features; over-all phrase level feature
      x = tf.reduce_max(x, -1)                                # [b, 23, dim_d]
      print(x)
      return x
    
    def get_config(self):
      """
      This method collects the input shape and other information about the layer.
      """
      config = {
          'dim_d': self.dim_d
      }
      base_config = super().get_config()
      return dict(list(base_config.items()) + list(config.items()))