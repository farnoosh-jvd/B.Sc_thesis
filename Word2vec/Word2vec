import tensorflow as tf
import numpy as np
import math

batch_size =32
vocabulary_size=1000
embedding_size=64
num_sampled=32


train_data, val_data = {("the", "good"), ("good", "boy"), ("good", "the"), ("go","to") , ("go", "boy"), ('Boy', 'go'), ("to","school")},\
                       [1,2,3,4]
#get data
print("Number of training examples :", batch_size*len(train_data))
print("Number of validation examples :", len (val_data))




def skipgram():

    batch_inputs = tf.placeholder(tf.int32 , shape=[batch_size, ])
    batch_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    val_dataset = tf.constant(val_data, dtype=tf.int32)

    with tf.variable_scope("word2vec") as scope:
        embeddings= tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1 , 1))
        batch_embeddings = tf.nn.embedding_lookup(embeddings, batch_inputs)
        w= tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                           stddev=1.0 / math.sqrt(embedding_size)))
        b= tf.Variable(tf.zeros([vocabulary_size]))

        loss= tf.reduce_mean (tf.nn.nce_loss(weights=w, biases=b, labels=batch_labels,
                             inputs=batch_inputs, num_sampled=num_sampled,
                             num_classes=vocabulary_size))
        norm= tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings= embeddings/norm

        val_embeddings = tf.nn.embedding_lookup(normalized_embeddings, val_dataset)
        similarity = tf.matmul(val_embeddings/1.0, normalized_embeddings/1.0, transpose_b=True)


    return batch_inputs, batch_labels, normalized_embeddings, loss, similarity

def run():

    batch_inputs, batch_labels , normalized_embeddings , loss, similarity = skipgram()
    optimizer= tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    init= tf.global_variables_initializer()

    with tf.session as sess:
        sess.run(init)
        average_loss=0.0

        for step , batch_data in enumerate(train_data):
            inputs , labels= batch_data
            feed_dict= {batch_inputs: inputs, batch_labels:labels}

            _,loss_val=sess.run([optimizer, loss], feed_dict)
            average_loss+= loss_val

            if step%100==0:
                if step>0:
                    average_loss/=100
                print("Loss at iter", step ,":  ", average_loss)
                average_loss=0

            sim=similarity.eval()
            for i in tf.xrange(len(val_data)):
                top_k=5
                nearest= (sim[i,:]).argsort()[1:top_k+1]

        final_embeddings=normalized_embeddings.eval()
    return final_embeddings


run()


