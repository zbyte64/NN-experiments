import tensorflow as tf

from operations import OperationsFromMatrix


def assemble_evaluation(last_layer, target, n_classes):
    #TODO handle different incomming shapes
    #print('last_layer , target:', last_layer.get_shape(), target.get_shape())
    ldim, tdim = last_layer.get_shape().ndims, target.get_shape().ndims
    if ldim > tdim + 1:
        last_layer = tf.contrib.layers.flatten(last_layer)
    
    #shape into logits (single dim classification)
    if last_layer.get_shape() != (None, n_classes):
        try:
            logits = tf.contrib.layers.fully_connected(last_layer, n_classes) #pretty much ensures we can train
        except:
            print('last_layer , target:', last_layer.get_shape(), target.get_shape())
            print(target.get_shape()[0], type(target.get_shape()[0]))
            raise
    else:
        logits = last_layer

    ldim, tdim = logits.get_shape().ndims, target.get_shape().ndims
    assert ldim == tdim + 1, "%i != %i + 1" % (ldim, tdim)
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, target))
    #loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits, target)

    #convert into scalar
    #http://stackoverflow.com/questions/37638777/does-tf-nn-softmax-cross-entropy-with-logits-account-for-batch-size
    #loss = tf.reduce_mean(loss)
    #print('loss:', loss.get_shape())

    # Create a tensor for training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='RMSProp',
        learning_rate=0.001)

    return tf.argmax(logits, 1), loss, train_op


def assemble_genome(genome, n_classes, eval_f=assemble_evaluation):
    def conv_model(features, target, mode):
        input_layer = features
        #target = tf.one_hot(target, n_classes) #or? tf.contrib.layers.one_hot_encoding(target, num_classes=n_classes)
        last_layer = OperationsFromMatrix(input_layer, genome)
        return eval_f(last_layer, target, n_classes)
    return conv_model