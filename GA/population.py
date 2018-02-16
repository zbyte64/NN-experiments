import numpy as np
import tensorflow as tf
from sklearn import metrics
import hashlib
import itertools
import time
import traceback

from genome import assemble_genome
from reproduction import match_maker
from operations import ReprFromMatrix


def initialize_world(world_shape, genome_shape):
    return np.random.uniform(high=np.finfo(np.float64).max, size=(*world_shape, *genome_shape))

def tensor_size(t):
    all_vars = set(v for key in t.graph.get_all_collection_keys() for v in t.graph.get_collection_ref(key) if hasattr(v, 'get_shape'))
    var_sizes = (np.product([1 if d is None else d for d in v.get_shape().as_list()])*v.dtype.size for v in all_vars)
    return sum(var_sizes)
    
    var_sizes = (np.product(list(map(int, v.get_shape())))*v.dtype.size if hasattr(v, 'get_shape') else 0
                 for key in t.graph.get_all_collection_keys() for v in t.graph.get_collection_ref(key))
    return sum(var_sizes)

def tuplize(x):
    if isinstance(x, tuple):
        return tuple(tuplize(y) for y in x)
    if isinstance(x, dict):
        return tuple( (tuplize(k), tuplize(v)) for k, v in x.items() ) 
    if isinstance(x, list):
        return tuple(tuplize(y) for y in x)
    return x

hashGenome = lambda g, l: hash(tuplize(ReprFromMatrix(g, l)))


def train_and_test(X_train, y_train, X_test, y_test, model_fn, steps=2000, 
                   batch_size=None, max_batch_size=1000, mem_size=1*1024**3, max_mem_size=2*1024**3):
    if not batch_size:
        with tf.Graph().as_default() as g: 
            psuedo_input = tf.placeholder(tf.float32, shape=(None, *X_train.shape[1:]))
            psuedo_output = tf.placeholder(tf.int32, shape=(None,))
            res = model_fn(psuedo_input, psuedo_output, 'training')
        #tf.reset_default_graph()
        #print("example:", example_individual, example_individual.graph==g)
        example_individual = res[1]
        example_size = tensor_size(example_individual)
        assert example_size < max_mem_size
        max_batch_size = min(max_batch_size, len(X_train))
        batch_size = max(min(int(mem_size/(example_size**1.5)), max_batch_size), 3)
        print("Individual size (MBytes):", example_size/(1024**2), batch_size)
        
    start_time = time.time()
    #tf.argmax(logits, 1), loss, train_op = model_fn(features, target, mode)
    classifier = tf.contrib.learn.SKCompat(tf.contrib.learn.Estimator(model_fn=model_fn))
    classifier.fit(X_train,
                   y_train,
                   batch_size=batch_size,
                   steps=steps)
    end_time = time.time()
    elapsed_time = end_time - start_time
    score = metrics.accuracy_score(y_test,
                                 list(classifier.predict(X_test)))
    #print('Accuracy: {0:f}'.format(score))
    return classifier, score, elapsed_time

def evolve_classifier(X_train_grad, y_train_grad, X_test_grad, y_test_grad, n_classes_grad, 
                      gradient_mask=None, steps_mask=None, batch_mask=None, passports=None,
                      world=None, world_shape=(5,10), genome_shape=(6,5)):
    #TODO tick mask, stress mask, footprint
    #genome resizes can happen outside
    tf.logging.set_verbosity(tf.logging.FATAL)
    
    if world is None:
        world = initialize_world(world_shape, genome_shape)
    else:
        world_shape = world.shape[:-len(genome_shape)]
    
    if gradient_mask is None:
        n_grads = len(n_classes_grad)
        p_size = int(world_shape[0] / n_grads)
        gradient_mask = np.zeros(world_shape, dtype='int32')
        for i in range(n_grads):
            gradient_mask[i*p_size:(i+1)*p_size] = i
    
    fitness = None
    fitness_history = dict() #TODO with expiration (ie last 100 accessed)
    
    generation = 0
    while True:
        print("GENERATION:", generation)
        generation += 1
        fitness = evaluate_world(world, world_shape, X_train_grad, y_train_grad, X_test_grad, y_test_grad, n_classes_grad, 
                                 gradient_mask, fitness_history, steps_mask, batch_mask, passports)
        #TODO viability
        if passports is not None:
            passport = dict()
        else:
            passport = None
        world = match_maker(world, fitness, passport=passport)
        if passports is not None:
            passports.append(passport)
        yield world, fitness

def evaluate_world(world, world_shape, X_train_grad, y_train_grad, X_test_grad, y_test_grad, n_classes_grad, 
                   gradient_mask, fitness_history, steps_mask=None, batch_mask=None, passports=None):
    fitness = np.zeros(shape=(*world_shape, 2))
    for i, j in itertools.product(range(world_shape[0]), range(world_shape[1])):
        genome = world[i, j]
        steps = steps_mask[i, j] if steps_mask is not None else 500
        batch_size = batch_mask[i, j] if batch_mask is not None else 250
        train_index = gradient_mask[i, j]
        #compute performance hash (based on phenotype and environment)
        psuedo_input = tf.placeholder(tf.float32, shape=(None, *X_train_grad[train_index].shape))
        perf_index = (hashGenome(genome, psuedo_input), steps, batch_size, train_index)
        if perf_index in fitness_history:
            accuracy, elapsed_time = fitness_history[perf_index]
        else:
            X_train, y_train, n_classes = X_train_grad[train_index], y_train_grad[train_index], n_classes_grad[train_index]
            X_test, y_test = X_test_grad[train_index], y_test_grad[train_index]
            try:
                gpu_options = tf.GPUOptions(allow_growth=True) #per_process_gpu_memory_fraction=0.333
                with tf.Session(
                    config=tf.ConfigProto(
                        log_device_placement=True, 
                        allow_soft_placement=True,
                        gpu_options=gpu_options)) as sess:
                    res = evaluate(genome, X_train, y_train, X_test, y_test, n_classes, steps, batch_size)
            except Exception as error:
                print("Catastrophic error during evaluation, stoping:", error)
                traceback.print_stack()
                print('--------------')
                traceback.print_exc()
                return fitness
            if res is None:
                accuracy, elapsed_time = np.finfo(np.float64).min, np.finfo(np.float64).max
            else:
                accuracy, elapsed_time = res
                fitness_history[perf_index] = accuracy, elapsed_time
        fitness[i, j] = [accuracy, 1.0/elapsed_time]
    return fitness

def evaluate(genome, X_train, y_train, X_test, y_test, n_classes, steps, batch_size):
    print("Eval genome:")
    psuedo_input = tf.placeholder(tf.float32, shape=X_train.shape)
    print(ReprFromMatrix(genome, psuedo_input))
    try:
        model_fn = assemble_genome(genome, n_classes)
    except Exception as error:
        print("Individual aborted:", error)
        return None
    else:
        try:
            classifier, accuracy, elapsed_time = train_and_test(X_train, y_train, X_test, y_test, model_fn, steps=steps, batch_size=batch_size)
        except tf.python.errors.UnimplementedError as error:
            print("Individiual died using unimplemented feature")
            return None
        except tf.python.errors.ResourceExhaustedError as error:
            print("Individual died from exhaustion")
            return None
        except Exception as error:
            if 'Dst tensor is not initialized' in str(error):
                print('Individual was too big')
                return None
            if 'Resource exhausted:' in str(error):
                print("Resource exhaustion detected.")
                #TODO fix this somehow, gc collect ineffective, no session left over, ??
                raise
            print("Individual died:", error)
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            return None
    print("Acc/Time", accuracy, elapsed_time)
    return accuracy, elapsed_time

def heritage(passports, point, t=None, d=1):
    if t is None:
        t = len(passports)-1
    if not t:
        return
    parents = None
    ti = 0
    for ti in range(t, -1, -1):
        if ti < len(passports) and point in passports[ti]:
            parents = passports[ti][point]
            break
    if parents is None:
        return None
    yield (d, ti, point, parents)
    for parent in parents:
        for p in heritage(passports, parent, ti-1, d=d+1):
            yield p
    #woot
