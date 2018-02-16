from population import evaluate, initialize_world
from operations import ReprFromMatrix
from reproduction import match_maker
import traceback
import numpy as np
import multiprocessing as mp
import itertools

#RPC to keep data & masks in memory, receive genome updates

class Farm(object):
    @classmethod
    def spawn(cls):
        in_que, out_que = mp.Queue(), mp.Queue(1)
        p = mp.Process(target=cls.run, args=(in_que, out_que))
        p.start()
        return p, in_que, out_que
            
    @classmethod
    def run(cls, in_que, out_que):
        tf.logging.set_verbosity(tf.logging.FATAL)
        farm = cls(in_que, out_que)
        farm.loop()
    
    def __init__(self, in_que, out_que):
        self.in_que = in_que
        self.out_que = out_que
        self.fitness_history = dict() #TODO with expiration (ie last 100 accessed)
    
    def loop(self):
        try:
            while True:
                msg_type, msg_payload = self.in_que.get()
                if msg_type == 'train_test':
                    self.update_train_test(msg_payload)
                elif msg_type == 'world':
                    self.update_world(msg_payload)
                elif msg_type == 'masks':
                    self.update_masks(msg_payload)
                elif msg_type == 'grad':
                    self.update_gradient_mask(msg_payload)
                elif msg_type == 'tick':
                    try:
                        res = self.tick()
                    except Exception as error:
                        print("Unhandled exception at a farm:", error)
                        traceback.print_stack()
                        print('--------------')
                        traceback.print_exc()
                        self.out_que.put(None)
                        return
                    self.out_que.put(res)
                else:
                    assert False, msg_type
        except mp.Queue.Empty:
            return
        finally:
            self.out_que.close()
    
    def update_train_test(self, train_test):
        self.train_test = train_test
    
    def update_gradient_mask(self, gradient_mask):
        self.gradient_mask = gradient_mask
    
    def update_world(self, world):
        self.world = world
    
    def update_masks(self, masks):
        self.masks = masks
    
    @property
    def world_shape(self):
        return self.world.shape
    
    def world_data_selector(self, x, y):
        return self.gradient_mask[x,y]
    
    @classmethod
    def hash_genome(cls, genome):
        def tuplize(x):
            if isinstance(x, tuple):
                return tuple(tuplize(y) for y in x)
            if isinstance(x, dict):
                return tuple( (tuplize(k), tuplize(v)) for k, v in x.items() ) 
            if isinstance(x, list):
                return tuple(tuplize(y) for y in x)
            return x
        return hash(tuplize(ReprFromMatrix(genome)))
    
    def tick(self):
        world_shape = self.world_shape
        fitness = np.zeros(shape=(*world_shape, 2))
    
        for i, j in itertools.product(range(world_shape[0]), range(world_shape[1])):
            genome = self.world[i, j]
            steps = self.masks['steps'][i, j] if 'steps' in self.masks is not None else 500
            batch_size = self.masks['batch'][i, j] if 'batch' in self.masks is not None else 250
            train_index = self.world_data_selector(i, j)
            #compute performance hash (based on phenotype and environment)
            perf_index = (self.hash_genome(genome), steps, batch_size, train_index)
            if perf_index in self.fitness_history:
                accuracy, elapsed_time = self.fitness_history[perf_index]
            else:
                X_train, y_train, X_test, y_test, n_classes = [ s[train_index] for s in self.train_test ]
                res = evaluate(genome, X_train, y_train, X_test, y_test, n_classes, steps, batch_size)
                if res is None:
                    accuracy, elapsed_time = np.finfo(np.float64).min, np.finfo(np.float64).max
                else:
                    accuracy, elapsed_time = res
                self.fitness_history[perf_index] = accuracy, elapsed_time
            fitness[i, j] = [accuracy, 1.0/elapsed_time]
        return fitness

class MassFarming(object):
    def __init__(self, X_train_grad, y_train_grad, X_test_grad, y_test_grad, n_classes_grad, 
                      gradient_mask=None, steps_mask=None, batch_mask=None, passports=None,
                      world=None, world_shape=(5,10), genome_shape=(6,5)):
        #TODO tick mask, stress mask
        if not world:
            world = initialize_world(world_shape, genome_shape)
        else:
            world_shape = world.shape[:-len(genome_shape)]

        if gradient_mask is None:
            n_grads = len(n_classes_grad)
            p_size = int(world_shape[0] / n_grads)
            gradient_mask = np.zeros(world_shape, dtype='int32')
            for i in range(n_grads):
                gradient_mask[i*p_size:(i+1)*p_size] = i
        
        self.passports = passports
        self.world = world
        self.gradient_mask = gradient_mask
        self.genome_shape = genome_shape
        self.world_shape = world_shape
        self.masks = dict()
        if steps_mask is not None:
            self.masks['steps'] = steps_mask
        if batch_mask is not None:
            self.masks['batch'] = batch_mask
        self.train_test = (X_train_grad, y_train_grad, X_test_grad, y_test_grad, n_classes_grad)
        self.farms = []
    
    def create_local_farms(self, n=1):
        farms = []
        for i in range(n):
            p, in_que, out_que = self.new_farm(i, n)
            farms.append((p, in_que, out_que))
        return farms
   
    def new_farm(self, i, n=None):
        if n is None:
            n = len(self.farms)
        w = self.world_shape[0]/n
        start, stop = w*i, w*(i+1)
        p, in_que, out_que = Farm.spawn()
        in_que.put(('train_test', self.train_test))
        in_que.put(('masks', { k: self.masks[k][start:stop] for k, m in self.masks.items() } ))
        in_que.put(('grad', self.gradient_mask[start:stop]))
        return p, in_que, out_que
    
    def tick_farm(self, i, tries=3):
        assert tries > 0
        p, in_que, out_que = self.farms[i]
        if not p.is_alive():
            in_que.close()
            out_que.close()
            p.join(1.0)
            self.new_farm(i)
            self.tick_farm(i, tries-1)
            return
        
        n = len(self.farms)
        w = self.world_shape[0]/n
        start, stop = w*i, w*(i+1)
        
        in_que.put(('world', self.world[start:stop]))
        in_que.put(('tick', 0))
    
    def harvest(self, i, tries=3):
        assert tries > 0
        p, in_que, out_que = self.farms[i]
        if not p.is_alive():
            in_que.close()
            out_que.close()
            p.join(1.0)
            self.new_farm(i)
            self.tick_farm(i)
            return self.harvest(i, tries-1)
        retval = out_que.get()
        if retval is None:
            in_que.close()
            out_que.close()
            p.join(1.0)
            self.new_farm(i)
            self.tick_farm(i)
            return self.harvest(i, tries-1)
        return retval
    
    def tick(self):
        for i in range(len(self.farms)):
            self.tick_farm(i)
        
        fitness = np.zeros((*self.world_shape, 2))
        w = self.world_shape[0]/len(self.farms)
        for i in range(len(self.farms)):
            start, stop = w*i, w*(i+1)
            fitness[start:stop] = self.harvest(i)
        return fitness
                
    
    def breed(self, fitness):
        print("BEST ACCURACY:")
        print(np.max(fitness[:,:,0]))
        
        top_dog_index = np.argmax(fitness[:,:,0])
        top_dog = self.world.__getitem__(divmod(top_dog_index, self.world_shape[1]))
        print("TOP DOG")
        #print(top_dog_index, top_dog)
        print(ReprFromMatrix(top_dog))
        #TODO viability
        if self.passports is not None:
            passport = dict()
        else:
            passport = None
        world = match_maker(self.world, fitness, passport=passport)
        if self.passports is not None:
            self.passports.append(passport)
        return world
    
    def shutdown(self):
        for p, in_que, out_que in self.farms:
            in_que.close()
            out_que.close()
            p.join(1.0)
            if p.is_alive():
                p.terminate()
                p.join()
        self.farms = list()
    
    def __del__(self):
        self.shutdown()

    def evolve(self, generations):
        if not len(self.farms):
            self.farms = self.create_local_farms()
        for i in range(generations):
            self.fitness = self.tick()
            self.world = self.breed(self.fitness)