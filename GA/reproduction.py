import random as rnd
import numpy as np
from scipy.ndimage import generic_filter
import scipy
import itertools


def match_maker(genomes, fitness, target_pop=None, viability=lambda x: True, passport=None):
    if target_pop is None:
        target_pop = genomes.shape[0:2]
    _index_shape = list(target_pop)
    #_index_shape.reverse()
    n_categories = fitness.shape[-1]
    n_dims = fitness.ndim
   
    #all normalization is local
    #do spatial combat & marriage
    #turnover kills off localized low fittness 
    # 3x3 -> is center * alpha > median = if true we survive (compute all medians first)
    # on empty -> randomly select mating from neighboring sectors
    footprint = [
        [[0,0], [1,1], [0,0]],
        [[1,1], [1,1], [1,1]],
        [[0,0], [1,1], [0,0]],
    ]
    #print("FITNESS:")
    #print(fitness)
    def relative_survival(a, **kwargs):
        ptp_norm = lambda x: (x - x.min(0)) / x.ptp(0)
        a.shape = (int(a.size/2),2)
        mid_index = int(a.shape[0]/2) + 1
        acc = a[:,1]
        tim = a[:,0]
        acc_norm = ptp_norm(acc)
        tim_norm = ptp_norm(tim)
        return acc_norm[mid_index] + tim_norm[mid_index]/3 - 2./3 # 1/2+1/(2*3)
        acc_md = np.median(acc)
        tim_md = np.median(tim)
        return (acc[mid_index] - acc_md) + (tim[mnd_index] - tim_md)*(acc_md/tim_md)
    survival_scores = generic_filter(fitness, relative_survival, footprint=footprint, mode='wrap')    
    survival_scores = survival_scores[:,:,::2]
    survival_scores = np.nan_to_num(survival_scores)
    #print("SURVIVAL_SCORES:")
    #print(survival_scores)
    
    #for all survival scores < 0 => reproduce
    indexes = itertools.product(*(range(size) for size in _index_shape))
    next_generation = np.zeros(genomes.shape)
    getItem = lambda m, c: m.__getitem__(c)
    for cords in indexes:
        v = getItem(survival_scores, cords)
        if v < 0:
            n_cords = list(neighbor_cords(cords, _index_shape))
            #print("neighbor cords:", cords, '=>', n_cords, target_pop)
            n_s = np.array([getItem(survival_scores, nc) for nc in n_cords])
            #print("neighbor scores:", n_s)
            #roullette with the neighbors
            parent_cords = list(roullette(n_cords, n_s, k=2))
            pair = list(map(lambda c: getItem(genomes, c), parent_cords))
            #TODO passport: t,x,y = [i,j], [k,l] #tracks genetic heritage, can answer where's the DNA from?
            if passport is not None:
                passport[cords] = parent_cords
            #TODO compute stress
            stress = rnd.uniform(0, 2)
            g = offspring(pair, stress=stress)
            #print("New Offspring:", g)
        else:
            g = getItem(genomes, cords)
        next_generation.__setitem__(cords, g)
    return next_generation

def roullette(labels, scores, k=1):
    '''
    labels = 1D array of samples
    scores = 1D array of fitness scores
    k = sample size
    '''
    labels = list(labels)
    scores = np.nan_to_num(scores)
    for i in range(k):
        roll = rnd.uniform(0, scores.max(0))
        index = 0
        while roll > 0 and index < len(scores):
            if scores[index] > 0:
                roll -= scores[index]
            if roll <= 0:
                break
            index += 1
        if index >= len(labels):
            print("roullette is broken, did not consome roll")
            yield rnd.choice(labels)
        else:
            yield labels[index]
            if len(labels):
                labels.pop(index)
                scores = np.concatenate([scores[:index], scores[index+1:]])

def neighbor_cords(cords, shape):
    for i in range(len(cords)):
        c = list(cords)
        c[i] += 1
        if c[i] >= shape[i]:
            c[i] = 0
        yield tuple(c)
        
        d = list(cords)
        d[i] -= 1
        if d[i] < 0:
            d[i] = shape[i]-1
        yield tuple(d)


def offspring(genomes, stress=1.0):
    genome = np.array(list(mingle_dna(genomes)), dtype=genomes[0].dtype)
    genome = mutate(genome, stress=stress)
    return genome


def mutate(genome, stress=1.0):
    #more stress, more mutations
    shape = genome.shape
    n_mutations = rnd.uniform(0, stress*3)
    bad_luck, n_mutations = n_mutations - int(n_mutations), int(n_mutations)
    for roll_n in range(n_mutations):
        roll = int(rnd.uniform(0, 20) + bad_luck)
        if roll < 4:
            genome = random_change(genome, rate=bad_luck)
        elif roll < 8:
            genome = random_insert(genome)
        elif roll < 12:
            genome = random_delete(genome)
        elif roll < 16:
            genome = random_swap(genome)
        else:
            genome = random_copy_and_paste(genome)
    if shape != genome.shape:
        genome = genome[0:shape[0], 0:shape[1]]
    genome = np.nan_to_num(genome)
    return genome

def mingle_dna(genomes):
    for expressions in zip(*genomes):
        yield rnd.choice(expressions)
        
def random_change(genome, rate=.2):
    changes = scipy.sparse.random(*genome.shape, density=rate)
    genome = genome + (changes * np.finfo(np.float64).max)
    return genome
            
def random_insert(genome):
    #insert works on all but the last one
    index = rnd.randint(0, len(genome)-1)
    new_row = np.random.uniform(size=genome.shape[1:])
    genome = np.insert(genome, index, new_row, axis=0)
    return genome

def random_delete(genome):
    index = rnd.randint(0, len(genome)-1)
    genome[index:-1] = genome[index+1:]
    genome[len(genome)-1] = np.zeros(genome[-1].shape, dtype=genome.dtype)
    return genome

def random_swap(genome, max_length=4):
    if len(genome) < 3:
        return np.flip(genome, 0)
    from_p = rnd.randint(0, len(genome)-1)
    to_p = rnd.randint(0, len(genome)-1)
    datum = genome[from_p]
    genome[from_p] = genome[to_p]
    genome[to_p] = datum
    return genome

def random_copy_and_paste(genome, max_length=4):
    if len(genome) < 3:
        return np.flip(genome, 0)
    index = rnd.randint(0, len(genome)-2)
    length = rnd.randint(1, min(len(genome)-index-1, max_length))
    target = rnd.randint(0, len(genome)-length)
    genome[target:target+length] = genome[index:index+length]
    return genome
