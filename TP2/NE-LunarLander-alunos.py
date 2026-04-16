import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue

# CONFIG
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
TEST_EPISODES = 1000
STEPS = 500

NUM_PROCESSES = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue = Queue()


nInputs = 8
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100
PROB_CROSSOVER = 0.9

PROB_MUTATION = 1.0/GENOTYPE_SIZE
STD_DEV = 0.1


ELITE_SIZE = 1

def network(shape, observation,ind):
    #Computes the output of the neural network given the observation and the genotype
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y)
    return x

def check_successful_landing(observation):
    #Checks the success of the landing based on the observation
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        return True
    return False

def objective_function(observation_history):
    # Trajectory-aware fitness: reward stable approach and good final touchdown state.
    final_obs = observation_history[-1]
    success = check_successful_landing(final_obs)

    cumulative_penalty = 0.0
    for obs in observation_history[:-1]:
        x = obs[0]
        y = obs[1]
        vx = obs[2]
        vy = obs[3]
        theta = obs[4]
        omega = obs[5]

        # Keep the lander close to center and descend in a controlled attitude.
        cumulative_penalty += (
            0.60 * abs(x)
            + 0.20 * abs(y)
            + 0.35 * abs(vx)
            + 0.50 * max(0.0, -vy)
            + 0.35 * abs(theta)
            + 0.20 * abs(omega)
        )

    avg_penalty = cumulative_penalty / max(1, len(observation_history) - 1)

    x = final_obs[0]
    y = final_obs[1]
    vx = final_obs[2]
    vy = final_obs[3]
    theta = final_obs[4]
    omega = final_obs[5]
    leg_contacts = final_obs[6] + final_obs[7]

    final_state_penalty = (
        2.00 * abs(x)
        + 0.70 * abs(y)
        + 1.00 * abs(vx)
        + 1.40 * max(0.0, -vy)
        + 1.00 * abs(theta)
        + 0.60 * abs(omega)
    )

    # Reward touching with both legs and strongly reward successful landings.
    contact_bonus = 8.0 * leg_contacts
    landing_bonus = 150.0 if success else 0.0

    fitness = landing_bonus + contact_bonus - final_state_penalty - avg_penalty
    return fitness, success

def simulate(genotype, render_mode = None, seed=None, env = None):
    #Simulates an episode of Lunar Lander, evaluating an individual
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
        
    observation, info = env.reset(seed=seed)

    observation_history = [observation]
    for _ in range(STEPS):
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        
        observation_history.append(observation)

        if terminated == True or truncated == True:
            break
    
    if env_was_none:    
        env.close()

    return objective_function(observation_history)

def evaluate(evaluationQueue, evaluatedQueue):
    #Evaluates individuals until it receives None
    #This function runs on multiple processes
    
    env = gym.make("LunarLander-v3", render_mode =None, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
    while True:
        ind = evaluationQueue.get()

        if ind is None:
            break
            
        ind['fitness'] = simulate(ind['genotype'], seed = None, env = env)[0]
                
        evaluatedQueue.put(ind)
    env.close()
    
def evaluate_population(population):
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop

def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1
        
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

def parent_selection(population):
    # Tournament selection: sample a few individuals and pick the fittest.
    tournament_size = min(5, len(population))
    candidates = random.sample(population, tournament_size)
    parent = max(candidates, key=lambda ind: ind['fitness'])
    return copy.deepcopy(parent)

def crossover(p1, p2):
    # One-point crossover on the genotype vectors.
    g1 = p1['genotype']
    g2 = p2['genotype']

    if len(g1) <= 1:
        child_genotype = g1[:]
    else:
        cut = random.randint(1, len(g1) - 1)
        child_genotype = g1[:cut] + g2[cut:]

    return {'genotype': child_genotype, 'fitness': None}

def mutation(p):
    # Gaussian mutation applied independently to each gene.
    genotype = p['genotype'][:]
    for i in range(len(genotype)):
        if random.random() < PROB_MUTATION:
            genotype[i] += random.gauss(0.0, STD_DEV)
            genotype[i] = max(-1.0, min(1.0, genotype[i]))

    return {'genotype': genotype, 'fitness': None}
    
def survival_selection(population, offspring):
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution():
    #Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    #Create initial population
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        
        #create offspring
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)

            else:
                ni = parent_selection(population)
                
            ni = mutation(ni)
            offspring.append(ni)
            
        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)
        
        #Print and save the best of the current generation
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        print(f'Best of generation {gen}: {best[1]}')

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    #Return the list of bests
    return bests

def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests

if __name__ == '__main__':

    #Pick a setting from below
    #--to evolve the controller--    
    evolve = True
    render_mode = None

    #--to test the evolved controller without visualisation--
    #evolve = False
    #render_mode = None

    #--to test the evolved controller with visualisation--
    #evolve = False
    #render_mode = 'human'
    
    
    if evolve:
        #evolve individuals
        n_runs = 5
        seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]
        for i in range(n_runs):    
            random.seed(seeds[i])
            bests = evolution()
            with open(f'log{i}.txt', 'w') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')

                
    else:
        #test evolved individuals
        #pick the file to test
        filename = 'log0.txt'
        bests = load_bests(filename)
        b = bests[-1]
        SHAPE = b[1]
        ind = b[2]
            
        ind = {'genotype': ind, 'fitness': None}
            
            
        ntests = TEST_EPISODES

        fit, success = 0, 0
        for i in range(1,ntests+1):
            f, s = simulate(ind['genotype'], render_mode=render_mode, seed = None)
            fit += f
            success += s
        print(fit/ntests, success/ntests)
