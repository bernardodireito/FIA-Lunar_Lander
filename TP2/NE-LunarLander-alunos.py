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
EVALUATION_EPISODES = int(os.environ.get('EVALUATION_EPISODES', 10))
EVALUATION_SEED_OFFSET = int(os.environ.get('EVALUATION_SEED_OFFSET', 0))
USE_FIXED_EVALUATION_SEEDS = os.environ.get('USE_FIXED_EVALUATION_SEEDS', '0').lower() in ('1', 'true', 'yes')
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

# Valor base do enunciado para a experiencia 1/3/5/7.
# Para as experiencias 2/4/6/8, trocar para 0.05.
PROB_MUTATION = float(os.environ.get('PROB_MUTATION', 1.0/GENOTYPE_SIZE))
STD_DEV = float(os.environ.get('STD_DEV', 0.1))


ELITE_SIZE = 1

def network(shape, observation,ind):
    # Computes the output of the neural network given the observation and the genotype.
    # O genotype tem todos os pesos da rede numa unica lista.
    # Como ha pesos para varias camadas, usamos weight_index para saber
    # em que parte dessa lista estamos.
    x = observation[:]

    # Entrada constante que funciona como bias.
    # Sem isto, quando as observacoes estao perto de zero, a rede tende a
    # produzir acoes perto de zero. No Lunar Lander isso pode deixar o motor
    # principal desligado precisamente quando era preciso travar a descida.
    if len(x) < shape[0]:
        x = np.append(x, 1.0)

    weight_index = 0
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[weight_index + k+j*len(x)]
        weight_index += len(x) * shape[i]
        x = np.tanh(y)
    return x

def controller_action(shape, observation, genotype):
    # Centraliza a escolha da acao num so sitio.
    # Neste momento usamos diretamente a saida da rede, que ja esta em [-1, 1].
    return network(shape, observation, genotype)

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
    # Esta funcao da uma pontuacao (fitness) ao individuo.
    # Quanto maior for o fitness, melhor foi o comportamento da nave.
    # Usamos a ultima observacao porque e nela que o ambiente regista o
    # resultado final depois de a nave tocar no chao ou terminar o episodio.
    final_observation = observation_history[-1]
    recent_observations = observation_history[-50:]

    # Cada posicao do vetor de observacao representa uma informacao da nave.
    # x/y: posicao; vx/vy: velocidade; theta/vtheta: angulo e rotacao.
    x = final_observation[0]
    y = final_observation[1]
    vx = final_observation[2]
    vy = final_observation[3]
    theta = final_observation[4]
    vtheta = final_observation[5]
    contact_left = final_observation[6]
    contact_right = final_observation[7]

    # Verifica se a nave cumpriu as condicoes formais de aterragem com sucesso.
    successful_landing = check_successful_landing(final_observation)
    legs_touching = contact_left == 1 and contact_right == 1
    on_landing_pad = abs(x) <= 0.2
    slow_vertical_speed = vy > -0.2
    upright = abs(theta) < np.deg2rad(20)

    # Para alem do ultimo estado, tambem olhamos para a parte final da
    # trajetoria. Isto ajuda a distinguir uma aproximacao controlada de uma
    # queda que por acaso termina perto da plataforma.
    mean_abs_x = np.mean([abs(obs[0]) for obs in recent_observations])
    mean_abs_vx = np.mean([abs(obs[2]) for obs in recent_observations])
    mean_abs_theta = np.mean([abs(obs[4]) for obs in recent_observations])
    mean_fast_fall = np.mean([max(0.0, -obs[3] - 0.25) for obs in recent_observations])

    # Comecamos a pontuacao em zero.
    # Depois retiramos pontos por comportamentos maus e damos pontos por
    # comportamentos bons. Penalizar abs(valor) significa que tanto valores
    # negativos como positivos sao maus quando queremos estar perto de zero.
    fitness = 0.0

    # Penaliza fortemente estar longe da plataforma.
    # O debugger mostrou que muitos individuos aterravam fora da zona correta.
    fitness -= 250.0 * abs(x)

    # Penaliza ainda mais quando passa para fora da margem da plataforma.
    if not on_landing_pad:
        fitness -= 600.0 * (abs(x) - 0.2)

    # Penaliza estar longe do solo, mas menos do que estar longe do centro.
    fitness -= 40.0 * abs(y)

    # Penaliza velocidades altas. Para aterrar bem, a nave deve chegar devagar.
    fitness -= 120.0 * abs(vx)
    fitness -= 220.0 * abs(vy)

    # Penaliza estar inclinada ou a rodar muito.
    fitness -= 160.0 * abs(theta)
    fitness -= 30.0 * abs(vtheta)

    # Penaliza uma aproximacao final instavel. Estes termos usam varios passos
    # recentes, nao apenas o ultimo frame.
    fitness -= 80.0 * mean_abs_x
    fitness -= 50.0 * mean_abs_vx
    fitness -= 60.0 * mean_abs_theta
    fitness -= 180.0 * mean_fast_fall

    # Da premios por cumprir partes da aterragem, mas apenas quando fazem
    # sentido. O debugger mostrou que premiar "estar lento" fora da plataforma
    # ainda deixava aterragens erradas com bom fitness.
    if on_landing_pad:
        fitness += 150.0
        if slow_vertical_speed:
            fitness += 150.0
        if upright:
            fitness += 150.0
        if legs_touching:
            fitness += 250.0
    elif legs_touching:
        # Tocar com as duas pernas fora da plataforma nao deve parecer uma boa
        # solucao para o algoritmo evolucionario.
        fitness -= 300.0

    # Grande bonus se a aterragem for considerada bem sucedida.
    # Isto ajuda a evolucao a preferir claramente individuos que aterram.
    if successful_landing:
        fitness += 2000.0

    # A funcao devolve duas coisas:
    # 1) o fitness usado pelo algoritmo evolucionario;
    # 2) True/False a dizer se a aterragem foi bem sucedida.
    return fitness, successful_landing

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
        action = controller_action(SHAPE, observation, genotype)
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
            
        # Avaliamos o mesmo individuo em varios episodios e usamos a media.
        # Isto reduz o efeito de "teve sorte numa tentativa" e favorece
        # controladores que funcionam de forma mais consistente.
        fitness = 0.0
        for episode in range(EVALUATION_EPISODES):
            # Por defeito usamos episodios aleatorios para evitar overfitting a
            # um pequeno conjunto de seeds. Se quiseres experimentar seeds fixas:
            # USE_FIXED_EVALUATION_SEEDS=1 EVALUATION_SEED_OFFSET=0
            seed = None
            if USE_FIXED_EVALUATION_SEEDS:
                seed = EVALUATION_SEED_OFFSET + episode
            fitness += simulate(ind['genotype'], seed = seed, env = env)[0]
        ind['fitness'] = fitness / EVALUATION_EPISODES
                
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
    # Selecao por torneio:
    # escolhemos alguns individuos ao acaso e, entre esses, fica o melhor.
    # Assim damos vantagem a quem tem bom fitness, mas sem eliminar totalmente
    # a aleatoriedade, que e importante para manter diversidade.
    tournament_size = 3
    tournament = random.sample(population, tournament_size)
    winner = max(tournament, key=lambda x: x['fitness'])

    # Devolvemos uma copia para evitar alterar diretamente o individuo original.
    return copy.deepcopy(winner)

def crossover(p1, p2):
    # Crossover aritmetico:
    # cria um filho misturando os pesos (genes) dos dois pais.
    # Cada gene do filho fica entre o valor do gene do pai 1 e do pai 2.
    genotype = []

    # O genotype e a lista de pesos da rede neuronal.
    # gene1 e gene2 sao o mesmo peso/ligacao, mas em pais diferentes.
    for gene1, gene2 in zip(p1['genotype'], p2['genotype']):
        # Usamos um alpha diferente para cada gene. Isto gera filhos mais
        # variados do que usar a mesma mistura para a rede inteira.
        alpha = random.random()
        genotype.append(alpha * gene1 + (1.0 - alpha) * gene2)

    # O filho ainda nao foi avaliado, por isso o fitness comeca como None.
    return {'genotype': genotype, 'fitness': None}

def mutation(p):
    # Mutacao:
    # percorremos todos os pesos da rede e, com uma pequena probabilidade,
    # alteramos ligeiramente esse peso.
    p['fitness'] = None
    for i in range(len(p['genotype'])):
        if random.random() < PROB_MUTATION:
            # random.gauss(0.0, STD_DEV) gera uma pequena alteracao aleatoria.
            # A maioria das alteracoes fica perto de 0, mas algumas podem ser
            # um pouco maiores. Isto permite explorar solucoes novas.
            p['genotype'][i] += random.gauss(0.0, STD_DEV)

            # Limita os pesos para evitar valores extremos que saturam a tanh
            # e fazem a rede devolver quase sempre -1 ou 1.
            p['genotype'][i] = max(-5.0, min(5.0, p['genotype'][i]))
    return p    
    
def survival_selection(population, offspring):
    # Selecao elitista de sobreviventes:
    # mantemos os melhores individuos da geracao anterior (elite) e
    # completamos a nova populacao com os melhores filhos.
    offspring.sort(key = lambda x: x['fitness'], reverse=True)

    # A elite e reavaliada porque o Lunar Lander e estocastico:
    # o mesmo individuo pode ter resultados ligeiramente diferentes.
    p = evaluate_population(population[:ELITE_SIZE])

    # Precisamos de POPULATION_SIZE individuos no total.
    # Se ELITE_SIZE = 1, ficamos com 1 elite + 99 melhores filhos.
    number_of_offspring = POPULATION_SIZE - ELITE_SIZE
    new_population = p + offspring[:number_of_offspring]
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
    #evolve = True
    #render_mode = None

    #--to test the evolved controller without visualisation--
    evolve = False
    render_mode = None

    # Permite correr experiencias sem editar o ficheiro:
    # EVOLVE=1 N_RUNS=1 LOG_PREFIX=candidate_ python NE-LunarLander-alunos.py
    evolve = os.environ.get('EVOLVE', str(evolve)).lower() in ('1', 'true', 'yes')
    render_mode = os.environ.get('RENDER_MODE', render_mode)

    #--to test the evolved controller with visualisation--
    #evolve = False
    #render_mode = 'human'
    
    
    if evolve:
        #evolve individuals
        n_runs = int(os.environ.get('N_RUNS', 5))
        log_prefix = os.environ.get('LOG_PREFIX', 'log')
        seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]
        for i in range(n_runs):    
            random.seed(seeds[i])
            bests = evolution()
            with open(f'{log_prefix}{i}.txt', 'w') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')

                
    else:
        #test evolved individuals
        #pick the file to test
        filename = os.environ.get('TEST_LOG', 'log0.txt')
        bests = load_bests(filename)
        # O ultimo individuo guardado nem sempre e o melhor de todos.
        # Como a avaliacao tem aleatoriedade, escolhemos o melhor fitness
        # registado no ficheiro de log.
        b = max(bests, key=lambda x: x[0])
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
