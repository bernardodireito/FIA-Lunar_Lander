"""Neuroevolution controller for Gymnasium LunarLander-v3.

High-level flow:
- define network/controller
- define fitness + simulation
- run the evolutionary loop and logging
- CLI entrypoint for train/test
"""

import argparse
import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue

# -----------------------------------------------------------------------------
# Config and experiment settings
# -----------------------------------------------------------------------------
# CONFIG
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
TEST_EPISODES = 1000
EVALUATION_EPISODES = int(os.environ.get('EVALUATION_EPISODES', 20))
EVALUATION_SEED_OFFSET = int(os.environ.get('EVALUATION_SEED_OFFSET', 0))
USE_FIXED_EVALUATION_SEEDS = os.environ.get('USE_FIXED_EVALUATION_SEEDS', '0').lower() in ('1', 'true', 'yes')
STEPS = 500
USE_ACTION_ASSIST = os.environ.get('USE_ACTION_ASSIST', '0').lower() in ('1', 'true', 'yes')
SUCCESS_BONUS_PER_EPISODE = float(os.environ.get('SUCCESS_BONUS_PER_EPISODE', 500.0))
CONSISTENCY_STD_PENALTY = float(os.environ.get('CONSISTENCY_STD_PENALTY', 0.12))

NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', os.cpu_count() or 1))
evaluationQueue = Queue()
evaluatedQueue = Queue()


nInputs = 9
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100
PROB_CROSSOVER = float(os.environ.get('PROB_CROSSOVER', 0.9))

# Valor base do enunciado para a experiencia 1/3/5/7.
# Para as experiencias 2/4/6/8, trocar para 0.05.
PROB_MUTATION = float(os.environ.get('PROB_MUTATION', 1.0/GENOTYPE_SIZE))
STD_DEV = float(os.environ.get('STD_DEV', 0.1))


ELITE_SIZE = int(os.environ.get('ELITE_SIZE', 1))
LOG_ROOT = os.environ.get('LOG_ROOT', 'logs')

EXPERIMENTS = [
    {'id': 1, 'mutation': 0.008, 'crossover': 0.5, 'elite': 0},
    {'id': 2, 'mutation': 0.05,  'crossover': 0.5, 'elite': 0},
    {'id': 3, 'mutation': 0.008, 'crossover': 0.9, 'elite': 0},
    {'id': 4, 'mutation': 0.05,  'crossover': 0.9, 'elite': 0},
    {'id': 5, 'mutation': 0.008, 'crossover': 0.5, 'elite': 1},
    {'id': 6, 'mutation': 0.05,  'crossover': 0.5, 'elite': 1},
    {'id': 7, 'mutation': 0.008, 'crossover': 0.9, 'elite': 1},
    {'id': 8, 'mutation': 0.05,  'crossover': 0.9, 'elite': 1},
]

# -----------------------------------------------------------------------------
# Neural network controller
# -----------------------------------------------------------------------------
def network(shape, observation,ind):
    """Forward pass of the fixed-topology tanh network."""
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
    """Returns the action chosen by the network, with optional assist."""
    # Centraliza a escolha da acao num so sitio.
    # A rede continua a ser a parte principal do controlador, mas juntamos uma
    # pequena correcao proporcional para evitar quedas muito rapidas e desvios
    # laterais que ja nao dao tempo de recuperar.
    action = np.array(network(shape, observation, genotype), dtype=float)

    if not USE_ACTION_ASSIST:
        return action

    x = observation[0]
    y = observation[1]
    vx = observation[2]
    vy = observation[3]
    theta = observation[4]
    vtheta = observation[5]

    # Motor principal: se a velocidade vertical esta demasiado negativa,
    # aumenta a potencia. Perto do chao esta correcao fica mais forte.
    falling_too_fast = max(0.0, -vy - 0.22)
    low_altitude = max(0.0, 0.65 - y)
    desired_main = 0.15 + 1.15 * falling_too_fast + 0.45 * low_altitude * falling_too_fast
    desired_main -= 0.20 * abs(theta)
    desired_main = np.clip(desired_main, -1.0, 1.0)

    # Motor lateral: se esta a direita ou a mover-se para a direita, aplica
    # acao negativa; se esta a esquerda ou a mover-se para a esquerda, aplica
    # acao positiva. Isto contraria o movimento e tenta recentrar a nave cedo.
    desired_side = -(1.35 * x + 0.95 * vx + 0.25 * theta + 0.12 * vtheta)
    desired_side = np.clip(desired_side, -1.0, 1.0)

    # Mistura a decisao evoluida com a correcao. A rede ainda pode adaptar-se,
    # mas deixa de poder ignorar completamente a travagem e o recentramento.
    action[0] = 0.45 * action[0] + 0.55 * desired_main
    action[1] = 0.55 * action[1] + 0.45 * desired_side

    return np.clip(action, -1.0, 1.0)

def check_successful_landing(observation):
    """Checks if the final observation meets landing success criteria."""
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

# -----------------------------------------------------------------------------
# Fitness function
# -----------------------------------------------------------------------------
def objective_function(observation_history):
    """Compute fitness and success flag from the episode observations."""
    # Esta funcao da uma pontuacao (fitness) ao individuo.
    # Quanto maior for o fitness, melhor foi o comportamento da nave.
    # Usamos a ultima observacao porque e nela que o ambiente regista o
    # resultado final depois de a nave tocar no chao ou terminar o episodio.
    final_observation = observation_history[-1]
    recent_observations = observation_history[-50:]
    approach_observations = observation_history[-100:]

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
    mean_outside_pad = np.mean([max(0.0, abs(obs[0]) - 0.2) for obs in approach_observations])
    mean_outward_drift = np.mean([max(0.0, obs[0] * obs[2]) for obs in approach_observations])
    mean_offcenter_fast_fall = np.mean([
        abs(obs[0]) * max(0.0, -obs[3] - 0.20)
        for obs in approach_observations
    ])
    mean_low_altitude_lateral_error = np.mean([
        max(0.0, 0.45 - obs[1]) * abs(obs[0])
        for obs in approach_observations
    ])
    mean_low_altitude_fast_fall = np.mean([
        max(0.0, 0.40 - obs[1]) * max(0.0, -obs[3] - 0.18)
        for obs in approach_observations
    ])
    mean_low_altitude_side_speed = np.mean([
        max(0.0, 0.40 - obs[1]) * abs(obs[2])
        for obs in approach_observations
    ])
    mean_low_altitude_tilt = np.mean([
        max(0.0, 0.40 - obs[1]) * abs(obs[4])
        for obs in approach_observations
    ])

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
    fitness -= 300.0 * abs(vy)

    # Penaliza estar inclinada ou a rodar muito.
    fitness -= 160.0 * abs(theta)
    fitness -= 30.0 * abs(vtheta)

    # Penaliza uma aproximacao final instavel. Estes termos usam varios passos
    # recentes, nao apenas o ultimo frame.
    fitness -= 110.0 * mean_abs_x
    fitness -= 70.0 * mean_abs_vx
    fitness -= 60.0 * mean_abs_theta
    fitness -= 260.0 * mean_fast_fall

    # Penaliza situacoes em que a nave ja vem sem margem para corrigir:
    # afastada do centro, a descer depressa, ou ainda a mover-se para fora.
    fitness -= 300.0 * mean_outside_pad
    fitness -= 420.0 * mean_outward_drift
    fitness -= 500.0 * mean_offcenter_fast_fall
    fitness -= 160.0 * mean_low_altitude_lateral_error
    fitness -= 520.0 * mean_low_altitude_fast_fall
    fitness -= 150.0 * mean_low_altitude_side_speed
    fitness -= 140.0 * mean_low_altitude_tilt

    # No estado final, estar fora da plataforma e ainda ter velocidade para
    # fora e especialmente mau: e exatamente o caso em que ja nao ha tempo
    # para recuperar a trajetoria.
    final_outward_drift = max(0.0, x * vx)
    if not on_landing_pad:
        fitness -= 700.0 * final_outward_drift

    # Da premios por cumprir partes da aterragem, mas apenas quando fazem
    # sentido. O debugger mostrou que premiar "estar lento" fora da plataforma
    # ainda deixava aterragens erradas com bom fitness.
    if on_landing_pad:
        fitness += 90.0
        if abs(x) <= 0.10:
            fitness += 110.0
        if abs(vx) <= 0.15:
            fitness += 90.0
        if slow_vertical_speed:
            fitness += 110.0
        if upright:
            fitness += 100.0
        if legs_touching and slow_vertical_speed and upright:
            fitness += 180.0
        elif legs_touching:
            fitness -= 120.0
    elif legs_touching:
        # Tocar com as duas pernas fora da plataforma nao deve parecer uma boa
        # solucao para o algoritmo evolucionario.
        fitness -= 500.0

    # Aterragens falhadas precisam de parecer claramente piores do que uma
    # boa aproximacao. Isto reduz a tentacao de "raspar" o chao so para
    # acumular alguns bonus locais e ajuda a diminuir overfitting.
    if not successful_landing:
        if not legs_touching:
            fitness -= 320.0
        if not on_landing_pad:
            fitness -= 260.0 + 500.0 * max(0.0, abs(x) - 0.2)
        if not slow_vertical_speed:
            fitness -= 550.0 * max(0.0, -vy - 0.2)
        if not upright:
            fitness -= 180.0 * abs(theta)

    # Grande bonus se a aterragem for considerada bem sucedida.
    # Isto ajuda a evolucao a preferir claramente individuos que aterram.
    if successful_landing:
        fitness += 1200.0

    # A funcao devolve duas coisas:
    # 1) o fitness usado pelo algoritmo evolucionario;
    # 2) True/False a dizer se a aterragem foi bem sucedida.
    return fitness, successful_landing

# -----------------------------------------------------------------------------
# Simulation and evaluation
# -----------------------------------------------------------------------------
def simulate(genotype, render_mode = None, seed=None, env = None):
    """Run one episode and return (fitness, success)."""
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
    """Worker process: evaluate individuals sent through the queue."""
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
        episode_fitnesses = []
        successes = 0
        for episode in range(EVALUATION_EPISODES):
            # Por defeito usamos episodios aleatorios para evitar overfitting a
            # um pequeno conjunto de seeds. Se quiseres experimentar seeds fixas:
            # USE_FIXED_EVALUATION_SEEDS=1 EVALUATION_SEED_OFFSET=0
            seed = None
            if USE_FIXED_EVALUATION_SEEDS:
                seed = EVALUATION_SEED_OFFSET + episode
            episode_fitness, success = simulate(ind['genotype'], seed = seed, env = env)
            episode_fitnesses.append(episode_fitness)
            successes += int(success)

        mean_fitness = float(np.mean(episode_fitnesses))
        std_fitness = float(np.std(episode_fitnesses))
        success_rate = successes / EVALUATION_EPISODES

        ind['fitness'] = (
            mean_fitness
            + SUCCESS_BONUS_PER_EPISODE * success_rate
            - CONSISTENCY_STD_PENALTY * std_fitness
        )
                
        evaluatedQueue.put(ind)
    env.close()
    
def evaluate_population(population):
    """Evaluate a population using the worker processes."""
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop

# -----------------------------------------------------------------------------
# Genetic operators
# -----------------------------------------------------------------------------
def generate_initial_population():
    """Create the initial population with random genotypes."""
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
    """Tournament selection with deep copy of the winner."""
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
    """Arithmetic crossover across all genes."""
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
    """Gaussian mutation with clipping to keep weights bounded."""
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
    """Elitist survivor selection using current elite and best offspring."""
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
    """Main evolutionary loop that returns the bests per generation."""
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

# -----------------------------------------------------------------------------
# Logging and experiments
# -----------------------------------------------------------------------------
def load_bests(fname):
    """Load best individuals from a log file."""
    #Load bests from file
    bests = []
    with open(resolve_log_path(fname), 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests

def experiment_log_dir(experiment_id, log_root=LOG_ROOT):
    return os.path.join(log_root, f'log_exp{experiment_id}')

def experiment_log_path(experiment_id, run_index, log_prefix='log', log_root=LOG_ROOT):
    filename = f'{log_prefix}_exp{experiment_id}_run{run_index}.txt'
    return os.path.join(experiment_log_dir(experiment_id, log_root), filename)

def resolve_log_path(fname):
    if os.path.exists(fname):
        return fname

    basename = os.path.basename(fname)
    for root, _, files in os.walk(LOG_ROOT):
        if basename in files:
            return os.path.join(root, basename)

    return fname

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    """Parse CLI arguments for training/testing runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--evolve', action='store_true', help='Treina controladores.')
    parser.add_argument('--test', action='store_true', help='Testa um controlador ja treinado.')
    parser.add_argument('--all-experiments', action='store_true', help='Corre as 8 experiencias da tabela.')
    parser.add_argument('--experiment-id', type=int, choices=range(1, 9), help='Corre apenas uma experiencia da tabela.')
    parser.add_argument('--runs', type=int, default=None, help='Numero de runs por experiencia.')
    parser.add_argument('--run-index', type=int, default=None, help='Corre apenas uma run especifica.')
    parser.add_argument('--log-prefix', default=None, help='Prefixo dos ficheiros de log.')
    parser.add_argument('--log-root', default=None, help='Pasta base dos logs.')
    parser.add_argument('--overwrite-logs', action='store_true', help='Volta a treinar mesmo que o log ja exista.')
    parser.add_argument('--num-processes', type=int, default=None, help='Processos usados para avaliar individuos.')
    parser.add_argument('--generations', type=int, default=None, help='Numero de geracoes.')
    parser.add_argument('--population-size', type=int, default=None, help='Tamanho da populacao.')
    parser.add_argument('--evaluation-episodes', type=int, default=None, help='Episodios usados para avaliar cada individuo durante o treino.')
    parser.add_argument('--test-log', default=None, help='Log a testar.')
    parser.add_argument('--test-episodes', type=int, default=None, help='Episodios usados no teste.')
    parser.add_argument('--render-mode', default=None, help='Render mode do Gymnasium, por exemplo human.')
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()

    #Pick a setting from below
    #--to evolve the controller--    
    #evolve = True
    #render_mode = None

    #--to test the evolved controller without visualisation--
    evolve = False
    render_mode = None

    # Permite correr experiencias sem editar o ficheiro, a partir do terminal:
    # EVOLVE=1 python NE-LunarLander-alunos.py
    # EVOLVE=1 RUN_ALL_EXPERIMENTS=0 EXPERIMENT_ID=1 N_RUNS=1 python NE-LunarLander-alunos.py
    evolve = os.environ.get('EVOLVE', str(evolve)).lower() in ('1', 'true', 'yes')
    if args.evolve:
        evolve = True
    if args.test:
        evolve = False

    render_mode = os.environ.get('RENDER_MODE', render_mode)
    if args.render_mode is not None:
        render_mode = args.render_mode

    if args.num_processes is not None:
        NUM_PROCESSES = args.num_processes
    if args.generations is not None:
        NUMBER_OF_GENERATIONS = args.generations
    if args.population_size is not None:
        POPULATION_SIZE = args.population_size
    if args.evaluation_episodes is not None:
        EVALUATION_EPISODES = args.evaluation_episodes
    if args.log_root is not None:
        LOG_ROOT = args.log_root

    #--to test the evolved controller with visualisation--
    #evolve = False
    #render_mode = 'human'
    
    
    if evolve:
        #evolve individuals
        n_runs = args.runs if args.runs is not None else int(os.environ.get('N_RUNS', 5))
        log_prefix = args.log_prefix if args.log_prefix is not None else os.environ.get('LOG_PREFIX', 'log')
        log_root = args.log_root if args.log_root is not None else LOG_ROOT
        run_all_experiments = os.environ.get('RUN_ALL_EXPERIMENTS', '1').lower() in ('1', 'true', 'yes')
        if args.all_experiments:
            run_all_experiments = True
        if args.experiment_id is not None:
            run_all_experiments = False
        experiment_id = args.experiment_id if args.experiment_id is not None else os.environ.get('EXPERIMENT_ID', None)
        seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]

        if args.run_index is not None and args.run_index >= len(seeds):
            raise ValueError(f'Run invalida: {args.run_index}. Maximo: {len(seeds) - 1}')
        if n_runs > len(seeds):
            raise ValueError(f'N_RUNS demasiado grande: {n_runs}. Maximo: {len(seeds)}')

        if run_all_experiments:
            experiments_to_run = EXPERIMENTS
        elif experiment_id is not None:
            experiments_to_run = [exp for exp in EXPERIMENTS if exp['id'] == int(experiment_id)]
            if len(experiments_to_run) == 0:
                raise ValueError(f'Experiencia invalida: {experiment_id}')
        else:
            experiments_to_run = [{
                'id': 'custom',
                'mutation': PROB_MUTATION,
                'crossover': PROB_CROSSOVER,
                'elite': ELITE_SIZE,
            }]

        for experiment in experiments_to_run:
            PROB_MUTATION = experiment['mutation']
            PROB_CROSSOVER = experiment['crossover']
            ELITE_SIZE = experiment['elite']

            print(
                f"Experiencia {experiment['id']}: "
                f"mutacao={PROB_MUTATION}, "
                f"crossover={PROB_CROSSOVER}, "
                f"elitismo={ELITE_SIZE}, "
                f"populacao={POPULATION_SIZE}, "
                f"geracoes={NUMBER_OF_GENERATIONS}, "
                f"processos={NUM_PROCESSES}, "
                f"episodios_avaliacao={EVALUATION_EPISODES}"
            )

            run_indexes = [args.run_index] if args.run_index is not None else range(n_runs)
            for i in run_indexes:
                log_path = experiment_log_path(experiment['id'], i, log_prefix, log_root)
                if os.path.exists(log_path) and not args.overwrite_logs:
                    print(f"Run {i + 1}/{n_runs} da experiencia {experiment['id']} ja existe: {log_path}")
                    continue

                print(f"Run {i + 1}/{n_runs} da experiencia {experiment['id']}")
                random.seed(seeds[i])
                bests = evolution()
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'w') as f:
                    for b in bests:
                        f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')
                print(f"Log guardado em {log_path}")

                
    else:
        #test evolved individuals
        #pick the file to test
        filename = args.test_log if args.test_log is not None else os.environ.get('TEST_LOG', 'log1.txt')
        bests = load_bests(filename)
        # O ultimo individuo guardado nem sempre e o melhor de todos.
        # Como a avaliacao tem aleatoriedade, escolhemos o melhor fitness
        # registado no ficheiro de log.
        b = max(bests, key=lambda x: x[0])
        SHAPE = b[1]
        ind = b[2]
            
        ind = {'genotype': ind, 'fitness': None}
            
            
        ntests = args.test_episodes if args.test_episodes is not None else TEST_EPISODES

        fit, success = 0, 0
        for i in range(1,ntests+1):
            f, s = simulate(ind['genotype'], render_mode=render_mode, seed = None)
            fit += f
            success += s
        print(fit/ntests, success/ntests)
