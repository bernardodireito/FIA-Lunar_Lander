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

# Deve ficar desligado nos testes oficiais. Foi mantido apenas para experiencias
# de debug, porque adiciona uma correcao manual por cima da rede neuronal.
USE_ACTION_ASSIST = os.environ.get('USE_ACTION_ASSIST', '0').lower() in ('1', 'true', 'yes')
SUCCESS_BONUS_PER_EPISODE = float(os.environ.get('SUCCESS_BONUS_PER_EPISODE', 1000.0))
SUCCESS_RATE_BONUS = float(os.environ.get('SUCCESS_RATE_BONUS', 800.0))
CONSISTENCY_STD_PENALTY = float(os.environ.get('CONSISTENCY_STD_PENALTY', 0.12))

NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', os.cpu_count() or 1))


nInputs = 8
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100
PROB_CROSSOVER = float(os.environ.get('PROB_CROSSOVER', 0.9))

# Estes valores so sao usados em runs custom. Nas experiencias oficiais, os
# valores de mutacao/crossover/elitismo sao definidos pela tabela EXPERIMENTS.
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
    # O genotype tem todos os pesos da rede numa unica lista.
    # Como ha pesos para varias camadas, usamos weight_index para saber
    # em que parte dessa lista estamos.
    x = observation[:]

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
    # Quanto maior for o fitness, melhor foi o comportamento da nave.
    # A funcao esta dividida em quatro ideias simples:
    # 1) penalizacoes base por estados perigosos;
    # 2) penalizacoes no penultimo estado, para evitar impactos bruscos;
    # 3) recompensas progressivas por cumprir partes da aterragem;
    # 4) bonus por contacto sustentado e por sucesso formal.
    final_observation = observation_history[-1]
    pre_final_observation = observation_history[-2] if len(observation_history) >= 2 else final_observation
    recent_observations = observation_history[-60:]
    contact_observations = observation_history[-25:]

    x = final_observation[0]
    vx = final_observation[2]
    vy = final_observation[3]
    theta = final_observation[4]
    vtheta = final_observation[5]
    contact_left = final_observation[6]
    contact_right = final_observation[7]

    pre_x = pre_final_observation[0]
    pre_y = pre_final_observation[1]
    pre_vx = pre_final_observation[2]
    pre_vy = pre_final_observation[3]
    pre_theta = pre_final_observation[4]


    successful_landing = check_successful_landing(final_observation)
    legs_touching = contact_left == 1 and contact_right == 1
    on_landing_pad = abs(x) <= 0.2
    slow_vertical_speed = vy > -0.2
    upright = abs(theta) < np.deg2rad(20)

    def progressive_score(value, limit):
        return max(0.0, 1.0 - abs(value) / limit)

    def downward_speed_score(value, limit):
        return max(0.0, 1.0 - max(0.0, -value) / limit)

    fitness = 0.0

    # Penalizacoes base: estas mantem a nave perto do centro, lenta e direita.
    # longe do centro
    fitness -= 240.0 * abs(x)
    # andar para o lado depressa
    fitness -= 120.0 * abs(vx)
    # cair depressa
    fitness -= 280.0 * max(0.0, -vy)
    # inclinacao
    fitness -= 160.0 * abs(theta)
    # rotacao
    fitness -= 35.0 * abs(vtheta)
    # longe do centro perto do chao
    fitness -= 420.0 * max(0.0, abs(x) - 0.2)


    # Historico recente: cair depressa perto do fim deve ser mau mesmo que o
    # ultimo estado fique "limpo" depois da terminacao do ambiente.
    mean_fast_fall = np.mean([max(0.0, -obs[3] - 0.25) for obs in recent_observations])
    mean_outside_pad = np.mean([max(0.0, abs(obs[0]) - 0.2) for obs in recent_observations])
    fitness -= 360.0 * mean_fast_fall
    fitness -= 260.0 * mean_outside_pad

    # Penultimo estado: aproximações baixas ainda descentradas, inclinadas ou
    # rapidas sao a causa principal das falhas observadas no debugger/render.
    pre_low_altitude = max(0.0, 0.35 - pre_y)
    fitness -= 240.0 * pre_low_altitude * abs(pre_vx)
    fitness -= 300.0 * pre_low_altitude * max(0.0, abs(pre_x) - 0.16)
    fitness -= 360.0 * pre_low_altitude * max(0.0, -pre_vy - 0.35)
    fitness -= 140.0 * pre_low_altitude * abs(pre_theta)

    # Cinco recompensas progressivas, uma por cada condicao desejada.
    pad_score = progressive_score(x, 0.45)
    vx_score = progressive_score(vx, 0.45)
    vy_score = downward_speed_score(vy, 0.45)
    angle_score = progressive_score(theta, np.deg2rad(30))
    legs_score = 0.5 * (contact_left + contact_right)

    fitness += 160.0 * pad_score
    fitness += 120.0 * vx_score
    fitness += 170.0 * vy_score
    fitness += 130.0 * angle_score
    fitness += 180.0 * legs_score

    # Bonus combinado: so fica grande quando as cinco condicoes estao boas ao
    # mesmo tempo, evitando que o controlador maximize uma delas isoladamente.
    landing_quality = pad_score * vx_score * vy_score * angle_score * max(0.25, legs_score)
    fitness += 900.0 * landing_quality

    # Contacto sustentado: recompensa ficar pousado com ambas as pernas durante
    # varios passos, em vez de apenas tocar bem num instante final.
    sustained_both_legs = np.mean([
        1.0 if obs[6] == 1 and obs[7] == 1 else 0.0
        for obs in contact_observations
    ])
    fitness += 600.0 * sustained_both_legs * pad_score * angle_score

    # Penalizacoes discretas para falhas formais no ultimo estado.
    if not successful_landing:
        if not legs_touching:
            fitness -= 250.0
        if not on_landing_pad:
            fitness -= 320.0 + 420.0 * max(0.0, abs(x) - 0.2)
        if not slow_vertical_speed:
            fitness -= 450.0 * max(0.0, -vy - 0.2)
        if not upright:
            fitness -= 160.0 * abs(theta)

    if successful_landing:
        fitness += 1600.0

    return fitness, successful_landing

# -----------------------------------------------------------------------------
# Simulation and evaluation
# -----------------------------------------------------------------------------
def simulate(genotype, render_mode = None, seed=None, env = None):
    """Run one episode and return (fitness, success)."""
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
        
    observation, info = env.reset(seed=seed)

    observation_history = [observation]
    for _ in range(STEPS):
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
    # Cada processo reutiliza o seu ambiente para evitar criar um Gym novo em
    # todas as avaliacoes.
    
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
            + SUCCESS_RATE_BONUS * (success_rate ** 2)
            - CONSISTENCY_STD_PENALTY * std_fitness
        )
                
        evaluatedQueue.put(ind)
    env.close()
    
def evaluate_population(population, evaluation_queue, evaluated_queue):
    """Evaluate a population using the worker processes."""
    for i in range(len(population)):
        evaluation_queue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluated_queue.get()
        new_pop.append(ind)
    return new_pop

# -----------------------------------------------------------------------------
# Genetic operators
# -----------------------------------------------------------------------------
def generate_initial_population():
    """Create the initial population with random genotypes."""
    population = []
    for i in range(POPULATION_SIZE):
        # A populacao inicial tem de ser aleatoria: nao usamos controladores
        # guardados de experiencias anteriores.
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

def parent_selection(population):
    """Rank-biased parent selection with deep copy of the chosen individual."""
    # A populacao esta ordenada por fitness. Usamos selecao por ranking para dar
    # maior probabilidade aos melhores individuos sem impedir que solucoes
    # medianas contribuam para a diversidade genetica.
    selection_pressure = 2.5
    index = int((random.random() ** selection_pressure) * len(population))
    winner = population[min(index, len(population) - 1)]

    # Devolvemos uma copia para evitar alterar diretamente o individuo original.
    return copy.deepcopy(winner)

def crossover(p1, p2):
    """BLX-alpha crossover for real-valued genotypes."""
    # Crossover BLX-alpha:
    # para cada gene, sorteamos dentro do intervalo definido pelos pais e
    # permitimos uma pequena extrapolacao. Isto combina boas solucoes, mas
    # tambem explora pesos proximos que nenhum dos pais tinha exatamente.
    genotype = []
    alpha = 0.25

    # O genotype e a lista de pesos da rede neuronal.
    # gene1 e gene2 sao o mesmo peso/ligacao, mas em pais diferentes.
    for gene1, gene2 in zip(p1['genotype'], p2['genotype']):
        low = min(gene1, gene2)
        high = max(gene1, gene2)
        interval = high - low
        child_gene = random.uniform(low - alpha * interval, high + alpha * interval)
        genotype.append(max(-5.0, min(5.0, child_gene)))

    # O filho ainda nao foi avaliado, por isso o fitness comeca como None.
    return {'genotype': genotype, 'fitness': None}

def mutation(p):
    """Gaussian mutation with rare resets and clipping."""
    # Mutacao:
    # percorremos todos os pesos da rede e, com uma pequena probabilidade,
    # alteramos ligeiramente esse peso.
    p['fitness'] = None
    for i in range(len(p['genotype'])):
        if random.random() < PROB_MUTATION:
            # Na maioria dos casos fazemos uma pequena mutacao gaussiana para
            # refinar. Raramente fazemos um reset ou um passo maior para escapar
            # de zonas onde a populacao ficou pouco diversa.
            if random.random() < 0.03:
                p['genotype'][i] = random.uniform(-1.0, 1.0)
            else:
                step = STD_DEV
                if random.random() < 0.15:
                    step *= 3.0
                p['genotype'][i] += random.gauss(0.0, step)

        # Limita os pesos para evitar valores extremos que saturam a tanh e
        # fazem a rede devolver quase sempre -1 ou 1.
        p['genotype'][i] = max(-5.0, min(5.0, p['genotype'][i]))
    return p    
    
def survival_selection(population, offspring, evaluation_queue, evaluated_queue):
    """Elitist survivor selection using current elite and best offspring."""
    # Selecao elitista de sobreviventes:
    # mantemos os melhores individuos da geracao anterior (elite) e
    # completamos a nova populacao com os melhores filhos.
    offspring.sort(key = lambda x: x['fitness'], reverse=True)

    # A elite e reavaliada porque o Lunar Lander e estocastico:
    # o mesmo individuo pode ter resultados ligeiramente diferentes.
    p = evaluate_population(population[:ELITE_SIZE], evaluation_queue, evaluated_queue)

    # Precisamos de POPULATION_SIZE individuos no total.
    # O resto da populacao vem dos melhores filhos.
    number_of_offspring = POPULATION_SIZE - ELITE_SIZE
    new_population = p + offspring[:number_of_offspring]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution():
    """Main evolutionary loop that returns the bests per generation."""
    evaluation_queue = Queue()
    evaluated_queue = Queue()

    # Cria os processos que avaliam individuos em paralelo dentro desta run.
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluation_queue, evaluated_queue)))
        evaluation_processes[-1].start()

    try:
        bests = []
        population = list(generate_initial_population())
        population = evaluate_population(population, evaluation_queue, evaluated_queue)
        population.sort(key = lambda x: x['fitness'], reverse=True)
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        
        for gen in range(NUMBER_OF_GENERATIONS):
            offspring = []
            
            while len(offspring) < POPULATION_SIZE:
                if random.random() < PROB_CROSSOVER:
                    p1 = parent_selection(population)
                    p2 = parent_selection(population)
                    ni = crossover(p1, p2)

                else:
                    ni = parent_selection(population)
                    
                ni = mutation(ni)
                offspring.append(ni)
                
            offspring = evaluate_population(offspring, evaluation_queue, evaluated_queue)

            population = survival_selection(population, offspring, evaluation_queue, evaluated_queue)
            
            best = (population[0]['genotype']), population[0]['fitness']
            bests.append(best)
            print(f'Best of generation {gen}: {best[1]}')
    finally:
        # Garante que os processos fecham mesmo se houver erro durante a run.
        for i in range(NUM_PROCESSES):
            evaluation_queue.put(None)
        for p in evaluation_processes:
            p.join()

    return bests

# -----------------------------------------------------------------------------
# Logging and experiments
# -----------------------------------------------------------------------------
def load_bests(fname):
    """Load best individuals from a log file."""
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

def current_training_settings(log_root):
    """Collect CLI-adjustable training settings for child run processes."""
    return {
        'population_size': POPULATION_SIZE,
        'number_of_generations': NUMBER_OF_GENERATIONS,
        'evaluation_episodes': EVALUATION_EPISODES,
        'num_processes': NUM_PROCESSES,
        'log_root': log_root,
        'success_bonus_per_episode': SUCCESS_BONUS_PER_EPISODE,
        'success_rate_bonus': SUCCESS_RATE_BONUS,
        'consistency_std_penalty': CONSISTENCY_STD_PENALTY,
    }

def apply_training_settings(experiment, settings):
    """Apply experiment and CLI settings inside the current process."""
    global PROB_MUTATION
    global PROB_CROSSOVER
    global ELITE_SIZE
    global POPULATION_SIZE
    global NUMBER_OF_GENERATIONS
    global EVALUATION_EPISODES
    global NUM_PROCESSES
    global LOG_ROOT
    global SUCCESS_BONUS_PER_EPISODE
    global SUCCESS_RATE_BONUS
    global CONSISTENCY_STD_PENALTY

    PROB_MUTATION = experiment['mutation']
    PROB_CROSSOVER = experiment['crossover']
    ELITE_SIZE = experiment['elite']
    POPULATION_SIZE = settings['population_size']
    NUMBER_OF_GENERATIONS = settings['number_of_generations']
    EVALUATION_EPISODES = settings['evaluation_episodes']
    NUM_PROCESSES = settings['num_processes']
    LOG_ROOT = settings['log_root']
    SUCCESS_BONUS_PER_EPISODE = settings['success_bonus_per_episode']
    SUCCESS_RATE_BONUS = settings['success_rate_bonus']
    CONSISTENCY_STD_PENALTY = settings['consistency_std_penalty']

def save_bests(log_path, bests):
    """Write the best individual of each generation to a log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        for b in bests:
            f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')

def train_single_run(experiment, run_index, n_runs, seed, log_path, settings):
    """Train one run. Used directly or as a multiprocessing target."""
    apply_training_settings(experiment, settings)
    print(f"Run {run_index + 1}/{n_runs} da experiencia {experiment['id']}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    bests = evolution()
    save_bests(log_path, bests)
    print(f"Log guardado em {log_path}", flush=True)

def run_training_jobs(jobs, parallel_runs):
    """Run training jobs sequentially or in batches of parallel processes."""
    if parallel_runs <= 1:
        for job in jobs:
            train_single_run(*job)
        return

    for start in range(0, len(jobs), parallel_runs):
        batch_jobs = jobs[start:start + parallel_runs]
        processes = []
        for job in batch_jobs:
            process = Process(target=train_single_run, args=job)
            process.start()
            processes.append(process)

        failed = []
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failed.append(process.exitcode)

        if failed:
            raise RuntimeError(f'{len(failed)} run(s) falharam com exit codes: {failed}')

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
    parser.add_argument('--parallel-runs', type=int, default=None, help='Runs treinadas em paralelo.')
    parser.add_argument('--generations', type=int, default=None, help='Numero de geracoes.')
    parser.add_argument('--population-size', type=int, default=None, help='Tamanho da populacao.')
    parser.add_argument('--evaluation-episodes', type=int, default=None, help='Episodios usados para avaliar cada individuo durante o treino.')
    parser.add_argument('--success-bonus-per-episode', type=float, default=None, help='Bonus linear aplicado a taxa de sucesso durante a avaliacao.')
    parser.add_argument('--success-rate-bonus', type=float, default=None, help='Bonus quadratico aplicado a taxa de sucesso durante a avaliacao.')
    parser.add_argument('--consistency-std-penalty', type=float, default=None, help='Penalizacao pelo desvio padrao do fitness durante a avaliacao.')
    parser.add_argument('--test-log', default=None, help='Log a testar.')
    parser.add_argument('--test-episodes', type=int, default=None, help='Episodios usados no teste.')
    parser.add_argument('--render-mode', default=None, help='Render mode do Gymnasium, por exemplo human.')
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()

    # Por defeito treina. A flag --test muda para modo de teste.
    evolve = True
    render_mode = None

    # Tambem e possivel configurar por variaveis de ambiente, mas a CLI tem
    # prioridade quando os argumentos sao passados no terminal.
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
    if args.success_bonus_per_episode is not None:
        SUCCESS_BONUS_PER_EPISODE = args.success_bonus_per_episode
    if args.success_rate_bonus is not None:
        SUCCESS_RATE_BONUS = args.success_rate_bonus
    if args.consistency_std_penalty is not None:
        CONSISTENCY_STD_PENALTY = args.consistency_std_penalty
    if args.log_root is not None:
        LOG_ROOT = args.log_root

    if evolve:
        n_runs = args.runs if args.runs is not None else int(os.environ.get('N_RUNS', 5))
        log_prefix = args.log_prefix if args.log_prefix is not None else os.environ.get('LOG_PREFIX', 'log')
        log_root = args.log_root if args.log_root is not None else LOG_ROOT
        parallel_runs = args.parallel_runs if args.parallel_runs is not None else int(os.environ.get('PARALLEL_RUNS', 1))
        if parallel_runs < 1:
            raise ValueError('--parallel-runs tem de ser pelo menos 1')

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
                f"runs_paralelas={parallel_runs}, "
                f"episodios_avaliacao={EVALUATION_EPISODES}, "
                f"bonus_sucesso={SUCCESS_BONUS_PER_EPISODE}, "
                f"bonus_taxa_sucesso={SUCCESS_RATE_BONUS}, "
                f"penalizacao_std={CONSISTENCY_STD_PENALTY}"
            )

            jobs = []
            settings = current_training_settings(log_root)
            run_indexes = [args.run_index] if args.run_index is not None else range(n_runs)
            for i in run_indexes:
                log_path = experiment_log_path(experiment['id'], i, log_prefix, log_root)
                if os.path.exists(log_path) and not args.overwrite_logs:
                    print(f"Run {i + 1}/{n_runs} da experiencia {experiment['id']} ja existe: {log_path}")
                    continue

                jobs.append((experiment.copy(), i, n_runs, seeds[i], log_path, settings.copy()))

            run_training_jobs(jobs, parallel_runs)

                
    else:
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
