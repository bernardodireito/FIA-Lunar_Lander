import gymnasium as gym
import numpy as np
import pygame


ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
#RENDER_MODE = 'human'
RENDER_MODE = None # seleccione esta opção para não visualizar o ambiente (testes mais rápidos)
EPISODES = 1000

# Inicialização do ambiente 
env = gym.make("LunarLander-v3", render_mode=RENDER_MODE, 
    continuous=True, gravity=GRAVITY, 
    enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
    turbulence_power=TURBULENCE_POWER)

def check_successful_landing(observation):
    # Extração das variáveis de estado
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    # Condições de sucesso
    legs_touching = contact_left == 1 and contact_right == 1
    on_landing_pad = abs(x) <= 0.2
    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        return True     
    return False
        
def simulate(steps=1000, seed=None, policy=None):    
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        action = policy(observ) # Obter a ação do agente reativo
        observ, _, term, trunc, _ = env.step(action) # Executar a ação
        if term or trunc:
            break
    success = check_successful_landing(observ)
    return step, success


# ==========================================
# PERCEPÇÕES (Regras Booleanas Limiares)
# ==========================================
def perigo_capotar_esquerda(obs): 
    return obs[4] > 0.15 or obs[5] > 0.1

def perigo_capotar_direita(obs): 
    return obs[4] < -0.15 or obs[5] < -0.1

def a_fugir_direita(obs): 
    return obs[0] > 0.08 or obs[2] > 0.08

def a_fugir_esquerda(obs): 
    return obs[0] < -0.08 or obs[2] < -0.08

def inclinado_suficiente_esquerda(obs): 
    return obs[4] > 0.1

def inclinado_suficiente_direita(obs): 
    return obs[4] < -0.1

def a_subir(obs): 
    return obs[3] > 0.0

def queda_perigosa(obs): 
    return obs[3] < -0.3

def queda_rapida(obs): 
    return obs[3] < -0.15

def quase_no_chao(obs): 
    return obs[1] < 0.3

def em_cima_da_plataforma_estavel(obs):
    # Só considera que está pronta para aterrar se estiver centrada e quase parada na horizontal
    return abs(obs[0]) <= 0.15 and abs(obs[2]) <= 0.1

def desalinhado_esquerda(obs): 
    return obs[4] > 0.05 or obs[5] > 0.02

def desalinhado_direita(obs): 
    return obs[4] < -0.05 or obs[5] < -0.02

def quase_chao_fora_do_sitio(obs): 
    return obs[1] < 0.4 and abs(obs[0]) > 0.15


# ==========================================
# AÇÕES (Comandos discretos dos motores)
# ==========================================
# O motor esquerdo (> 0.5) roda a nave para a direita. O direito (< -0.5) roda para a esquerda.
def ligar_motor_esquerdo(): return np.array([0.0, 1.0])
def ligar_motor_direito(): return np.array([0.0, -1.0])
def ligar_motor_principal(): return np.array([1.0, 0.0])
def desligar_motores(): return np.array([0.0, 0.0])


# ==========================================
# AGENTE REATIVO (Sistema de Produções)
# ==========================================
def reactive_agent(observation):
    action = desligar_motores()
    
    # ------------------------------------------
    # 1. CONTROLO LATERAL E DE ORIENTAÇÃO
    # ------------------------------------------
    if perigo_capotar_esquerda(observation):
        action = ligar_motor_esquerdo()
    elif perigo_capotar_direita(observation):
        action = ligar_motor_direito()

    # Fase Final: Agora exige que a nave esteja ESTÁVEL
    elif em_cima_da_plataforma_estavel(observation):
        if desalinhado_esquerda(observation):
            action = ligar_motor_esquerdo()
        elif desalinhado_direita(observation):
            action = ligar_motor_direito()

    # Correção de Trajetória: Usa FORÇA MÁXIMA (1.0) para compensar desvios fortes
    elif a_fugir_direita(observation):
        if not inclinado_suficiente_esquerda(observation):
            action = ligar_motor_direito()
        elif not a_subir(observation):
            action = ligar_motor_principal()
            
    elif a_fugir_esquerda(observation):
        if not inclinado_suficiente_direita(observation):
            action = ligar_motor_esquerdo()
        elif not a_subir(observation):
            action = ligar_motor_principal()

    # ------------------------------------------
    # 2. CONTROLO VERTICAL
    # ------------------------------------------
    if a_subir(observation):
        action[0] = 0.0

    elif quase_chao_fora_do_sitio(observation):
        action[0] = 1.0

    elif queda_perigosa(observation):
        action[0] = 1.0
        
    elif queda_rapida(observation):
        # Apenas aumenta a potência, não corta a potência lateral se já for 1.0
        if action[0] < 0.6: 
            action[0] = 0.6
            
    elif quase_no_chao(observation) and observation[3] < -0.05:
        # Aterragem muuuuito mais suave (potência 0.8 no fim)
        if action[0] < 0.8: 
            action[0] = 0.8
            
    return action
    

#=============================================================================
# CICLO PRINCIPAL DE AVALIAÇÃO
# =============================================================================

success = 0.0
steps = 0.0

for i in range(EPISODES):
    st, su = simulate(steps=1000000, policy=reactive_agent)

    if su:
        steps += st
    success += su

    aterrou = "True" if su else "False"
    taxa = success / (i + 1) * 100
    print(f"Episódio {i+1:>4} | Aterrou: {aterrou} | Taxa de sucesso: {taxa:.1f}%")

print("\n" + "="*50)
print(f"  Taxa de sucesso final : {success/EPISODES*100:.1f}%")
if success > 0:
    print(f"  Média de passos (aterragens bem sucedidas): {steps/success:.1f}")
print("="*50)