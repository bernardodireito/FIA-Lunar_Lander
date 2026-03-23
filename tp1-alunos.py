import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
#RENDER_MODE = 'human'
RENDER_MODE = None #seleccione esta opção para não visualizar o ambiente (testes mais rápidos)
EPISODES = 1000

env = gym.make("LunarLander-v3", render_mode =RENDER_MODE, 
    continuous=True, gravity=GRAVITY, 
    enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
    turbulence_power=TURBULENCE_POWER)

def check_successful_landing(observation):
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
        
def simulate(steps=1000,seed=None, policy = None):    
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        action = policy(observ)
        observ, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    success = check_successful_landing(observ)
    return step, success


# ==========================================
# Perceptions (Regras Booleanas Limiares)
# ==========================================
def perigo_capotar_esquerda(obs):
    # theta positivo é esquerda, vtheta positivo é rotação anti-horária 
    return obs[4] > 0.15 or obs[5] > 0.1

def perigo_capotar_direita(obs):
    # theta negativo é direita, vtheta negativo é rotação horária 
    return obs[4] < -0.15 or obs[5] < -0.1

def a_fugir_direita(obs):
    # x > 0.05 e vx > -0.05, ou vx > 0.08.
    return (obs[0] > 0.05 and obs[2] > -0.05) or obs[2] > 0.08

def a_fugir_esquerda(obs):
    # x < -0.05 e vx < 0.05, ou vx < -0.08.
    return (obs[0] < -0.05 and obs[2] < 0.05) or obs[2] < -0.08

def inclinado_suficiente_esquerda(obs):
    # Mais inclinação para tornar o empurrão horizontal mais eficaz.
    return obs[4] > 0.1

def inclinado_suficiente_direita(obs):
    return obs[4] < -0.1

def a_subir(obs):
    # vy é positivo quando sobe
    return obs[3] > 0.0

def queda_perigosa(obs):
    # A cair muito rápido, precisa de travagem a fundo
    return obs[3] < -0.4

def queda_rapida(obs):
    # A cair rápido, mas não o suficiente para prego a fundo.
    # O limite de sucesso é > -0.2, por isso disparamos aos -0.2.
    return obs[3] < -0.2

def quase_no_chao(obs):
    return obs[1] < 0.2


# ==========================================
# Actions (Comandos discretos dos motores)
# ==========================================
# Nota: O motor esquerdo (> 0.5) roda a nave para a direita. O direito (< -0.5) roda para a esquerda[cite: 61, 62].
def ligar_motor_esquerdo(): return np.array([0.0, 1.0])
def ligar_motor_direito(): return np.array([0.0, -1.0])
def ligar_motor_principal(): return np.array([1.0, 0.0])
def ligar_motor_principal_suave(): return np.array([0.5, 0.0])
def desligar_motores(): return np.array([0.0, 0.0])


# ==========================================
# Agente Reativo (Sistema de Produções)
# ==========================================
def reactive_agent(observation):
    action = desligar_motores()
    
    # 1. CONTROLO LATERAL / ROTAÇÃO
    # Prioridade Absoluta: Não capotar
    if perigo_capotar_esquerda(observation):
        action = ligar_motor_esquerdo()
    elif perigo_capotar_direita(observation):
        action = ligar_motor_direito()
        
    # Prioridade Secundária: Corrigir Trajetória (Agora reage mais cedo)
    elif a_fugir_direita(observation):
        # Precisa de ir para a esquerda, logo tem de se inclinar para a esquerda.
        if not inclinado_suficiente_esquerda(observation):
            action = ligar_motor_direito() # Inclina para a esquerda
        elif not a_subir(observation):
            action = ligar_motor_principal_suave() # Empurra de volta ao centro
            
    elif a_fugir_esquerda(observation):
        # Precisa de ir para a direita, logo tem de se inclinar para a direita.
        if not inclinado_suficiente_direita(observation):
            action = ligar_motor_esquerdo() # Inclina para a direita
        elif not a_subir(observation):
            action = ligar_motor_principal_suave() # Empurra de volta ao centro

    # 2. CONTROLO VERTICAL (Sobrepõe-se ao action[0])
    # Regra de ouro: se está a subir, corta o motor principal.
    if a_subir(observation):
        action[0] = 0.0
    elif queda_perigosa(observation):
        action[0] = 1.0
    elif queda_rapida(observation):
        action[0] = 0.6
    elif quase_no_chao(observation) and observation[3] < -0.05:
        # Pouso suave.
        if action[0] < 0.5:
            action[0] = 0.5
            
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