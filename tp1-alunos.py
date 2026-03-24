import gymnasium as gym
import numpy as np
import pygame


ENABLE_WIND = True
WIND_POWER = 20.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
#RENDER_MODE = 'human'
RENDER_MODE = None # seleccione esta opção para não visualizar o ambiente (testes mais rápidos)
EPISODES = 1000

env = gym.make("LunarLander-v3", render_mode=RENDER_MODE, 
    continuous=True, gravity=GRAVITY, 
    enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
    turbulence_power=TURBULENCE_POWER)

# ==========================================
# MEMÓRIA E ESTIMATIVA (Função Original Aprovada)
# ==========================================
agent_memory = {
    'prev_obs': None,
    'prev_estimate': 0.0
}

def calc_wind_push(prev_obs, obs, prev_estimate=0.0, alpha=0.18):
    """Estimativa exata enviada ao Professor (Mantém-se puramente reativo por modelo)"""
    if prev_obs is None:
        return prev_estimate
    drift = obs[2] - prev_obs[2]
    return (1.0 - alpha) * prev_estimate + alpha * drift


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
        
def simulate(steps=1000, seed=None, policy=None, ep_idx=0): 
    global agent_memory
    agent_memory['prev_obs'] = None
    agent_memory['prev_estimate'] = 0.0
       
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        if RENDER_MODE == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    import sys; sys.exit(0)

        action = policy(observ) 
        observ, _, term, trunc, _ = env.step(action) 
        if term or trunc:
            break
            
    success = check_successful_landing(observ)

    # ==========================================
    # DEBUGGER DE FALHAS
    # ==========================================
    if not success:
        motivo = "Desconhecido"
        if trunc or step >= 999:
            motivo = "LIMITE DE PASSOS ATINGIDO (Timeout)"
        elif abs(observ[0]) > 1.0 or observ[1] > 1.5:
            motivo = "SAIU DO ECRÃ (Levado pelo vento)"
        elif observ[6] == 0 and observ[7] == 0:
            motivo = "CAIU DESAMPARADA / CAPOTOU"
        else:
            motivo = "CRASH (Bateu rápido demais ou muito inclinada)"
            
        print(f"\n[DEBUG] FALHA NO EP {ep_idx+1:>3} | Passos: {step:>4} | Motivo: {motivo}")
        print(f"        Estado Final: x={observ[0]:+.3f} | y={observ[1]:+.3f} | vx={observ[2]:+.3f} | vy={observ[3]:+.3f} | theta={np.rad2deg(observ[4]):+.1f}º")
    
    return step, success


# =============================================================================
# PERCEPÇÕES (Sensores do Agente)
# =============================================================================
# As perceções transformam os valores contínuos do ambiente em verdades lógicas 
# (Verdadeiro/Falso) para o Sistema de Produções tomar decisões simples.
# 
# Observação: [x, y, vx, vy, theta, vtheta, left_leg, right_leg]
#   x          = obs[0]   # posição horizontal (- esq, + dir)
#   y          = obs[1]   # altitude
#   vx         = obs[2]   # velocidade horizontal (- esq, + dir)
#   vy         = obs[3]   # velocidade vertical (- desce)
#   theta      = obs[4]   # orientação (- dir, + esq)
#   vtheta     = obs[5]   # velocidade angular
#   left_leg   = obs[6]   # sensor da perna esquerda (1.0 = no chão, 0.0 = no ar)
#   right_leg  = obs[7]   # sensor da perna direita (1.0 = no chão, 0.0 = no ar)
# =============================================================================

# --- 1. ESTADOS DE EMERGÊNCIA (Capotamento) ---
def perigo_capotar_esquerda(obs): 
    limiar_theta = 0.6 if sente_turbulencia() else 0.25
    limiar_vtheta = 0.3 if sente_turbulencia() else 0.15
    return obs[4] > limiar_theta or obs[5] > limiar_vtheta

def perigo_capotar_direita(obs): 
    limiar_theta = -0.6 if sente_turbulencia() else -0.25
    limiar_vtheta = -0.3 if sente_turbulencia() else -0.15
    return obs[4] < limiar_theta or obs[5] < limiar_vtheta

# --- 2. NAVEGAÇÃO E POSICIONAMENTO ---
def a_fugir_direita(obs): 
    limiar_v = 0.02 if sente_turbulencia() else 0.08
    limiar_x = 0.08 if sente_turbulencia() else 0.1 
    return obs[0] > limiar_x or obs[2] > limiar_v

def a_fugir_esquerda(obs): 
    limiar_v = -0.02 if sente_turbulencia() else -0.08
    limiar_x = -0.08 if sente_turbulencia() else -0.1
    return obs[0] < limiar_x or obs[2] < limiar_v

def em_cima_da_plataforma_estavel(obs):
    limiar_v = 0.05 if sente_turbulencia() else 0.1
    return abs(obs[0]) <= 0.15 and abs(obs[2]) <= limiar_v

# --- 3. MEMÓRIA DO VENTO (Alínea do Agente com Estado) ---
def sente_vento_forte_direita(): 
    return agent_memory['prev_estimate'] > 0.005

def sente_vento_forte_esquerda(): 
    return agent_memory['prev_estimate'] < -0.005

def sente_turbulencia():
    # O AGENTE PENSA: "Se o módulo da minha estimativa for superior a 0.005,
    # então chego à conclusão de que estou num ambiente com vento forte!"
    return abs(agent_memory['prev_estimate']) > 0.005

# --- 4. COMBATE AO VENTO (Agressividade de Inclinação) ---
def inclinado_suficiente_esquerda(obs): 
    limiar = 0.45 if sente_turbulencia() else 0.1
    return obs[4] > limiar

def inclinado_suficiente_direita(obs): 
    limiar = -0.45 if sente_turbulencia() else -0.1
    return obs[4] < limiar

def desalinhado_esquerda(obs): 
    return obs[4] > 0.05 or obs[5] > 0.02

def desalinhado_direita(obs): 
    return obs[4] < -0.05 or obs[5] < -0.02

# --- 5. CONTROLO VERTICAL (Altitude e Queda) ---
def a_subir(obs): 
    limiar = 0.15 if sente_turbulencia() else 0.0
    return obs[3] > limiar

def queda_perigosa(obs):
    limiar = -0.55 if sente_turbulencia() else -0.35
    return obs[3] < limiar

def queda_rapida(obs):
    limiar = -0.3 if sente_turbulencia() else -0.15
    return obs[3] < limiar

def quase_no_chao(obs):
    limiar = 0.15 if sente_turbulencia() else 0.2
    return obs[1] < limiar

def quase_chao_fora_do_sitio(obs):
    altura = 0.1 if sente_turbulencia() else 0.4
    limiar_x = 0.15 if sente_turbulencia() else 0.2
    return obs[1] < altura and abs(obs[0]) > limiar_x


# =============================================================================
# AÇÕES (Atuadores do Agente)
# =============================================================================
# O vetor de ação é composto por 2 valores [motor_principal, motores_laterais].
# motor_principal: de 0.0 a 1.0 (aponta sempre na direção "debaixo" da nave).
# motores_laterais: de -1.0 a 1.0 (-1 aciona o motor direito, 1 aciona o esquerdo).
# Estas funções traduzem decisões lógicas em comandos mecânicos fixos.
# =============================================================================

def ligar_motor_esquerdo(): 
    """Dispara o propulsor lateral esquerdo. Empurra a nave para a direita 
       e fá-la rodar no sentido dos ponteiros do relógio (diminui o theta)."""
    return np.array([0.0, 1.0])

def ligar_motor_direito(): 
    """Dispara o propulsor lateral direito. Empurra a nave para a esquerda 
       e fá-la rodar no sentido contrário aos ponteiros do relógio (aumenta o theta)."""
    return np.array([0.0, -1.0])

def ligar_motor_principal(): 
    """Dispara o propulsor inferior a 100%. Usado para travar quedas perigosas 
       ou empurrar a nave na direção em que o nariz está apontado."""
    return np.array([1.0, 0.0])

def desligar_motores(): 
    """Deixa a nave à mercê da inércia e da gravidade (ação por defeito - idle)."""
    return np.array([0.0, 0.0])

# ==========================================
# AGENTE REATIVO
# ==========================================
def reactive_agent(observation):

    global agent_memory
    action = desligar_motores()
    
    # 0. ATUALIZAR MODELO INTERNO
    # (Agora ele calcula SEMPRE. Se não houver vento, a estimativa fica perto de 0)
    agent_memory['prev_estimate'] = calc_wind_push(
        agent_memory['prev_obs'], 
        observation, 
        agent_memory['prev_estimate']
    )
    
    # 1. CONTROLO LATERAL E DE ORIENTAÇÃO
    if perigo_capotar_esquerda(observation):
        action = ligar_motor_esquerdo()
    elif perigo_capotar_direita(observation):
        action = ligar_motor_direito()

    elif em_cima_da_plataforma_estavel(observation):
        if desalinhado_esquerda(observation):
            action = ligar_motor_esquerdo()
        elif desalinhado_direita(observation):
            action = ligar_motor_direito()

    # Luta direta contra o vento (Depende exclusivamente da memória do robô!)
    elif sente_vento_forte_direita():
        if not inclinado_suficiente_esquerda(observation):
            action = ligar_motor_direito()
        elif not a_subir(observation):
            action = ligar_motor_principal()

    elif sente_vento_forte_esquerda():
        if not inclinado_suficiente_direita(observation):
            action = ligar_motor_esquerdo()
        elif not a_subir(observation):
            action = ligar_motor_principal()

    # Fugas normais
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

    # 2. CONTROLO VERTICAL (Travagem de última hora)
    if a_subir(observation):
        action[0] = 0.0

    elif quase_chao_fora_do_sitio(observation):
        action[0] = 1.0

    elif queda_perigosa(observation):
        action[0] = 1.0
        
    elif queda_rapida(observation):
        if action[0] < 0.6: action[0] = 0.6
    elif quase_no_chao(observation) and observation[3] < -0.05:
        if action[0] < 0.8: action[0] = 0.8
            
    # Guarda o estado para o frame seguinte
    agent_memory['prev_obs'] = observation
    return action
    

#=============================================================================
# CICLO PRINCIPAL DE AVALIAÇÃO
# =============================================================================

success = 0.0
steps = 0.0

for i in range(EPISODES):
    st, su = simulate(steps=1000000, policy=reactive_agent, ep_idx=i)

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
