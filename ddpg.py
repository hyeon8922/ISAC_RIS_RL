import torch
import numpy as np
from torch import nn, optim

############################################################
# 0) Ornstein-Uhlenbeck Noise Class (batch-aware)
############################################################
class OUNoise:
    """
    Ornstein–Uhlenbeck process (batch-aware).
    """
    def __init__(self, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.x0 = x0
        self.X = None  # internal state (np.ndarray)

    def _ensure_state(self, shape):
        if (self.X is None) or (self.X.shape != tuple(shape)):
            if self.x0 is None:
                self.X = np.zeros(shape, dtype=np.float32)
            else:
                self.X = np.broadcast_to(self.x0, shape).astype(np.float32).copy()

    def reset(self, shape=None, x0=None):
        if x0 is not None:
            self.x0 = x0
        if shape is None:
            self.X = None
        else:
            self._ensure_state(shape)

    def __call__(self, shape):
        self._ensure_state(shape)
        mu = np.broadcast_to(self.mu, shape).astype(np.float32)
        randn = np.random.randn(*shape).astype(np.float32)
        dx = self.theta * (mu - self.X) * self.dt + self.sigma * np.sqrt(self.dt) * randn
        self.X = self.X + dx
        return self.X


############################################################
# 1) ReplayBuffer (state는 1차원 벡터)
############################################################
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, data):
        """
        data: (state, action, reward, next_state, done)
        - state:        torch.Tensor [state_dim]
        - action:       torch.Tensor [N] (complex)
        - reward:       float
        - next_state:   torch.Tensor [state_dim]
        - done:         float (0.0 or 1.0)
        """
        if len(self.storage) == self.max_size:
            self.storage[self.ptr % self.max_size] = data
        else:
            self.storage.append(data)
        self.ptr += 1

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in ind]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),                                      # [B, state_dim]
            torch.stack(actions),                                     # [B, N] (complex)
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),  # [B, 1]
            torch.stack(next_states),                                 # [B, state_dim]
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)     # [B, 1]
        )

    def __len__(self):
        return len(self.storage)


############################################################
# 2) Actor
############################################################
class Actor(nn.Module):
    def __init__(self, N, state_dim):
        super(Actor, self).__init__()
        self.N = N
        self.state_dim = state_dim

        self.fc1 = nn.Linear(state_dim, 128)   # ← 여기 state_dim
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, N)

    def forward(self, x):
        """
        x: [B, state_dim]
        return: theta, thetaH (각 [B, N], complex)
        """
        x = x.view(-1, self.state_dim)         # ← 여기도 state_dim
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        phase = self.fc3(x)                    # [B, N]

        theta_real = torch.cos(phase)
        theta_imag = torch.sin(phase)
        theta  = (theta_real + 1j * theta_imag).to(torch.complex64)
        thetaH = (theta_real - 1j * theta_imag).to(torch.complex64)
        return theta, thetaH

    def get_phase(self, x):
        """
        x: [B, state_dim]
        return: phase [B, N]
        """
        x = x.view(-1, self.state_dim)         # ← 여기도 state_dim
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        phase = self.fc3(x)
        return phase


############################################################
# 3) Critic
############################################################
class Critic(nn.Module):
    def __init__(self, N, state_dim):
        super(Critic, self).__init__()
        self.N = N
        self.state_dim = state_dim

        # state_dim + action_real(N) + action_imag(N) = state_dim + 2N
        self.fc1 = nn.Linear(state_dim + 2 * N, 512)   # ← 여기도 state_dim
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, action):
        """
        x:      [B, state_dim]
        action: [B, N] (complex)
        """
        x = x.view(-1, self.state_dim)                 # ← 여기도 state_dim
        action_cat = torch.cat([torch.real(action), torch.imag(action)], dim=1)
        x = torch.cat([x, action_cat], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


############################################################
# 4) DDPG Agent
############################################################
class DDPGAgent:
    def __init__(self,
                 N,
                 device,
                 lr=1e-3,
                 gamma=0.99,
                 tau=0.005,
                 max_buffer_size=1e6,
                 use_ou_noise=True,
                 ou_rho=0.90,
                 ou_sigma=0.20
                 ):
        self.N = N
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # state = flatten(4 x N x N) + 1 (이전 SNR dB)
        self.state_dim = 4 * N * N + 2

        # Actor
        self.actor = Actor(N, self.state_dim).to(device)
        self.actor_target = Actor(N, self.state_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic
        self.critic = Critic(N, self.state_dim).to(device)
        self.critic_target = Critic(N, self.state_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(max_size=max_buffer_size)

        # OU Noise (스텝 기준)
        self.use_ou_noise = use_ou_noise
        if self.use_ou_noise:
            theta = float(-np.log(max(min(ou_rho, 0.9999), 1e-6)))  # theta = -log(rho)
            self.ou_noise = OUNoise(mu=0.0, theta=theta, sigma=float(ou_sigma), dt=1.0, x0=None)
            self.ou_noise.reset()
        else:
            self.ou_noise = None

    def reset_noise(self, shape=None):
        if self.use_ou_noise and (self.ou_noise is not None):
            self.ou_noise.reset(shape=None)

    @torch.no_grad()
    def select_action(self, state, noise=True):
        """
        state: [B, state_dim]
        return: theta, thetaH (복소수)  [B, N]
        """
        state = state.to(self.device)
        phase = self.actor.get_phase(state)  # [B, N]

        if noise and self.use_ou_noise and (self.ou_noise is not None):
            phase_np = phase.cpu().numpy()
            noise_val = self.ou_noise(phase_np.shape)  # 스텝당 OU
            phase = torch.from_numpy(phase_np + noise_val).to(self.device)

        theta_real = torch.cos(phase)
        theta_imag = torch.sin(phase)
        theta  = (theta_real + 1j * theta_imag).to(torch.complex64)
        thetaH = (theta_real - 1j * theta_imag).to(torch.complex64)
        return theta, thetaH

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Critic 업데이트
        with torch.no_grad():
            next_phase = self.actor_target.get_phase(next_states)
            next_theta_real = torch.cos(next_phase)
            next_theta_imag = torch.sin(next_phase)
            next_actions = (next_theta_real + 1j * next_theta_imag).to(torch.complex64)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 업데이트
        current_phase = self.actor.get_phase(states)
        cur_theta_real = torch.cos(current_phase)
        cur_theta_imag = torch.sin(current_phase)
        current_actions = (cur_theta_real + 1j * cur_theta_imag).to(torch.complex64)

        actor_loss = -self.critic(states, current_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
