'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below. 

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

import torch 
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
import math
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER 
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.agent import *
from typing import Optional, Type, List, Tuple

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,

            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=0,
                                      n_steps=30*90*20,
                                      batch_size=16,
                                      ent_coef=0.05,
                                      policy_kwargs=policy_kwargs)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
       
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action
    
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, hidden_dim: int = 256):
        super().__init__(observation_space, features_dim)
        in_dim = int(np.prod(observation_space.shape))
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, features_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = 0.01 if m is self.model[-1] else 2**0.5
                nn.init.orthogonal_(m.weight, gain=gain); nn.init.constant_(m.bias, 0.)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() > 2:
            obs = obs.view(obs.size(0), -1)
        return self.model(obs)

    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 128, hidden_dim: int = 256) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim),
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=nn.SiLU,
            ortho_init=True,
        )
    
class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)
    
    def _initialize(self) -> None:
        if self.extractor is None:
            raise ValueError("CustomAgent requires an extractor; pass extractor=MLPExtractor (or similar).")

        if self.file_path is None:
            self.model = self.sb3_class(
                "MlpPolicy",
                self.env,
                policy_kwargs=self.extractor.get_policy_kwargs(),
                verbose=0,
                n_steps=30 * 90 * 3,
                batch_size=128,
                ent_coef=0.01,
            )
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward / 140


# In[ ]:


def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt


def stage1_health_distance_reward(
    env: WarehouseBrawl,
    close_target: float = 2.5,
    far_target: float = 6.0,
    sigma: float = 1.6
) -> float:
    """
    Encourage the agent to approach when healthier and disengage when behind.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    horizontal_distance = abs(player.body.position.x - opponent.body.position.x)
    health_margin = opponent.damage - player.damage  # positive → player healthier
    sigma = max(1e-3, sigma)

    if health_margin >= 0:
        weight = min(health_margin / 120.0, 1.0)
        score = math.exp(-((horizontal_distance - close_target) ** 2) / (2 * sigma ** 2))
        return weight * score * env.dt
    else:
        weight = min((-health_margin) / 120.0, 1.0)
        spread = sigma * 1.5
        score = math.exp(-((horizontal_distance - far_target) ** 2) / (2 * spread ** 2))
        return weight * score * env.dt


def stage1_ground_contact_reward(
    env: WarehouseBrawl,
    danger_threshold: float = 3.8
) -> float:
    """
    Reward staying within safe vertical bounds and lightly favour grounded states.
    """
    player: Player = env.objects["player"]
    grounded = player.is_on_floor() or player.on_platform is not None

    if player.body.position.y > danger_threshold:
        return -0.25 * env.dt
    return (0.12 if grounded else 0.0) * env.dt


def stage1_horizontal_velocity_reward(
    env: WarehouseBrawl,
    scale: float = 0.2
) -> float:
    """
    Provide a small incentive for horizontal movement to combat idling.
    """
    player: Player = env.objects["player"]
    velocity = abs(player.body.velocity.x)
    return min(velocity, 6.0) * scale * env.dt


def _stage1_get_jump_tracker(env: WarehouseBrawl) -> dict:
    tracker = getattr(env, "_stage1_jump_tracker", None)
    if tracker is None:
        tracker = {
            "tracking_jump": False,
            "pre_jump_height_diff": None,
            "frames_since_jump": 0,
            "penalty_cooldown": 0,
            "intent_reward_granted": False,
        }
        env._stage1_jump_tracker = tracker
    return tracker


def stage1_proximity_delta_reward(
    env: WarehouseBrawl,
    approach_scale: float = 1.5,
    hold_bonus: float = 0.1,
    preferred_distance: float = 2.2
) -> float:
    """
    Reward the agent for shrinking the horizontal gap to the opponent and for
    staying within a preferred melee range once achieved.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    current_distance = abs(player.body.position.x - opponent.body.position.x)
    previous_distance = getattr(env, "_stage1_prev_distance", current_distance)
    env._stage1_prev_distance = current_distance

    if previous_distance is None:
        return 0.0

    distance_delta = previous_distance - current_distance
    reward = distance_delta * approach_scale
    if current_distance <= preferred_distance:
        reward += hold_bonus

    return reward * env.dt


def stage1_distance_potential_reward(
    env: WarehouseBrawl,
    max_distance: float = 10.0
) -> float:
    """
    Dense potential-based reward that encourages staying close to the opponent.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    distance = abs(player.body.position.x - opponent.body.position.x)
    closeness = 1.0 - min(max(distance / max_distance, 0.0), 1.0)
    return closeness * env.dt


def stage1_distance_penalty_reward(
    env: WarehouseBrawl,
    min_distance: float = 0.2,
    max_distance: float = 10.0
) -> float:
    """
    Apply a penalty proportional to the current distance from the opponent.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    distance = abs(player.body.position.x - opponent.body.position.x)
    distance_clamped = min(max(distance, min_distance), max_distance)
    return -distance_clamped * env.dt


def stage1_directional_velocity_reward(
    env: WarehouseBrawl,
    max_speed: float = 6.0,
    deadzone: float = 0.4
) -> float:
    """
    Encourage moving horizontally toward the opponent when far away.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    dx = opponent.body.position.x - player.body.position.x
    if abs(dx) <= deadzone:
        return 0.0

    desired_dir = np.sign(dx)
    normalized_speed = np.clip(player.body.velocity.x / max_speed, -1.0, 1.0)
    return desired_dir * normalized_speed * env.dt


def stage1_vertical_alignment_reward(
    env: WarehouseBrawl,
    max_height_diff: float = 4.0,
    hold_bonus: float = 0.2
) -> float:
    """
    Encourage matching the opponent's vertical position.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    height_diff = abs(player.body.position.y - opponent.body.position.y)
    normalized = 1.0 - min(height_diff / max_height_diff, 1.0)
    reward = normalized * env.dt
    if height_diff <= 0.5:
        reward += hold_bonus * env.dt
    return reward


def stage1_stagnation_penalty_reward(
    env: WarehouseBrawl,
    min_speed: float = 0.15
) -> float:
    """
    Penalize pressing horizontal controls without producing movement.
    """
    player: Player = env.objects["player"]
    action_vec = player.cur_action
    act_helper = env.act_helper
    if action_vec is None or act_helper is None:
        return 0.0

    horizontal_keys = []
    for key in ('a', 'd'):
        idx = act_helper.sections.get(key)
        if idx is not None:
            horizontal_keys.append(idx)

    if not horizontal_keys:
        return 0.0

    if any(action_vec[idx] > 0.5 for idx in horizontal_keys):
        if abs(player.body.velocity.x) < min_speed:
            return -1.0 * env.dt
    return 0.0


def stage1_jump_penalty_reward(
    env: WarehouseBrawl,
    min_horizontal_speed: float = 1.0,
    close_distance: float = 1.2,
    vertical_tolerance: float = 0.25,
    penalty_value: float = -0.4,
    cooldown_frames: int = 8
) -> float:
    """
    Penalize clearly wasteful jumps (e.g., neutral hops while grounded and not
    attempting to close a vertical gap), while leaving room for tactical jumps.
    """
    player: Player = env.objects["player"]
    action_vec = player.cur_action
    if action_vec is None or len(action_vec) == 0:
        return 0.0

    act_helper = env.act_helper
    if act_helper is None or 'space' not in act_helper.sections:
        return 0.0

    space_idx = act_helper.sections['space']
    if action_vec[space_idx] > 0.5:
        grounded = player.is_on_floor() or player.on_platform is not None
        opponent: Player = env.objects["opponent"]
        player_below = player.body.position.y > opponent.body.position.y + vertical_tolerance
        horizontal_gap = abs(player.body.position.x - opponent.body.position.x)
        horizontal_speed = abs(player.body.velocity.x)

        tracker = _stage1_get_jump_tracker(env)
        tracker["intent_reward_granted"] = tracker["intent_reward_granted"] and grounded
        if tracker["penalty_cooldown"] > 0:
            tracker["penalty_cooldown"] -= 1
            return 0.0

        if grounded and not player_below and horizontal_gap > close_distance and horizontal_speed < min_horizontal_speed:
            tracker["penalty_cooldown"] = cooldown_frames
            return penalty_value * env.dt
    return 0.0


def stage1_jump_alignment_reward(
    env: WarehouseBrawl,
    min_improvement: float = 0.15,
    jump_window: int = 18,
    improvement_scale: float = 2.0,
) -> float:
    """
    Provide a positive signal when a jump meaningfully reduces the vertical gap
    to the opponent soon after takeoff.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    action_vec = player.cur_action
    act_helper = env.act_helper

    if action_vec is None or act_helper is None:
        return 0.0

    tracker = _stage1_get_jump_tracker(env)
    space_idx = act_helper.sections.get('space')
    if space_idx is None:
        return 0.0

    grounded = player.is_on_floor() or player.on_platform is not None
    height_diff = abs(player.body.position.y - opponent.body.position.y)

    if action_vec[space_idx] > 0.5 and grounded:
        tracker["tracking_jump"] = True
        tracker["pre_jump_height_diff"] = height_diff
        tracker["frames_since_jump"] = 0
        return 0.0

    if not tracker["tracking_jump"]:
        return 0.0

    tracker["frames_since_jump"] += 1
    previous_diff = tracker.get("pre_jump_height_diff")
    if previous_diff is None:
        tracker["tracking_jump"] = False
        return 0.0

    improvement = previous_diff - height_diff
    reward = 0.0

    if improvement >= min_improvement:
        reward = improvement * improvement_scale * env.dt
        tracker["tracking_jump"] = False
        tracker["pre_jump_height_diff"] = None
    elif tracker["frames_since_jump"] > jump_window:
        tracker["tracking_jump"] = False
        tracker["pre_jump_height_diff"] = None

    return reward


def stage1_jump_initiation_reward(
    env: WarehouseBrawl,
    vertical_tolerance: float = 0.35,
    reward_value: float = 0.6
) -> float:
    """
    Reward pressing jump while grounded when the opponent is clearly above.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    action_vec = player.cur_action
    act_helper = env.act_helper

    if action_vec is None or act_helper is None:
        return 0.0

    space_idx = act_helper.sections.get('space')
    if space_idx is None:
        return 0.0

    tracker = _stage1_get_jump_tracker(env)
    grounded = player.is_on_floor() or player.on_platform is not None
    player_below = player.body.position.y > opponent.body.position.y + vertical_tolerance

    if action_vec[space_idx] > 0.5 and grounded and player_below:
        if not tracker["intent_reward_granted"]:
            tracker["intent_reward_granted"] = True
            return reward_value * env.dt
    elif grounded:
        tracker["intent_reward_granted"] = False

    if not grounded:
        tracker["intent_reward_granted"] = False

    return 0.0


def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return env.dt
    return 0

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0
    
def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 2.0
        elif env.objects["player"].weapon == "Spear":
            return 1.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0


STAGE1_ALLOWED_KEYS = ["a", "d", "space"]

def gen_stage1_reward_manager() -> RewardManager:
    reward_functions = {
        'stage1_distance_potential_reward': RewTerm(func=stage1_distance_potential_reward, weight=6.0),
        'stage1_distance_penalty_reward': RewTerm(func=stage1_distance_penalty_reward, weight=1.0),
        'stage1_proximity_delta_reward': RewTerm(func=stage1_proximity_delta_reward, weight=0.5),
        'stage1_directional_velocity_reward': RewTerm(func=stage1_directional_velocity_reward, weight=1.2),
        'stage1_horizontal_velocity_reward': RewTerm(func=stage1_horizontal_velocity_reward, weight=0.6),
        'stage1_ground_contact_reward': RewTerm(func=stage1_ground_contact_reward, weight=0.25),
        'stage1_health_distance_reward': RewTerm(func=stage1_health_distance_reward, weight=0.6),
        'stage1_vertical_alignment_reward': RewTerm(func=stage1_vertical_alignment_reward, weight=1.0),
        'stage1_stagnation_penalty_reward': RewTerm(func=stage1_stagnation_penalty_reward, weight=0.6),
        'stage1_jump_initiation_reward': RewTerm(func=stage1_jump_initiation_reward, weight=1.2),
        'stage1_jump_alignment_reward': RewTerm(func=stage1_jump_alignment_reward, weight=0.8),
        'stage1_jump_penalty_reward': RewTerm(func=stage1_jump_penalty_reward, weight=0.3),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.5),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=0.3),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=25)),
    }
    return RewardManager(reward_functions, signal_subscriptions)

'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager():
    reward_functions = {
        #'target_height_reward': RewTerm(func=base_height_l2, weight=0.0, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.5),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=1.0),
        #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
        #'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.05),
        'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-0.04, params={'desired_state': AttackState}),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.01),
        #'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=8)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=5)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=10)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=15))
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':
    # Create agent
    my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

    # Start here if you want to train from scratch. e.g:
    # my_agent = RecurrentPPOAgent()

    # Start here if you want to train from a specific timestep. e.g:
    #my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_3/rl_model_120006_steps.zip')

    # Reward manager
    reward_manager = gen_stage1_reward_manager()
    # Set save settings here:
    save_handler = SaveHandler(
        agent=my_agent, # Agent to save
        save_freq=100_000, # Save frequency
        max_saved=40, # Maximum number of saved models
        save_path='checkpoints', # Save path
        run_name='stage1_experiment',
        mode=SaveHandlerMode.FORCE # Save mode, FORCE or RESUME
    )

    # Set opponent settings here:
    opponent_specification = {
                    'constant_agent': (1.0, partial(ConstantAgent)),
                }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    stage1_wrappers = [
        lambda env: Stage1InitializationWrapper(env),
        lambda env: ObservationNormalizationWrapper(env),
    ]

    train(my_agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        CameraResolution.LOW,
        train_timesteps=400_000,
        train_logging=TrainLogging.PLOT,
        allowed_keys=STAGE1_ALLOWED_KEYS,
        env_wrappers=stage1_wrappers
    )
