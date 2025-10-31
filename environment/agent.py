from environment.environment import ActHelper, AirTurnaroundState, Animation, AnimationSprite2D, AttackState, BackDashState, Camera, CameraResolution, Capsule, CapsuleCollider, Cast, CastFrameChangeHolder, CasterPositionChange, CasterVelocityDampXY, CasterVelocitySet, CasterVelocitySetXY, CompactMoveState, DashState, DealtPositionTarget, DodgeState, Facing, GameObject, Ground, GroundState, HurtboxPositionChange, InAirState, KOState, KeyIconPanel, KeyStatus, MalachiteEnv, MatchStats, MoveManager, MoveType, ObsHelper, Particle, Player, PlayerInputHandler, PlayerObjectState, PlayerStats, Power, RenderMode, Result, Signal, SprintingState, Stage, StandingState, StunState, Target, TauntState, TurnaroundState, UIHandler, WalkingState, WarehouseBrawl, hex_to_rgb

import warnings
from typing import TYPE_CHECKING, Any, Generic, \
 SupportsFloat, TypeVar, Type, Optional, List, Dict, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, MISSING
from collections import defaultdict
from functools import partial
from typing import Tuple, Any

from PIL import Image, ImageSequence
import matplotlib.pyplot as plt

import gdown, os, math, random, shutil, json

import numpy as np
import torch
from torch import nn

import gymnasium
from gymnasium import spaces

import pygame
import pygame.gfxdraw
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d

import cv2
import skimage.transform as st
import skvideo
import skvideo.io
from IPython.display import Video

from stable_baselines3.common.monitor import Monitor


# ## Agents

# ### Agent Abstract Base Class

# In[ ]:


SelfAgent = TypeVar("SelfAgent", bound="Agent")

class Agent(ABC):

    def __init__(
            self,
            file_path: Optional[str] = None
        ):

        # If no supplied file_path, load from gdown (optional file_path returned)
        if file_path is None:
            file_path = self._gdown()

        self.file_path: Optional[str] = file_path
        self.initialized = False

    def get_env_info(self, env):
        if isinstance(env, Monitor):
            self_env = env.env
        else:
            self_env = env
        self.observation_space = self_env.observation_space
        self.obs_helper = self_env.obs_helper
        self.action_space = self_env.action_space
        self.act_helper = self_env.act_helper
        self.env = env
        self._initialize()
        self.initialized = True

    def get_num_timesteps(self) -> int:
        if hasattr(self, 'model'):
            return self.model.num_timesteps
        else:
            return 0

    def update_num_timesteps(self, num_timesteps: int) -> None:
        if hasattr(self, 'model'):
            self.model.num_timesteps = num_timesteps

    @abstractmethod
    def predict(self, obs) -> spaces.Space:
        pass

    def save(self, file_path: str) -> None:
        return

    def reset(self) -> None:
        return

    def _initialize(self) -> None:
        """

        """
        return

    def _gdown(self) -> Optional[str]:
        """
        Loads the necessary file from Google Drive, returning a file path.
        Or, returns None, if the agent does not require loaded files.

        :return:
        """
        return


# ### Agent Classes

# In[ ]:


class ConstantAgent(Agent):
    '''
    ConstantAgent:
    - The ConstantAgent simply is in an IdleState (action_space all equal to zero.)
    As such it will not do anything, DON'T use this agent for your training.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.action_space.sample())
        return action

class RandomAgent(Agent):
    '''
    RandomAgent:
    - The RandomAgent (as it name says) simply samples random actions.
    NOT used for training
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action


# ## StableBaselines3 Integration

# ### Reward Configuration

# In[ ]:

@dataclass
class RewTerm():
    """Configuration for a reward term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the reward signals as torch float tensors of
    shape (num_envs,).
    """

    weight: float = MISSING
    """The weight of the reward term.

    This is multiplied with the reward term's value to compute the final
    reward.

    Note:
        If the weight is zero, the reward term is ignored.
    """

    params: dict[str, Any] = field(default_factory=dict)
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict.

    .. note::
        If the value is a :class:`SceneEntityCfg` object, the manager will query the scene entity
        from the :class:`InteractiveScene` and process the entity's joints and bodies as specified
        in the :class:`SceneEntityCfg` object.
    """



# In[ ]:


class RewardManager():
    """Reward terms for the MDP."""

    # (1) Constant running reward
    def __init__(self,
                 reward_functions: Optional[Dict[str, RewTerm]]=None,
                 signal_subscriptions: Optional[Dict[str, Tuple[str, RewTerm]]]=None) -> None:
        self.reward_functions = reward_functions
        self.signal_subscriptions = signal_subscriptions
        self.total_reward = 0.0
        self.collected_signal_rewards = 0.0

    def subscribe_signals(self, env) -> None:
        if self.signal_subscriptions is None:
            return
        for _, (name, term_cfg) in self.signal_subscriptions.items():
            getattr(env, name).connect(partial(self._signal_func, term_cfg))

    def _signal_func(self, term_cfg: RewTerm, *args, **kwargs):
        term_partial = partial(term_cfg.func, **term_cfg.params)
        self.collected_signal_rewards += term_partial(*args, **kwargs) * term_cfg.weight

    def process(self, env, dt) -> float:
        # reset computation
        reward_buffer = 0.0
        term_logs: dict[str, float] = {}
        # iterate over all the reward terms
        if self.reward_functions is not None:
            for name, term_cfg in self.reward_functions.items():
                # skip if weight is zero (kind of a micro-optimization)
                if term_cfg.weight == 0.0:
                    continue
                # compute term's value
                value = term_cfg.func(env, **term_cfg.params) * term_cfg.weight
                # update total reward
                reward_buffer += value
                term_logs[name] = float(value)

        reward = reward_buffer + self.collected_signal_rewards
        self.collected_signal_rewards = 0.0

        self.total_reward += reward

        log = env.logger[0]
        log['reward'] = f'{reward_buffer:.3f}'
        log['total_reward'] = f'{self.total_reward:.3f}'
        if term_logs:
            log['reward_terms'] = {k: f'{v:.3f}' for k, v in term_logs.items()}
        signal_contrib = reward - reward_buffer
        if abs(signal_contrib) > 0:
            log['signal_reward'] = f'{signal_contrib:.3f}'
        env.logger[0] = log
        return reward

    def reset(self):
        self.total_reward = 0.0
        self.collected_signal_rewards = 0.0


# ### Training Wrappers

class ActionSubsetWrapper(gymnasium.ActionWrapper):
    """
    Mask out any buttons not listed in ``allowed_keys`` by zeroing their entries
    in the action vector. This keeps the network interface identical while
    effectively shrinking the usable action set.
    """

    def __init__(self, env, allowed_keys: list[str], threshold: float = 0.4):
        super().__init__(env)
        if not allowed_keys:
            raise ValueError("ActionSubsetWrapper requires at least one allowed key.")

        self.raw_env = getattr(env, "raw_env", getattr(env, "unwrapped", env))
        self.act_helper = getattr(env, "act_helper", getattr(self.raw_env, "act_helper", None))

        key_map = getattr(env, "act_helper", None)
        if key_map is None:
            raise AttributeError("Wrapped environment does not expose act_helper; cannot subset actions.")

        self.key_indices: list[int] = []
        self.threshold = threshold
        for key in allowed_keys:
            key = key.lower()
            if key not in key_map.sections:
                raise KeyError(f"Key '{key}' not present in action space. Available keys: {list(key_map.sections.keys())}")
            self.key_indices.append(key_map.sections[key])

    def action(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        filtered = np.zeros_like(action)
        filtered[self.key_indices] = action[self.key_indices]
        return (filtered >= self.threshold).astype(np.int32)


class BinaryActionWrapper(gymnasium.ActionWrapper):
    """
    Convert continuous action vectors into binary button presses by thresholding.
    """

    def __init__(self, env, threshold: float = 0.4):
        super().__init__(env)
        self.threshold = threshold

    def action(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        return (action >= self.threshold).astype(np.int32)

def _resolve_allowed_action_indices(act_helper: ActHelper | None, allowed_keys: list[str]) -> Optional[list[int]]:
    if act_helper is None:
        return None
    indices = []
    for key in allowed_keys:
        key = key.lower()
        if key not in act_helper.sections:
            raise KeyError(f"Key '{key}' not present in action space. Available keys: {list(act_helper.sections.keys())}")
        indices.append(act_helper.sections[key])
    return indices


def _filter_action_vector(action: np.ndarray, allowed_indices: list[int], threshold: float = 0.4) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    filtered = np.zeros_like(action)
    filtered[allowed_indices] = action[allowed_indices]
    return (filtered >= threshold).astype(np.int32)


class Stage1InitializationWrapper(gymnasium.Wrapper):
    """
    Episode wrapper used for the first curriculum phase. It randomizes starting
    health values and spawn locations to avoid overfitting to the default reset.
    """

    def __init__(
        self,
        env,
        player_damage_range: tuple[float, float] = (0.0, 80.0),
        opponent_damage_range: tuple[float, float] = (0.0, 80.0),
        spawn_range: tuple[float, float] = (-5.0, 5.0),
    ):
        super().__init__(env)
        self.raw_env = getattr(env, "raw_env", getattr(env, "unwrapped", env))
        self.player_damage_range = player_damage_range
        self.opponent_damage_range = opponent_damage_range
        self.spawn_range = spawn_range
        # Expose base environment attributes so downstream wrappers can still access them.
        self.act_helper = getattr(env, "act_helper", None)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._randomize_health_and_positions()
        return self._refresh_observation(obs), info

    def _refresh_observation(self, original_obs):
        raw_env = getattr(self, "raw_env", None)
        if raw_env is None:
            return original_obs

        if hasattr(self.env, "opponent_obs"):
            self.env.opponent_obs = raw_env.observe(1)
        return raw_env.observe(0)

    def _randomize_health_and_positions(self):
        raw_env = getattr(self, "raw_env", None)
        if raw_env is None:
            return

        player = raw_env.objects.get("player")
        opponent = raw_env.objects.get("opponent")
        if player is None or opponent is None:
            return

        self._set_damage(player, random.uniform(*self.player_damage_range))
        self._set_damage(opponent, random.uniform(*self.opponent_damage_range))

        # Randomize horizontal spawn positions with a minimum separation.
        min_sep = 1.5
        left, right = self.spawn_range
        for _ in range(10):
            p1_x = random.uniform(left, right)
            p2_x = random.uniform(left, right)
            if abs(p1_x - p2_x) >= min_sep:
                break
        else:
            p1_x, p2_x = -2.5, 2.5

        self._set_position(player, (p1_x, 0.0))
        self._set_position(opponent, (p2_x, 0.0))
        setattr(raw_env, "_stage1_prev_distance", abs(player.body.position.x - opponent.body.position.x))

    @staticmethod
    def _set_damage(player, value: float):
        value = max(0.0, float(value))
        player.damage = value
        player.damage_taken_this_stock = value
        player.damage_taken_total = value

    @staticmethod
    def _set_position(player, pos: tuple[float, float]):
        x, y = pos
        player.body.position = pymunk.Vec2d(x, y)
        player.body.velocity = pymunk.Vec2d(0.0, 0.0)
        player.prev_x = x
        player.prev_y = y
        player.start_position = [x, y]


class ObservationNormalizationWrapper(gymnasium.ObservationWrapper):
    """
    Linearly scale each observation component into roughly [-1, 1] using known
    physical limits, without modifying the underlying environment.
    """

    POS_LIMIT = np.array([18.0, 7.0], dtype=np.float32)
    VEL_LIMIT = np.array([10.0, 10.0], dtype=np.float32)
    SPAWNER_TYPE_LIMIT = 3.0
    JUMPS_LIMIT = 2.0
    STATE_LIMIT = 12.0
    MOVE_TYPE_LIMIT = 11.0
    STOCK_LIMIT = 3.0
    WEAPON_TYPE_LIMIT = 3.0
    DODGE_LIMIT = 82.0
    STUN_LIMIT = 80.0

    def __init__(self, env):
        super().__init__(env)
        self.raw_env = getattr(env, "raw_env", getattr(env, "unwrapped", env))
        self.act_helper = getattr(env, "act_helper", getattr(self.raw_env, "act_helper", None))
        self.obs_helper = getattr(env, "obs_helper", getattr(self.raw_env, "obs_helper", None))
        if self.obs_helper is None:
            raise AttributeError("ObservationNormalizationWrapper requires an environment exposing obs_helper.")

        self.section_slices: dict[str, tuple[int, int]] = self.obs_helper.sections
        self.section_affine: dict[str, tuple[np.ndarray, np.ndarray]] = {
            name: self._compute_affine(name, start, end)
            for name, (start, end) in self.section_slices.items()
        }

        self.observation_space = env.observation_space

    def observation(self, observation):
        obs = np.array(observation, dtype=np.float32, copy=True)
        for name, (start, end) in self.section_slices.items():
            scale, offset = self.section_affine[name]
            if scale.size == 0:
                continue
            obs_section = obs[start:end]
            obs[start:end] = np.clip(obs_section * scale + offset, -1.0, 1.0)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def _compute_affine(self, name: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        length = end - start
        if length == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        def _ensure_shape(arr: np.ndarray | float, size: int) -> np.ndarray:
            arr_np = np.asarray(arr, dtype=np.float32)
            if arr_np.shape == ():
                arr_np = np.full(size, float(arr_np), dtype=np.float32)
            return np.broadcast_to(arr_np, (size,)).astype(np.float32)

        if "_pos" in name and "spawner" not in name:
            low = -self.POS_LIMIT
            high = self.POS_LIMIT
        elif "_vel" in name and "spawner" not in name:
            low = -self.VEL_LIMIT
            high = self.VEL_LIMIT
        elif "_facing" in name or "_grounded" in name or "_aerial" in name or "_damage" in name:
            low = np.zeros(length, dtype=np.float32)
            high = np.ones(length, dtype=np.float32)
        elif "_jumps_left" in name:
            low = np.zeros(length, dtype=np.float32)
            high = np.full(length, self.JUMPS_LIMIT, dtype=np.float32)
        elif "_state" in name:
            low = np.zeros(length, dtype=np.float32)
            high = np.full(length, self.STATE_LIMIT, dtype=np.float32)
        elif "_move_type" in name:
            low = np.zeros(length, dtype=np.float32)
            high = np.full(length, self.MOVE_TYPE_LIMIT, dtype=np.float32)
        elif "_stocks" in name:
            low = np.zeros(length, dtype=np.float32)
            high = np.full(length, self.STOCK_LIMIT, dtype=np.float32)
        elif "_recoveries_left" in name:
            low = np.zeros(length, dtype=np.float32)
            high = np.ones(length, dtype=np.float32)
        elif "_dodge_timer" in name:
            low = np.zeros(length, dtype=np.float32)
            high = np.full(length, self.DODGE_LIMIT, dtype=np.float32)
        elif "_stun_frames" in name:
            low = np.zeros(length, dtype=np.float32)
            high = np.full(length, self.STUN_LIMIT, dtype=np.float32)
        elif "_weapon_type" in name:
            low = np.zeros(length, dtype=np.float32)
            high = np.full(length, self.WEAPON_TYPE_LIMIT, dtype=np.float32)
        elif "_spawner_" in name:
            low = np.array([-self.POS_LIMIT[0], -self.POS_LIMIT[1], 0.0], dtype=np.float32)
            high = np.array([self.POS_LIMIT[0], self.POS_LIMIT[1], self.SPAWNER_TYPE_LIMIT], dtype=np.float32)
        else:
            low = np.zeros(length, dtype=np.float32)
            high = np.ones(length, dtype=np.float32)

        low = _ensure_shape(low, length)
        high = _ensure_shape(high, length)
        denom = np.maximum(high - low, 1e-6)
        scale = 2.0 / denom
        offset = -(high + low) / denom
        return scale, offset


# ### Save, Self-play, and Opponents

# In[ ]:


class SaveHandlerMode(Enum):
    FORCE = 0
    RESUME = 1

class SaveHandler():
    """Handles saving.

    Args:
        agent (Agent): Agent to save.
        save_freq (int): Number of steps between saving.
        max_saved (int): Maximum number of saved models.
        save_dir (str): Directory to save models.
        name_prefix (str): Prefix for saved models.
    """

    # System for saving to internet

    def __init__(
            self,
            agent: Agent,
            save_freq: int=10_000,
            max_saved: int=20,
            run_name: str='experiment_1',
            save_path: str='checkpoints',
            name_prefix: str = "rl_model",
            mode: SaveHandlerMode=SaveHandlerMode.FORCE
        ):
        self.agent = agent
        self.save_freq = save_freq
        self.run_name = run_name
        self.max_saved = max_saved
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.mode = mode

        self.steps_until_save = save_freq
        # Get model paths from exp_path, if it exists
        exp_path = self._experiment_path()
        self.history: List[str] = []
        if self.mode == SaveHandlerMode.FORCE:
            # Clear old dir
            if os.path.exists(exp_path) and len(os.listdir(exp_path)) != 0:
                while True:
                    answer = input(f"Would you like to clear the folder {exp_path} (SaveHandlerMode.FORCE): yes (y) or no (n): ").strip().lower()
                    if answer in ('y', 'n'):
                        break
                    else:
                        print("Invalid input, please enter 'y' or 'n'.")

                if answer == 'n':
                    raise ValueError('Please switch to SaveHandlerMode.FORCE or use a new run_name.')
                print(f'Clearing {exp_path}...')
                if os.path.exists(exp_path):
                    shutil.rmtree(exp_path)
            else:
                print(f'{exp_path} empty or does not exist. Creating...')

            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
        elif self.mode == SaveHandlerMode.RESUME:
            if os.path.exists(exp_path):
                # Get all model paths
                self.history = [os.path.join(exp_path, f) for f in os.listdir(exp_path) if os.path.isfile(os.path.join(exp_path, f))]
                # Filter any non .csv
                self.history = [f for f in self.history if f.endswith('.zip')]
                if len(self.history) != 0:
                    self.history.sort(key=lambda x: int(os.path.basename(x).split('_')[-2].split('.')[0]))
                    if max_saved != -1: self.history = self.history[-max_saved:]
                    print(f'Best model is {self.history[-1]}')
                else:
                    print(f'No models found in {exp_path}.')
                    raise FileNotFoundError
            else:
                print(f'No file found at {exp_path}')


    def update_info(self) -> None:
        self.num_timesteps = self.agent.get_num_timesteps()

    def _experiment_path(self) -> str:
        """
        Helper to get experiment path for each type of checkpoint.

        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, self.run_name)

    def _checkpoint_path(self, extension: str = '') -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self._experiment_path(), f"{self.name_prefix}_{self.num_timesteps}_steps.{extension}")

    def save_agent(self) -> None:
        print(f"Saving agent to {self._checkpoint_path()}")
        model_path = self._checkpoint_path('zip')
        self.agent.save(model_path)
        self.history.append(model_path)
        if self.max_saved != -1 and len(self.history) > self.max_saved:
            os.remove(self.history.pop(0))

    def process(self) -> bool:
        self.num_timesteps += 1

        if self.steps_until_save <= 0:
            # Save agent
            self.steps_until_save = self.save_freq
            self.save_agent()
            return True
        self.steps_until_save -= 1

        return False

    def get_random_model_path(self) -> str:
        if len(self.history) == 0:
            return None
        return random.choice(self.history)

    def get_latest_model_path(self) -> str:
        if len(self.history) == 0:
            return None
        return self.history[-1]

class SelfPlayHandler(ABC):
    """Handles self-play."""

    def __init__(self, agent_partial: partial):
        self.agent_partial = agent_partial
    
    def get_model_from_path(self, path) -> Agent:
        if path:
            try:
                opponent = self.agent_partial(file_path=path)
            except FileNotFoundError:
                print(f"Warning: Self-play file {path} not found. Defaulting to constant agent.")
                opponent = ConstantAgent()
        else:
            print("Warning: No self-play model saved. Defaulting to constant agent.")
            opponent = ConstantAgent()
        opponent.get_env_info(self.env)
        return opponent

    @abstractmethod
    def get_opponent(self) -> Agent:
        pass

class SelfPlayLatest(SelfPlayHandler):
    def __init__(self, agent_partial: partial):
        super().__init__(agent_partial)
    
    def get_opponent(self) -> Agent:
        assert self.save_handler is not None, "Save handler must be specified for self-play"
        chosen_path = self.save_handler.get_latest_model_path()
        return self.get_model_from_path(chosen_path)

class SelfPlayRandom(SelfPlayHandler):
    def __init__(self, agent_partial: partial):
        super().__init__(agent_partial)
    
    def get_opponent(self) -> Agent:
        assert self.save_handler is not None, "Save handler must be specified for self-play"
        chosen_path = self.save_handler.get_random_model_path()
        return self.get_model_from_path(chosen_path)

@dataclass
class OpponentsCfg():
    """Configuration for opponents.

    Args:
        swap_steps (int): Number of steps between swapping opponents.
        opponents (dict): Dictionary specifying available opponents and their selection probabilities.
    """
    swap_steps: int = 10_000
    opponents: dict[str, Any] = field(default_factory=lambda: {
                'random_agent': (0.8, partial(RandomAgent)),
                'constant_agent': (0.2, partial(ConstantAgent)),
                #'recurrent_agent': (0.1, partial(RecurrentPPOAgent, file_path='skibidi')),
            })

    def validate_probabilities(self) -> None:
        total_prob = sum(prob if isinstance(prob, float) else prob[0] for prob in self.opponents.values())

        if abs(total_prob - 1.0) > 1e-5:
            print(f"Warning: Probabilities do not sum to 1 (current sum = {total_prob}). Normalizing...")
            self.opponents = {
                key: (value / total_prob if isinstance(value, float) else (value[0] / total_prob, value[1]))
                for key, value in self.opponents.items()
            }

    def process(self) -> None:
        pass

    def on_env_reset(self) -> Agent:

        agent_name = random.choices(
            list(self.opponents.keys()),
            weights=[prob if isinstance(prob, float) else prob[0] for prob in self.opponents.values()]
        )[0]

        # If self-play is selected, return the trained model
        print(f'Selected {agent_name}')
        if agent_name == "self_play":
            selfplay_handler: SelfPlayHandler = self.opponents[agent_name][1]
            return selfplay_handler.get_opponent()
        else:
            # Otherwise, return an instance of the selected agent class
            opponent = self.opponents[agent_name][1]()

        opponent.get_env_info(self.env)
        return opponent


# ### Self-Play Warehouse Brawl

# In[ ]:


class SelfPlayWarehouseBrawl(gymnasium.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 reward_manager: Optional[RewardManager]=None,
                 opponent_cfg: OpponentsCfg=OpponentsCfg(),
                 save_handler: Optional[SaveHandler]=None,
                 render_every: int | None = None,
                 resolution: CameraResolution=CameraResolution.LOW):
        """
        Initializes the environment.

        Args:
            reward_manager (Optional[RewardManager]): Reward manager.
            opponent_cfg (OpponentCfg): Configuration for opponents.
            save_handler (SaveHandler): Configuration for self-play.
            render_every (int | None): Number of steps between a demo render (None if no rendering).
        """
        super().__init__()

        self.reward_manager = reward_manager
        self.save_handler = save_handler
        self.opponent_cfg = opponent_cfg
        self.render_every = render_every
        self.resolution = resolution

        self.games_done = 0


        # Give OpponentCfg references, and normalize probabilities
        self.opponent_cfg.env = self
        self.opponent_cfg.validate_probabilities()

        # Check if using self-play
        for key, value in self.opponent_cfg.opponents.items():
            if isinstance(value[1], SelfPlayHandler):
                assert self.save_handler is not None, "Save handler must be specified for self-play"

                # Give SelfPlayHandler references
                selfplay_handler: SelfPlayHandler = value[1]
                selfplay_handler.save_handler = self.save_handler
                selfplay_handler.env = self       

        self.raw_env = WarehouseBrawl(resolution=resolution, train_mode=True)
        self.action_space = self.raw_env.action_space
        self.act_helper = self.raw_env.act_helper
        self.observation_space = self.raw_env.observation_space
        self.obs_helper = self.raw_env.obs_helper

    def on_training_start(self):
        # Update SaveHandler
        if self.save_handler is not None:
            self.save_handler.update_info()

    def on_training_end(self):
        if self.save_handler is not None:
            self.save_handler.agent.update_num_timesteps(self.save_handler.num_timesteps)
            self.save_handler.save_agent()

    def step(self, action):

        full_action = {
            0: action,
            1: self.opponent_agent.predict(self.opponent_obs),
        }

        observations, rewards, terminated, truncated, info = self.raw_env.step(full_action)

        if self.save_handler is not None:
            self.save_handler.process()

        if self.reward_manager is None:
            reward = rewards[0]
        else:
            reward = self.reward_manager.process(self.raw_env, 1 / 30.0)

        return observations[0], reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset MalachiteEnv
        observations, info = self.raw_env.reset()

        self.reward_manager.reset()

        # Select agent
        new_agent: Agent = self.opponent_cfg.on_env_reset()
        if new_agent is not None:
            self.opponent_agent: Agent = new_agent
        self.opponent_obs = observations[1]


        self.games_done += 1
        #if self.games_done % self.render_every == 0:
            #self.render_out_video()

        return observations[0], info

    def render(self):
        img = self.raw_env.render()
        return img

    def close(self):
        pass


# ## Run Match

# In[ ]:


from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

def run_match(agent_1: Agent | partial,
              agent_2: Agent | partial,
              max_timesteps=30*90,
              video_path: Optional[str]=None,
              agent_1_name: Optional[str]=None,
              agent_2_name: Optional[str]=None,
              resolution = CameraResolution.LOW,
              reward_manager: Optional[RewardManager]=None,
              train_mode=False,
              allowed_keys: Optional[List[str]] = None,
              env_wrappers: Optional[List[Callable[[gymnasium.Env], gymnasium.Env]]] = None,
              ) -> MatchStats:
    # Initialize env

    env = WarehouseBrawl(resolution=resolution, train_mode=train_mode)

    allowed_indices: Optional[List[int]] = None

    if env_wrappers and isinstance(env, gymnasium.Env):
        for wrapper_fn in env_wrappers:
            env = wrapper_fn(env)
    elif env_wrappers:
        warnings.warn("env_wrappers ignored: WarehouseBrawl is not a gymnasium.Env instance.")

    if allowed_keys is not None:
        if isinstance(env, gymnasium.Env):
            env = ActionSubsetWrapper(env, allowed_keys=allowed_keys)
        else:
            allowed_indices = _resolve_allowed_action_indices(getattr(env, "act_helper", None), allowed_keys)

    observations, infos = env.reset()
    obs_1 = observations[0]
    obs_2 = observations[1]
    print("RUN MATCH IS RUNNING")

    if reward_manager is not None:
        reward_manager.reset()
        base_raw_env = getattr(env, "raw_env", None)
        if base_raw_env is None and hasattr(env, "env"):
            base_raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
        reward_manager.subscribe_signals(base_raw_env or env)

    if agent_1_name is None:
        agent_1_name = 'agent_1'
    if agent_2_name is None:
        agent_2_name = 'agent_2'

    env.agent_1_name = agent_1_name
    env.agent_2_name = agent_2_name


    writer = None
    if video_path is None:
        print("video_path=None -> Not rendering")
    else:
        print(f"video_path={video_path} -> Rendering")
        # Initialize video writer
        writer = skvideo.io.FFmpegWriter(video_path, outputdict={
            '-vcodec': 'libx264',  # Use H.264 for Windows Media Player
            '-pix_fmt': 'yuv420p',  # Compatible with both WMP & Colab
            '-preset': 'fast',  # Faster encoding
            '-crf': '20',  # Quality-based encoding (lower = better quality)
            '-r': '30',  # Frame rate
            '-vf': 'transpose=1,hflip'
        })

    # If partial
    if callable(agent_1):
        agent_1 = agent_1()
    if callable(agent_2):
        agent_2 = agent_2()

    # Initialize agents
    if not agent_1.initialized: agent_1.get_env_info(env)
    if not agent_2.initialized: agent_2.get_env_info(env)
    # 596, 336
    platform1 = env.objects["platform1"]

    for time in tqdm(range(max_timesteps), total=max_timesteps):
      platform1.physics_process(0.05)
      full_action = {
          0: agent_1.predict(obs_1),
          1: agent_2.predict(obs_2)
      }

      if allowed_indices is not None:
          full_action = {
              agent_id: _filter_action_vector(act, allowed_indices)
              for agent_id, act in full_action.items()
          }

      observations, rewards, terminated, truncated, info = env.step(full_action)
      obs_1 = observations[0]
      obs_2 = observations[1]

      if reward_manager is not None:
          reward_manager.process(env, 1 / env.fps)

      if video_path is not None:
          img = env.render()
          writer.writeFrame(img)
          del img

      if terminated or truncated:
          break


    if video_path is not None:
        writer.close()
    env.close()

    # visualize
    # Video(video_path, embed=True, width=800) if video_path is not None else None
    player_1_stats = env.get_stats(0)
    player_2_stats = env.get_stats(1)

    if player_1_stats.lives_left > player_2_stats.lives_left:
        result = Result.WIN
    elif player_1_stats.lives_left < player_2_stats.lives_left:
        result = Result.LOSS
    else:
        result = Result.DRAW
    
    match_stats = MatchStats(
        match_time=env.steps / env.fps,
        player1=player_1_stats,
        player2=player_2_stats,
        player1_result=result
    )

    del env

    return match_stats


class ConstantAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.action_space.sample())
        return action

class RandomAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action


class BasedAgent(Agent):

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

        #if keys[pygame.K_q]:
        #    action = self.act_helper.press_keys(['q'], action)
        #if keys[pygame.K_v]:
        #    action = self.act_helper.press_keys(['v'], action)
        return action


class ClockworkAgent(Agent):

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
                (30, []),
                (7, ['d']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (20, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
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

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

class SB3Agent(Agent):

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

from sb3_contrib import RecurrentPPO

class RecurrentPPOAgent(Agent):

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


# ## Training Function
# A helper function for training.

# In[ ]:


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

class TrainLogging(Enum):
    NONE = 0
    TO_FILE = 1
    PLOT = 2

def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")

    weights = np.repeat(1.0, 50) / 50
    print(weights, y)
    y = np.convolve(y, weights, "valid")
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")

    # save to file
    plt.savefig(log_folder + title + ".png")

def train(agent: Agent,
          reward_manager: RewardManager,
          save_handler: Optional[SaveHandler]=None,
          opponent_cfg: OpponentsCfg=OpponentsCfg(),
          resolution: CameraResolution=CameraResolution.LOW,
          train_timesteps: int=400_000,
          train_logging: TrainLogging=TrainLogging.PLOT,
          allowed_keys: Optional[List[str]] = None,
          env_wrappers: Optional[List[Callable[[gymnasium.Env], gymnasium.Env]]] = None,
          ):
    # Create environment
    env = SelfPlayWarehouseBrawl(
        reward_manager=reward_manager,
        opponent_cfg=opponent_cfg,
        save_handler=save_handler,
        resolution=resolution
    )
    if env_wrappers:
        for wrapper_fn in env_wrappers:
            env = wrapper_fn(env)

    if allowed_keys is not None:
        env = ActionSubsetWrapper(env, allowed_keys=allowed_keys)

    base_raw_env = getattr(env, "raw_env", None)
    if base_raw_env is None and hasattr(env, "env"):
        base_raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    reward_manager.subscribe_signals(base_raw_env)

    if train_logging != TrainLogging.NONE:
        # Create log dir
        log_dir = f"{save_handler._experiment_path()}/" if save_handler is not None else "/tmp/gym/"
        os.makedirs(log_dir, exist_ok=True)

        # Logs will be saved in log_dir/monitor.csv
        env = Monitor(env, log_dir)

    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    try:
        agent.get_env_info(base_env)
        base_env.on_training_start()
        agent.learn(env, total_timesteps=train_timesteps, verbose=1)
        base_env.on_training_end()
    except KeyboardInterrupt:
        pass

    env.close()

    if save_handler is not None:
        save_handler.save_agent()

    if train_logging == TrainLogging.PLOT:
        plot_results(log_dir)

## Run Human vs AI match function
import pygame
from pygame.locals import QUIT

def run_real_time_match(
    agent_1: UserInputAgent,
    agent_2: Agent,
    max_timesteps=30*90,
    resolution=CameraResolution.LOW,
    allowed_keys: Optional[List[str]] = None,
    env_wrappers: Optional[List[Callable[[gymnasium.Env], gymnasium.Env]]] = None,
):
    pygame.init()

    pygame.mixer.init()

    # Load your soundtrack (must be .wav, .ogg, or supported format)
    pygame.mixer.music.load("environment/assets/soundtrack.mp3")

    # Play it on loop: -1 = loop forever
    pygame.mixer.music.play(-1)

    # Optional: set volume (0.0 to 1.0)
    pygame.mixer.music.set_volume(0.2)

    resolutions = {
        CameraResolution.LOW: (480, 720),
        CameraResolution.MEDIUM: (720, 1280),
        CameraResolution.HIGH: (1080, 1920)
    }
    
    screen = pygame.display.set_mode(resolutions[resolution][::-1])  # Set screen dimensions


    pygame.display.set_caption("AI Squared - Player vs AI Demo")

    clock = pygame.time.Clock()

    # Initialize environment
    env = WarehouseBrawl(resolution=resolution, train_mode=False)

    allowed_indices: Optional[List[int]] = None

    if env_wrappers and isinstance(env, gymnasium.Env):
        for wrapper_fn in env_wrappers:
            env = wrapper_fn(env)
    elif env_wrappers:
        warnings.warn("env_wrappers ignored: WarehouseBrawl is not a gymnasium.Env instance.")

    if allowed_keys is not None:
        if isinstance(env, gymnasium.Env):
            env = ActionSubsetWrapper(env, allowed_keys=allowed_keys)
        else:
            allowed_indices = _resolve_allowed_action_indices(getattr(env, "act_helper", None), allowed_keys)

    observations, _ = env.reset()
    obs_1 = observations[0]
    obs_2 = observations[1]

    if not agent_1.initialized: agent_1.get_env_info(env)
    if not agent_2.initialized: agent_2.get_env_info(env)

    # Run the match loop
    running = True
    timestep = 0
   # platform1 = env.objects["platform1"] #mohamed
    #stage2 = env.objects["stage2"]
    background_image = pygame.image.load('environment/assets/map/bg.jpg').convert() 
    while running and timestep < max_timesteps:
       
        # Pygame event to handle real-time user input 
       
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == pygame.VIDEORESIZE:
                 screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
       
        action_1 = agent_1.predict(obs_1)

        # AI input
        action_2 = agent_2.predict(obs_2)

        # Sample action space
        full_action = {0: action_1, 1: action_2}

        if allowed_indices is not None:
            full_action = {
                agent_id: _filter_action_vector(act, allowed_indices)
                for agent_id, act in full_action.items()
            }

        observations, rewards, terminated, truncated, info = env.step(full_action)
        obs_1 = observations[0]
        obs_2 = observations[1]

        # Render the game
        
        img = env.render()
        screen.blit(pygame.surfarray.make_surface(img), (0, 0))
     
        pygame.display.flip()

        # Control frame rate (30 fps)
        clock.tick(30)

        # If the match is over (either terminated or truncated), stop the loop
        if terminated or truncated:
            running = False

        timestep += 1

    # Clean up pygame after match
    pygame.quit()

    # Return match stats
    player_1_stats = env.get_stats(0)
    player_2_stats = env.get_stats(1)

    if player_1_stats.lives_left > player_2_stats.lives_left:
        result = Result.WIN
    elif player_1_stats.lives_left < player_2_stats.lives_left:
        result = Result.LOSS
    else:
        result = Result.DRAW
    
    match_stats = MatchStats(
        match_time=timestep / 30.0,
        player1=player_1_stats,
        player2=player_2_stats,
        player1_result=result
    )

    # Close environment
    env.close()

    return match_stats
