from environment.environment import RenderMode, CameraResolution
from environment.agent import run_match
from user.train_agent import (
    UserInputAgent,
    BasedAgent,
    ConstantAgent,
    ClockworkAgent,
    SB3Agent,
    RecurrentPPOAgent,
    STAGE1_ALLOWED_KEYS,
)  # add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame
pygame.init()

my_agent = UserInputAgent()

#Input your file path here in SubmittedAgent if you are loading a model:
opponent = SubmittedAgent(file_path="/home/littlestronomer/Documents/UTMIST/UTMIST-AI2-AgentSmiths/checkpoints/stage1_experiment/rl_model_405000_steps")

match_time = 99999

# Run a single real-time match
run_match(
    agent_1=opponent,
    agent_2=my_agent,
    max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
    video_path='tt_agent.mp4',
    allowed_keys=STAGE1_ALLOWED_KEYS,
)
