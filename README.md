DQN — MountainCar-v0
MountainCar has a notoriously sparse reward — the agent gets -1 every step and nothing else until it reaches the flag.

In practice this means random exploration almost never finds the goal, and the agent stalls early in training.

This project solves that with a custom reward wrapper that adds shaped bonuses based on the car's position and velocity. 

The agent still has to learn the swinging strategy on its own — the shaping just gives it enough signal to get started.

Built with PyTorch. 

Standard DQN setup: replay buffer, target network, epsilon-greedy exploration, Huber loss, gradient clipping. Trained over 500K steps, evaluated visually at the end using PIL rendering
