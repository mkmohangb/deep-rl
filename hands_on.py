import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

def evaluate(model: BaseAlgorithm,
             env: gym.Env,
             n_eval_episodes: int = 100,
             deterministic: bool = False
             )->float:
    """
    Evaluate an RL agent for `n_eval_episodes`.

    :param model: the RL Agent
    :param env: the gym Environment
    :param n_eval_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean episodic reward for the last `n_eval_episodes`
     (Mean over episodes of the cumulative episodic reward)
    """
    episode_cumulative_reward = []
    for _ in range(n_eval_episodes):
        cumulative_reward = 0.0
        done = False
        obs, _ = env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += reward
            done = terminated or truncated
            if done:
                episode_cumulative_reward.append(cumulative_reward)

    mean_episode_reward = np.mean(episode_cumulative_reward)
    std_episode_reward = np.std(episode_cumulative_reward)
    print(f"Mean episode reward: {mean_episode_reward:.2f} +/- {std_episode_reward:.2f}")

    return mean_episode_reward


def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode='rgb_array')])
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder, 
                                record_video_trigger=lambda step: step == 0,
                                video_length=video_length, name_prefix=prefix)
    obs = eval_env.reset()
    for _ in range(video_length):
        action,_ = model.predict(obs, deterministic=True)
        obs, _, _, _ = eval_env.step(action)

    eval_env.close()



env_id = "Pendulum-v1"
env = gym.make(env_id)

model = PPO("MlpPolicy", env, gamma=0.98, use_sde=True,
            sde_sample_freq=4, learning_rate=1e-3, verbose=1)

print(env.observation_space)
print(env.action_space)

# retrieve first observation
obs, _ = env.reset()

#predict the action to take given the observation
action, _ = model.predict(obs, deterministic=True)

assert env.action_space.contains(action)
print(action)

obs, reward, terminated, truncated, info = env.step(action)

print(f"obs_shape={obs.shape}, reward={reward}, done? {terminated or truncated}")

# manually evaluate
env.reset(seed=42)
mean_reward = evaluate(model, env, n_eval_episodes=20, deterministic=True) 
print(f"{mean_reward:.2f}")

# Using inbuilt evaluate
env.reset(seed=42)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# train the model
model.learn(total_timesteps=50000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"after training, mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

print('recording video...')
record_video('Pendulum-v1', model, video_length=500, prefix='ppo-pendulum')
print('done')
