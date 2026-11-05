import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from torch.distributions import Normal

from pool_env import PoolEnv


class RewardWrapper(gym.Wrapper):
    def step(self, action):
        best = self.unwrapped.best_shot
        env = self.unwrapped

        reward = 0.0
        obs, _, term, trunc, info = self.env.step(action)

        # 1. Angle Alignment Reward
        v_opt = np.array([best[0], best[1]]) / np.hypot(best[0], best[1])
        v_chosen = np.array([np.cos(action[0] * np.pi), np.sin(action[0] * np.pi)])
        alignment = (np.clip(np.dot(v_opt, v_chosen), -1.0, 1.0) + 1.0) / 2.0
        reward += 25.0 * (alignment**2)

        # 2. Power Reward (only if agent hits the right suit)
        target_ball = info.get("target_ball")
        target_pocket = env.best_target_pocket

        if target_ball is not None and target_ball.ballclass == env.suit:
            final_dist = (
                np.hypot(
                    target_ball.body.position.x - target_pocket[0],
                    target_ball.body.position.y - target_pocket[1],
                )
                if target_ball not in info.get("ball_pocketed", [])
                else 0.0
            )  # Distance in px between target and pocket

            not_enough_power_penalty = final_dist / 10.0
            too_much_power_penalty = 5.0 * action[1]
            reward += 75.0 - not_enough_power_penalty - too_much_power_penalty

        # 3. Game Events (Penalties only)
        reward -= 75.0 if info.get("cue_ball_pocketed") else 0.0
        reward -= 100.0 if info.get("black_ball_pocketed") else 0.0

        # Clip and normalize
        reward = np.clip(reward, -150.0, 150.0) / 100.0

        return obs, reward, term, trunc, info


class TransformerAgent(nn.Module):
    def __init__(self, hidden_dim=64, n_head=2, n_layers=1):
        super().__init__()
        self.proj = nn.Linear(10, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.0,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.pool = nn.Linear(hidden_dim, 1)
        self.mean = nn.Linear(hidden_dim, 3)
        self.log_std = nn.Linear(hidden_dim, 3)

    def forward(self, state):
        batch_size = state.size(0)
        x, y, c = state[:, 0::3], state[:, 1::3], (state[:, 2::3] * 4).long()

        # Inject pocket coordinates (normalized) as virtual tokens
        pockets_x = (
            torch.tensor([0.0, 1.0, 1.0, 0.0, 0.5, 0.5], device=state.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        pockets_y = (
            torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 1.0], device=state.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        pockets_c = torch.full((batch_size, 6), 5, dtype=torch.long, device=state.device)

        x = torch.cat([x, pockets_x], dim=1)
        y = torch.cat([y, pockets_y], dim=1)
        c = torch.cat([c, pockets_c], dim=1)

        # Calculate relative coordinates from cue ball (class == 3)
        cue_mask = c == 3
        cue_x = torch.sum(x * cue_mask, dim=1, keepdim=True) / (
            torch.sum(cue_mask, dim=1, keepdim=True) + 1e-8
        )
        cue_y = torch.sum(y * cue_mask, dim=1, keepdim=True) / (
            torch.sum(cue_mask, dim=1, keepdim=True) + 1e-8
        )

        rel_x = x - cue_x
        rel_y = y - cue_y

        features = torch.cat(
            [
                torch.stack([x, y, rel_x, rel_y], dim=2),
                nn.functional.one_hot(c, 6).float(),
            ],
            dim=2,
        )

        encoded = self.transformer(self.proj(features))
        pooled = torch.sum(encoded * torch.softmax(self.pool(encoded), dim=1), dim=1)

        mean = self.mean(pooled)
        mean[:, :2] = torch.tanh(mean[:, :2])
        return mean, torch.exp(torch.clamp(self.log_std(pooled), -20, 2))

    def get_action(self, state):
        mean, std = self(state)
        dist = Normal(mean, std)
        sample = dist.sample()

        s, c, p = sample[..., 0], sample[..., 1], sample[..., 2]
        angle = torch.atan2(s, c) / np.pi
        power = torch.sigmoid(p)
        # power = torch.full_like(angle, 1.0)

        action_2d = torch.stack([angle, power], dim=-1)
        return sample, action_2d, dist.log_prob(sample).sum(-1)


def make_env(env_id, phase):
    def _thunk():
        return RewardWrapper(
            PoolEnv(
                algo="GRPO",
                obs_type="vector",
                training=True,
                num_balls=2,
                curriculum_phase=phase,
                env_id=env_id,
            )
        )

    return _thunk


def train():
    NUM_ENVS = 64
    ROLLOUT_STEPS = 10
    ITERS = 250
    BATCH_SIZE = 128
    EPOCHS = 10
    PHASE = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training phase {PHASE} on {DEVICE} with {NUM_ENVS} parallel environments.")

    envs = AsyncVectorEnv(
        [make_env(i, PHASE) for i in range(NUM_ENVS)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    agent = TransformerAgent(
        envs.single_observation_space.shape[0], n_head=4, n_layers=2
    ).to(DEVICE)
    optimizer = optim.AdamW(agent.parameters(), lr=3e-4)

    if PHASE > 2:
        prev_ckpt_path = f"checkpoints/grpo_pool_final_phase{PHASE - 1}.pt"
        # Fallback to the older naming convention if it's picking up from original phase 1
        if not os.path.exists(prev_ckpt_path) and os.path.exists(
            "checkpoints/grpo_pool_final.pt"
        ):
            prev_ckpt_path = "checkpoints/grpo_pool_final.pt"

        if os.path.exists(prev_ckpt_path):
            print(f"Loading weights from previous phase: {prev_ckpt_path}")
            checkpoint = torch.load(
                prev_ckpt_path, map_location=DEVICE, weights_only=False
            )
            agent.load_state_dict(checkpoint["model_state_dict"])
            # Intentionally not loading the optimizer state to allow fresh momentum for the new phase
        else:
            print(
                f"Warning: Could not find previous phase checkpoint ({prev_ckpt_path}). Training from scratch."
            )

    obs = torch.tensor(envs.reset()[0], dtype=torch.float32, device=DEVICE)
    rewards_history = np.zeros((ITERS, ROLLOUT_STEPS, NUM_ENVS))

    for i in range(ITERS):
        b_obs, b_acts, b_logprobs, b_rewards, b_dones = [], [], [], [], []

        # 1. Collect Rollouts
        with torch.no_grad():
            for step in range(ROLLOUT_STEPS):
                sample, action, log_prob = agent.get_action(obs)
                next_obs, reward, term, trunc, _ = envs.step(action.cpu().numpy())

                b_obs.append(obs)
                b_acts.append(sample)
                b_logprobs.append(log_prob)
                b_rewards.append(torch.tensor(reward, dtype=torch.float32, device=DEVICE))
                b_dones.append(
                    torch.tensor(
                        np.logical_or(term, trunc), dtype=torch.float32, device=DEVICE
                    )
                )

                rewards_history[i, step] = reward
                obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)

        b_obs, b_acts = torch.stack(b_obs), torch.stack(b_acts)
        b_logprobs, b_rewards, b_dones = (
            torch.stack(b_logprobs),
            torch.stack(b_rewards),
            torch.stack(b_dones),
        )

        # 2. Advantages (Discounted Returns & Group Normalization)
        returns = torch.zeros_like(b_rewards)
        R = 0
        for t in reversed(range(ROLLOUT_STEPS)):
            R = b_rewards[t] + 0.99 * R * (1.0 - b_dones[t])
            returns[t] = R

        adv = (returns - returns.mean(dim=1, keepdim=True)) / (
            returns.std(dim=1, keepdim=True) + 1e-8
        )

        # 3. Filter for Curriculum (CAPO logic)
        flat_obs, flat_acts = b_obs.view(-1, b_obs.size(-1)), b_acts.view(-1, 3)
        flat_logprobs, flat_adv = b_logprobs.view(-1), adv.view(-1)

        valid = (
            flat_adv >= 0
            if i < int(ITERS * 0.3)
            else torch.ones_like(flat_adv, dtype=torch.bool)
        )
        if valid.sum() > 0:
            flat_obs, flat_acts = flat_obs[valid], flat_acts[valid]
            flat_logprobs, flat_adv = flat_logprobs[valid], flat_adv[valid]

        # 4. Optimize
        for _ in range(EPOCHS):
            indices = torch.randperm(flat_obs.size(0), device=DEVICE)
            for start in range(0, flat_obs.size(0), BATCH_SIZE):
                idx = indices[start : start + BATCH_SIZE]
                mean, std = agent(flat_obs[idx])
                new_logprobs = Normal(mean, std).log_prob(flat_acts[idx]).sum(-1)

                ratio = torch.exp(new_logprobs - flat_logprobs[idx])
                surr1 = ratio * flat_adv[idx]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * flat_adv[idx]

                loss = -torch.min(surr1, surr2).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        print(f"Iter {i + 1}/{ITERS} | Mean Group Return: {b_rewards.mean().item():.2f}")

        # 5. Checkpointing
        if (i + 1) % 50 == 0 or (i + 1) == ITERS:
            ckpt_name = (
                f"grpo_pool_final_phase{PHASE}"
                if i + 1 == ITERS
                else f"checkpoint_phase{PHASE}_{i + 1}"
            )
            torch.save(
                {
                    "iteration": i + 1,
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rewards": rewards_history,
                },
                f"checkpoints/{ckpt_name}.pt",
            )
            print(f"--> Checkpoint saved: checkpoints/{ckpt_name}.pt")

    envs.close()
    print("Training Complete.")


if __name__ == "__main__":
    train()
