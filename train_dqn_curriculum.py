import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import argparse
from env.custom_taxi_env import CustomTaxiEnv
from env.simple_custom_taxi_env import SimpleTaxiEnv
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
# Ensure directory exists for saving models
os.makedirs("models", exist_ok=True)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

# Define DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

        # self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return self.fc2(x)

# Define DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, buffer_size=100000, batch_size=512, 
                 gamma=0.99, tau=1e-3, lr=5e-4, update_every=4, weight_decay=1e-4, 
                 scheduler_type='cosine', min_lr=1e-5, scheduler_steps=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.scheduler_steps = scheduler_steps
        
        # Q-Networks
        self.qnetwork_local = DQN(state_size, action_size, hidden_size).to(device)
        self.qnetwork_target = DQN(state_size, action_size, hidden_size).to(device)
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.qnetwork_local.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=scheduler_steps, 
                eta_min=min_lr
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.5, 
                patience=100,
                min_lr=min_lr
            )
        else:
            self.scheduler = None
        
        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size)
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        
        # Track losses and learning rates for monitoring
        self.losses = []
        self.current_loss = 0
        self.lr_history = []
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            
            # Step the scheduler if using cosine annealing
            if self.scheduler is not None and self.scheduler_type == 'cosine':
                self.scheduler.step()
                # Record current learning rate
                self.lr_history.append(self.get_lr())
    
    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.current_loss = loss.item()
        self.losses.append(self.current_loss)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        # Step the plateau scheduler if using it
        if self.scheduler is not None and self.scheduler_type == 'plateau':
            self.scheduler.step(Q_expected.mean().item())
            # Record current learning rate
            self.lr_history.append(self.get_lr())
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def get_lr(self):
        """Return the current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def save(self, filename):
        """Save the DQN model"""
        checkpoint = {
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'losses': self.losses,
            'lr_history': self.lr_history,
            'scheduler_type': self.scheduler_type
        }
        
        # Save scheduler state if it exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filename)
    
    def load(self, filename):
        """Load the DQN model"""
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
            self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load losses if available
            if 'losses' in checkpoint:
                self.losses = checkpoint['losses']
            
            # Load learning rate history if available
            if 'lr_history' in checkpoint:
                self.lr_history = checkpoint['lr_history']
            
            # Load scheduler if it exists in the checkpoint
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            return True
        return False

def detect_environment_type(state):
    """Detect if we're in the simple environment or custom environment"""
    # Simple environment has a state length of 16 with specific structure
    if isinstance(state, tuple):
        state_vector = state[0]
    else:
        state_vector = state
    
    state_length = len(state_vector) if hasattr(state_vector, '__len__') else 0
    
    if state_length <= 5:
        return "simple", 5  # Simple environment with grid size 5
    else:
        return "custom", 50  # Custom environment with default grid size 50

def preprocess_state(state, grid_size=50, env_type="custom"):
    """Convert the state to a normalized numpy array for the DQN"""
    # Extract the state vector from the tuple if needed
    if isinstance(state, tuple) and len(state) == 2:
        state_vector = state[0]
    else:
        state_vector = state
    
    # Convert to numpy array
    state_array = np.array(state_vector, dtype=np.float32)
    
    # Special handling for simple environment
    if env_type == "simple":
        # Simple environment has a different state structure
        # We need to convert it to match the format expected by our model
        # Typically has 4 or 5 elements: taxi_row, taxi_col, passenger_idx, destination_idx, (optional: passenger_in_taxi)
        simple_state = np.zeros(16, dtype=np.float32)
        
        # Map the simple state to our expected format
        if len(state_array) >= 2:
            # Taxi position (row, col)
            simple_state[0] = state_array[0]  # taxi row
            simple_state[1] = state_array[1]  # taxi col
            
            # Passenger location (using passenger_idx to determine location)
            if len(state_array) >= 3 and state_array[2] < 4:  # passenger at a location
                passenger_idx = int(state_array[2])
                # Map passenger index to a position (simplified mapping)
                passenger_locations = [(0, 0), (0, 4), (4, 0), (4, 4)]  # R, G, Y, B locations
                if passenger_idx < len(passenger_locations):
                    simple_state[2] = passenger_locations[passenger_idx][0]  # passenger row
                    simple_state[3] = passenger_locations[passenger_idx][1]  # passenger col
            
            # Destination (using destination_idx)
            if len(state_array) >= 4:
                dest_idx = int(state_array[3])
                # Map destination index to a position
                dest_locations = [(0, 0), (0, 4), (4, 0), (4, 4)]  # R, G, Y, B locations
                if dest_idx < len(dest_locations):
                    simple_state[4] = dest_locations[dest_idx][0]  # destination row
                    simple_state[5] = dest_locations[dest_idx][1]  # destination col
            
            # Passenger in taxi flag
            if len(state_array) >= 5:
                simple_state[10] = 1.0 if state_array[2] == 4 else 0.0  # passenger in taxi
        
        state_array = simple_state
    else:
        # Pad the state array if it's too short for the custom environment
        if len(state_array) < 16:
            padded_array = np.zeros(16, dtype=np.float32)
            padded_array[:len(state_array)] = state_array
            state_array = padded_array
    
    # Normalize spatial coordinates by grid size to help generalization
    # Positions (taxi, stations) are in the first 10 elements
    for i in range(min(10, len(state_array))):
        state_array[i] = state_array[i] / grid_size
    
    return state_array

def train_curriculum(agent, env_class, grid_sizes, episodes_per_size, max_steps=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Train the agent using curriculum learning with increasing grid sizes.
    
    Args:
        agent: The DQN agent
        env_class: The environment class to use (CustomTaxiEnv or SimpleTaxiEnv)
        grid_sizes: List of grid sizes to train on
        episodes_per_size: List of number of episodes to train for each grid size
        max_steps: Maximum steps per episode
        eps_start: Starting epsilon for exploration
        eps_end: Minimum epsilon
        eps_decay: Decay rate for epsilon
    """
    scores = []
    eps = eps_start
    
    # For each grid size in the curriculum
    for i, grid_size in enumerate(grid_sizes):
        # Get number of episodes for this grid size
        num_episodes = episodes_per_size[i]
        
        # Create environment with current grid size
        if env_class == SimpleTaxiEnv:
            env = env_class(fuel_limit=grid_size*grid_size*2)
        else:
            env = env_class(grid_size=grid_size, fuel_limit=grid_size*grid_size*2)
        
        # Train for specified number of episodes at this grid size
        with tqdm(total=num_episodes, desc=f"Training on grid size {grid_size}x{grid_size}", 
                  position=0, leave=True) as pbar:
            for episode in range(1, num_episodes + 1):
                state, _ = env.reset()
                
                # Detect environment type and preprocess state
                env_type, _ = detect_environment_type(state)
                state = preprocess_state(state, grid_size, env_type)
                
                score = 0
                
                for step in range(max_steps):
                    # Select action
                    action = agent.act(state, eps)
                    
                    # Take action
                    next_state, reward, done, _, _ = env.step(action)
                    
                    # Preprocess next state
                    next_state = preprocess_state(next_state, grid_size, env_type)
                    
                    # Store experience and learn
                    agent.step(state, action, reward, next_state, done)
                    
                    # Update state and score
                    state = next_state
                    score += reward
                    
                    if done:
                        break
                
                # Decay epsilon
                eps = max(eps_end, eps_decay * eps)
                
                # Save score
                scores.append(score)
                
                # Calculate statistics for display
                avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                current_lr = agent.get_lr()
                
                # Update progress bar with current episode info
                pbar.set_description(f"Training on {grid_size}x{grid_size} - Episode {episode}/{num_episodes}")
                pbar.set_postfix({
                    "score": f"{score:.2f}", 
                    "avg_score": f"{avg_score:.2f}",
                    "epsilon": f"{eps:.4f}", 
                    "lr": f"{current_lr:.6f}",
                    "steps": step+1
                })
                pbar.update(1)
                
                # Save model periodically
                if episode % 500 == 0:
                    agent.save(f"models/dqn_taxi_{grid_size}x{grid_size}_{env_class.__name__}_ep{episode}.pth")
                    print(f"Model saved to models/dqn_taxi_{grid_size}x{grid_size}_{env_class.__name__}_ep{episode}.pth")
        
        # Save final model for this grid size
        agent.save(f"models/dqn_taxi_{grid_size}x{grid_size}_{env_class.__name__}_final.pth")
        print(f"Model saved to models/dqn_taxi_{grid_size}x{grid_size}_{env_class.__name__}_final.pth")
    
    # Save the final model
    agent.save(f"models/best_dqn_taxi_{env_class.__name__}_final.pth")
    print(f"Model saved to models/best_dqn_taxi_{env_class.__name__}_final.pth")
    return scores

def plot_scores(scores):
    """Plot the scores during training"""
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Training Scores')
    plt.savefig('results/training_scores.png')
    plt.show()
    plt.close()
    print(f"Training scores saved to results/training_scores.png")

def plot_training_metrics(agent):
    """Plot the loss and learning rate history"""
    if not hasattr(agent, 'losses') or len(agent.losses) == 0:
        print("No loss history available to plot")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot losses
    ax1.plot(agent.losses)
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Apply smoothing for better visualization
    window_size = min(100, len(agent.losses) // 10)
    if window_size > 1:
        smoothed_losses = np.convolve(agent.losses, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(agent.losses)), smoothed_losses, 'r-', linewidth=2)
        ax1.legend(['Raw Loss', 'Smoothed Loss'])
    
    # If we have learning rate history, plot it
    if hasattr(agent, 'lr_history') and len(agent.lr_history) > 0:
        ax2.plot(agent.lr_history)
        ax2.set_ylabel('Learning Rate')
        ax2.set_xlabel('Training Steps')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)
        ax2.set_yscale('log')  # Log scale for learning rate
    else:
        ax2.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('results/training_metrics.png')
    plt.show()
    plt.close()
    print(f"Training metrics saved to results/training_metrics.png")

def test_agent(agent, env_class, grid_size=50, num_episodes=10, render=True):
    """Test the trained agent"""
    if env_class == SimpleTaxiEnv:
        env = env_class(fuel_limit=grid_size*grid_size*2)
    else:
        env = env_class(grid_size=grid_size, fuel_limit=grid_size*grid_size*2)
    
    
    total_score = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        # Detect environment type and preprocess state
        env_type, _ = detect_environment_type(state)
        state = preprocess_state(state, grid_size, env_type)
        
        score = 0
        done = False
        step = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Preprocess next state
            next_state = preprocess_state(next_state, grid_size, env_type)
            
            if render and episode == 0:
                clear_output(wait=True)
                print(f"Step {step}, Action: {env.get_action_name(action)}, Reward: {reward}")
                env.render()
                time.sleep(0.1)
            
            state = next_state
            score += reward
            step += 1
        
        total_score += score
        print(f"Episode {episode+1}/{num_episodes}, Score: {score}")
    
    avg_score = total_score / num_episodes
    print(f"Average Score: {avg_score}")
    return avg_score

def main():
    parser = argparse.ArgumentParser(description='Train a DQN agent with curriculum learning')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--test', action='store_true', help='Test the agent')
    parser.add_argument('--render', action='store_true', help='Render the environment during testing')
    parser.add_argument('--load', type=str, default='models/best_dqn_taxi_50x50.pth', help='Load a saved model')
    parser.add_argument('--env', type=str, default='custom', choices=['custom', 'simple'], help='Environment to use')
    
    # Add new arguments for optimizer and scheduler
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'none'], 
                        help='Learning rate scheduler type')
    parser.add_argument('--scheduler_steps', type=int, default=10000, 
                        help='Number of steps for cosine annealing scheduler')
    
    args = parser.parse_args()
    
    # Define state and action sizes
    state_size = 16  # Size of the state vector
    action_size = 6  # Number of possible actions
    
    # Choose environment class based on argument
    env_class = CustomTaxiEnv if args.env == 'custom' else SimpleTaxiEnv
    
    # Create agent with command line parameters
    scheduler_type = None if args.scheduler == 'none' else args.scheduler
    
    agent = DQNAgent(
        state_size=state_size, 
        action_size=action_size, 
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        scheduler_type=scheduler_type,
        scheduler_steps=args.scheduler_steps
    )
    
    # Load model if specified
    if args.load:
        if os.path.isfile(args.load):
            print(f"Loading model from {args.load}")
            agent.load(args.load)
        else:
            print(f"Model file {args.load} does not exist, using random weights")
    
    # Train or test based on arguments
    if args.train:
        # Define curriculum (gradually increasing grid sizes)
        if args.env == 'custom':
            grid_sizes = [5, 10]
            episodes_per_size = [1000, 500]
            # grid_sizes = [5, 10, 25, 35, 50]
            # episodes_per_size = [400, 400, 200, 150, 100]
        else:
            # For simple environment, just train on grid size 5
            grid_sizes = [5]
            episodes_per_size = [1000]
        
        # Train with curriculum learning
        print(f"Starting curriculum training on {args.env} environment...")
        scores = train_curriculum(agent, env_class, grid_sizes, episodes_per_size)
        
        # Plot training scores
        os.makedirs("results", exist_ok=True)
        plot_scores(scores)
        plot_training_metrics(agent)
    
    if args.test:
        # Test the agent
        print(f"Testing agent on {args.env} environment...")
        grid_size = 50 if args.env == 'custom' else 5
        test_agent(agent, env_class, grid_size=grid_size, render=args.render)

if __name__ == "__main__":
    main() 