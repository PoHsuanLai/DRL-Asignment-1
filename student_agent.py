# Remember to adjust your student ID in meta.xml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os

# Patch for NumPy compatibility issue
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Global agent instance
agent = None
model_path = "best_dqn_taxi_CustomTaxiEnv_final.pth"  # Path to the best DQN model

# Check for CUDA availability but default to CPU for compatibility
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-Network - we only need the local network for inference
        self.qnetwork = DQN(state_size, action_size, hidden_size).to(device)
        
        # Load the model
        self.model_loaded = self.load(model_path)
        if self.model_loaded:
            print(f"Successfully loaded model from {model_path}")
        else:
            print(f"Failed to load model from {model_path}, will use random actions")
        
    def act(self, state, eps=0.0):
        # If model failed to load, use random actions
        if not self.model_loaded:
            return random.randint(0, self.action_size - 1)
            
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def load(self, filename):
        """Load the DQN model"""
        if os.path.isfile(filename):
            try:
                # Use map_location to ensure model loads even if trained on different device
                # Use weights_only=True for security and to avoid warnings
                checkpoint = torch.load(filename, map_location=device, weights_only=True)
                
                # Check if the checkpoint contains the expected keys
                if 'qnetwork_local_state_dict' in checkpoint:
                    # Load the local network weights (the one used for inference)
                    self.qnetwork.load_state_dict(checkpoint['qnetwork_local_state_dict'])
                    print("Loaded qnetwork_local_state_dict successfully")
                    return True
                elif 'qnetwork_state_dict' in checkpoint:
                    # Alternative key that might be used
                    self.qnetwork.load_state_dict(checkpoint['qnetwork_state_dict'])
                    print("Loaded qnetwork_state_dict successfully")
                    return True
                else:
                    # If the checkpoint structure is different, try to load the first state_dict found
                    for key in checkpoint:
                        if 'state_dict' in key:
                            try:
                                self.qnetwork.load_state_dict(checkpoint[key])
                                print(f"Loaded {key} successfully")
                                return True
                            except:
                                continue
                    
                    print("Could not find a compatible state_dict in the checkpoint")
                    return False
            except Exception as e:
                print(f"Error loading DQN model: {e}")
                return False
        else:
            print(f"DQN model file not found: {filename}")
            return False

def preprocess_state(state, grid_size=50):
    """Convert the state to a normalized numpy array for the DQN"""
    # Extract the state vector from the tuple if needed
    if isinstance(state, tuple) and len(state) == 2:
        state_vector = state[0]
    else:
        state_vector = state
    
    # Convert to numpy array
    state_array = np.array(state_vector, dtype=np.float32)
    
    # Pad the state array if it's too short
    if len(state_array) < 16:
        padded_array = np.zeros(16, dtype=np.float32)
        padded_array[:len(state_array)] = state_array
        state_array = padded_array
    
    # Normalize spatial coordinates by grid size to help generalization
    # Positions (taxi, stations) are in the first 10 elements
    for i in range(min(10, len(state_array))):
        state_array[i] = state_array[i] / grid_size
    
    return state_array

def get_action(obs):
    """Return an action based on the current observation using the DQN model"""
    global agent
    
    # Initialize DQN agent if it doesn't exist
    if agent is None:
        # State size is 16 (taxi environment state vector)
        # Action size is 6 (north, east, south, west, pickup, dropoff)
        agent = DQNAgent(state_size=16, action_size=6, hidden_size=64)
    
    # Extract grid_size from info dict if available
    grid_size = 50  # Default to custom environment grid size
    if isinstance(obs, tuple) and len(obs) == 2:
        if isinstance(obs[1], dict) and 'grid_size' in obs[1]:
            grid_size = obs[1]['grid_size']
    
    try:
        # Preprocess the state
        state = preprocess_state(obs, grid_size)
        
        # Get action from DQN with no exploration for testing
        action = agent.act(state, eps=0.0)
        
        # Check if action mask is available
        if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict) and 'action_mask' in obs[1]:
            action_mask = obs[1]['action_mask']
            if action_mask[action] == 0:  # Invalid action
                # Choose a random valid action instead
                valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                if valid_actions:
                    return random.choice(valid_actions)
        
        return action
    except Exception as e:
        # If any error occurs, return a random action
        print(f"Error in get_action: {e}")
        return random.randint(0, 5)

# # If this file is run directly, test the agent
# if __name__ == "__main__":
#     try:
#         from env.simple_custom_taxi_env import SimpleTaxiEnv
#         env_class = SimpleTaxiEnv
#         print("Using SimpleTaxiEnv for testing")
#     except ImportError:
#         print("Could not import SimpleTaxiEnv")
#         exit(1)
    
#     # Create environment - use grid size 5 for SimpleTaxiEnv
#     grid_size = 5
#     env_config = {
#         "grid_size": grid_size,
#         "fuel_limit": grid_size * grid_size * 2
#     }
#     env = env_class(**env_config)
    
#     # Test the trained agent
#     print("\nTesting trained agent...")
#     print(f"Environment: {env_class.__name__}, Grid Size: {grid_size}")
    
#     state, info = env.reset()
#     print(f"Initial state shape: {np.array(state).shape if hasattr(state, 'shape') else len(state)}")
#     print(f"Initial state: {state}")
#     if isinstance(info, dict) and 'grid_size' in info:
#         print(f"Grid size from info: {info['grid_size']}")
    
#     total_reward = 0
#     done = False
#     step = 0
    
#     while not done and step < 1000:  # Add step limit to prevent infinite loops
#         action = get_action((state, {'grid_size': grid_size}))
#         next_state, reward, done, _, info = env.step(action)
#         print(f"Step {step}: Action={action}, Reward={reward}")
        
#         if step % 10 == 0:  # Print state every 10 steps to avoid too much output
#             print(f"State: {next_state}")
        
#         state = next_state
#         total_reward += reward
#         step += 1
    
#     print(f"Test completed with total reward: {total_reward} after {step} steps")
    
