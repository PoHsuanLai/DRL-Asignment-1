import os
import numpy as np
import torch
import argparse
import time
from IPython.display import clear_output

# Import both environments
from custom_taxi_env import CustomTaxiEnv
from env.simple_custom_taxi_env import SimpleTaxiEnv

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def preprocess_state(state, grid_size=5):
    """Convert the state to a normalized numpy array for the DQN"""
    # Convert to numpy array
    state_array = np.array(state, dtype=np.float32)
    
    # Normalize spatial coordinates by grid size to help generalization
    # Positions (taxi, stations) are in the first 10 elements
    for i in range(min(10, len(state_array))):
        state_array[i] = state_array[i] / grid_size
    
    return state_array

def load_dqn_model(model_path, state_size=16, action_size=6):
    """Load a trained DQN model"""
    from train_dqn_curriculum import DQN
    
    # Create model with the same hidden size as used during training
    model = DQN(state_size, action_size, hidden_size=256).to(device)
    
    # Load weights
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        model.eval()
        print(f"Successfully loaded model from {model_path}")
        return model
    else:
        print(f"Model file not found: {model_path}")
        return None

def get_action(model, state, grid_size=5):
    """Get action from the model using the state"""
    # Preprocess state
    processed_state = preprocess_state(state, grid_size)
    
    # Convert to tensor
    state_tensor = torch.from_numpy(processed_state).float().unsqueeze(0).to(device)
    
    # Get action values
    with torch.no_grad():
        action_values = model(state_tensor)
    
    # Return best action
    return np.argmax(action_values.cpu().data.numpy())

def test_in_custom_env(model, grid_size=50, num_episodes=3, render=True):
    """Test the model in the custom environment"""
    print("\n=== Testing in Custom Environment ===")
    env = CustomTaxiEnv(grid_size=grid_size, fuel_limit=grid_size*grid_size*2)
    
    total_score = 0
    success_count = 0
    max_steps = 100  # Limit to 100 steps to prevent infinite loops
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        score = 0
        done = False
        step = 0
        
        while not done and step < max_steps:  # Limit to max_steps
            action = get_action(model, state, grid_size)
            next_state, reward, done, _, _ = env.step(action)
            
            if render:
                clear_output(wait=True)
                print(f"Episode {episode+1}/{num_episodes}, Step {step}")
                print(f"Action: {env.get_action_name(action)}, Reward: {reward}")
                env.render()
                time.sleep(0.1)
            
            state = next_state
            score += reward
            step += 1
            
            # Check if passenger was successfully delivered
            if done and reward > 0:
                success_count += 1
        
        # Check if we hit the step limit
        if step >= max_steps:
            print(f"Episode {episode+1} reached the maximum number of steps ({max_steps})")
        
        total_score += score
        print(f"Episode {episode+1}/{num_episodes}, Score: {score}, Steps: {step}")
    
    avg_score = total_score / num_episodes
    success_rate = success_count / num_episodes * 100
    print(f"Custom Env - Average Score: {avg_score:.2f}, Success Rate: {success_rate:.2f}%")
    
    return avg_score, success_rate

def test_in_simple_env(model, num_episodes=3, render=True):
    """Test the model in the simple environment"""
    print("\n=== Testing in Simple Environment ===")
    env = SimpleTaxiEnv(fuel_limit=5000)
    
    total_score = 0
    success_count = 0
    max_steps = 100  # Limit to 100 steps to prevent infinite loops
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        score = 0
        done = False
        step = 0
        
        while not done and step < max_steps:  # Limit to max_steps
            action = get_action(model, state, grid_size=5)  # Simple env is always 5x5
            next_state, reward, done, _, _ = env.step(action)
            
            if render:
                clear_output(wait=True)
                print(f"Episode {episode+1}/{num_episodes}, Step {step}")
                print(f"Action: {env.get_action_name(action)}, Reward: {reward}")
                
                # Extract taxi position for rendering
                taxi_row, taxi_col = next_state[0], next_state[1]
                env.render_env((taxi_row, taxi_col), action=action, step=step, fuel=env.current_fuel)
                time.sleep(0.1)
            
            state = next_state
            score += reward
            step += 1
            
            # Check if passenger was successfully delivered
            if done and reward > 0:
                success_count += 1
        
        # Check if we hit the step limit
        if step >= max_steps:
            print(f"Episode {episode+1} reached the maximum number of steps ({max_steps})")
        
        total_score += score
        print(f"Episode {episode+1}/{num_episodes}, Score: {score}, Steps: {step}")
    
    avg_score = total_score / num_episodes
    success_rate = success_count / num_episodes * 100
    print(f"Simple Env - Average Score: {avg_score:.2f}, Success Rate: {success_rate:.2f}%")
    
    return avg_score, success_rate

def main():
    parser = argparse.ArgumentParser(description='Test DQN agent compatibility between environments')
    parser.add_argument('--model', type=str, default="models/best_dqn_taxi_50x50.pth", help='Path to the model file')
    parser.add_argument('--grid_size', type=int, default=50, help='Grid size for custom environment')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to test')
    parser.add_argument('--render', action='store_true', help='Render the environment during testing')
    parser.add_argument('--custom_only', action='store_true', help='Test only in custom environment')
    parser.add_argument('--simple_only', action='store_true', help='Test only in simple environment')
    args = parser.parse_args()
    
    # Load model
    model = load_dqn_model(args.model)
    if model is None:
        return
    
    # Test in environments
    if not args.simple_only:
        custom_score, custom_success = test_in_custom_env(model, args.grid_size, args.episodes, args.render)
    
    if not args.custom_only:
        simple_score, simple_success = test_in_simple_env(model, args.episodes, args.render)
    
    # Print summary if tested in both environments
    if not args.simple_only and not args.custom_only:
        print("\n=== Compatibility Summary ===")
        print(f"Custom Environment ({args.grid_size}x{args.grid_size}) - Avg Score: {custom_score:.2f}, Success Rate: {custom_success:.2f}%")
        print(f"Simple Environment (5x5) - Avg Score: {simple_score:.2f}, Success Rate: {simple_success:.2f}%")

if __name__ == "__main__":
    main() 