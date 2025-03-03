# Custom Taxi Environment with Curriculum Learning

This repository contains a custom implementation of the Taxi environment with additional features:

1. **Custom Grid Size**: Unlike the standard Taxi-v3 environment which is fixed at 5x5, this environment supports any grid size.
2. **Reward Shaping**: The environment provides rewards for:
   - Moving toward the passenger
   - Successfully picking up the passenger
   - Moving toward the destination after pickup
   - Successfully dropping off the passenger at the destination
3. **Curriculum Learning**: Training script that gradually increases grid size from 5x5 up to 50x50.
4. **Compatibility**: The trained agent is compatible with both the custom environment and the original simple_custom_taxi_env.

## Files

- `custom_taxi_env.py`: The custom taxi environment implementation
- `student_agent.py`: A sample agent implementation
- `train_dqn_curriculum.py`: Script for training a DQN agent with curriculum learning
- `test_compatibility.py`: Script for testing compatibility between environments
- `simple_custom_taxi_env.py`: The original taxi environment (for reference)

## Environment Details

### Observation Space

The observation is a 16-dimensional vector, matching the format of simple_custom_taxi_env.py:
```
(taxi_row, taxi_col, 
 station0_row, station0_col, station1_row, station1_col, 
 station2_row, station2_col, station3_row, station3_col,
 obstacle_north, obstacle_south, obstacle_east, obstacle_west, 
 passenger_look, destination_look)
```

### Action Space

The action space consists of 6 discrete actions:
- 0: Move south
- 1: Move north
- 2: Move east
- 3: Move west
- 4: Pickup passenger
- 5: Dropoff passenger

### Rewards

The environment provides the following rewards:
- +0.5 for moving toward the passenger (when passenger not in taxi)
- -0.2 for moving away from the passenger (when passenger not in taxi)
- +10 for successfully picking up the passenger
- -5 for invalid pickup attempt
- +0.5 for moving toward the destination (when passenger in taxi)
- -0.2 for moving away from the destination (when passenger in taxi)
- +50 for successfully dropping off the passenger at the destination
- -5 for invalid dropoff attempt
- -0.1 base movement penalty (to encourage efficiency)
- -10 for running out of fuel

### Terminal States

The episode ends when:
1. The passenger is successfully dropped off at the destination
2. The taxi runs out of fuel

## Usage

### Creating an Environment

```python
from custom_taxi_env import CustomTaxiEnv

# Create environment with custom parameters
env = CustomTaxiEnv(
    grid_size=7,  # Custom grid size (default is 5)
    fuel_limit=100,  # Fuel limit (default is 5000)
    render_mode='human'  # Render mode (default is 'ansi')
)

# Reset the environment
obs, info = env.reset()

# Take a step
action = 0  # Move south
obs, reward, done, truncated, info = env.step(action)

# Render the environment
env.render()
```

### Training with Curriculum Learning

```bash
# Train the agent with curriculum learning
python train_dqn_curriculum.py --train

# Test the trained agent
python train_dqn_curriculum.py --test --render --load models/best_dqn_taxi_50x50.pth
```

The curriculum learning process:
1. Starts with a 5x5 grid (like the original environment)
2. Gradually increases the grid size: 5x5 → 10x10 → 15x15 → 20x20 → 30x30 → 40x40 → 50x50
3. Trains longer on larger grid sizes
4. Saves models at each stage of training

### Testing Compatibility

```bash
# Test compatibility between environments
python test_compatibility.py --render

# Test only in custom environment
python test_compatibility.py --custom_only --grid_size 50 --render

# Test only in simple environment
python test_compatibility.py --simple_only --render
```

## DQN Agent

The DQN agent implementation includes:
- Experience replay buffer
- Target network for stable learning
- Epsilon-greedy exploration
- Soft updates for target network
- State normalization for better generalization across grid sizes

## Customization

You can customize the environment further by modifying `custom_taxi_env.py`. Some possible extensions:
- Add obstacles
- Change the station locations
- Modify the reward structure
- Add fuel stations
- Implement more complex passenger behavior 