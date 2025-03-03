import gym
import numpy as np
from gym import spaces
import time
from IPython.display import clear_output

# Patch for NumPy compatibility issue
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

class CustomTaxiEnv(gym.Env):
    """
    Custom Taxi Environment that supports:
    1. Custom grid sizes
    2. Rewards for moving toward passenger
    3. Rewards for picking up passenger
    4. Rewards for moving toward destination after pickup
    
    The state observation format is compatible with simple_custom_taxi_env.py
    """
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, grid_size=5, fuel_limit=5000, render_mode='ansi'):
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.render_mode = render_mode
        
        # Define the action space (same as Taxi-v3)
        # 0: Move south, 1: Move north, 2: Move east, 3: Move west, 4: Pickup, 5: Dropoff
        self.action_space = spaces.Discrete(6)
        
        # Define the observation space
        # This matches the format from simple_custom_taxi_env.py
        # (taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, 
        #  station2_row, station2_col, station3_row, station3_col,
        #  obstacle_north, obstacle_south, obstacle_east, obstacle_west, 
        #  passenger_look, destination_look)
        self.observation_space = spaces.Box(
            low=np.zeros(16),
            high=np.ones(16) * max(grid_size, 1),
            dtype=np.int32
        )
        
        # Define the stations (pickup/dropoff locations)
        self.stations = [
            (0, 0),                      # Bottom-left
            (0, self.grid_size - 1),     # Bottom-right
            (self.grid_size - 1, 0),     # Top-left
            (self.grid_size - 1, self.grid_size - 1)  # Top-right
        ]
        
        # Initialize state variables
        self.taxi_pos = None
        self.passenger_loc = None
        self.destination = None
        self.passenger_picked_up = False
        
        # For tracking previous distances (to calculate rewards for moving toward targets)
        self.prev_passenger_distance = None
        self.prev_destination_distance = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset fuel
        self.current_fuel = self.fuel_limit
        
        # Randomly place taxi
        self.taxi_pos = (
            self.np_random.integers(0, self.grid_size),
            self.np_random.integers(0, self.grid_size)
        )
        
        # Randomly select passenger location from stations
        passenger_idx = self.np_random.integers(0, len(self.stations))
        self.passenger_loc = self.stations[passenger_idx]
        
        # Randomly select destination from stations (different from passenger location)
        destination_idx = passenger_idx
        while destination_idx == passenger_idx:
            destination_idx = self.np_random.integers(0, len(self.stations))
        self.destination = self.stations[destination_idx]
        
        # Reset passenger status
        self.passenger_picked_up = False
        
        # Initialize previous distances
        self.prev_passenger_distance = self._manhattan_distance(self.taxi_pos, self.passenger_loc)
        self.prev_destination_distance = self._manhattan_distance(self.taxi_pos, self.destination)
        
        # Return observation and info
        return self.get_state(), self._get_info()
    
    def get_state(self):
        """
        Return the state in the same format as simple_custom_taxi_env.py
        """
        taxi_row, taxi_col = self.taxi_pos
        passenger_x, passenger_y = self.passenger_loc
        destination_x, destination_y = self.destination
        
        # Obstacle flags (boundaries)
        obstacle_north = int(taxi_row == 0)
        obstacle_south = int(taxi_row == self.grid_size - 1)
        obstacle_east = int(taxi_col == self.grid_size - 1)
        obstacle_west = int(taxi_col == 0)
        
        # Check if passenger is visible from current position
        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
        
        # Check if destination is visible from current position
        destination_loc_north = int((taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int((taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east = int((taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west = int((taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle = int((taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle
        
        # Create state tuple in the same format as simple_custom_taxi_env.py
        state = (
            taxi_row, taxi_col, 
            self.stations[0][0], self.stations[0][1],
            self.stations[1][0], self.stations[1][1],
            self.stations[2][0], self.stations[2][1],
            self.stations[3][0], self.stations[3][1],
            obstacle_north, obstacle_south, obstacle_east, obstacle_west, 
            passenger_look, destination_look
        )
        
        return state
    
    def _get_info(self):
        """Return additional information"""
        return {
            "fuel_remaining": self.current_fuel,
            "passenger_picked_up": self.passenger_picked_up,
            "passenger_distance": self._manhattan_distance(self.taxi_pos, self.passenger_loc),
            "destination_distance": self._manhattan_distance(self.taxi_pos, self.destination),
            "grid_size": self.grid_size
        }
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def step(self, action):
        """Take a step in the environment with the given action"""
        # Initialize reward, done, and truncated flags
        reward = 0
        done = False
        truncated = False
        
        # Store current position for distance calculation
        old_taxi_pos = self.taxi_pos
        
        # Process the action
        if action == 0:  # Move south
            self.taxi_pos = (min(self.taxi_pos[0] + 1, self.grid_size - 1), self.taxi_pos[1])
        elif action == 1:  # Move north
            self.taxi_pos = (max(self.taxi_pos[0] - 1, 0), self.taxi_pos[1])
        elif action == 2:  # Move east
            self.taxi_pos = (self.taxi_pos[0], min(self.taxi_pos[1] + 1, self.grid_size - 1))
        elif action == 3:  # Move west
            self.taxi_pos = (self.taxi_pos[0], max(self.taxi_pos[1] - 1, 0))
        elif action == 4:  # Pickup
            if self.taxi_pos == self.passenger_loc and not self.passenger_picked_up:
                self.passenger_picked_up = True
                reward += 10  # Reward for successful pickup
            else:
                reward -= 5  # Penalty for invalid pickup
        elif action == 5:  # Dropoff
            if self.taxi_pos == self.destination and self.passenger_picked_up:
                self.passenger_picked_up = False
                reward += 50  # Reward for successful dropoff
                done = True  # Episode ends on successful dropoff
            else:
                reward -= 5  # Penalty for invalid dropoff
        
        # Update passenger position if picked up
        if self.passenger_picked_up:
            self.passenger_loc = self.taxi_pos
        
        # Consume fuel
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            reward -= 10  # Penalty for running out of fuel
            done = True
        
        # Calculate rewards for moving toward targets
        if not self.passenger_picked_up:
            # If passenger not picked up, reward for moving toward passenger
            current_passenger_distance = self._manhattan_distance(self.taxi_pos, self.passenger_loc)
            if current_passenger_distance < self.prev_passenger_distance:
                reward += 0.5  # Small reward for moving toward passenger
            elif current_passenger_distance > self.prev_passenger_distance:
                reward -= 0.2  # Small penalty for moving away from passenger
            self.prev_passenger_distance = current_passenger_distance
        else:
            # If passenger picked up, reward for moving toward destination
            current_destination_distance = self._manhattan_distance(self.taxi_pos, self.destination)
            if current_destination_distance < self.prev_destination_distance:
                reward += 0.5  # Small reward for moving toward destination
            elif current_destination_distance > self.prev_destination_distance:
                reward -= 0.2  # Small penalty for moving away from destination
            self.prev_destination_distance = current_destination_distance
        
        # Base movement penalty
        if action < 4:  # Movement actions
            reward -= 0.1  # Small penalty for each movement to encourage efficiency
        
        # Return observation, reward, done, truncated, and info
        return self.get_state(), reward, done, truncated, self._get_info()
    
    def render(self):
        """Render the environment"""
        if self.render_mode == 'human':
            clear_output(wait=True)
        
        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        # Place station markers
        grid[0][0] = 'R'  # Red station
        grid[0][self.grid_size-1] = 'G'  # Green station
        grid[self.grid_size-1][0] = 'Y'  # Yellow station
        grid[self.grid_size-1][self.grid_size-1] = 'B'  # Blue station
        
        # Place passenger if not in taxi
        if not self.passenger_picked_up:
            py, px = self.passenger_loc
            grid[py][px] = 'P'
        
        # Place destination
        dy, dx = self.destination
        grid[dy][dx] = 'D'
        
        # Place taxi (overwrites other markers)
        ty, tx = self.taxi_pos
        if self.passenger_picked_up:
            grid[ty][tx] = '🚖P'  # Taxi with passenger
        else:
            grid[ty][tx] = '🚖'  # Empty taxi
        
        # Print grid
        for row in grid:
            print(" ".join(row))
        
        # Print additional info
        print(f"\nFuel remaining: {self.current_fuel}")
        print(f"Passenger in taxi: {self.passenger_picked_up}")
        if not self.passenger_picked_up:
            print(f"Distance to passenger: {self._manhattan_distance(self.taxi_pos, self.passenger_loc)}")
        else:
            print(f"Distance to destination: {self._manhattan_distance(self.taxi_pos, self.destination)}")
        
        return grid
    
    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"

def run_agent(agent_file, env_config, render=False):
    """Run an agent in the custom taxi environment"""
    import importlib.util
    
    # Load the agent module
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)
    
    # Create environment
    env = CustomTaxiEnv(**env_config)
    
    # Reset environment
    obs, info = env.reset()
    
    # Initialize tracking variables
    total_reward = 0
    done = False
    step_count = 0
    
    # Render initial state
    if render:
        env.render()
        time.sleep(0.5)
    
    # Run episode
    while not done:
        # Get action from agent
        action = student_agent.get_action(obs)
        
        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Update tracking variables
        total_reward += reward
        step_count += 1
        
        # Render if requested
        if render:
            print(f"\nStep {step_count}: {env.get_action_name(action)}, Reward: {reward}")
            env.render()
            time.sleep(0.5)
    
    print(f"\nAgent finished in {step_count} steps with total reward: {total_reward}")
    return total_reward 