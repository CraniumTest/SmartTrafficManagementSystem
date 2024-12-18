# Smart Traffic Management System (STMS) - README

## Overview
The Smart Traffic Management System (STMS) project aims to develop a dynamic traffic signal optimization system leveraging deep reinforcement learning (DRL) to improve traffic flow efficiency in urban areas. The system simulates a traffic environment using SUMO (Simulation of Urban Mobility) and trains a Deep Q-Network (DQN) agent to optimize traffic light signals at intersections.

## Project Structure
- **SmartTrafficManagementSystem**: The main project directory.
- **requirements.txt**: Lists all the dependencies required for the project, ensuring a consistent setup.
- **main.py**: The core script where the implementation of the traffic simulation and the DQN agent resides.
- **sumo_config.sumocfg**: Placeholder for the SUMO configuration file, defining the simulation parameters.

## Key Components
1. **Traffic Simulation**: Utilizes SUMO to simulate various traffic scenarios and traci to interact with the simulation in real-time.
2. **Reinforcement Learning Model**: Implements a DQN agent that uses TensorFlow to learn optimal traffic light phase adjustments through trial and error.
3. **Agent Training**: Runs multiple episodes to train the agent, allowing it to learn strategies that minimize vehicle wait times and improve traffic flow.

## Setup Instructions
1. Clone the repository and navigate to the `SmartTrafficManagementSystem` directory.
2. Install the Python dependencies listed in `requirements.txt` using the command:
   ```
   pip install -r requirements.txt
   ```
3. Ensure SUMO is installed and configured on your system, with the `SUMO_BINARY` path correctly specified in the `main.py` file.
4. Update the `sumo_config.sumocfg` placeholder with a realistic configuration for the traffic simulation.

## Running the Simulation
- Execute the `main.py` script to start the traffic light optimization simulation. The script will initialize the traffic environment and train the DQN agent over multiple episodes.
- The system will output performance metrics such as score and epsilon value for each episode, indicating the agent's learning progress.

## Future Development
- **Validation**: Enhance the simulation with real-world traffic data to validate the model's efficacy.
- **Deployment**: Test the trained models in a real-world setup to evaluate performance benefits.
- **Enhancements**: Further refine the learning algorithms and incorporate additional state-action pairs for more complex traffic scenarios.

## Contribution
Contributors are welcome to expand the functionality of the STMS. For code contributions, please follow the project guidelines and ensure compatibility with the existing codebase.

## Acknowledgments
The STMS project is built upon the SUMO traffic simulator and leverages TensorFlow for machine learning. Special thanks to the open-source community for providing these essential tools and libraries.
