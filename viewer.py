import MDAnalysis as mda
import nglview as nv
import numpy as np
from IPython.display import display

# Define parameters
num_steps = 1000  # Number of simulation steps
num_particles = 10  # Number of spherical molecules
dt = 0.1  # Time step
diffusion_coefficient = 1.0  # Diffusion coefficient for Brownian motion
update_interval = 500  # Visualization update interval

# Generate initial positions for particles randomly within a box
initial_positions = np.random.rand(num_particles, 3) * 10.0  # Adjust the box size as needed

# Create a Universe to hold the particles and create a trajectory
universe = mda.Universe.empty(num_particles, trajectory=True)
universe.atoms.positions = initial_positions

trajectory = []  # Store positions for the trajectory

# Create a trajectory viewer for visualization
view = nv.show_mdanalysis(universe)
display(view)  # Display the initial visualization

# Function to calculate forces based on your custom equation
def calculate_custom_forces(positions):
    # Example: Calculate forces based on a simple harmonic potential
    k = 1.0  # Spring constant
    equilibrium_distance = 2.0  # Equilibrium distance for the harmonic potential
    
    # Calculate forces using Hooke's Law (F = -k * x)
    displacements = positions - equilibrium_distance  # Calculate displacements from equilibrium
    forces = -k * displacements  # Calculate forces based on Hooke's Law
    
    return forces

# Function to perform simulation steps
def run_simulation():
    for i in range(num_steps):
        # Generate random displacements for Brownian motion
        brownian_displacements = np.sqrt(2 * diffusion_coefficient * dt) * np.random.randn(num_particles, 3)

        # Calculate forces based on the custom equation
        current_positions = universe.atoms.positions
        custom_forces = calculate_custom_forces(current_positions)

        # Update particle positions incorporating Brownian motion and custom forces
        total_displacements = brownian_displacements + (custom_forces * dt)
        universe.atoms.positions += total_displacements
        
        # Append current positions to the trajectory
        trajectory.append(universe.atoms.positions.copy())

        # Update visualization at intervals
        if i % update_interval == 0:
            view.coordinates = universe.atoms.positions
            display(view)  # Display the updated visualization

# Run the simulation
run_simulation()

# Save the trajectory to a new DCD file
with mda.Writer("trajectory.dcd", num_particles) as writer:
    for frame in trajectory:
        universe.atoms.positions = frame
        writer.write(universe.atoms)

# Load the trajectory and visualize it
trajectory_universe = mda.Universe("trajectory.dcd")
trajectory_view = nv.show_mdanalysis(trajectory_universe)
display(trajectory_view)