class System:
    def __init__(self):
        pass

    def __update__(self, states_dot, parameters, inputs):
        for bdx in range(parameters.shape[0]):
            states_dot[bdx, :] = parameters[bdx, 0]*inputs[bdx, :] # Change the update
        return states_dot
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import mplcursors
    fs = 1e3
    nbatches = 10
    tstart = 0.0
    tend = 1.0
    ninputs = 1
    input_range = (-20, 20)
    parameter_range = (-1, 1)

    ntimesteps = int(fs*(tend - tstart))
    time_values = np.asarray(np.linspace(tstart, tend, ntimesteps, endpoint=False), dtype=np.float32)
    times = np.broadcast_to(time_values.reshape(1, ntimesteps, 1), (nbatches, ntimesteps, ninputs))

    input_values = np.asarray(np.linspace(input_range[0], input_range[1], ntimesteps, endpoint=False), dtype=np.float32)
    inputs = np.broadcast_to(input_values.reshape(1, ntimesteps, 1), (nbatches, ntimesteps, ninputs))

    parameter_values = np.asarray(np.linspace(parameter_range[0], parameter_range[1], nbatches, endpoint=False), dtype=np.float32)
    parameters = parameter_values.reshape(nbatches, 1)

    states = np.zeros(inputs.shape, dtype=np.float32)
    states_dot = np.zeros(inputs.shape, dtype=np.float32)

    sys = System()

    def propagate(sys, times, states, states_dot, parameters, inputs):
        ntimesteps = times.shape[1]
        for t in range(1, ntimesteps):
            dt = times[:, t:t+1, :] - times[:, t-1:t, :]
            states_dot[0:, t-1, :] = sys.__update__(states_dot[:, t-1, :], parameters[0:, :], inputs[0:, t-1, 0:])
            states[:, t, :] = states[:, t-1, :] + dt[:, 0, :]*states_dot[:, t-1, :]
        return states, states_dot
    
    propagate(sys, times, states, states_dot, parameters, inputs)
    
    # Creating subplots
    fig, axs = plt.subplots(2, 1)

    # Plotting Time vs State in the first subplot
    for bdx in range(nbatches):
        axs[0].plot(times[bdx, :-1, 0], states[bdx, :-1, 0], label=f'Parameter: {parameters[bdx, 0]:.2f}')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('State')
    axs[0].set_title('Time vs State across different parameters')
    axs[0].legend()

    # Plotting State vs State Dot in the second subplot
    for bdx in range(nbatches):
        axs[1].plot(states[bdx, :-1, 0], states_dot[bdx, :-1, 0])
        # Creating a grid of state values
        state_vals = np.linspace(states[bdx, :-1, 0].min(), states[bdx, :-1, 0].max(), 20)
        
        # Calculating state_dot for each state in the grid
        state_dot_vals = parameters[bdx, 0] * np.square(state_vals)
        
        # Plotting the vector field using quiver
        axs[1].quiver(state_vals, np.zeros_like(state_vals), np.zeros_like(state_vals), state_dot_vals, angles='xy', scale_units='xy', scale=1, color='grey', alpha=0.5)

    axs[1].set_xlabel('State')
    axs[1].set_ylabel('State Dot')
    axs[1].set_title('State Dot vs State across different parameters')

    # Enabling interactive data selection cursor
    mplcursors.cursor(hover=True)

    plt.tight_layout()
    plt.show()