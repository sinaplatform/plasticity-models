import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Different loading path generation strategies 

def deformation(M, tmax, Fmax, Fscale, steps):

    t_interp = np.linspace(0, tmax, steps)
    seed=None

    if seed is not None:
        np.random.seed(seed)
        
    # Divide time interval [0, 1] into M intervals
    t = np.sort(np.random.uniform(0, 1, M))
    t = np.insert(t, 0, 0)  # Ensure t^0 = 0
    t = np.append(t, 1)      # Ensure t^M = 1

    # Initialize F_ij arrays (for i,j=1:3)
    F = np.zeros((M + 2))

    # Initialize velocity increments
    v = np.random.choice([-1, 1], size=(M + 1))  # v_ij are Rademacher variables


    for m in range(1, M + 2):
        delta_t = np.sqrt(t[m] - t[m - 1])
        F[m] = F[ m - 1] + v[m - 1] * Fmax * delta_t

    # Create interpolated paths for F_ij(t)
    interpolated_F = {}

    # Cubic Hermite spline interpolation
    spline = CubicHermiteSpline(t, F, np.gradient(F, t))
    interpolated_F = spline
    derivative_F = spline.derivative(nu=1)
    
    # Interpolate the values
    F_interp = interpolated_F(t_interp)
    F_prim = derivative_F(t_interp)

    max_value =np.max(np.abs(F_interp))
    F_interp = F_interp / max_value
    F_interp = F_interp * Fscale

    F_prim = F_prim / max_value
    F_prim = F_prim * Fscale

    return t_interp, F_interp, F_prim, t, (F / max_value) * Fscale

# Generate custom cyclic strain path (from user-provided function)
def generate_cyclic_strain(max_strain, num_cycles, points_per_cycle=100):
    """
    Generate cyclic strain history.
    """
    lcnd=np.tile([-max_strain,max_strain],num_cycles)
    lcnd=np.insert(lcnd,0,0)
    lcnd=np.insert(lcnd,1,max_strain)

    # Create arrays for strain points
    strain_history = []
    for i in range(len(lcnd)-1):
        if i==0:
            segment = np.linspace(lcnd[i], lcnd[i+1], np.int8(points_per_cycle/2))
        else:
            segment = np.linspace(lcnd[i], lcnd[i+1], points_per_cycle)
        strain_history.extend(segment[:-1])  # Exclude last point to avoid duplicates
    strain_history.append(lcnd[-1])  # Add final point
    steps = np.linspace(0, len(strain_history), len(strain_history))

    return steps, strain_history


def generate_positive_strain_path_with_trend(initial_strain, strain_increment, linear_slope, num_cycles, points_per_segment=100):
    strain = []
    
    # First segment: 0 to initial_strain
    first_segment = np.linspace(0, initial_strain, points_per_segment)[:-1]
    strain.extend(first_segment)
    
    for cycle in range(num_cycles):
        current_max = initial_strain + cycle * strain_increment
        
        # Down to zero
        down_segment = np.linspace(current_max, 0, points_per_segment)[:-1]
        strain.extend(down_segment)
        
        # Up to next max
        next_max = initial_strain + (cycle + 1) * strain_increment
        up_segment = np.linspace(0, next_max, points_per_segment)[:-1]
        strain.extend(up_segment)
    
    # Add final point
    strain.append(initial_strain + (num_cycles - 1) * strain_increment)
    
    # Add linear trend
    strain = np.array(strain)
    time = np.arange(len(strain))
    trend = linear_slope * time
    strain_with_trend = strain + trend
    steps = np.linspace(0, len(strain_with_trend), len(strain_with_trend))

    return steps, strain_with_trend

# Main code to calculate the state of materials under defomation loading (deformation control)

def system_of_odes(state, strain, d_strain):
    """
    System of ODEs for combined isotropic-kinematic hardening plasticity with saturations.

    Parameters:
    - t: Time
    - state: [sigma, chi, R] (stress, back stress, isotropic hardening variable)
    - strain: Current strain at time t
    - d_strain: Current strain rate at time t
    """
    sigma, chi, R, lambda_pl, strain_pl = state

    f_yield = abs(sigma - chi) - (sigma_y + R)
    
    if f_yield < 0 or f_yield == 0:  # Elastic step
        # Stress evolves elastically
        d_sigma = E * d_strain
        d_chi = 0
        d_R = 0
        d_lambda = 0
        d_strain_pl = 0
        return [d_sigma, d_chi, d_R, d_lambda, d_strain_pl]
    elif f_yield > 0 :
        # Plastic step
        n = np.sign(sigma - chi)
        H = E + c + b

        d_lambda = f_yield / H
        d_strain_pl = d_lambda * n

        # Stress evolution
        d_sigma = E * (d_strain - d_strain_pl)
        #  stress evolution (includes decay term)
        d_chi = c * d_strain_pl - gamma * chi * d_lambda
        # Isotropic hardening evolution
        d_R = b * (Q - R) * d_lambda

        return [d_sigma, d_chi, d_R, d_lambda, d_strain_pl]
    

def append_or_save(file_path, new_data):
    if os.path.exists(file_path):
        existing_data = np.load(file_path)
        new_data = np.expand_dims(new_data, axis=0)
        combined_data = np.concatenate((existing_data, new_data), axis=0)
        np.save(file_path, combined_data)
    else:
        # If the file doesn't exist, expand dims and save
        new_data = np.expand_dims(new_data, axis=0)
        np.save(file_path, new_data)


# Example usage
if __name__ == "__main__":

    # Material properties
    E = 200e3  # Elastic modulus (MPa)
    sigma_y = 150  # Initial yield stress (MPa)
    c = 10000    # Kinematic hardening parameter
    gamma = 20  # Back stress saturation parameter
    Q = 150  # Saturation value for isotropic hardening (MPa)
    b = 5  # Rate of isotropic hardening

    # Strain path generation parameters
    initial_strain = 0.01   # Maximum strain amplitude
    strain_increment = 0
    linear_slope = 1e-5
    num_cycles = 20         # Number of cycles
    points_per_cycle = 200  # Points per cycle

    M = 12
    Fmax = 1.0
    Fscale = 0.02
    tmax = 1
    steps = 2000

    # Generate sample discrete data
    file_time = 'time.npy'
    file_strain = 'strain.npy'
    file_stress = 'stress.npy'
    file_chi = 'chi.npy'
    file_R = 'R.npy'
    file_lambda = 'lambda.npy'
    file_strainpl = 'strainpl.npy'

    

    for i in range(5000):

        # Generate strain and strain rate history (activate one the below 1- cyclic, 2- augmented cyclic, 3- randomised):
        # steps, strain_history = generate_cyclic_strain(initial_strain, num_cycles, points_per_cycle)
        # steps, strain_history = generate_positive_strain_path_with_trend(initial_strain, strain_increment, linear_slope, num_cycles, points_per_cycle)
        time, strain_history, dstrain, t_point, strain_point = deformation(M, tmax, Fmax, Fscale, steps)

        strain_rate = np.diff(strain_history)  # Difference between consecutive elements
        strain_rate = np.insert(strain_rate, 0, 0)

        # Initial state variables
        state = [0, 0, 0, 0, 0]  # Initial conditions: [sigma, alpha, R, lambda, strain_pl]
        sigma_history = []
        chi_history = []
        R_history = []
        Lambda_history = []
        strain_pl_history = []

        # Solve for each strain step
        for i, s in enumerate(time):
            strain = strain_history[i]
            strain_rate_t = strain_rate[i]
            sol = solve_ivp(
                lambda t, y: system_of_odes(y, strain, strain_rate_t),
                [0, 1],
                state,
                method='RK45'
            )
            state = sol.y[:, -1]
            sigma_history.append(state[0])
            chi_history.append(state[1])
            R_history.append(state[2])
            Lambda_history.append(state[3])
            strain_pl_history.append(state[4])

        # Convert results to arrays
        sigma_history = np.array(sigma_history)
        chi_history = np.array(chi_history)
        R_history = np.array(R_history)
        Lambda_history = np.array(Lambda_history)
        strain_pl_history = np.array(strain_pl_history)

        append_or_save(file_time, time)
        append_or_save(file_strain, strain_history)
        append_or_save(file_stress, sigma_history)
        append_or_save(file_chi, chi_history)
        append_or_save(file_R, R_history)
        append_or_save(file_lambda, Lambda_history)
        append_or_save(file_strainpl, strain_pl_history)
 