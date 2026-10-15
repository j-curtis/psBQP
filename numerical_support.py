import numpy as np

from nambu_keldysh_class import NambuKeldyshTensor, get_expansion_tensor


class ThermalConvolutions:

    def __init__(self, temperature, grid_parameters):
        self.temperature = temperature
        self.grid_parameters = grid_parameters
        self._generate_time_grid()

        self.tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        self.expansion_tensor = get_expansion_tensor(self.ntpoints)

        self.thermal_dist = self._build_thermal_dist()
        self.thermal_integral = self._build_thermal_integral()
        self.log_two_time = self._build_log_two_time()
        self.thermal_sum_left, self.thermal_sum_right = self._build_thermal_sums()
 
    # ========== Grid and thermal quantities ==========

    def _generate_time_grid(self):
        grid_parameters = self.grid_parameters

        if 'time_sampling' in grid_parameters:
            self.ntpoints = grid_parameters['time_sampling']
        elif 'n_tpoints' in grid_parameters:
            self.ntpoints = grid_parameters['n_tpoints']
        else:
            raise ValueError("grid_parameters must contain 'time_sampling' or 'n_tpoints'")

        if 'time_duration' in grid_parameters:
            self.tmax = grid_parameters['time_duration']
        elif 't_max' in grid_parameters:
            self.tmax = grid_parameters['t_max']
        else:
            raise ValueError("grid_parameters must contain 'time_duration' or 't_max'")

        self.delta_t = grid_parameters['dt'] if 'dt' in grid_parameters else self.tmax / (self.ntpoints - 1)
        self.time_grid = np.linspace(-self.tmax, 0, self.ntpoints)

    def _tau_matrix(self):
        t_i, t_j = np.meshgrid(self.time_grid, self.time_grid, indexing='ij')
        return t_i - t_j

    def _build_thermal_dist(self):
        tau_matrix = self._tau_matrix()
        f_two_time = np.zeros((self.ntpoints, self.ntpoints), dtype=complex)
        mask = np.abs(tau_matrix) > 1e-6
        f_two_time[mask] = -1j * self.temperature / np.sinh(np.pi * tau_matrix[mask] * self.temperature)
        return NambuKeldyshTensor(f_two_time, pauli_channel=0)

    def _build_thermal_integral(self):
        tau_upper = self._tau_matrix()
        tau_lower = np.full_like(tau_upper, -2 * self.tmax)

        def integral_from_minus_infinity(tau_values):
            result = np.zeros_like(tau_values, dtype=complex)
            mask = np.abs(tau_values) > 1e-6
            x = np.pi * tau_values[mask] * self.temperature
            result[mask] = -1j / np.pi * np.log(np.tanh(x / 2.0) + 0j)
            return result

        F_two_time = 1j * np.imag(integral_from_minus_infinity(tau_upper) - integral_from_minus_infinity(tau_lower))
        F_two_time[np.abs(tau_upper) < 1e-6] = 0.0
        return NambuKeldyshTensor(F_two_time, pauli_channel=0)

    def _build_log_two_time(self):
        tau_matrix = self._tau_matrix()
        log_two_time = np.zeros_like(tau_matrix, dtype=complex)
        mask = np.abs(tau_matrix) > 1e-6
        log_two_time[mask] = -1j / np.pi * np.log(np.abs(np.pi * tau_matrix[mask] * self.temperature / 2.0))
        return NambuKeldyshTensor(log_two_time, pauli_channel=0)

    def _build_thermal_sums(self):
        N_t = self.ntpoints
        N_t_extended = 2 * N_t - 1
        dt = self.delta_t

        extended_time_grid = np.append(np.linspace(-2 * self.tmax, -self.tmax, N_t - 1, endpoint=False), self.time_grid)
        t_i, t_j = np.meshgrid(extended_time_grid, self.time_grid, indexing='ij')
        tau_matrix = t_i - t_j

        f_two_time = np.zeros((N_t_extended, N_t), dtype=complex)
        mask = np.abs(tau_matrix) > 1e-6
        f_two_time[mask] = -1j * self.temperature / np.sinh(np.pi * tau_matrix[mask] * self.temperature)
        f_two_time_nk = NambuKeldyshTensor(f_two_time, pauli_channel=0)

        ones_data_left_minus = np.ones(N_t_extended)
        ones_data_left_minus[-1] = 0
        ones_data_left_minus[-2] = 1 / 2
        ones_data_left = np.ones(N_t_extended)
        ones_data_left[-1] = 1 / 2

        ones_tensor_left_minus = NambuKeldyshTensor([ones_data_left_minus], pauli_channel=0)
        ones_tensor_left = NambuKeldyshTensor([ones_data_left], pauli_channel=0)
        extension_indices = get_expansion_tensor((1, N_t))

        f_sum_left_minus = (ones_tensor_left_minus @ f_two_time_nk) * dt
        f_sum_left = (ones_tensor_left @ f_two_time_nk) * dt

        row_indices = np.arange(N_t)[:, np.newaxis]
        col_indices = np.arange(N_t)[np.newaxis, :]
        ones_tensor = NambuKeldyshTensor((row_indices <= col_indices).astype(complex), pauli_channel=0)

        f_sum_right_minus = (f_two_time_nk[-2:-1, :] @ ones_tensor - extension_indices * f_two_time_nk[-2:-1, :] * 0.5) * dt
        f_sum_right = (f_two_time_nk[-1:, :] @ ones_tensor - extension_indices * f_two_time_nk[-1:, :] * 0.5) * dt

        thermal_sum_left = NambuKeldyshTensor(np.append(f_sum_right_minus.data[0, 0], f_sum_right.data[0, 0], axis=0), pauli_channel=0)
        thermal_sum_right = NambuKeldyshTensor(np.append(f_sum_left_minus.data[0, 0], f_sum_left.data[0, 0], axis=0), pauli_channel=0)
        return thermal_sum_left, thermal_sum_right

    def _log_antiderivative(self, u):
        u_safe = np.where(u == 0, 1.0, u)
        return np.where(u == 0, 0.0, -1j / np.pi * u * (np.log(np.abs(np.pi * u_safe * self.temperature / 2.0)) - 1.0))

    def thermal_gap_term(self, gap_tensor):
        return -self.thermal_integral * gap_tensor - gap_tensor * self.thermal_integral

    def convolve_thermal_dist_left(self, g_row, time_index=-1):
        N_t = self.ntpoints
        dt = self.delta_t
        k = time_index % N_t
        k_sum = time_index % self.thermal_sum_right.shape[2]
        f, F = self.thermal_dist, self.thermal_integral

        first_endpoint = g_row[:, 0:1] * f[0:1, :]
        last_endpoint = g_row[:, k:k + 1] * f[k:k + 1, :]
        result_std = (g_row @ f) * dt - 0.5 * dt * first_endpoint - 0.5 * dt * last_endpoint

        #* smooth factor at the point of t'' <= t_k closest to the pole t''=t': the pole itself for t' <= t_k, the upper end t''=t_k for t' > t_k
        #* g_row vanishes for t' > t_k (retarded), so with g_row[t'] the subtraction was inactive there and the trapezoid sum had an O(1) error one step beyond t_k
        g_for_reg_data = g_row.data.copy()
        g_for_reg_data[:, :, :, k + 1:] = g_row.data[:, :, :, k:k + 1]
        g_for_reg = NambuKeldyshTensor(g_for_reg_data)

        result_fact = g_for_reg * self.thermal_sum_right[k_sum:k_sum + 1, :]
        result_anal = g_for_reg * F[k:k + 1, :]

        dg_dt = g_row.gradient(dt, axis=1)
        gradient_correction = dg_dt[-1:, :] * dt * -1j / np.pi

        #* singular point inside the range: weight 1, at the end point t'=t_k: 1/2 (trapezoid weight of an end point), beyond the edge: 0
        #* with 0 at t'=t_k the lattice FDT misses the equilibrium g^K diagonal at O(dt), with 1/2 it agrees at O(dt^2)
        edge_scaling = np.ones((1, N_t))
        #~ edge_scaling[0, k:] = 0.5
        edge_scaling[0, k + 1:] = 0.0
        edge_scaling[0, k] = 0.5
        scaling_tensor = NambuKeldyshTensor(edge_scaling, pauli_channel=0)

        return result_std + (-result_fact + result_anal) + scaling_tensor * gradient_correction

    def convolve_thermal_dist_right(self, g, time_index=-1):
        N_t = self.ntpoints
        dt = self.delta_t
        k = time_index % N_t
        k_sum = time_index % self.thermal_sum_left.shape[2]
        f_row = self.thermal_dist[k:k + 1, :]
        F_row = self.thermal_integral[k:k + 1, :]
        g_diagonal = g.diagonal_time()

        first_endpoint = f_row[:, 0:1] * g[0:1, :]
        last_endpoint = f_row[:, :] * g_diagonal
        result_std = (f_row @ g) * dt - 0.5 * dt * first_endpoint - 0.5 * dt * last_endpoint

        #* smooth factor at the point of t'' <= t' closest to the pole t''=t_k: the upper end t''=t' for t' <= t_k, the pole itself (last ga element) for t' > t_k
        #* g(t_k, t') vanishes for t' < t_k (advanced), so with it the subtraction was inactive and the trapezoid sum had an O(1/m^2) error m steps below t_k
        #~ g_for_reg = g[k:k + 1, :]
        g_for_reg_data = g_diagonal.data.copy()
        g_for_reg_data[:, :, k + 1:] = g.data[:, :, k, k + 1:]
        g_for_reg = NambuKeldyshTensor(g_for_reg_data)

        result_fact = self.thermal_sum_left[k_sum:k_sum + 1, :] * g_for_reg
        result_anal = (-F_row) * g_for_reg

        dg_dt = g.gradient(dt, axis=0)
        gradient_correction = dg_dt[k:k + 1, :] * dt * 1j / np.pi

        #* singular point beyond the edge (t' < t_k) and at the edge: 0, inside the range (t' > t_k): 1
        filter_tensor = np.zeros((1, N_t))
        #* pole t''=t_k inside the interval for t' > t_k: weight 1 (checked against exact integrals: -0.5 is the worst of {-0.5, 0, 0.5, 1})
        #~ filter_tensor[0, k + 1:] = -0.5
        filter_tensor[0, k + 1:] = 1.0
        #* end point t'=t_k: trapezoid weight 1/2 (same value as in convolve_thermal_dist_left)
        filter_tensor[0, k] = 0.5 
        filter_tensor = NambuKeldyshTensor(filter_tensor, pauli_channel=0)

        return result_std + (-result_fact + result_anal) + filter_tensor * gradient_correction

    # ========== Convolutions with thermal_gap_term = -(F*gap + gap*F), log regularized ==========

    def convolve_thermal_int_left(self, g_row, gap_tensor, time_index=-1):
        k = time_index % self.ntpoints
        dt = self.delta_t
        thermal_gap_term = self.thermal_gap_term(gap_tensor)

        first_endpoint = g_row[:, 0:1] * thermal_gap_term[0:1, :]
        last_endpoint = g_row[:, k:k + 1] * thermal_gap_term[k:k + 1, :]
        result = (g_row @ thermal_gap_term) * dt - 0.5 * dt * first_endpoint - 0.5 * dt * last_endpoint

        L_mat = self.log_two_time.data[0, 0]
        t0, t1 = self.time_grid[0], self.time_grid[k]
        trapezoid_weights = np.ones(k + 1)
        trapezoid_weights[0] -= 0.5
        trapezoid_weights[-1] -= 0.5
        log_bracket = (self._log_antiderivative(t1 - self.time_grid) - self._log_antiderivative(t0 - self.time_grid) - dt * (trapezoid_weights @ L_mat[:k + 1, :]))
        log_bracket_tensor = NambuKeldyshTensor(log_bracket[np.newaxis, :], pauli_channel=0)

        return result + (-2 * g_row * gap_tensor * log_bracket_tensor)

    def convolve_thermal_int_right(self, g, gap_tensor, time_index=-1):
        k = time_index % self.ntpoints
        dt = self.delta_t
        thermal_gap_term = self.thermal_gap_term(gap_tensor)
        thermal_gap_row = thermal_gap_term[k:k + 1, :]
        g_diagonal = g.diagonal_time()

        first_endpoint = thermal_gap_term[k, 0] * g[0:1, :]
        last_endpoint = thermal_gap_row * g_diagonal
        result = (thermal_gap_row @ g) * dt - 0.5 * dt * first_endpoint - 0.5 * dt * last_endpoint

        L_row = self.log_two_time.data[0, 0][k, :]
        t0, t1 = self.time_grid[0], self.time_grid[k]
        trapezoid_sum = dt * (np.cumsum(L_row) - 0.5 * L_row[0] - 0.5 * L_row)
        log_bracket = (self._log_antiderivative(self.time_grid - t1) - self._log_antiderivative(t0 - t1) - trapezoid_sum)
        log_bracket_tensor = NambuKeldyshTensor(log_bracket[np.newaxis, :], pauli_channel=0)
        gap_sum = gap_tensor + gap_tensor[k] * self.expansion_tensor

        return result + log_bracket_tensor * (-1 * gap_sum * g_diagonal)

    # ========== Sum of the f and F convolutions ==========

    def convolve_thermal_full_left(self, g_row, gap_tensor, time_index=-1):
        return (self.convolve_thermal_dist_left(g_row, time_index) * self.tau3 * 2 + self.convolve_thermal_int_left(g_row, gap_tensor, time_index))

    def convolve_thermal_full_right(self, g, gap_tensor, time_index=-1):
        return (2 * self.tau3 * self.convolve_thermal_dist_right(g, time_index) + self.convolve_thermal_int_right(g, gap_tensor, time_index))

# ========== Interpolation of a state onto a different time grid ==========

def interpolate_two_time_state(tensor, T_old, T_new, N_new, hermitian):
    from scipy.interpolate import RegularGridInterpolator

    data = tensor.data if isinstance(tensor, NambuKeldyshTensor) else np.asarray(tensor)
    N_old = data.shape[2]

    # Rectangle in (t, tau = t - t'): H[i, k] = X[i, i-k]; tau beyond the available history is filled
    # with the oldest available time at the same tau (stationary extension)
    i_idx, k_idx = np.meshgrid(np.arange(N_old), np.arange(N_old), indexing='ij')
    rows = np.where(k_idx <= i_idx, i_idx, k_idx)
    cols = np.where(k_idx <= i_idx, i_idx - k_idx, 0)
    rectangle = np.moveaxis(data[:, :, rows, cols], (0, 1), (2, 3))

    dt_old = T_old / (N_old - 1)
    interpolator = RegularGridInterpolator((np.linspace(-T_old, 0, N_old), dt_old * np.arange(N_old)), rectangle, method='linear')

    time_new = np.linspace(-T_new, 0, N_new)
    t_query = np.clip(np.repeat(time_new[:, None], N_new, axis=1), -T_old, 0)
    tau_full = time_new[:, None] - time_new[None, :]
    tau_query = np.clip(tau_full, 0, T_old)
    lower = np.tril(np.ones((N_new, N_new), dtype=bool))

    values = interpolator(np.stack([t_query[lower], tau_query[lower]], axis=-1))
    # beyond the loaded history the functions have decayed: tau > T_old is set to zero
    values[tau_full[lower] > T_old * (1 + 1e-9)] = 0.0
    result = np.zeros((2, 2, N_new, N_new), dtype=complex)
    result[:, :, lower] = np.moveaxis(values, 0, -1)

    if not hermitian:
        return NambuKeldyshTensor(result)

    strict_lower = np.tril(np.ones((N_new, N_new)), k=-1)
    diagonal = np.eye(N_new)
    lower_part = NambuKeldyshTensor(result * strict_lower)
    diagonal_part = NambuKeldyshTensor(result * diagonal)
    full = lower_part.data + lower_part.involution().data + 0.5 * (diagonal_part.data + diagonal_part.involution().data)
    return NambuKeldyshTensor(full)
