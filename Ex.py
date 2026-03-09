import numpy as np
import matplotlib.pyplot as plt

class Hydrocode1D:
    """
    A simple 1D Eulerian hydrocode (fixed grid) for better stability.
    Uses upwind scheme for advection.
    """
    
    def __init__(self, n_cells, x_min=0.0, x_max=1.0):
        self.n = n_cells
        self.x_min = x_min
        self.x_max = x_max
        self.dx = (x_max - x_min) / n_cells
        
        # Cell centers (fixed Eulerian grid)
        self.x = np.linspace(x_min + self.dx/2, x_max - self.dx/2, n_cells)
        
        # Conserved variables
        self.rho = np.zeros(n_cells)   # Density
        self.mom = np.zeros(n_cells)   # Momentum (rho * u)
        self.E = np.zeros(n_cells)     # Total energy
        
        # Primitive variables
        self.u = np.zeros(n_cells)     # Velocity
        self.p = np.zeros(n_cells)     # Pressure
        self.e = np.zeros(n_cells)     # Specific internal energy
        
        # Physical constants
        self.gamma = 1.4
        
    def set_initial_conditions(self, rho_func, u_func, p_func):
        """Set initial conditions."""
        for i in range(self.n):
            self.rho[i] = rho_func(self.x[i])
            self.u[i] = u_func(self.x[i])
            self.p[i] = p_func(self.x[i])
        
        # Convert to conserved variables
        self.mom = self.rho * self.u
        self.e = self.p / (self.rho * (self.gamma - 1.0))
        self.E = self.rho * (self.e + 0.5 * self.u**2)
        
    def conserved_to_primitive(self):
        """Convert conserved to primitive variables."""
        self.u = self.mom / self.rho
        kinetic = 0.5 * self.rho * self.u**2
        self.e = (self.E - kinetic) / self.rho
        self.p = (self.gamma - 1.0) * self.rho * self.e
        
        # Ensure positivity
        self.rho = np.maximum(self.rho, 1e-10)
        self.p = np.maximum(self.p, 1e-10)
        self.e = np.maximum(self.e, 1e-10)
        
    def time_step(self):
        """Calculate stable time step using CFL condition."""
        cs = np.sqrt(self.gamma * self.p / self.rho)
        max_speed = np.max(np.abs(self.u) + cs)
        cfl = 0.4
        dt = cfl * self.dx / (max_speed + 1e-10)
        return dt
    
    def step(self, dt):
        """Advance one time step using simple upwind scheme."""
        
        # Create temporary arrays for fluxes
        rho_new = self.rho.copy()
        mom_new = self.mom.copy()
        E_new = self.E.copy()
        
        # Simple donor cell (upwind) advection
        for i in range(1, self.n - 1):
            # Calculate fluxes at interfaces
            # Left interface (i-1/2)
            if self.u[i] > 0:
                # Flow from left
                flux_rho_L = self.rho[i-1] * self.u[i-1]
                flux_mom_L = self.mom[i-1] * self.u[i-1] + self.p[i-1]
                flux_E_L = (self.E[i-1] + self.p[i-1]) * self.u[i-1]
            else:
                # Flow from right (current cell)
                flux_rho_L = self.rho[i] * self.u[i]
                flux_mom_L = self.mom[i] * self.u[i] + self.p[i]
                flux_E_L = (self.E[i] + self.p[i]) * self.u[i]
            
            # Right interface (i+1/2)
            if self.u[i+1] > 0:
                # Flow from left (current cell)
                flux_rho_R = self.rho[i] * self.u[i]
                flux_mom_R = self.mom[i] * self.u[i] + self.p[i]
                flux_E_R = (self.E[i] + self.p[i]) * self.u[i]
            else:
                # Flow from right
                flux_rho_R = self.rho[i+1] * self.u[i+1]
                flux_mom_R = self.mom[i+1] * self.u[i+1] + self.p[i+1]
                flux_E_R = (self.E[i+1] + self.p[i+1]) * self.u[i+1]
            
            # Update conserved variables
            rho_new[i] = self.rho[i] - (dt / self.dx) * (flux_rho_R - flux_rho_L)
            mom_new[i] = self.mom[i] - (dt / self.dx) * (flux_mom_R - flux_mom_L)
            E_new[i] = self.E[i] - (dt / self.dx) * (flux_E_R - flux_E_L)
        
        # Update variables
        self.rho = rho_new
        self.mom = mom_new
        self.E = E_new
        
        # Convert back to primitive variables
        self.conserved_to_primitive()
        
    def run(self, t_end, save_snapshots=False, max_steps=50000):
        """Run simulation."""
        t = 0.0
        step_count = 0
        
        snapshots = []
        snapshot_times = [0.0, t_end/4, t_end/2, 3*t_end/4, t_end]
        next_snap_idx = 1
        
        # Save initial state
        if save_snapshots:
            snapshots.append({
                'time': 0.0,
                'x': self.x.copy(),
                'rho': self.rho.copy(),
                'u': self.u.copy(),
                'p': self.p.copy(),
                'e': self.e.copy()
            })
        
        print(f"Starting simulation to t={t_end}...")
        print(f"Grid: {self.n} cells, dx={self.dx:.4f}")
        print(f"Initial dt ~ {self.time_step():.2e}")
        
        last_print_step = 0
        
        while t < t_end and step_count < max_steps:
            dt = self.time_step()
            
            if t + dt > t_end:
                dt = t_end - t
            
            self.step(dt)
            t += dt
            step_count += 1
            
            # Progress updates
            if step_count - last_print_step >= 200:
                print(f"  Step {step_count}, t={t:.6f} ({100*t/t_end:.1f}%), dt={dt:.2e}")
                last_print_step = step_count
            
            # Save snapshots
            if save_snapshots and next_snap_idx < len(snapshot_times):
                if t >= snapshot_times[next_snap_idx]:
                    snapshots.append({
                        'time': t,
                        'x': self.x.copy(),
                        'rho': self.rho.copy(),
                        'u': self.u.copy(),
                        'p': self.p.copy(),
                        'e': self.e.copy()
                    })
                    print(f"  Snapshot saved at t={t:.6f}")
                    next_snap_idx += 1
        
        if step_count >= max_steps:
            print(f"Stopped at max_steps={max_steps}, reached t={t:.6f}")
        else:
            print(f"Complete! {step_count} steps, final t={t:.6f}")
        
        return snapshots if save_snapshots else None


def plot_results(snapshots):
    """Plot snapshots from the simulation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for idx, snap in enumerate(snapshots):
        t = snap['time']
        color = colors[idx % len(colors)]
        axes[0, 0].plot(snap['x'], snap['rho'], label=f't={t:.3f}', color=color, linewidth=2)
        axes[0, 1].plot(snap['x'], snap['u'], label=f't={t:.3f}', color=color, linewidth=2)
        axes[1, 0].plot(snap['x'], snap['p'], label=f't={t:.3f}', color=color, linewidth=2)
        axes[1, 1].plot(snap['x'], snap['e'], label=f't={t:.3f}', color=color, linewidth=2)
    
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title('Sod Shock Tube Solution', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_ylabel('Velocity', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_ylabel('Pressure', fontsize=12)
    axes[1, 0].set_xlabel('Position', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_ylabel('Internal Energy', fontsize=12)
    axes[1, 1].set_xlabel('Position', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()


# Example: Sod shock tube problem
if __name__ == "__main__":
    # Eulerian grid is much more stable - can use more cells
    hydro = Hydrocode1D(n_cells=100, x_min=0.0, x_max=1.0)
    
    # Sod shock tube initial conditions
    def rho_init(x):
        return 1.0 if x < 0.5 else 0.125
    
    def u_init(x):
        return 0.0
    
    def p_init(x):
        return 1.0 if x < 0.5 else 0.1
    
    hydro.set_initial_conditions(rho_init, u_init, p_init)
    
    # Run simulation
    snapshots = hydro.run(t_end=0.2, save_snapshots=True)
    
    # Plot results
    if snapshots:
        print("\nGenerating plots...")
        plot_results(snapshots)