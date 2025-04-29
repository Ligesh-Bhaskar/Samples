import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance, cKDTree
import os
import time
import argparse
import pickle
from multiprocessing import Pool
import random
import math
import contextlib
from collections import defaultdict

class Box:
    def __init__(self, Lx, Ly, Lz):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
    
    def wrap_position(self, pos):
        x, y, z = pos
        x = x - self.Lx * round(x / self.Lx)
        y = y - self.Ly * round(y / self.Ly)
        z = z - self.Lz * round(z / self.Lz)
        return np.array([x, y, z])
    
    def wrap_vector(self, vec):
        x, y, z = vec
        x = x - self.Lx * round(x / self.Lx)
        y = y - self.Ly * round(y / self.Ly)
        z = z - self.Lz * round(z / self.Lz)
        return np.array([x, y, z])
    
    def pbc_distance(self, pos1, pos2):
        delta = pos1 - pos2
        delta = self.wrap_vector(delta)
        return np.linalg.norm(delta)
    
    def pbc_vector(self, pos1, pos2):
        delta = pos1 - pos2
        delta = self.wrap_vector(delta)
        return delta
    
    @property
    def volume(self):
        return self.Lx * self.Ly * self.Lz

class Particle:
    def __init__(self, position, charge=-1.0, mass=1.0, particle_type=0, chain_id=None, monomer_id=None):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.force = np.zeros(3, dtype=np.float64)
        self.charge = charge
        self.mass = mass
        self.type = particle_type
        self.chain_id = chain_id
        self.monomer_id = monomer_id
        self.crosslinks = set()  
    
    def reset_force(self):
        self.force = np.zeros(3, dtype=np.float64)

class Bond:
    def __init__(self, i, j, bond_type="polymer"):
        self.i = i  
        self.j = j  
        self.type = bond_type

class Angle:
    def __init__(self, i, j, k, angle_type="polymer"):
        self.i = i 
        self.j = j 
        self.k = k  
        self.type = angle_type

class ForceField:
    def __init__(self):
        pass
    
    def compute(self, particles, bonds, angles, box):
        pass

class FENEBond(ForceField):
    def __init__(self, k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0, 
                 k_cross=15.0, r0_cross=1.8, epsilon_cross=0.5, sigma_cross=1.2):
        super().__init__()
        
        self.params = {
            "polymer": {"k": k, "r0": r0, "epsilon": epsilon, "sigma": sigma, "delta": delta},
            "crosslink": {"k": k_cross, "r0": r0_cross, "epsilon": epsilon_cross, "sigma": sigma_cross, "delta": delta}
        }
    
    def compute(self, particles, bonds, angles, box):
        for bond in bonds:
            i, j = bond.i, bond.j
            param = self.params[bond.type]

            ri = particles[i].position
            rj = particles[j].position
            
            rij = box.pbc_vector(ri, rj)
            r = np.linalg.norm(rij)
            
            # FENE attractive part
            if r < param["r0"]:
                fene_force = -param["k"] * rij / (1 - (r/param["r0"])**2)
            else:
                fene_force = np.zeros(3)
            
            # WCA repulsive part
            rcut = 2**(1/6) * param["sigma"]
            if r < rcut + param["delta"]:
                sigma_r = param["sigma"] / (r - param["delta"])
                lj_force = 4 * param["epsilon"] * (12 * sigma_r**12 - 6 * sigma_r**6) * rij / r**2
            else:
                lj_force = np.zeros(3)
            
            # Apply forces
            particles[i].force += fene_force + lj_force
            particles[j].force -= fene_force + lj_force

class HarmonicAngle(ForceField):
	
    def __init__(self, k=20.0, theta0=np.pi):
        super().__init__()
        self.k = k
        self.theta0 = theta0
    
    def compute(self, particles, bonds, angles, box):
		
        for angle in angles:
            i, j, k = angle.i, angle.j, angle.k
            
            ri = particles[i].position
            rj = particles[j].position
            rk = particles[k].position
            
            rji = box.pbc_vector(rj, ri)
            rjk = box.pbc_vector(rj, rk)
            
            rji_norm = np.linalg.norm(rji)
            rjk_norm = np.linalg.norm(rjk)
            
            if rji_norm < 1e-10 or rjk_norm < 1e-10:
                continue  
            
            rji_unit = rji / rji_norm
            rjk_unit = rjk / rjk_norm
            
            #Calculate cosine of angle
            cos_theta = np.dot(rji_unit, rjk_unit)
            cos_theta = np.clip(cos_theta, -1.0, 1.0) 
            theta = np.arccos(cos_theta)
            
            #Calculate force magnitude
            force_mag = -self.k * (theta - self.theta0)
            
            #Calculate perpendicular components for force direction
            if abs(cos_theta) < 0.999:  
                perp_i = rji_unit - cos_theta * rjk_unit
                perp_i /= np.linalg.norm(perp_i)
                
                perp_k = rjk_unit - cos_theta * rji_unit
                perp_k /= np.linalg.norm(perp_k)
                
                #Apply forces
                particles[i].force += force_mag * perp_i / rji_norm
                particles[k].force += force_mag * perp_k / rjk_norm
                particles[j].force -= force_mag * (perp_i / rji_norm + perp_k / rjk_norm)

class WCARepulsion(ForceField):
    def __init__(self, epsilon=1.0, sigma=1.0, cutoff=None):
        super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff if cutoff is not None else 2**(1/6) * sigma
    
    def compute(self, particles, bonds, angles, box, neighbor_list=None):
        n = len(particles)
        
        if neighbor_list is None:

            for i in range(n):
                for j in range(i+1, n):
                    ri = particles[i].position
                    rj = particles[j].position
                    rij = box.pbc_vector(ri, rj)
                    r = np.linalg.norm(rij)
                    
                    if r < self.cutoff:
                        self._add_force(particles[i], particles[j], rij, r)
        else:

            for i, neighbors in enumerate(neighbor_list):
                for j in neighbors:
                    if j > i:  
                        ri = particles[i].position
                        rj = particles[j].position
                        rij = box.pbc_vector(ri, rj)
                        r = np.linalg.norm(rij)
                        
                        if r < self.cutoff:
                            self._add_force(particles[i], particles[j], rij, r)
    
    def _add_force(self, particle_i, particle_j, rij, r):

        if r < self.cutoff:
            sigma_r = self.sigma / r
            force_mag = 4 * self.epsilon * (12 * sigma_r**12 - 6 * sigma_r**6) / r**2
            force_vec = force_mag * rij
            
            particle_i.force += force_vec
            particle_j.force -= force_vec

class SoftRepulsion(ForceField):
    def __init__(self, A=1.0, cutoff=1.5):
        super().__init__()
        self.A = A
        self.cutoff = cutoff
        
    def compute(self, particles, bonds, angles, box, neighbor_list=None):
        n = len(particles)
        
        if neighbor_list is None:
            for i in range(n):
                for j in range(i+1, n):
                    ri = particles[i].position
                    rj = particles[j].position
                    rij = box.pbc_vector(ri, rj)
                    r = np.linalg.norm(rij)
                    
                    if r < self.cutoff:
                        self._add_force(particles[i], particles[j], rij, r)
        else:
            for i, neighbors in enumerate(neighbor_list):
                for j in neighbors:
                    if j > i:
                        ri = particles[i].position
                        rj = particles[j].position
                        rij = box.pbc_vector(ri, rj)
                        r = np.linalg.norm(rij)
                        
                        if r < self.cutoff:
                            self._add_force(particles[i], particles[j], rij, r)
    
    def _add_force(self, particle_i, particle_j, rij, r):
        if r < self.cutoff:
            
            force_mag = self.A * np.pi / self.cutoff * np.sin(np.pi * r / self.cutoff) / r
            force_vec = force_mag * rij
            
            particle_i.force -= force_vec  
            particle_j.force += force_vec

class YukawaElectrostatics(ForceField):

    def __init__(self, epsilon=1.0, kappa=1.0, cutoff=3.0):
        super().__init__()
        self.epsilon = epsilon
        self.kappa = kappa 
        self.cutoff = cutoff
    
    def compute(self, particles, bonds, angles, box, neighbor_list=None):

        n = len(particles)
        
        if neighbor_list is None:

            for i in range(n):
                for j in range(i+1, n):
                    ri = particles[i].position
                    rj = particles[j].position
                    rij = box.pbc_vector(ri, rj)
                    r = np.linalg.norm(rij)
                    
                    if r < self.cutoff:
                        qi = particles[i].charge
                        qj = particles[j].charge
                        
                        #Yukawa force
                        yukawa_force = self.epsilon * qi * qj * np.exp(-self.kappa * r) * (1 + self.kappa * r) / r**2
                        force_vec = yukawa_force * rij
                        
                        particles[i].force += force_vec
                        particles[j].force -= force_vec
        else:

            for i, neighbors in enumerate(neighbor_list):
                for j in neighbors:
                    if j > i:  
                        ri = particles[i].position
                        rj = particles[j].position
                        rij = box.pbc_vector(ri, rj)
                        r = np.linalg.norm(rij)
                        
                        if r < self.cutoff:
                            qi = particles[i].charge
                            qj = particles[j].charge
                            
                            #Yukawa force
                            yukawa_force = self.epsilon * qi * qj * np.exp(-self.kappa * r) * (1 + self.kappa * r) / r**2
                            force_vec = yukawa_force * rij
                            
                            particles[i].force += force_vec
                            particles[j].force -= force_vec

class VicsekActiveForce(ForceField):

    def __init__(self, active_force=1.0, align_radius=3.0, align_strength=0.5):
        super().__init__()
        self.active_force = active_force
        self.align_radius = align_radius
        self.align_strength = align_strength
    
    def compute(self, particles, bonds, angles, box, neighbor_list=None):

        n = len(particles)
        positions = np.array([p.position for p in particles])
        velocities = np.array([p.velocity for p in particles])
        
        tree = cKDTree(positions, boxsize=[box.Lx, box.Ly, box.Lz])
        
        for i in range(n):
            #Find neighbors within align_radius
            nearby_indices = tree.query_ball_point(particles[i].position, self.align_radius)
            
            if len(nearby_indices) > 1:  # If there are neighbors (including self)
                #Calculate average velocity direction
                avg_vel = np.mean(velocities[nearby_indices], axis=0)
                if np.linalg.norm(avg_vel) > 0:
                    avg_vel = avg_vel / np.linalg.norm(avg_vel)
                
                #Current velocity direction
                current_vel = velocities[i]
                if np.linalg.norm(current_vel) > 0:
                    current_vel = current_vel / np.linalg.norm(current_vel)
                else:
                    current_vel = np.random.normal(0, 1, 3)
                    current_vel = current_vel / np.linalg.norm(current_vel)
                
                #Combine directions with alignment strength
                direction = (1 - self.align_strength) * current_vel + self.align_strength * avg_vel
                
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    
                    #Apply active force in the calculated direction
                    particles[i].force += self.active_force * direction

class PolymerSystem:

    def __init__(self, box_size, kT=1.0, dt=0.01, damping=1.0):
        self.box = Box(box_size, box_size, box_size)
        self.particles = []
        self.bonds = []
        self.angles = []
        self.force_fields = []
        self.kT = kT
        self.dt = dt
        self.damping = damping 
        self.time = 0
        self.neighbor_list = None
        self.neighbor_list_update_freq = 10
        self.step_count = 0
        
        #Statistics
        self.energy_history = []
        self.rg_history = []
        self.crosslink_history = []
        
        #Analysis data
        self.initial_positions = None
    
    def add_force_field(self, force_field):

        self.force_fields.append(force_field)
        
    def load_xyz(self, xyz_file, monomers_per_chain=None):
        try:
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
            
            n_particles = int(lines[0].strip())
            
            positions = []
            types = []
            
            for i in range(2, len(lines)):
                line = lines[i].strip()
                if not line: 
                    continue
                    
                parts = line.split()
                if len(parts) >= 4:  
                    ptype = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    
                    positions.append([x, y, z])
                    types.append(ptype)
            
            if len(positions) != n_particles:
                print(f"Warning: Expected {n_particles} particles but found {len(positions)}")
            
            if monomers_per_chain is None or monomers_per_chain <= 0:
                monomers_per_chain = len(positions)
                n_chains = 1
            else:
                n_chains = len(positions) // monomers_per_chain
                if len(positions) % monomers_per_chain != 0:
                    print(f"Warning: {len(positions)} particles don't divide evenly into chains of {monomers_per_chain} monomers")
                    
                    n_chains = (len(positions) + monomers_per_chain - 1) // monomers_per_chain
            
            print(f"Creating {n_chains} chains with {monomers_per_chain} monomers per chain")
            
            for i, (pos, typ) in enumerate(zip(positions, types)):
                chain_id = i // monomers_per_chain
                monomer_id = i % monomers_per_chain
                
                self.particles.append(Particle(pos, charge=-1.0, particle_type=typ,
                                              chain_id=chain_id, monomer_id=monomer_id))
            
            for chain in range(n_chains):
                start = chain * monomers_per_chain
                end = min((chain + 1) * monomers_per_chain, len(positions))
                
                for i in range(start, end - 1):
                    self.bonds.append(Bond(i, i + 1))
                    
                    if i < end - 2:
                        self.angles.append(Angle(i, i + 1, i + 2))
            
            print(f"Loaded {len(self.particles)} particles, {len(self.bonds)} bonds, and {len(self.angles)} angles")
            
        except Exception as e:
            print(f"Error loading XYZ file: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    def save_polymer_config(self, filename):
        with open(filename, 'w') as f:
            # Write timestep
            f.write(f"{int(self.time/self.dt)}\n")
            
            # Write total number of particles
            f.write(f"{len(self.particles)}\n")
            
            # Write particle data: type X Y Z
            for p in self.particles:
                x, y, z = p.position
                f.write(f"{p.type} {x:.6f} {y:.6f} {z:.6f}\n")               

    def save_lammpstrj(self, filename):
        with open(filename, 'w') as f:
            #Write header
            f.write(f"ITEM: TIMESTEP\n{int(self.time/self.dt)}\n")
            f.write(f"ITEM: NUMBER OF ATOMS\n{len(self.particles)}\n")
            
            #Write box
            f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{-self.box.Lx/2:.6f} {self.box.Lx/2:.6f}\n")
            f.write(f"{-self.box.Ly/2:.6f} {self.box.Ly/2:.6f}\n")
            f.write(f"{-self.box.Lz/2:.6f} {self.box.Lz/2:.6f}\n")
            
            #Write atoms: id type x y z vx vy vz
            f.write(f"ITEM: ATOMS id type x y z vx vy vz\n")
            for i, p in enumerate(self.particles):
                x, y, z = p.position
                vx, vy, vz = p.velocity
                f.write(f"{i+1} {p.type+1} {x:.6f} {y:.6f} {z:.6f} {vx:.6f} {vy:.6f} {vz:.6f}\n")
    
    def save_analysis_data(self, prefix, format='txt'):
        extension = format.lower()
        
        #Radius of gyration
        if self.rg_history:
            with open(f"{prefix}_rg.{extension}", 'w') as f:
                f.write("# Time Rg\n")
                for time, rg in self.rg_history:
                    f.write(f"{time:.6f} {rg:.6f}\n")
        
        #Crosslinks
        if self.crosslink_history:
            with open(f"{prefix}_crosslinks.{extension}", 'w') as f:
                f.write("# Time NumCrosslinks\n")
                for time, count in self.crosslink_history:
                    f.write(f"{time:.6f} {count}\n")
        
        #MSD
        if self.initial_positions is not None:
            msd = self.calculate_msd()
            with open(f"{prefix}_msd.{extension}", 'w') as f:
                f.write(f"# Time MSD\n")
                f.write(f"{self.time:.6f} {msd:.6f}\n")
        
        #Structure factor
        q_values, sq = self.calculate_structure_factor()
        with open(f"{prefix}_sq.{extension}", 'w') as f:
            f.write("# q S(q)\n")
            for q, s in zip(q_values, sq):
                f.write(f"{q:.6f} {s:.6f}\n")
        
        #RDF
        r_values, rdf = self.calculate_rdf()
        with open(f"{prefix}_rdf.{extension}", 'w') as f:
            f.write("# r g(r)\n")
            for r, g in zip(r_values, rdf):
                f.write(f"{r:.6f} {g:.6f}\n")
        
        #Network analysis if we have crosslinks
        if any(p.crosslinks for p in self.particles):
            network = self.analyze_network()
            with open(f"{prefix}_network.{extension}", 'w') as f:
                f.write(f"# Network analysis\n")
                f.write(f"NumComponents {network['n_components']}\n")
                f.write(f"LargestComponent {network['largest_component']}\n")
                f.write(f"AvgComponentSize {network['avg_component_size']:.6f}\n")
                f.write("# Component sizes\n")
                for i, size in enumerate(network['component_sizes']):
                    f.write(f"{i} {size}\n")
        
        #Crosslink analysis - count links per bead
        links_per_bead = [len(p.crosslinks) for p in self.particles]
        with open(f"{prefix}_crosslink_per_bead.{extension}", 'w') as f:
            f.write("# BeadID NumCrosslinks\n")
            for i, count in enumerate(links_per_bead):
                f.write(f"{i} {count}\n")
            
            #Summary statistics
            f.write("\n# Summary Statistics\n")
            f.write(f"MaxLinks {max(links_per_bead)}\n")
            f.write(f"AvgLinks {np.mean(links_per_bead):.6f}\n")
            f.write(f"StdLinks {np.std(links_per_bead):.6f}\n")
                                          
    def update_neighbor_list(self, cutoff, padding=0.5):
        positions = np.array([p.position for p in self.particles])
        tree = cKDTree(positions, boxsize=[self.box.Lx, self.box.Ly, self.box.Lz])
        
        self.neighbor_list = []
        for i, pos in enumerate(positions):
            neighbors = tree.query_ball_point(pos, cutoff + padding)
            self.neighbor_list.append(neighbors)
    
    def compute_forces(self):
        for p in self.particles:
            p.reset_force()

        if self.step_count % self.neighbor_list_update_freq == 0:
            max_cutoff = max([f.cutoff for f in self.force_fields if hasattr(f, 'cutoff')] or [0])
            if max_cutoff > 0:
                self.update_neighbor_list(max_cutoff)
        
        for force_field in self.force_fields:
            if hasattr(force_field, 'cutoff'):
                force_field.compute(self.particles, self.bonds, self.angles, self.box, self.neighbor_list)
            else:
                force_field.compute(self.particles, self.bonds, self.angles, self.box)
    
    def brownian_dynamics_step(self):

        self.compute_forces()
        
        for p in self.particles:
            #Drift term (force/friction)
            drift = self.dt * p.force / self.damping
            
            #Diffusion term (thermal noise)
            sigma = np.sqrt(2 * self.kT * self.dt / self.damping)
            noise = np.random.normal(0, sigma, 3)
            
            #Update position
            p.position += drift + noise
            
            #Wrap position within periodic box
            p.position = self.box.wrap_position(p.position)
            
            #Update velocity estimation for active forces
            p.velocity = drift / self.dt
        
        self.time += self.dt
        self.step_count += 1
    
    def langevin_dynamics_step(self):

        self.compute_forces()
        
        for p in self.particles:
            #Random force term
            sigma = np.sqrt(2 * self.damping * self.kT / self.dt)
            random_force = np.random.normal(0, sigma, 3)
            
            #Update velocity 
            p.velocity += 0.5 * self.dt * (p.force - self.damping * p.velocity + random_force) / p.mass
            
            #Update position
            p.position += self.dt * p.velocity
            
            #Wrap position within periodic box
            p.position = self.box.wrap_position(p.position)
        
        #Recompute forces with new positions
        self.compute_forces()
        
        #Update velocities
        for p in self.particles:
            sigma = np.sqrt(2 * self.damping * self.kT / self.dt)
            random_force = np.random.normal(0, sigma, 3)
            p.velocity += 0.5 * self.dt * (p.force - self.damping * p.velocity + random_force) / p.mass
        
        self.time += self.dt
        self.step_count += 1
    
    def run(self, steps, brownian=True, dump_period=None, data_dump_period=None,
             output_prefix=None, snapshots_dir=None, phase="production",
             crosslink_period=None, breaking_rate=None, target_crosslinks=None,
             max_links_per_bead=None, crosslink_distance=None, crosslink_probability=None):
    
        if output_prefix and (dump_period or data_dump_period):
            os.makedirs(os.path.dirname(output_prefix) or '.', exist_ok=True)
            
        traj_file = None
        xyz_file = None
        if dump_period and output_prefix:
            traj_filename = f"{output_prefix}_{phase}.lammpstrj"
            traj_file = open(traj_filename, 'w')
            
            xyz_filename = f"{output_prefix}_{phase}.xyz"
            xyz_file = open(xyz_filename, 'w')
        
        #Track crosslink dynamics
        if crosslink_period:
            crosslink_dynamics = []
        
        for step in range(steps):
            if brownian:
                self.brownian_dynamics_step()
            else:
                self.langevin_dynamics_step()
                
            if self.step_count % 100 == 0:
                self.collect_statistics()
            
            if crosslink_period and step > 0 and step % crosslink_period == 0:
                broken, created = self.manage_dynamic_crosslinks(
                    breaking_rate=breaking_rate,
                    target_crosslinks=target_crosslinks,
                    max_links_per_bead=max_links_per_bead,
                    max_distance=crosslink_distance,
                    crosslink_probability=crosslink_probability
                )
                
                crosslink_count = sum(len(p.crosslinks) for p in self.particles) // 2
                crosslink_dynamics.append((self.time, broken, created, crosslink_count))
                print(f"Step {step}: Broke {broken}, created {created} crosslinks. Total: {crosslink_count}")
            
            if dump_period and output_prefix and step % dump_period == 0:
                if xyz_file:
                    xyz_file.write(f"{len(self.particles)}\n")
                    xyz_file.write(f"Timestep: {self.time} Box: {self.box.Lx} {self.box.Ly} {self.box.Lz}\n")
                    
                    for p in self.particles:
                        ptype = p.type
                        x, y, z = p.position
                        xyz_file.write(f"{ptype} {x} {y} {z}\n")
                    xyz_file.flush()
                    
                if traj_file:
                    traj_file.write(f"ITEM: TIMESTEP\n{int(self.time/self.dt)}\n")
                    traj_file.write(f"ITEM: NUMBER OF ATOMS\n{len(self.particles)}\n")
                    traj_file.write(f"ITEM: BOX BOUNDS pp pp pp\n")
                    traj_file.write(f"{-self.box.Lx/2:.6f} {self.box.Lx/2:.6f}\n")
                    traj_file.write(f"{-self.box.Ly/2:.6f} {self.box.Ly/2:.6f}\n")
                    traj_file.write(f"{-self.box.Lz/2:.6f} {self.box.Lz/2:.6f}\n")
                    traj_file.write(f"ITEM: ATOMS id type x y z vx vy vz\n")
                    for i, p in enumerate(self.particles):
                        x, y, z = p.position
                        vx, vy, vz = p.velocity
                        traj_file.write(f"{i+1} {p.type+1} {x:.6f} {y:.6f} {z:.6f} {vx:.6f} {vy:.6f} {vz:.6f}\n")
                    traj_file.flush()
                
            if data_dump_period and output_prefix and step % data_dump_period == 0:
                if snapshots_dir:
                    data_filename = f"{snapshots_dir}/snapshot_{phase}_step_{self.step_count}.data"
                else:
                    data_filename = f"{output_prefix}_{phase}_step_{self.step_count}.data"
                self.save_lammps_data(data_filename)
        
        if crosslink_period and output_prefix:
            dynamics_filename = f"{output_prefix}_{phase}_crosslink_dynamics.txt"
            with open(dynamics_filename, 'w') as f:
                f.write("# Time Broken Created Total\n")
                for time, broken, created, total in crosslink_dynamics:
                    f.write(f"{time:.6f} {broken} {created} {total}\n")
        
        if traj_file:
            traj_file.close()
        if xyz_file:
            xyz_file.close()    
    
    def collect_statistics(self):

        #Radius of gyration
        rg = self.calculate_rg()[0]
        self.rg_history.append((self.time, rg))
        
        #Count crosslinks
        crosslink_count = sum(len(p.crosslinks) for p in self.particles) // 2
        self.crosslink_history.append((self.time, crosslink_count))
    
    def save_trajectory(self, filename):

        with open(filename, 'w') as f:
            f.write(f"{len(self.particles)}\n")
            f.write(f"Timestep: {self.time} Box: {self.box.Lx} {self.box.Ly} {self.box.Lz}\n")
            
            for p in self.particles:
                ptype = p.type
                x, y, z = p.position
                f.write(f"{ptype} {x} {y} {z}\n")
    
    def save_lammps_data(self, filename):

        with open(filename, 'w') as f:
            #Header
            f.write(f"LAMMPS data file from Python simulation at time {self.time}\n\n")
            
            #Counts
            f.write(f"{len(self.particles)} atoms\n")
            max_type = max(p.type for p in self.particles) + 1
            f.write(f"{max_type} atom types\n")
            f.write(f"{len(self.bonds)} bonds\n")
            bond_types = set(b.type for b in self.bonds)
            f.write(f"{len(bond_types)} bond types\n")
            f.write(f"{len(self.angles)} angles\n")
            angle_types = set(a.type for a in self.angles)
            f.write(f"{len(angle_types)} angle types\n\n")
            
            #Box 
            f.write(f"{-self.box.Lx/2:.6f} {self.box.Lx/2:.6f} xlo xhi\n")
            f.write(f"{-self.box.Ly/2:.6f} {self.box.Ly/2:.6f} ylo yhi\n")
            f.write(f"{-self.box.Lz/2:.6f} {self.box.Lz/2:.6f} zlo zhi\n\n")
            
            #Masses
            f.write("Masses\n\n")
            for i in range(max_type):
                f.write(f"{i+1} 1.0\n")
            f.write("\n")
            
            #Atoms
            f.write("Atoms\n\n")
            for i, p in enumerate(self.particles):
                f.write(f"{i+1} {p.type+1} {p.charge:.1f} {p.position[0]:.6f} {p.position[1]:.6f} {p.position[2]:.6f}\n")
            f.write("\n")
            
            #Bonds
            if self.bonds:
                f.write("Bonds\n\n")
                bond_type_dict = {bt: i+1 for i, bt in enumerate(bond_types)}
                for i, bond in enumerate(self.bonds):
                    f.write(f"{i+1} {bond_type_dict[bond.type]} {bond.i+1} {bond.j+1}\n")
                f.write("\n")
            
            #Angles
            if self.angles:
                f.write("Angles\n\n")
                angle_type_dict = {at: i+1 for i, at in enumerate(angle_types)}
                for i, angle in enumerate(self.angles):
                    f.write(f"{i+1} {angle_type_dict[angle.type]} {angle.i+1} {angle.j+1} {angle.k+1}\n")
    
    def calculate_rg(self):

        chain_particles = defaultdict(list)
        for i, p in enumerate(self.particles):
            if p.chain_id is not None:
                chain_particles[p.chain_id].append(i)
        
        rg_values = []

        for chain_id, indices in chain_particles.items():
            if len(indices) < 2:
                continue
                
            positions = np.array([self.particles[i].position for i in indices])
            com = np.mean(positions, axis=0)

            r2 = 0
            for idx in indices:
                r_i = self.box.pbc_vector(self.particles[idx].position, com)
                r2 += np.dot(r_i, r_i)
            
            rg2 = r2 / len(indices)
            rg_values.append(np.sqrt(rg2))
        
        if rg_values:
            return np.mean(rg_values), np.std(rg_values)
        else:
            return 0, 0
    
    def calculate_structure_factor(self, qmax=10.0, nq=100):
        positions = np.array([p.position for p in self.particles])
        q_values = np.linspace(0.1, qmax, nq)
        sq = np.zeros_like(q_values)
        
        for i, q in enumerate(q_values):
            phase = np.exp(1j * q * positions[:, 0])
            sq[i] = np.abs(np.sum(phase))**2 / len(positions)
        
        return q_values, sq
    
    def calculate_rdf(self, dr=0.1, rmax=None):

        if rmax is None:
            rmax = min(self.box.Lx, self.box.Ly, self.box.Lz) / 2
        
        positions = np.array([p.position for p in self.particles])
        nbins = int(rmax / dr)
        r_values = np.linspace(dr/2, rmax-dr/2, nbins)
        gr = np.zeros_like(r_values)
        
        tree = cKDTree(positions, boxsize=[self.box.Lx, self.box.Ly, self.box.Lz])
        

        for i, pos in enumerate(positions):
            indices = tree.query_ball_point(pos, rmax)
            indices.remove(i)  
            
            if not indices:
                continue
                
            dists = [self.box.pbc_distance(pos, positions[j]) for j in indices]
            
            hist, _ = np.histogram(dists, bins=nbins, range=(0, rmax))
            gr += hist
        
        density = len(positions) / self.box.volume
        norm = 4/3 * np.pi * density * ((r_values + dr/2)**3 - (r_values - dr/2)**3)
        gr = gr / (len(positions) * norm)
        
        return r_values, gr

    def create_crosslinks(self, crosslink_probability=0.05, max_distance=1.5, exclude_same_chain=True, 
                         max_links_per_bead=None, target_crosslinks=None):
                         
        positions = np.array([p.position for p in self.particles])
        tree = cKDTree(positions, boxsize=[self.box.Lx, self.box.Ly, self.box.Lz])
        
        #Current number of crosslinks
        current_crosslinks = sum(len(p.crosslinks) for p in self.particles) // 2
        
        #Crosslinks to create
        crosslinks_to_create = float('inf')
        if target_crosslinks is not None:
            crosslinks_to_create = max(0, target_crosslinks - current_crosslinks)
            if crosslinks_to_create <= 0:
                return 0
        
        pairs = []
        for i, pos in enumerate(positions):
            if max_links_per_bead and len(self.particles[i].crosslinks) >= max_links_per_bead:
                continue
                
            indices = tree.query_ball_point(pos, max_distance)
            
            for j in indices:
                if i >= j:  # Avoid duplicates
                    continue
                    
                if max_links_per_bead and len(self.particles[j].crosslinks) >= max_links_per_bead:
                    continue
                    
                if exclude_same_chain and self.particles[i].chain_id == self.particles[j].chain_id:
                    continue
                    
                already_bonded = False
                for bond in self.bonds:
                    if (bond.i == i and bond.j == j) or (bond.i == j and bond.j == i):
                        already_bonded = True
                        break
                
                if not already_bonded:
                    pairs.append((i, j))
        
        random.shuffle(pairs)
        
        crosslink_count = 0
        for i, j in pairs:
            if np.random.random() < probability:
                self.bonds.append(Bond(i, j, bond_type="crosslink"))
                self.particles[i].crosslinks.add(j)
                self.particles[j].crosslinks.add(i)
                crosslink_count += 1
                
                if crosslink_count >= crosslinks_to_create:
                    break
        
        return crosslink_count
    
    def break_crosslinks(self, breaking_rate=0.01):
    
        crosslink_bonds = [i for i, bond in enumerate(self.bonds) if bond.type == "crosslink"]
        
        if not crosslink_bonds:
            return 0
        
        broken_count = 0
        bonds_to_remove = []
        
        for bond_idx in crosslink_bonds:
            if np.random.random() < breaking_rate:
                bonds_to_remove.append(bond_idx)
        
        for bond_idx in sorted(bonds_to_remove, reverse=True):
            bond = self.bonds[bond_idx]
            i, j = bond.i, bond.j
            
            if j in self.particles[i].crosslinks:
                self.particles[i].crosslinks.remove(j)
            if i in self.particles[j].crosslinks:
                self.particles[j].crosslinks.remove(i)
            
            self.bonds.pop(bond_idx)
            broken_count += 1
        
        return broken_count
    
    def manage_dynamic_crosslinks(self, breaking_rate, target_crosslinks, max_links_per_bead,
                                max_distance, crosslink_probability):
    
        broken_count = self.break_crosslinks(breaking_rate)
        
        created_count = self.create_crosslinks(
            crosslink_probability=crosslink_probability,
            max_distance=max_distance,
            exclude_same_chain=True,
            max_links_per_bead=max_links_per_bead,
            target_crosslinks=target_crosslinks
        )
        
        return broken_count, created_count
            
    def analyze_network(self):

        n = len(self.particles)
        adj_list = [[] for _ in range(n)]
        
        for bond in self.bonds:
            i, j = bond.i, bond.j
            adj_list[i].append(j)
            adj_list[j].append(i)
        
        visited = [False] * n
        components = []
        
        for start in range(n):
            if visited[start]:
                continue
                
            component = []
            queue = [start]
            visited[start] = True
            
            while queue:
                node = queue.pop(0)
                component.append(node)
                
                for neighbor in adj_list[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            components.append(component)
        
        sizes = [len(c) for c in components]
        
        return {
            "n_components": len(components),
            "largest_component": max(sizes),
            "avg_component_size": np.mean(sizes),
            "component_sizes": sizes
        }
    
    def store_initial_positions(self):

        self.initial_positions = np.array([p.position.copy() for p in self.particles])
    
    def calculate_msd(self):

        if self.initial_positions is None:
            print("No initial positions stored")
            return 0
        
        current_positions = np.array([p.position for p in self.particles])
        n = len(self.particles)
        
        sq_disp = 0
        for i in range(n):
            delta = self.box.pbc_vector(current_positions[i], self.initial_positions[i])
            sq_disp += np.dot(delta, delta)
        
        return sq_disp / n
    
    def plot_statistics(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if self.rg_history:
            times, rgs = zip(*self.rg_history)
            ax1.plot(times, rgs, 'o-')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Radius of Gyration')
            ax1.set_title('Radius of Gyration vs Time')
        
        if self.crosslink_history:
            times, crosslinks = zip(*self.crosslink_history)
            ax2.plot(times, crosslinks, 'o-')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Number of Crosslinks')
            ax2.set_title('Crosslinks vs Time')
        
        plt.tight_layout()
        return fig
    
    def visualize(self, show_bonds=True, show_crosslinks=True, alpha=0.8):
		
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = np.array([p.position for p in self.particles])
        colors = np.array([p.chain_id for p in self.particles])
        
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                            c=colors, cmap='tab20', s=50, alpha=alpha)
        
        if show_bonds:
            for bond in self.bonds:
                if bond.type == "crosslink" and not show_crosslinks:
                    continue
                    
                i, j = bond.i, bond.j
                pos_i = self.particles[i].position
                pos_j = self.particles[j].position
                
                delta = self.box.pbc_vector(pos_i, pos_j)
                pos_j_adj = pos_i - delta
                
                color = 'black' if bond.type == "polymer" else 'red'
                ax.plot([pos_i[0], pos_j_adj[0]], 
                        [pos_i[1], pos_j_adj[1]], 
                        [pos_i[2], pos_j_adj[2]], 
                        color=color, linewidth=1, alpha=0.6)
        
        Lx, Ly, Lz = self.box.Lx, self.box.Ly, self.box.Lz
        box_x = [-Lx/2, Lx/2]
        box_y = [-Ly/2, Ly/2]
        box_z = [-Lz/2, Lz/2]
        
        for x in box_x:
            for y in box_y:
                ax.plot([x, x], [y, y], box_z, 'k-', alpha=0.2)
        for x in box_x:
            for z in box_z:
                ax.plot([x, x], box_y, [z, z], 'k-', alpha=0.2)
        for y in box_y:
            for z in box_z:
                ax.plot(box_x, [y, y], [z, z], 'k-', alpha=0.2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Polymer System Visualization')
        
        #Set equal aspect ratio
        max_range = max(Lx, Ly, Lz)
        ax.set_xlim(-max_range/2, max_range/2)
        ax.set_ylim(-max_range/2, max_range/2)
        ax.set_zlim(-max_range/2, max_range/2)
        
        return fig

def count_overlaps(system, threshold=0.8):
    count = 0
    positions = np.array([p.position for p in system.particles])
    
    tree = cKDTree(positions, boxsize=[system.box.Lx, system.box.Ly, system.box.Lz])
    
    for i, pos in enumerate(positions):
        neighbors = tree.query_ball_point(pos, threshold)
        count += len([j for j in neighbors if j > i])
    
    return count

def run_overlap_removal(system, steps=1000000, n_stages=10):

    print("Starting overlap removal using a soft potential...")
    
    original_force_fields = system.force_fields.copy()
    
    r_cut_soft = next((ff.r0 for ff in system.force_fields if isinstance(ff, FENEBond)), 1.5)
    soft_potential = SoftRepulsion(A=0.1, cutoff=r_cut_soft)
    
    bond_forces = [ff for ff in system.force_fields if isinstance(ff, FENEBond)]
    system.force_fields = bond_forces + [soft_potential]
    
    steps_per_stage = steps // n_stages
    for stage in range(n_stages):
        soft_potential.A = 0.1 + 0.9 * (stage / (n_stages - 1))
        print(f"Overlap removal stage {stage+1}/{n_stages}, A = {soft_potential.A:.2f}")
        system.run(steps_per_stage, brownian=True)
        
        overlaps = count_overlaps(system, threshold=0.8)
        print(f"  Remaining overlaps: {overlaps}")
    
    system.force_fields = original_force_fields
    print("Overlap removal completed!")

def run_simulation(args):
    system = PolymerSystem(box_size=args.box_size, kT=args.kT, dt=args.dt, damping=args.damping)
            
    if args.input_xyz:
        if os.path.exists(args.input_xyz):
            system.load_xyz(args.input_xyz, args.monomers)
        else:
            print(f"ERROR: File {args.input_xyz} not found!")
            return None
                 
    #Force fields
    system.add_force_field(FENEBond(
        k=args.bond_k, r0=args.bond_r0, 
        epsilon=args.wca_epsilon, sigma=args.wca_sigma,
        k_cross=args.fene_k_cross, r0_cross=args.fene_r0_cross,
        epsilon_cross=args.wca_epsilon_cross, sigma_cross=args.wca_sigma_cross
    ))
    
    system.add_force_field(HarmonicAngle(k=args.angle_k, theta0=args.angle_theta0))
    system.add_force_field(WCARepulsion(epsilon=args.wca_epsilon, sigma=args.wca_sigma, cutoff=args.wca_cutoff))
            
    if args.electrostatics:
        system.add_force_field(YukawaElectrostatics(epsilon=args.yukawa_epsilon, kappa=args.yukawa_kappa, cutoff=args.yukawa_cutoff))
            
    if args.active:
        system.add_force_field(VicsekActiveForce(active_force=args.active_force, align_radius=args.align_radius))
            
    #Initial positions for MSD
    system.store_initial_positions()
    
    # Overlap removal
    run_overlap_removal(system, steps=10000, n_stages=5)
            
    #Equilibration
    print(f"Equilibrating for {args.equil_steps} steps...")
    system.run(args.equil_steps, brownian=args.brownian,
               dump_period=args.dump_period,
               data_dump_period=args.data_dump_period,
               output_prefix=args.output_prefix,
               snapshots_dir=args.snapshots_dir,
               phase="relaxation")
            
    #Create initial crosslinks
    if args.total_crosslinks > 0:
        print(f"Creating initial {args.total_crosslinks} crosslinks...")
        created = system.create_crosslinks(
            probability=args.crosslink_prob, 
            max_distance=args.crosslink_dist,
            max_links_per_bead=args.maximum_crosslinks_per_bead,
            target_crosslinks=args.total_crosslinks
        )
        print(f"Created {created} initial crosslinks")
            
    #Production run
    print(f"Running production for {args.prod_steps} steps with dynamic crosslinks...")
    system.run(
        args.prod_steps, 
        brownian=args.brownian,
        dump_period=args.dump_period,
        data_dump_period=args.data_dump_period,
        output_prefix=args.output_prefix,
        snapshots_dir=args.snapshots_dir,
        phase="production",
        # Dynamic crosslink parameters
        crosslink_period=args.crosslink_period,
        breaking_rate=args.breaking_rate,
        target_crosslinks=args.total_crosslinks,
        max_links_per_bead=args.maximum_crosslinks_per_bead,
        crosslink_distance=args.crosslink_dist,
        crosslink_probability=args.crosslink_prob
    )
            
    #Postprocessing
    rg, rg_std = system.calculate_rg()
    print(f"Final radius of gyration: {rg:.4f} ± {rg_std:.4f}")
            
    msd = system.calculate_msd()
    print(f"Mean squared displacement: {msd:.4f}")
            
    if args.total_crosslinks > 0:
        network = system.analyze_network()
        print(f"Network analysis: {network['n_components']} components, largest has {network['largest_component']} particles")
            
    if args.output_prefix:
        system.save_trajectory(f"{args.output_prefix}_final.xyz")
        system.save_lammps_data(f"{args.output_prefix}_final.data")
        system.save_lammpstrj(f"{args.output_prefix}_final.lammpstrj")
                    
        system.save_analysis_data(f"{args.analysis_dir}/analysis", args.analysis_output_format)
            
        with open(f"{args.analysis_dir}/stats.pkl", 'wb') as f:
            pickle.dump({
                'rg_history': system.rg_history,
                'crosslink_history': system.crosslink_history,
                'final_rg': rg,
                'msd': msd
            }, f)
                    
        fig = system.plot_statistics()
        fig.savefig(f"{args.analysis_dir}/stats.png", dpi=150)
                    
        if args.visualize:
            fig = system.visualize()
            fig.savefig(f"{args.analysis_dir}/visual.png", dpi=150)
            
    if args.visualize and not args.no_display:
        fig = system.visualize()
        plt.show()
            
    return system

def parse_arguments(param_file='Parameters.txt'):
    
    args = argparse.Namespace(
        box_size=20.0, chains=10, monomers=50, kT=1.0, dt=0.01, damping=1.0,
        bond_length=1.0, bond_k=30.0, bond_r0=1.5, angle_k=20.0, angle_theta0=np.pi,
        wca_epsilon=1.0, wca_sigma=1.0, wca_cutoff=2.5, electrostatics=False,
        yukawa_epsilon=1.0, yukawa_kappa=1.0, yukawa_cutoff=3.0, active=False,
        active_force=1.0, align_radius=3.0, crosslink_prob=0.0, crosslink_dist=1.5,
        equil_steps=1000, prod_steps=10000, brownian=True, input_xyz=None,
        output_prefix='polymer', run_number = 0, visualize=False, no_display=False,
        dump_period=1000, data_dump_period=10000, analysis_output_format='txt',
        fene_k_cross=15.0, fene_r0_cross=1.8, wca_epsilon_cross=0.5, wca_sigma_cross=1.2,
        total_crosslinks=0, breaking_rate=0.01, crosslink_period=1000, 
        maximum_crosslinks_per_bead=3
    )
    
    params = {}
    try:
        with open(param_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    try:
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        else:
                            value = eval(value)
                    except:
                        pass
                    
                    params[key] = value
        
        mapping = {
            'box_size': 'box_size',
            'n_chains': 'chains',
            'monomers_per_chain': 'monomers',
            'kT': 'kT',
            'dt': 'dt',
            'brownian_gamma': 'damping',
            'bond_length': 'bond_length',
            'fene_k': 'bond_k',
            'fene_r0': 'bond_r0',
            'persistence_length': 'angle_k',
            'wca_epsilon': 'wca_epsilon',
            'wca_sigma': 'wca_sigma',
            'wca_rcut': 'wca_cutoff',
            'coulomb_epsilon': 'yukawa_epsilon',
            'coulomb_kappa': 'yukawa_kappa',
            'rcut_crosslink': 'crosslink_dist',
            'crosslink_rate': 'crosslink_prob',
            'relaxation_steps': 'equil_steps',
            'mainrun_steps': 'prod_steps',
            'active_force': 'active_force',
            'align_radius': 'align_radius',
            'input_file': 'input_xyz',
            'output_prefix': 'output_prefix',
            'run_number': 'run_number',
            'dump_period': 'dump_period',
            'data_dump_period': 'data_dump_period',
            'analysis_format': 'analysis_output_format',
            'fene_k_cross': 'fene_k_cross',
            'fene_r0_cross': 'fene_r0_cross',
            'wca_epsilon_cross': 'wca_epsilon_cross',
            'wca_sigma_cross': 'wca_sigma_cross',
            'total_crosslinks': 'total_crosslinks',
            'breaking_rate': 'breaking_rate',
            'crosslink_period': 'crosslink_period',
            'maximum_crosslinks_per_bead': 'maximum_crosslinks_per_bead'
        }
        
        for param_key, arg_key in mapping.items():
            if param_key in params:
                setattr(args, arg_key, params[param_key])
        
        if params.get('coulomb_epsilon', 0) > 0:
            args.electrostatics = True
            
        if params.get('active_force', 0) > 0:
            args.active = True
            
        args.equil_steps = int(args.equil_steps)
        args.prod_steps = int(args.prod_steps)
        
        if isinstance(args.output_prefix, int):
            args.output_prefix = f"polymer_{args.output_prefix}"
        
    except Exception as e:
        print(f"Error reading parameter file: {e}")
        print("Using default parameters")
    
    return args

def setup_directory_structure(run_number):

    master_dir = f"Sim_Run{run_number}"
    os.makedirs(master_dir, exist_ok=True)
    
    analysis_dir = os.path.join(master_dir, "Analysis")
    snapshots_dir = os.path.join(master_dir, "Snapshots")
    
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)
    
    return master_dir, analysis_dir, snapshots_dir

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def main(param_file='Parameters.txt'):
    args = parse_arguments(param_file)
    
    master_dir, analysis_dir, snapshots_dir = setup_directory_structure(args.run_number)
    
    args.output_prefix = os.path.join(master_dir, f"polymer_{args.run_number}")
    args.analysis_dir = analysis_dir
    args.snapshots_dir = snapshots_dir
    
    log_file = os.path.join(master_dir, f"Simlog_Run{args.run_number}.txt")
    logger = Logger(log_file)
    
    original_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        print("=== Polymer Network Simulation ===")
        print(f"System: {args.chains} chains, {args.monomers} monomers per chain, {args.box_size}³ box")
        print(f"Temperature: {args.kT}, Time step: {args.dt}")
        print(f"Dynamics: {'Brownian' if args.brownian else 'Langevin'} dynamics")
        print(f"Output directories: {master_dir}, {analysis_dir}, {snapshots_dir}")
        
        start_time = time.time()
        
        system = run_simulation(args)
        
        elapsed = time.time() - start_time
        print(f"Simulation completed in {elapsed:.2f} seconds")
        
        return system
    
    finally:
        sys.stdout = original_stdout
        logger.close()

if __name__ == "__main__":
    import sys
    param_file = 'Parameters.txt'
    if len(sys.argv) > 1:
        param_file = sys.argv[1]
    main(param_file)
