import numpy as np
import pandas as pd
from pycbc.waveform import get_td_waveform
from pycbc.noise import gaussian
from pycbc.psd import aLIGOZeroDetHighPower
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle

class GWDatasetGenerator:
    def __init__(self, sample_rate=4096, duration=2.0, f_lower=20):
        """
        Initialize dataset generator
        
        Parameters:
        -----------
        sample_rate : float
            Sampling rate in Hz
        duration : float  
            Fixed duration for all waveforms in seconds
        f_lower : float
            Lower frequency cutoff in Hz
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.f_lower = f_lower
        self.delta_t = 1.0 / sample_rate
        self.n_samples = int(duration * sample_rate)
        
    def sample_parameters(self, n_samples):
        """
        Sample binary parameters from astrophysical distributions
        
        Returns:
        --------
        params : dict
            Dictionary with mass1, mass2, and derived parameters
        """
        np.random.seed(42)  # For reproducibility during development
        
        # Component mass ranges (solar masses)
        # Typical BBH: 5-100 M☉, focus on LIGO-detectable range
        m1_min, m1_max = 10.0, 80.0
        m2_min, m2_max = 10.0, 80.0
        
        # Sample uniformly in component masses for now
        # TODO: Use more realistic mass distributions later
        mass1 = np.random.uniform(m1_min, m1_max, n_samples)
        mass2 = np.random.uniform(m2_min, m2_max, n_samples)
        
        # Ensure m1 >= m2 by convention
        m1_new = np.maximum(mass1, mass2)
        m2_new = np.minimum(mass1, mass2)
        
        # Calculate derived parameters
        chirp_mass = self._chirp_mass(m1_new, m2_new)
        mass_ratio = m2_new / m1_new  # q <= 1
        total_mass = m1_new + m2_new
        
        # Coalescence time (relative to end of signal)
        # Sample uniformly in merger time within our duration window
        tc = np.random.uniform(0.1, 0.9, n_samples)  # Fraction of duration
        
        return {
            'mass1': m1_new,
            'mass2': m2_new, 
            'chirp_mass': chirp_mass,
            'mass_ratio': mass_ratio,
            'total_mass': total_mass,
            'tc_frac': tc  # Store as fraction for easier handling
        }
    
    def _chirp_mass(self, m1, m2):
        """Calculate chirp mass"""
        return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    
    def generate_waveform(self, mass1, mass2):
        """
        Generate a single waveform and crop/pad to fixed length
        
        Returns:
        --------
        strain : array
            Fixed-length strain time series
        """
        try:
            # Generate waveform  
            hp, hc = get_td_waveform(
                approximant="IMRPhenomD",
                mass1=mass1,
                mass2=mass2,
                delta_t=self.delta_t,
                f_lower=self.f_lower
            )
            
            strain = hp.numpy()
            
            # Crop or pad to fixed length
            if len(strain) >= self.n_samples:
                # Take the last n_samples (merger-centered)
                strain = strain[-self.n_samples:]
            else:
                # Zero-pad at the beginning
                padding = self.n_samples - len(strain)
                strain = np.pad(strain, (padding, 0), mode='constant')
                
            return strain
            
        except Exception as e:
            print(f"Failed to generate waveform for m1={mass1:.1f}, m2={mass2:.1f}: {e}")
            # Return zero array as fallback
            return np.zeros(self.n_samples)
    
    def add_noise(self, strain, snr_target=20):
        """
        Add realistic detector noise to achieve target SNR
        
        Parameters:
        -----------
        strain : array
            Clean waveform strain
        snr_target : float
            Target signal-to-noise ratio
            
        Returns:
        --------
        noisy_strain : array
            Strain with added noise
        actual_snr : float
            Actual achieved SNR
        """
        # Generate noise with aLIGO design sensitivity
        # This is a simplified version - real implementation would be more complex
        noise_std = np.max(np.abs(strain)) / snr_target
        noise = np.random.normal(0, noise_std, len(strain))
        
        noisy_strain = strain + noise
        
        # Calculate actual SNR
        signal_power = np.sum(strain**2)
        noise_power = np.sum(noise**2)
        actual_snr = np.sqrt(signal_power / noise_power) if noise_power > 0 else np.inf
        
        return noisy_strain, actual_snr
    
    def generate_dataset(self, n_samples, snr_range=(10, 50), save_path=None):
        """
        Generate complete dataset
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        snr_range : tuple
            (min_snr, max_snr) for random SNR sampling
        save_path : str
            Path to save dataset (optional)
            
        Returns:
        --------
        dataset : dict
            Contains 'waveforms', 'parameters', 'snr'
        """
        print(f"Generating {n_samples} waveforms...")
        
        # Sample parameters
        params = self.sample_parameters(n_samples)
        
        waveforms = []
        snr_values = []
        valid_indices = []
        
        for i in tqdm(range(n_samples)):
            # Generate clean waveform
            strain = self.generate_waveform(params['mass1'][i], params['mass2'][i])
            
            # Skip if waveform generation failed
            if np.all(strain == 0):
                continue
                
            # Add noise with random SNR
            target_snr = np.random.uniform(*snr_range)
            noisy_strain, actual_snr = self.add_noise(strain, target_snr)
            
            waveforms.append(noisy_strain)
            snr_values.append(actual_snr)
            valid_indices.append(i)
        
        # Filter parameters to valid samples only
        valid_params = {}
        for key, values in params.items():
            valid_params[key] = values[valid_indices]
        
        dataset = {
            'waveforms': np.array(waveforms),
            'parameters': valid_params,
            'snr': np.array(snr_values),
            'metadata': {
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'f_lower': self.f_lower,
                'n_samples_requested': n_samples,
                'n_samples_valid': len(waveforms)
            }
        }
        
        print(f"Successfully generated {len(waveforms)} valid waveforms")
        
        # Save if requested
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"Dataset saved to {save_path}")
        
        return dataset

# Test the dataset generator
if __name__ == "__main__":
    # Create generator
    gen = GWDatasetGenerator(duration=2.0)  # 2 second windows
    
    # Generate small test dataset
    print("Generating test dataset...")
    dataset = gen.generate_dataset(n_samples=100, save_path="test_dataset.pkl")
    
    # Print statistics
    params = dataset['parameters']
    print(f"\nDataset Statistics:")
    print(f"Number of samples: {len(dataset['waveforms'])}")
    print(f"Waveform shape: {dataset['waveforms'][0].shape}")
    print(f"Chirp mass range: {params['chirp_mass'].min():.1f} - {params['chirp_mass'].max():.1f} M☉")
    print(f"Mass ratio range: {params['mass_ratio'].min():.3f} - {params['mass_ratio'].max():.3f}")
    print(f"SNR range: {dataset['snr'].min():.1f} - {dataset['snr'].max():.1f}")
    
    # Plot a few examples
    plt.figure(figsize=(15, 8))
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        
        times = np.linspace(0, gen.duration, gen.n_samples)
        plt.plot(times, dataset['waveforms'][i])
        
        mc = params['chirp_mass'][i]
        q = params['mass_ratio'][i]
        snr = dataset['snr'][i]
        
        plt.title(f'Sample {i}: Mc={mc:.1f}M☉, q={q:.2f}, SNR={snr:.1f}')
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nExample plots saved as 'dataset_examples.png'")
    print("Test dataset saved as 'test_dataset.pkl'")