import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform
import pycbc.types

def generate_waveform(mass1, mass2, sample_rate=4096, f_lower=20):
    """
    Generate a gravitational waveform using IMRPhenomD
    
    Parameters:
    -----------
    mass1, mass2 : float
        Component masses in solar masses
    sample_rate : float
        Sampling rate in Hz
    f_lower : float
        Lower frequency cutoff in Hz
    
    Returns:
    --------
    times : array
        Time array
    strain : array 
        h_plus strain data
    """
    
    # Generate waveform
    hp, hc = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=mass1,
        mass2=mass2, 
        delta_t=1.0/sample_rate,
        f_lower=f_lower
    )
    
    # Convert to numpy arrays
    times = hp.sample_times.numpy()
    strain = hp.numpy()
    
    return times, strain

def calculate_chirp_mass(m1, m2):
    """Calculate chirp mass from component masses"""
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

def calculate_mass_ratio(m1, m2):
    """Calculate mass ratio q = m2/m1 (q <= 1 by convention)"""
    return min(m1, m2) / max(m1, m2)

# Test with example parameters
if __name__ == "__main__":
    # Binary black hole parameters
    m1, m2 = 30.0, 25.0  # solar masses
    
    print(f"Component masses: {m1:.1f}, {m2:.1f} M☉")
    print(f"Chirp mass: {calculate_chirp_mass(m1, m2):.2f} M☉")  
    print(f"Mass ratio: {calculate_mass_ratio(m1, m2):.3f}")
    
    # Generate waveform
    times, strain = generate_waveform(m1, m2)
    
    print(f"Waveform duration: {times[-1] - times[0]:.3f} seconds")
    print(f"Number of samples: {len(strain)}")
    
    # Plot the waveform
    plt.figure(figsize=(12, 6))
    
    # Full waveform
    plt.subplot(1, 2, 1)
    plt.plot(times, strain)
    plt.xlabel('Time (s)')
    plt.ylabel('Strain h_+')
    plt.title('Full Waveform')
    plt.grid(True, alpha=0.3)
    
    # Zoom on merger (last 0.2 seconds)
    plt.subplot(1, 2, 2)
    mask = times >= (times[-1] - 0.2)
    plt.plot(times[mask], strain[mask])
    plt.xlabel('Time (s)')
    plt.ylabel('Strain h_+')
    plt.title('Merger Phase (last 0.2s)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_waveform.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nWaveform generated successfully!")
    print("Plot saved as 'test_waveform.png'")