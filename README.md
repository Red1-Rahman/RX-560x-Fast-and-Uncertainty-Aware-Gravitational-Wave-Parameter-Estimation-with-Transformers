# RX 560x Fast and Uncertainty-Aware Gravitational Wave Parameter Estimation with Transformers

This project demonstrates a lightweight Transformer neural network for rapid gravitational wave parameter estimation from LIGO/Virgo detector data. The model efficiently extracts key physical parameters including chirp mass, mass ratio, and coalescence time from binary black hole merger signals.

**Key Features:**
- **Fast Inference**: Sub-100ms prediction time with only 121K parameters
- **Uncertainty Quantification**: Gaussian likelihood head provides prediction confidence intervals
- **Interactive Demo**: Streamlit web application for real-time parameter estimation
- **CPU Training**: Trained efficiently on consumer hardware in ~4 hours
- **Realistic Data**: Trained on 800 synthetic waveforms with detector noise

The Transformer architecture leverages attention mechanisms to capture temporal dependencies in gravitational wave signals, achieving competitive accuracy while maintaining computational efficiency. This approach enables real-time parameter estimation for gravitational wave astronomy applications.

**Created by [Redwan Rahman](https://redwan-rahman.netlify.app)**