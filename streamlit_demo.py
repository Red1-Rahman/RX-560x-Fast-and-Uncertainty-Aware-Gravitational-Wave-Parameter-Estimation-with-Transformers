import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import io
from test_transformer import GWTransformer

# Page configuration
st.set_page_config(
    page_title="GW Parameter Estimator",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = GWTransformer(seq_len=1024, d_model=64, nhead=4, num_layers=2)
        model.load_state_dict(torch.load('gw_transformer.pth', map_location='cpu'))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'gw_transformer.pth' not found. Please train the model first.")
        return None

@st.cache_data
def load_example_data():
    """Load example waveforms for demo"""
    try:
        with open('val_dataset.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        return None

def denormalize_chirp_mass(normalized_mc, mc_min=9.8, mc_max=68.9):
    """Convert normalized chirp mass back to solar masses"""
    return normalized_mc * (mc_max - mc_min) + mc_min

def predict_parameters(model, waveform):
    """Make parameter predictions"""
    with torch.no_grad():
        waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0)  # Add batch dimension
        predictions, log_var = model(waveform_tensor)
        
        # Convert to numpy
        pred_np = predictions.squeeze().numpy()
        uncertainty = torch.exp(0.5 * log_var).squeeze().numpy()  # Standard deviation
        
        return pred_np, uncertainty

def create_waveform_plot(waveform, title="Gravitational Waveform"):
    """Create interactive waveform plot"""
    times = np.linspace(0, 0.25, len(waveform))  # 0.25 seconds
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=waveform,
        mode='lines',
        name='Strain',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Strain",
        hovermode='x unified',
        showlegend=False,
        height=400
    )
    
    return fig

def create_parameter_plot(predictions, uncertainties, true_params=None):
    """Create parameter comparison plot"""
    param_names = ['Chirp Mass (M☉)', 'Mass Ratio', 'Coalescence Time']
    
    # Denormalize chirp mass for display
    display_pred = predictions.copy()
    display_pred[0] = denormalize_chirp_mass(predictions[0])
    
    display_unc = uncertainties.copy()
    display_unc[0] = uncertainties[0] * (68.9 - 9.8)  # Scale uncertainty too
    
    fig = go.Figure()
    
    # Predicted values with error bars
    fig.add_trace(go.Scatter(
        x=param_names,
        y=display_pred,
        error_y=dict(type='data', array=display_unc, visible=True),
        mode='markers',
        marker=dict(size=12, color='red'),
        name='Predicted'
    ))
    
    # True values if available
    if true_params is not None:
        display_true = true_params.copy()
        display_true[0] = denormalize_chirp_mass(true_params[0])
        
        fig.add_trace(go.Scatter(
            x=param_names,
            y=display_true,
            mode='markers',
            marker=dict(size=12, color='blue', symbol='diamond'),
            name='True Value'
        ))
    
    fig.update_layout(
        title="Parameter Predictions with Uncertainty",
        yaxis_title="Parameter Value",
        hovermode='x unified',
        height=400
    )
    
    return fig

def main():
    # Title and description
    st.title("🌊 Gravitational Wave Parameter Estimator")
    st.markdown("""
    **Fast and Uncertainty-Aware Parameter Estimation with Transformers**
    
    This demo uses a Transformer neural network to estimate physical parameters 
    from gravitational wave signals detected by LIGO/Virgo. Upload your own 
    waveform data or try the built-in examples!
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Choose Data Source:",
        ["Example Waveforms", "Upload Your Own", "Generate Random"]
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Input Waveform")
        
        waveform = None
        true_params = None
        
        if data_source == "Example Waveforms":
            example_data = load_example_data()
            if example_data:
                example_idx = st.selectbox(
                    "Select Example:",
                    range(min(20, len(example_data['waveforms']))),
                    format_func=lambda x: f"Example {x+1}"
                )
                
                waveform = example_data['waveforms'][example_idx]
                
                # Get true parameters
                params = example_data['parameters']
                true_params = np.array([
                    params['chirp_mass'][example_idx],
                    params['mass_ratio'][example_idx], 
                    params['tc_frac'][example_idx]
                ])
                
                # Normalize chirp mass to match model training
                mc_min, mc_max = 9.8, 68.9
                true_params[0] = (true_params[0] - mc_min) / (mc_max - mc_min)
                
                st.success(f"Loaded example waveform {example_idx + 1}")
            else:
                st.error("Example data not found. Please ensure 'val_dataset.pkl' exists.")
        
        elif data_source == "Upload Your Own":
            uploaded_file = st.file_uploader(
                "Upload waveform data (CSV, TXT, or NPY)",
                type=['csv', 'txt', 'npy'],
                help="Upload a file containing 1024 strain values"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.npy'):
                        waveform = np.load(uploaded_file)
                    else:
                        # Try to read as CSV/TXT
                        data = pd.read_csv(uploaded_file, header=None)
                        waveform = data.iloc[:, 0].values if data.shape[1] == 1 else data.values.flatten()
                    
                    # Ensure correct length
                    if len(waveform) != 1024:
                        if len(waveform) > 1024:
                            waveform = waveform[:1024]
                            st.warning("Waveform truncated to 1024 samples")
                        else:
                            # Zero pad
                            padding = 1024 - len(waveform)
                            waveform = np.pad(waveform, (padding, 0), mode='constant')
                            st.warning(f"Waveform zero-padded to 1024 samples")
                    
                    st.success("File uploaded successfully!")
                    
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        else:  # Generate Random
            if st.button("Generate Random Waveform"):
                # Generate a simple chirp-like signal for demo
                t = np.linspace(0, 0.25, 1024)
                frequency = 20 * (1 + 100 * t**2)  # Chirp frequency
                waveform = np.sin(2 * np.pi * frequency * t) * np.exp(-2 * t)
                waveform += 0.1 * np.random.randn(1024)  # Add noise
                st.success("Random waveform generated!")
        
        # Display waveform
        if waveform is not None:
            fig = create_waveform_plot(waveform)
            st.plotly_chart(fig, use_container_width=True)
            
            # Waveform statistics
            st.markdown("**Waveform Statistics:**")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric("Duration", "0.25 s")
            with stats_col2:
                st.metric("Samples", f"{len(waveform)}")
            with stats_col3:
                st.metric("Max Strain", f"{np.max(np.abs(waveform)):.2e}")
    
    with col2:
        st.subheader("🎯 Parameter Predictions")
        
        if waveform is not None:
            # Make predictions
            with st.spinner("Making predictions..."):
                predictions, uncertainties = predict_parameters(model, waveform)
            
            # Create parameter plot
            fig = create_parameter_plot(predictions, uncertainties, true_params)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display results in a nice format
            st.markdown("**Predicted Parameters:**")
            
            # Denormalize for display
            pred_mc = denormalize_chirp_mass(predictions[0])
            unc_mc = uncertainties[0] * (68.9 - 9.8)
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.metric(
                    "Chirp Mass",
                    f"{pred_mc:.1f} M☉",
                    f"±{unc_mc:.1f}"
                )
            
            with result_col2:
                st.metric(
                    "Mass Ratio",
                    f"{predictions[1]:.3f}",
                    f"±{uncertainties[1]:.3f}"
                )
            
            with result_col3:
                st.metric(
                    "Coalescence Time",
                    f"{predictions[2]:.3f}",
                    f"±{uncertainties[2]:.3f}"
                )
            
            # Show true values if available
            if true_params is not None:
                st.markdown("**True Parameters:**")
                true_mc = denormalize_chirp_mass(true_params[0])
                
                true_col1, true_col2, true_col3 = st.columns(3)
                with true_col1:
                    st.info(f"True Chirp Mass: {true_mc:.1f} M☉")
                with true_col2:
                    st.info(f"True Mass Ratio: {true_params[1]:.3f}")
                with true_col3:
                    st.info(f"True Coalescence Time: {true_params[2]:.3f}")
                
                # Calculate errors
                error_mc = abs(pred_mc - true_mc)
                error_q = abs(predictions[1] - true_params[1])
                error_tc = abs(predictions[2] - true_params[2])
                
                st.markdown("**Prediction Errors:**")
                error_col1, error_col2, error_col3 = st.columns(3)
                with error_col1:
                    color = "red" if error_mc > unc_mc else "green"
                    st.markdown(f":{color}[Chirp Mass: {error_mc:.1f} M☉]")
                with error_col2:
                    color = "red" if error_q > uncertainties[1] else "green"
                    st.markdown(f":{color}[Mass Ratio: {error_q:.3f}]")
                with error_col3:
                    color = "red" if error_tc > uncertainties[2] else "green"
                    st.markdown(f":{color}[Coalescence Time: {error_tc:.3f}]")
        
        else:
            st.info("👆 Select or upload a waveform to see parameter predictions")
    
    # Information section
    st.markdown("---")
    
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        **Model Architecture:** Lightweight Transformer with 4 attention heads and 2 layers
        
        **Training Data:** 800 synthetic binary black hole waveforms with realistic detector noise
        
        **Parameters Estimated:**
        - **Chirp Mass**: Combined mass parameter that determines the inspiral rate
        - **Mass Ratio**: Ratio of smaller to larger mass (q = m₂/m₁ ≤ 1)
        - **Coalescence Time**: Time of merger relative to observation window
        
        **Uncertainty Estimation:** Gaussian likelihood head provides prediction uncertainties
        
        **Performance:** Trained on CPU in ~4 hours, inference time <100ms
        """)
    
    with st.expander("🚀 Technical Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Model Specs:**
            - Parameters: 121,222
            - Model size: 0.5 MB
            - Sequence length: 1024
            - Embedding dimension: 64
            """)
        with col2:
            st.markdown("""
            **Training:**
            - Loss: Gaussian NLL
            - Optimizer: AdamW
            - Best validation loss: -0.893
            - Training time: ~4 hours (CPU)
            """)

    # Add footer at the end of the Streamlit app
    st.markdown("---")
    st.markdown("Created by [Redwan Rahman](https://redwan-rahman.netlify.app)")

if __name__ == "__main__":
    main()