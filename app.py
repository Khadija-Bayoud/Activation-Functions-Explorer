import numpy as np
import streamlit as st
import plotly.graph_objects as go
from activations.metadata import activation_functions


st.set_page_config(
    page_title="Activation Functions Explorer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Neural Network Activation Functions Explorer")
st.markdown("Interactive visualization of the most commonly used activation functions in deep learning")
st.markdown("---")

# Sidebar
st.sidebar.header("Select Activation Function")
selected_func = st.sidebar.selectbox(
    "Choose a function:",
    list(activation_functions.keys()),
    index=7
)

func_info = activation_functions[selected_func]

is_softmax = func_info.get("type") == "vector"


# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"📊 {selected_func} Function")
    
    # Display formula
    st.markdown("**Mathematical Formula:**")
    st.latex(func_info['latex'])
    st.markdown(f"**Output:** {func_info['range']}")
    
    # Interactive parameters if available
    params = {}
    if 'params' in func_info:
        st.markdown("### 🎛️ Adjust Parameters")
        for param_name, (default, min_val, max_val, step) in func_info['params'].items():
            params[param_name] = st.slider(
                f"{param_name.capitalize()}",
                min_value=min_val,
                max_value=max_val,
                value=default,
                step=step
            )
    
    if is_softmax:
        st.markdown("### 🎚️ Class Logits")

        num_classes = st.slider("Number of classes", 2, 6, 3)

        cols = st.columns(num_classes)
        
        logits = []
        for i in range(num_classes):
            with cols[i]:
                logit = st.slider(f"z{i}", -10.0, 10.0, 0.0, 0.1, label_visibility="collapsed")
                st.caption(f"Class {i}")  
                logits.append(logit)
        
        logits = np.array(logits)
        
    else:
        # Input value slider
        st.markdown("### 🎚️ Interactive Demo")
        input_val = st.slider(
            "Input value (x)",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )
    
        # Calculate output
        if params:
            output_val = func_info['func'](input_val, **params)
        else:
            output_val = func_info['func'](input_val)
        
        # Display input/output
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Input", f"{input_val:.2f}")
        with col_b:
            st.metric("Output", f"{output_val:.2f}")

with col2:
    st.subheader("📈 Function Visualization")
    
    if is_softmax:
        # Softmax
        exp_z = np.exp(logits - np.max(logits))
        probs = exp_z / exp_z.sum()

        # Plotly bar chart
        fig = go.Figure(
            data=[go.Bar(
                x=[f"Class {i}" for i in range(num_classes)],
                y=probs,
                text=[f"{p:.2f}" for p in probs],  
                textposition='auto',                 
                marker_color='skyblue'
            )]
        )

        fig.update_layout(
            title="Softmax Output Probabilities",
            yaxis_title="Probability",
            xaxis_title="Class",
            yaxis=dict(range=[0, 1.1]),  
            uniformtext_minsize=10,      
            uniformtext_mode='hide'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
    
        # Generate x values
        x = np.linspace(-10, 10, 1000)
        
        # Calculate y values
        if params:
            y = func_info['func'](x, **params)
        else:
            y = func_info['func'](x)
        
        # Create plot
        fig = go.Figure()
        
        # Add function curve
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=selected_func,
            line=dict(color='#1f77b4', width=3)
        ))
        
        # Add point for current input
        fig.add_trace(go.Scatter(
            x=[input_val],
            y=[output_val],
            mode='markers',
            name='Current Input',
            marker=dict(size=15, color='red', symbol='circle')
        ))
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            xaxis_title="Input (x)",
            yaxis_title="Output",
            height=400,
            hovermode='x unified',
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Description section
st.markdown("---")
st.subheader("📝 About " + selected_func)

col3, col4 = st.columns(2)

with col3:
    st.markdown("**Description:**")
    st.write(func_info['description'])
    
    st.markdown("**💡 Intuition:**")
    st.write(func_info['intuition'])
    
    st.markdown("**🎯 Common Use Cases:**")
    st.write(func_info['use_cases'])

with col4:
    st.markdown("**✅ Pros:**")
    for pro in func_info['pros']:
        st.write(f"• {pro}")
    
    st.markdown("**❌ Cons:**")
    for con in func_info['cons']:
        st.write(f"• {con}")

# Comparison section
st.markdown("---")
st.markdown('<div class="section-header">🔄 Compare Multiple Functions</div>', unsafe_allow_html=True)

col5, col6 = st.columns([2, 1])

with col5:
    activations_for_selection = [k for k in activation_functions.keys() if k != "Softmax"]

    selected_funcs = st.multiselect(
        "📊 Select functions to compare:",
        activations_for_selection,
        default=["ReLU", "Sigmoid", "Tanh"]
    )

with col6:
    x_range = st.slider(
        "📏 X-axis Range",
        min_value=2,
        max_value=20,
        value=10,
        step=1
    )

if selected_funcs:
    x_compare = np.linspace(-x_range, x_range, 1000)
    fig_compare = go.Figure()
    
    colors = ['#667eea', '#f093fb', '#4facfe', '#fa709a', '#30cfd0', '#a8edea', '#fed6e3', '#c471ed', '#12c2e9']
    
    for i, func_name in enumerate(selected_funcs):
        y_compare = activation_functions[func_name]['func'](x_compare)
        fig_compare.add_trace(go.Scatter(
            x=x_compare,
            y=y_compare,
            mode='lines',
            name=f"{activation_functions[func_name]['emoji']} {func_name}",
            line=dict(width=4, color=colors[i % len(colors)])
        ))
    
    fig_compare.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, line_width=1)
    fig_compare.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3, line_width=1)
    
    fig_compare.update_layout(
        xaxis_title="Input (x)",
        yaxis_title="Output f(x)",
        height=550,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13),
        xaxis=dict(gridcolor='rgba(200,200,200,0.3)', zeroline=True, zerolinecolor='gray'),
        yaxis=dict(gridcolor='rgba(200,200,200,0.3)', zeroline=True, zerolinecolor='gray'),
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # Quick comparison table
    st.markdown("### 📋 Quick Comparison Table")
    comparison_data = []
    for func_name in selected_funcs:
        comparison_data.append({
            "Function": f"{activation_functions[func_name]['emoji']} {func_name}",
            "Range": activation_functions[func_name]['range'],
            "Primary Use": activation_functions[func_name]['use_cases'][:50] + "..."
        })
    
    st.table(comparison_data)
else:
    st.info("👆 Select at least one function to compare")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 15px; margin-top: 2rem;">
    <h3 style="color: #667eea;">💡 About This Dashboard</h3>
    <p style="color: #6c757d;">This interactive tool helps you understand and visualize activation functions used in neural networks. 
    Experiment with different functions and parameters to build intuition about how they transform data in deep learning models.</p>
    <p style="color: #6c757d; font-size: 0.9rem; margin-top: 1rem;">
        <strong>Pro Tip:</strong> Try adjusting the input slider and parameter values to see real-time changes in the function behavior!
    </p>
</div>
""", unsafe_allow_html=True)
