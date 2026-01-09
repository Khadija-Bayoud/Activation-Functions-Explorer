from .functions import sigmoid, softmax, tanh, relu, leaky_relu, elu, swish, gelu, softplus, mish

activation_functions = {
    "Sigmoid": {
        "func": sigmoid,
        "formula": "σ(x) = 1 / (1 + e^(-x))",
        "latex": r"\sigma(x) = \frac{1}{1 + e^{-x}}",
        "description": "Squashes input values to a range between 0 and 1, creating an S-shaped curve.",
        "intuition": "Think of it as a smooth switch that gradually turns on as input increases. Historically used to model neuron firing rates.",
        "pros": [
            "Smooth gradient, easy to differentiate",
            "Clear probabilistic interpretation (outputs between 0-1)",
            "Historically important for understanding neural networks"
        ],
        "cons": [
            "Vanishing gradient problem for extreme values",
            "Not zero-centered (causes zig-zagging in gradient descent)",
            "Computationally expensive (exponential operation)"
        ],
        "range": "(0, 1)",
        "use_cases": "Binary classification (output layer).",
        "emoji": "📊"
    },
    "Softmax": {
        "func": softmax,
        "type": "vector",
        "emoji": "📊",
        "latex": r"\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}",
        "range": "(0, 1)",
        "description": "Converts a vector of logits into a probability distribution.",
        "intuition": "Highlights the most likely class while keeping probabilities normalized.",
        "use_cases": "Multi-class classification output layer. Recommendation systems (ranking user preferences). Sentiment analysis (online text, reviews, social comments)",
        "pros": [
            "Differentiable → works well with backpropagation",
            "Emphasizes the largest logits"
        ],
        "cons": [
            "Sensitive to large logits → can cause numerical instability",
            "Not zero-centered → may slow down learning",
            "Expensive for very large number of classes (computes exponentials and sums)"
        ]
    },
    "Tanh": {
        "func": tanh,
        "formula": "tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))",
        "latex": r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}",
        "description": "Squashes input values to a range between -1 and 1, zero-centered version of sigmoid.",
        "intuition": "Similar to sigmoid but centered at zero, making it easier for the network to learn. Like a balanced seesaw.",
        "pros": [
            "Zero-centered (better than sigmoid)",
            "Stronger gradients than sigmoid",
            "Bounded output helps with stability"
        ],
        "cons": [
            "Still suffers from vanishing gradients",
            "Computationally expensive",
            "Can saturate for large positivese/negative values",
        ],
        "range": "(-1, 1)",
        "use_cases": "Hidden layers in RNNs, traditional feedforward networks. Phoneme and word classification in speech recognition. Signal segmentation and classification (medicine, telecom)",
        "emoji": "〰️",
    },
    "ReLU": {
        "func": relu,
        "formula": "ReLU(x) = max(0, x)",
        "latex": r"\text{ReLU}(x) = \max(0, x)",
        "description": "Rectified Linear Unit - outputs the input directly if positive, otherwise outputs zero.",
        "intuition": "The simplest non-linearity: just clip negative values to zero. Like a one-way valve for signals.",
        "pros": [
            "Computationally efficient (simple threshold)",
            "No vanishing gradient for positive values",
            "Sparse activation (many zeros)",
            "Helps with faster convergence"
        ],
        "cons": [
            "Dying ReLU problem (neurons can get stuck at zero)",
            "Not zero-centered",
            "Unbounded output can lead to exploding activations"
        ],
        "range": "[0, ∞)",
        "use_cases": "Most common choice for hidden layers in CNNs, MLPs. Object detection (robust feature extraction). Text analysis and sentiment modeling",
        "emoji": "⚡",
    },
    "Leaky ReLU": {
        "func": leaky_relu,
        "formula": "LeakyReLU(x) = max(αx, x), where α ≈ 0.01",
        "latex": r"\text{LeakyReLU}(x) = \max(\alpha x, x) \text{, where } \alpha \approx 0.01",
        "description": "Variant of ReLU that allows a small gradient when the unit is not active.",
        "intuition": "Fixes the dying ReLU problem by allowing a small, non-zero gradient for negative values.",
        "pros": [
            "Prevents dying ReLU problem",
            "Computationally efficient",
            "Works well in practice"
        ],
        "cons": [
            "Not zero-centered",
            "Requires tuning the α parameter"
        ],
        "range": "(-∞, ∞)",
        "use_cases": "Alternative to ReLU when dying neurons are a problem. Image generation (stable and diverse sample distribution). Time series prediction (handling negative inputs)",
        "params": {"alpha": (0.01, 0.0, 0.3, 0.01)},
        "emoji": "💧",
    },
    "ELU": {
        "func": elu,
        "formula": "ELU(x) = x if x > 0 else α(e^x - 1)",
        "latex": r"\text{ELU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha(e^x - 1) & \text{if } x < 0 \end{cases}",
        "description": "Exponential Linear Unit - smooth approximation to ReLU with negative values.",
        "intuition": "Combines benefits of ReLU with smooth negative values that push mean activations closer to zero.",
        "pros": [
            "Smooth everywhere (differentiable at zero)",
            "Mean activations closer to zero",
            "No dying ReLU problem",
            "Better learning characteristics"
        ],
        "cons": [
            "Computationally more expensive (exponential)",
            "Saturation for very negative values"
        ],
        "range": "(-α, ∞)",
        "use_cases": "Alternative to ReLU for deep networks",
        "params": {"alpha": (1.0, 0.1, 3.0, 0.1)},
        "emoji": "🌊",
    },
    "Swish": {
        "func": swish,
        "formula": "Swish(x) = x · σ(βx)",
        "latex": r"\text{Swish}(x) = x \cdot \sigma(\beta x)",
        "description": "Self-gated activation function discovered by Google. Smooth, non-monotonic function.",
        "intuition": "Smoothly interpolates between linear and ReLU-like behavior. The function 'remembers' through its self-gating.",
        "pros": [
            "Smooth everywhere",
            "Self-gating property",
            "Often outperforms ReLU in deep networks",
            "Non-monotonic (can model more complex patterns)"
        ],
        "cons": [
            "Computationally more expensive",
            "Unbounded output",
            "Less interpretable"
        ],
        "range": "(-∞, ∞)",
        "use_cases": "Deep neural networks, mobile architectures (MobileNet)",
        "params": {"beta": (1.0, 0.1, 2.0, 0.1)},
        "emoji": "🔄",
    },
    "GELU": {
        "func": gelu,
        "formula": "GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))",
        "latex": r"\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)",
        "description": "Gaussian Error Linear Unit - weights inputs by their magnitude, used in transformers.",
        "intuition": "Probabilistically gates inputs based on their value. Think of it as a smooth, stochastic version of ReLU.",
        "pros": [
            "Smooth everywhere",
            "Stochastic regularization effect",
            "State-of-the-art in transformers (BERT, GPT)",
            "Non-monotonic"
        ],
        "cons": [
            "Computationally expensive",
            "More complex to implement",
            "Less intuitive"
        ],
        "range": "(-0.17, ∞)",
        "use_cases": "Transformer models (BERT, GPT), modern NLP architectures",
        "emoji": "🎯",
    },
    "Softplus": {
        "func": softplus,
        "formula": "Softplus(x) = ln(1 + e^x)",
        "latex": r"\text{Softplus}(x) = \ln(1 + e^x)",
        "description": "Smooth approximation to ReLU, always positive and differentiable.",
        "intuition": "A smooth version of ReLU that never quite reaches zero. Like ReLU with rounded corners.",
        "pros": [
            "Smooth everywhere",
            "Always positive",
            "Derivative is sigmoid"
        ],
        "cons": [
            "Computationally expensive",
            "Can saturate for large negative values",
            "Rarely outperforms ReLU in practice"
        ],
        "range": "(0, ∞)",
        "use_cases": "Variational autoencoders, when strictly positive outputs needed",
        "emoji": "➰",
    },
    "Mish": {
        "func": mish,
        "formula": "Mish(x) = x · tanh(softplus(x))",
        "latex": r"\text{Mish}(x) = x \cdot \tanh(\text{Softplus}(x))",
        "description": "Self-regularized non-monotonic activation function. Smooth, continuous, and unbounded above.",
        "intuition": "Combines smoothness of Swish with better properties. Self-regularizes due to its smooth nature.",
        "pros": [
            "Smooth and continuous",
            "Self-regularizing",
            "Often improves accuracy over ReLU/Swish",
            "Non-monotonic"
        ],
        "cons": [
            "Most computationally expensive",
            "Relatively new (less studied)",
            "Memory intensive"
        ],
        "range": "(-∞, ∞)",
        "use_cases": "Computer vision tasks, object detection (YOLOv4)",
        "emoji": "✨",
    }
}
