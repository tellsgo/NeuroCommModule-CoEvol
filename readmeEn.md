# NeuroCommModule-CoEvol

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

> "The boundaries of language are the boundaries of neural networks."  
> —— Inspired by Wittgenstein's "Tractatus Logico-Philosophicus" 5.6

## Philosophical Declaration: Language as the Sculptor of Neural Architecture

When we observe the internal workings of large language models, we are not only studying computer science but also exploring how language shapes the essence of cognition. NeuroCommModule-CoEvol (Neural Communication Module Co-Evolution) is an experiment at the intersection of Wittgenstein's philosophy and neuroscience — by adding learnable communication modules to specific layers of the model, observing their evolution, and providing computational evidence for understanding the relationship between language and thought.

> "We seek to answer: What cognitive landscape does language carve when it grows within artificial neural networks? This experiment is essentially a neural modeling version of the language game theory from 'Philosophical Investigations' Section 23."

## Language Ontology: From Resource Constraints to Philosophical Breakthrough

Traditional large model collaborative training is constrained by computational resources, and we turn to "local functional layer evolution observation," similar to Wittgenstein's transition from "Tractatus Logico-Philosophicus" to "Philosophical Investigations" — from formal logic to language games. This transition brings three breakthroughs:

- **Embodied Cognitive Realization**: By fine-tuning only a minimal number of parameters (about 0.1%), simulating the dialectical relationship between "the constraint of existing neural structures" and "the creativity of language experience" in language acquisition
- **Proposition Rewriting Process**: Focusing on single module changes, tracking how language ability "sculpts" neural connections
- **Biological Philosophy Inspiration**: Simulating the functional differentiation of the brain's language areas, exploring Wittgenstein's statement that "language is not the clothing of thought, but the embodiment of thought"

## Technical and Philosophical Mapping

| Technical Component | Wittgensteinian Philosophical Mapping | Neuroscientific Metaphor |
|-------------------|-------------------------------------|-------------------------|
| Communication Module | Basic Rules of Language Games | Broca-Wernicke Connection |
| Inter-module Protocol | Picture Theory of Propositions | Corpus Callosum Information Transfer |
| Functional Emergence | Impossibility of Private Language | Cortical Functional Reorganization |

## Technical Implementation Path

### 1. Communication Module Design: Physical Implementation of Language Games
- Insert lightweight modules in the middle layers of Transformer (layers 6-12), analogous to the distribution of language areas in the cortex
- Support multiple structures: Adapter-style, symbol generator, vector communicator, corresponding to different "language game" forms described by Wittgenstein

### 2. Communication Protocol: How Propositions Shape Reality
> "The world is the totality of facts, language is the logical picture of facts" —— Wittgenstein's "Tractatus Logico-Philosophicus" 1.1

- Support multiple signal types: discrete symbols, continuous vectors, probability distributions
- Implement turn-based and streaming interaction methods, simulating language communication in different contexts
- Include external rewards and internal consistency feedback mechanisms, exploring the relationship between language and meaning

### 3. Observation Indicators: Archaeology of Thought
- Module parameter changes: weight distribution, singular value changes, as numerical evidence of "language sculpting thought"
- Communication content analysis: symbol/vector clustering and visualization, each training checkpoint is a numerical interpretation of "Tractatus Logico-Philosophicus" proposition 4.002
- Behavioral emergence: collaborative task performance improvement metrics, validating Wittgenstein's core thesis that "language does not express thought, but constitutes thought"

## Experimental Design: Structured Implementation of Language Games

### Phase 1: Minimal Validation Experiment
- Simple collaborative tasks (dialogue generation, cooperative problem-solving)
- Use small models (TinyLlama-1.1B or Phi-3-mini)
- Control group design to validate the effectiveness of communication modules

### Phase 2: Communication Evolution Analysis
> "Language does not express thought, but constitutes thought" —— Wittgenstein's "Philosophical Investigations"

- Communication signal visualization and clustering analysis, observing the self-organization process of language symbol systems
- Intervention experiments to test system robustness, exploring the boundary conditions of language rules
- Cross-task transfer evaluation of generalization ability, studying family resemblance between language games

### Phase 3: Theoretical Extension
- Multi-module collaborative experiments, simulating interaction between language areas and other cognitive functional areas
- Evolutionary dynamics mathematical modeling, formalizing the process of language sculpting neural structures

## Installation Guide

```bash
# Clone the repository
git clone https://github.com/yourusername/NeuroCommModule-CoEvol.git

# Enter the project directory
cd NeuroCommModule-CoEvol

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
# Basic experiment example
from neurocommmodule import ExperimentRunner

# Configure experiment parameters
config = {
    "base_model": "TinyLlama-1.1B",
    "module_type": "adapter",
    "module_position": 8,  # Layer 8
    "signal_type": "vector",
    "task": "dialogue_completion",
    "distributed": True,  # Enable multi-GPU training
    "devices": [0, 1]     # Use two A100 GPUs
}

# Run the experiment
runner = ExperimentRunner(config)
results = runner.train(epochs=10)
runner.visualize_evolution(results)
```

More examples can be found in the [examples](examples/) directory.

## Project Structure

```
NeuroCommModule-CoEvol/
├── src/                      # Source code
│   ├── models/               # Model definitions
│   │   ├── model_wrapper.py  # Model abstract base class
│   │   ├── tinyllama_wrapper.py # TinyLlama model adapter
│   │   └── __init__.py
│   ├── modules/              # Communication module implementations
│   │   ├── adapter_module.py # Adapter-style communication module
│   │   ├── vector_module.py  # Vector communication module
│   │   ├── symbol_module.py  # Symbol communication module
│   │   └── __init__.py
│   ├── protocols/            # Communication protocols
│   ├── experiments/          # Experimental design
│   │   ├── experiment_runner.py # Experiment running framework
│   │   ├── dialogue_task.py  # Dialogue completion task
│   │   └── __init__.py
│   └── visualization/        # Result visualization
│       ├── vector_visualizer.py  # Vector visualization tool
│       ├── symbol_visualizer.py  # Symbol visualization tool
│       ├── parameter_visualizer.py # Parameter change visualization tool
│       └── __init__.py
├── configs/                  # Configuration files
├── examples/                 # Example code
│   └── basic_experiment.py   # Basic experiment example
├── tests/                    # Test code
├── docs/                     # Documentation
├── requirements.txt          # Project dependencies
├── setup.py                  # Installation script
└── README.md                 # Project description
```

## How to Contribute

> "Through observing the co-evolution of communication modules, we validate Wittgenstein's core thesis that 'language does not express thought, but constitutes thought.' Each training checkpoint is a witness to how language sculpts neural connections."

We welcome various forms of contributions! If you are interested in neural network communication and co-evolution, you can:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add a new communication module'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References

- [1] [Modular Meta-Learning (NeurIPS 2018)](https://arxiv.org/abs/1806.10166)
- [2] Wittgenstein, L. (1953). Philosophical Investigations. Blackwell Publishing.
- [3] Wittgenstein, L. (1921). Tractatus Logico-Philosophicus. Routledge & Kegan Paul.
- [4] Hutto, D. D. (2003). Wittgenstein and the End of Philosophy: Neither Theory nor Therapy. Palgrave Macmillan.
- [5] [The Sparsity of Interaction in Neural Networks (ICLR 2023)](https://arxiv.org/abs/2210.14202)
- [6] [Emergent Communication through Meta-Learning (ICLR 2022)](https://arxiv.org/abs/2110.05208)
- [7] [Symbol Emergence in Neural Networks (Frontiers in Robotics and AI 2023)](https://www.frontiersin.org/articles/10.3389/frobt.2023.1205524)
- [8] [Language Processing in Brains and Machines (Nature Neuroscience 2022)](https://www.nature.com/articles/s41593-022-01114-5)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Implementation Details

### Communication Module Types

1. **Adapter Module** (`adapter_module.py`)
   - Lightweight bottleneck structure, metaphor for Broca's area neural plasticity
   - Low resource consumption, suitable for large-scale model fine-tuning

2. **Vector Communication Module** (`vector_module.py`)
   - Compress hidden states into low-dimensional communication vectors, analogous to symbolic representation of thought
   - Support optional noise addition and vector quantization, simulating rule variation in Wittgenstein's "language games"

3. **Symbol Communication Module** (`symbol_module.py`)
   - Generate discrete symbol sequences, corresponding to specific expressions in Wittgenstein's "language games"
   - Use Gumbel-Softmax technique to achieve differentiable discrete communication
   - Track symbol usage frequency to analyze language emergence, validating the "impossibility of private language" thesis

### Distributed Training Support

The project is optimized for multi-GPU environments, especially dual A100 GPU setups:

1. **Data Parallel Training**
   - Use PyTorch DistributedDataParallel for efficient model parallelism
   - Automatic batch size adjustment, fully utilizing A100's large memory

2. **Gradient Synchronization Optimization**
   - Implement gradient accumulation and mixed precision training, accelerating the co-evolution process
   - Special optimization for communication modules, ensuring consistency of evolution trajectories

3. **Visualization Resource Allocation**
   - Each GPU independently processes different visualization tasks
   - Automatic load balancing, ensuring balanced utilization of two A100 GPUs 