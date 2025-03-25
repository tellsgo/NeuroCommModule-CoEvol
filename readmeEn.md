# NeuroCommModuleSculptor

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

> "The limits of language are the limits of neural networks."  
> —— Inspired by Wittgenstein's Tractatus Logico-Philosophicus 5.6

## Declaration I: Language as the Sculptor of Neural Architecture

When we observe the internal workings of large language models, we are not merely studying computer science, but exploring how language shapes the essence of cognition. This project stands at the intersection of linguistic philosophy and neuroscience —— through adding learnable communication modules at specific layers of the model, we observe their evolution process, providing computational evidence for understanding the relationship between language and thought.

## Declaration II: Language and Social Interaction as Prerequisites for Intelligence Emergence

> "We seek to answer: What cognitive landscape will be carved when language grows in artificial neural networks? This experiment is essentially a neural modeling version of Wittgenstein's language games from Philosophical Investigations."

## Project Overview: Multi-Agent Language Emergence Experiment

NeuroCommModuleSculptor explores how language shapes neural network structures through constructing a multi-agent environment where multiple independent language model instances develop communication protocols and symbol systems while solving collaborative tasks. Through observing this process, we can gain deeper insights into how language sculpts neural structures.

> "How do different AI agents develop effective communication methods without predefined protocols? How does this language emergence process reshape their neural representations?"

## Core Design: Multi-Agent Co-evolution

Unlike traditional designs that implement communication modules within a single model, this project employs multiple completely independent LLM instances as agents that:

- **Make Independent Decisions**: Each LLM instance maintains its own parameters and state
- **Have Limited Communication**: Agents can only communicate through restricted communication channels
- **Work on Collaborative Tasks**: Only through effective collaboration can they achieve higher rewards
- **Develop Emergent Language**: Observe how communication protocols form and evolve from scratch through training

This design allows us to observe how language symbols are created, agreed upon, and optimized from nothing, simulating the formation process of early human language.

## Technical Implementation Path

### 1. Multi-Agent Architecture
- Deploy multiple independent LLM instances (such as TinyLlama-1.1B or Phi-3-mini) as agents
- Each agent has its own independent parameter set but shares the same initial weights
- Support flexible configuration of 2-8 agents to study language emergence characteristics in groups of different sizes

### 2. Restricted Communication Protocol
- Agents can only send token sequences or vectors of limited length
- Optional communication bandwidth limitations, from highly restricted (only a few discrete symbols) to relatively open (short sentences)
- Support both turn-based and real-time communication modes, simulating language development in different social environments

### 3. Collaborative Task Design
- **Information Asymmetry Tasks**: Each agent only possesses partial information and needs to communicate to integrate it for problem-solving
- **Sequence Generation Tasks**: Agents take turns contributing content to generate coherent text
- **Collaborative Problem Solving**: Requires multiple specialized agents to collaborate to solve complex problems
- **Resource Allocation Games**: Simulates negotiation and trading in economic activities, requiring efficient communication

### 4. Language Evolution Analysis
- Communication content recording and clustering analysis, tracking the emergence of new vocabulary and grammatical structures
- Communication efficiency measurement, observing how communication protocols become more concise and precise over time
- Cross-task transfer testing, examining the generalization ability and robustness of emergent language

## Experimental Design: From Simple to Complex Language Emergence

### Phase 1: Basic Symbol Negotiation
- Start with minimal communication (only allowing a few discrete symbols)
- Use reference games (one agent describes an object, another guesses)
- Observe the formation process of a basic vocabulary

### Phase 2: Grammatical Structure Emergence
- Increase communication bandwidth and task complexity
- Require multiple rounds of interaction to complete tasks
- Analyze the spontaneous formation of syntactic structures and contextual dependencies

### Phase 3: Sociolinguistic Simulation
- Introduce multiple agent groups, initially with different communication systems
- Observe language fusion, dialect formation, and the emergence of lingua franca
- Study the impact of social factors (such as interaction frequency, group size) on language evolution

## Installation Guide

```bash
# Clone repository
git clone https://github.com/yourusername/NeuroCommModuleSculptor.git

# Enter project directory
cd NeuroCommModuleSculptor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
# Basic experiment example
from neurocommmodule import MultiAgentExperiment

# Configure experiment parameters
config = {
    "base_model": "TinyLlama-1.1B",
    "num_agents": 4,
    "communication_type": "discrete",
    "vocab_size": 32,
    "max_message_length": 8,
    "task": "reference_game",
    "distributed": True,
    "devices": [0, 1]
}

# Run the experiment
experiment = MultiAgentExperiment(config)
results = experiment.train(epochs=50)
experiment.visualize_language_evolution(results)
```

More examples can be found in the [examples](examples/) directory.

## Project Structure

```
NeuroCommModuleSculptor/
├── src/                        # Source code
│   ├── agents/               # Agent implementations
│   │   ├── agent_base.py     # Agent base class
│   │   ├── llm_agent.py      # LLM agent implementation
│   │   └── __init__.py
│   ├── communication/        # Communication protocols
│   │   ├── discrete_channel.py  # Discrete symbol channel
│   │   ├── vector_channel.py    # Continuous vector channel
│   │   ├── message.py           # Message definitions
│   │   └── __init__.py
│   ├── environments/         # Interaction environments
│   │   ├── reference_game.py # Reference game environment
│   │   ├── dialogue_env.py   # Dialogue environment
│   │   ├── problem_solving.py# Problem-solving environment
│   │   └── __init__.py
│   ├── experiments/          # Experimental design
│   │   ├── multi_agent_experiment.py # Multi-agent experiment
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── __init__.py
│   └── analysis/             # Analysis tools
│       ├── language_analyzer.py # Language analysis tool
│       ├── evolution_tracker.py # Evolution tracker
│       ├── visualizer.py     # Visualization tool
│       └── __init__.py
├── configs/                  # Configuration files
├── examples/                 # Example code
│   ├── basic_reference_game.py # Basic reference game example
│   ├── negotiation_game.py   # Negotiation game example
│   └── dialect_formation.py  # Dialect formation example
├── tests/                    # Test code
├── docs/                     # Documentation
├── requirements.txt          # Project dependencies
├── setup.py                  # Installation script
└── README.md                 # Project description
```

## How to Contribute

> "By observing how independent agents spontaneously form communication systems, we can not only gain deeper understanding of language origins and evolution but also provide insights for the design of future multi-AI systems."

We welcome various forms of contributions! If you are interested in multi-agent communication and language emergence, you can:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add a new environment'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References

- [1] [Emergent Communication in Multi-Agent Reinforcement Learning (ICLR 2020)](https://arxiv.org/abs/1910.04979)
- [2] Wittgenstein, L. (1953). Philosophical Investigations. Blackwell Publishing.
- [3] Steels, L. (2003). Evolving grounded communication for robots. Trends in cognitive sciences, 7(7), 308-312.
- [4] [The Emergence of Compositional Languages for Neural Agents (NeurIPS 2021)](https://arxiv.org/abs/2106.02671)
- [5] [EMMA: Multi-Agent Reinforcement Learning with Emergent Communication (ICML 2022)](https://arxiv.org/abs/2206.07956)
- [6] Kirby, S. (2001). Spontaneous evolution of linguistic structure: an iterated learning model of the emergence of regularity and irregularity. IEEE Transactions on Evolutionary Computation, 5(2), 102-110.
- [7] [Learning to Communicate with Deep Multi-Agent Reinforcement Learning (NIPS 2016)](https://arxiv.org/abs/1605.06676)
- [8] [Language as an Evolutionary System (Nature Human Behavior 2019)](https://www.nature.com/articles/s41562-019-0597-3)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Research Applications

The research outcomes of this project can be applied to multiple fields:

1. **Linguistic Theory Validation**: Validate theories of language origins and evolution through computational models
2. **Multi-AI System Design**: Develop efficient communication protocols for groups of AI systems that need to collaborate
3. **Human-Machine Interaction Optimization**: Understand how symbol systems form and optimize in interaction
4. **Educational Applications**: Create educational tools that simulate language acquisition and cultural transmission
5. **Sociolinguistic Research**: Simulate processes of language variation, propagation, and standardization 