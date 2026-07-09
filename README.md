# eval-observability-project
This is a project to probe the use of some common eval/observability tools in agent wielding projects. This project in particular is concerned with stock market predictions. The agents will be evaluated on tool usage and accuracy of their predictions.

## Tools
Observability and Evaluation tools implemented/tested in this project.

### Braintrust
Braintrust supports:
* Rubric-based evaluation of agent outputs 'Was the prediction reasonable given the evidence available at the time?'
* Consistency scoring across runs, which is crucial for agentic debate systems
* Human-in-the-loop validation, letting one know whether a label was 'correct', 'profitable', 'well-reasoned', or something else entirely

### Weights & Biases Traces
Provides timeline-style view:
* Agent chain-of-thought
* Every tool call (API, MCP, web search, data query, etc.)
* Latency, cost, failure mode
* Input/output for each agent loop (topic generation -> debate -> adjudication)

### MLflow AI
Excellent for model-centric workflows, but might not be ideal for agentic-centric implementations.
* Model tracking/versioning
* Training metrics
* Hyperparameters

# Outline

Similar to [FractionalAI's implementation](https://www.fractional.ai/case-study/augmenting-intelligence-with-ai-powered-trend-discovery-for-a-pe-backed-intelligence-platform), the project will have an affirmative, negative, and judge agent. The first two agents will be prompted to inspect a particular stock ticker/company, and determine if the stock is good to purchase or sell for a given time frame. The judge agent will listen to the reasonings given by the first two agents, and finally consolidate a purchase plan for the stock.

The agents' actions will all be recorded via the different observability tools mentioned in [Tools](##Tools).

# Currently Working on:

Model currently isn't returning the values expected. Working on applying Pydantic to enable mandated value output. Must work on the following:
* generate all Pydantic models for tools
* update all tool functions
* update all agent files
* update orchestrator to expect validated models
* add automatic JSON validation before sending tool results