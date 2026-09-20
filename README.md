# neo-light-core

Shared token optimiser core library for neo-light forks.

## Contents

- `config.py` — TokenOptimiserConfig schema & defaults
- `router.py` — TokenRouter, ITIL classification, parallelize/swarm scoring
- `boa.py` — Board of Advisors (6 seats), governance, TOS probing, swarm validation
- `monica_integration.py` — `optimise()` wrapper, calibration, learning log
- `mock_backends.py` — T1/T2/T3 mock backends + test harness
- `dashboard/` — FastAPI dashboard for live config & analytics
- `tests/` — 78 integration tests

## Installation

```bash
pip install -e .
```

## Usage

```python
from token_optimiser.config import config
from token_optimiser.router import TokenRouter
from token_optimiser.monica_integration import optimise

# User repo provides config_user.py that overrides config
decision = optimise("Your task here")
```

## Versioning

Semver tags: `v0.1.0`, `v0.2.0`, etc. User repos pin submodule to tags.
