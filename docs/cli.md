# Command-Line Interface

ProofOfThought provides a CLI for executing Z3 JSON DSL programs directly from the command line.

## Installation

The CLI is installed automatically with the package:

```bash
pip install proofofthought
```

## Usage

```bash
python -m z3adapter.cli <json_file> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `json_file` | Path to JSON configuration file (required) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--verify-timeout` | `10000` | Timeout for verification checks in milliseconds |
| `--optimize-timeout` | `100000` | Timeout for optimization in milliseconds |
| `--log-level` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

## Examples

### Basic Execution

```bash
python -m z3adapter.cli examples/syllogism.json
```

### With Custom Timeouts

```bash
python -m z3adapter.cli complex_problem.json --verify-timeout 30000 --optimize-timeout 200000
```

### Debug Mode

```bash
python -m z3adapter.cli my_program.json --log-level DEBUG
```

## JSON File Format

The CLI expects JSON files in the ProofOfThought DSL format. See the [DSL Specification](dsl-specification.md) for details.

### Minimal Example

```json
{
  "sorts": [
    {"name": "Animal", "type": "enum", "values": ["cat", "dog", "bird"]}
  ],
  "constants": [
    {"name": "pet", "sort": "Animal"}
  ],
  "rules": [
    {"constraint": "pet == Animal.cat"}
  ],
  "verifications": [
    {"description": "Pet is a cat", "check": "pet == Animal.cat"}
  ]
}
```

### Running the Example

```bash
# Save the above JSON to test.json
python -m z3adapter.cli test.json
```

Output:
```
INFO: Processing sorts...
INFO: Processing constants...
INFO: Processing rules...
INFO: Running verifications...
INFO: Verification 'Pet is a cat': sat
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (invalid JSON, Z3 error, etc.) |
| `130` | Interrupted by user (Ctrl+C) |

## Programmatic Alternative

For integration into Python applications, use the `Z3JSONInterpreter` class directly:

```python
from z3adapter.interpreter import Z3JSONInterpreter

interpreter = Z3JSONInterpreter(
    "my_program.json",
    verify_timeout=10000,
    optimize_timeout=100000,
)
interpreter.run()
```

## Limitations

- The CLI only supports the JSON backend, not SMT2
- For LLM-generated programs, use the `ProofOfThought` class instead
- Output is logged to stderr; use `--log-level WARNING` for quiet operation
