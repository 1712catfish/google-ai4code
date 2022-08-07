# Utilities to generate run script
from pathlib import Path


def load_code(path='.', modules=None):
    codes = dict()
    for module in Path(path).glob('*.py'):
        if modules is None or module.stem in modules:
            with open(module, 'r') as f:
                codes[module.stem] = f.read()
    return codes


def generate_run_script(code_string_vars, run_description=None):
    run_script = []
    if run_description is None:
        run_description = code_string_vars
    for i, (code_string_var, description) in enumerate(zip(code_string_vars, run_description)):
        run_script.append(f"print('{description}...'); exec({code_string_var})")
    return '\n'.join(run_script)