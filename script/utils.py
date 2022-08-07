# Utilities to generate run script

from pathlib import Path


def load_code(path='.', modules=None):
    codes = dict()
    for module in Path(path).glob('*.py'):
        if modules is None or module.stem in modules:
            with open(module, 'r') as f:
                codes[module.stem] = f.read()
    return codes


def generate_run_script(code_strings, run_description):
    run_script = []
    for i, (code_string, description) in enumerate(zip(code_strings, run_description)):
        run_script.append(f"print('{description}...'); ")
        run_script.append(f"exec({code_string})'])")
    return '\n'.join(run_script)
