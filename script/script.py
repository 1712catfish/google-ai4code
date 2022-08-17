from utils import *

MODULES = ['dependencies', 'setups', 'heavy_setups', 'utils', 'distributed']

RUN_DESCRIPTION = [
    'Importing dependencies',
    'Setting up light variables',
    'Setting up heavy variables',
    'Building utilities',
    'Running distributed data preprocessing'
]

CODES = load_code(path='/content/google-ai4code/prep', modules=MODULES)
print(generate_run_script(
    code_string_vars=[f"CODES['{key}']" for key in MODULES],
    run_description=RUN_DESCRIPTION
))

# print('Importing dependencies...'); exec(CODES['dependencies'])
# print('Setting up light variables...'); exec(CODES['setups'])
# print('Setting up heavy variables...'); exec(CODES['heavy_setups'])
# print('Building utilities...'); exec(CODES['utils'])
# print('Running distributed data preprocessing...'); exec(CODES['distributed'])

