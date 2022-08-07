from utils import *

MODULES = ['dependencies', 'setups', 'bottleneck_setups', 'utils', 'distributed']

PREP = load_code('/content/google-ai4code/prep')

RUN_DESCRIPTION = [
    'Importing dependencies',
    'Setting up light variables',
    'Setting up heavy variables',
    'Building utilities',
    'Running distributed data preprocessing'
]

generate_run_script([f"PREP['{key}']" for key in MODULES], RUN_DESCRIPTION)