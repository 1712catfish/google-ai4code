def runfile(path):
    with open(path, 'r') as f:
        exec(f.read())