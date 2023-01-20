import os

from src.io import *

filter = ("dir", "neu")

def cond(filename):
    return os.path.basename(filename).startswith(filter)

paramss = []
results = []
dirnames = [dirname[0] for dirname in os.walk("results") if cond(dirname[0])]
print(f"Number of results: {len(dirnames)}")
for dirname in dirnames:
    tag = os.path.basename(dirname)
    filename = os.path.join(dirname, f"{tag}_result.csv")
    params, result = load_result(filename)
    paramss.append(params)
    results.append(result)

save_results(paramss, results, "problems")