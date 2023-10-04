import io
import gurobipy as gp
import urllib.request as ulr
import pathlib as pl
import zipfile as zf
benchmark_path = pl.Path('mip2017_benchmark')


def ensure_downloaded():
    if not benchmark_path.exists():
        print('One-time download of MIPLIB-2017 benchmark may take a minute...')
        with ulr.urlopen('https://miplib.zib.de/downloads/benchmark.zip') as f:
            with io.BytesIO(f.read()) as data, zf.ZipFile(data) as zipped:
                zipped.extractall(benchmark_path)
        print('Done downloading and extracting benchmark files. Extracted to', benchmark_path)


class BenchmarkInstance:
    def __init__(self, opt, fn, score):
        self.known_optimum = opt == "=opt="
        self.score = score
        self.filename = fn

    def as_gurobi_model(self):
        return gp.read(benchmark_path / self.filename)


def get_instances():
    ensure_downloaded()
    instances = {}
    with open(benchmark_path / 'miplib2017-v23.solu.txt') as file:
        for line in file:
            parts = line.split(' ', 3)
            names = parts[1].split('.')
            if len(parts) == 2:
                score = float('inf') if parts[0] == '=inf=' else float('nan')
            else:
                score = float(parts[2])
            instances[names[0]] = BenchmarkInstance(parts[0], parts[1], score)
    return instances


