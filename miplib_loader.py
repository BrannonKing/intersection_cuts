import io
import gurobipy as gp
import urllib.request as ulr
import pathlib as pl
import zipfile as zf
benchmark_path = pl.Path('mip2017_benchmark')
solution_filename = 'miplib2017-v26.solu'


def ensure_downloaded():
    if not benchmark_path.exists():
        print('One-time download of MIPLIB-2017 benchmark may take a minute...')
        with ulr.urlopen('https://miplib.zib.de/downloads/benchmark.zip') as f:
            with io.BytesIO(f.read()) as data, zf.ZipFile(data) as zipped:
                zipped.extractall(benchmark_path)
        ulr.urlretrieve('https://miplib.zib.de/downloads/' + solution_filename, benchmark_path / solution_filename)
        print('Done downloading and extracting benchmark files. Extracted to', benchmark_path)


class BenchmarkInstance:
    def __init__(self, opt, fn, score):
        self.known_optimum = opt == "=opt="
        self.score = score
        self.filename = fn

    def as_gurobi_model(self):
        # for windows, may need to get https://sourceforge.net/projects/gzip-for-windows/ and put it in venv/Scripts
        return gp.read(str(self.filename))


def get_instances():
    ensure_downloaded()
    instances = {}
    files = {path.name.removesuffix('.mps.gz'): path for path in benchmark_path.rglob('*') if '.mps' in path.name}
    with open(benchmark_path / solution_filename) as file:
        for line in file:
            parts = line.split()
            if parts[1] not in files:
                continue
            if len(parts) == 2:
                score = float('inf') if parts[0] == '=inf=' else float('nan')
            else:
                score = float(parts[2])
            instances[parts[1]] = BenchmarkInstance(parts[0], files[parts[1]], score)
    return instances


