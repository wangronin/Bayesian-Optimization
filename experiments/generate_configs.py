import sys
import os
import json
from run_by_config import validate_optimizers


def generate_configs(experiment_config_file_name):
    with open(experiment_config_file_name, 'r') as f:
        config = json.load(f)
    results_folder = config['folder']
    fids = config['optimizers']
    iids = config['iids']
    dims = config['dims']
    reps = config['reps']
    optimizers = config['optimizers']
    lb, ub = config['lb'], config['ub']
    validate_optimizers(optimizers)
    runs_number = len(optimizers) * len(fids) * len(iids) * len(dims) * reps
    cur_config_number = 0
    dir_name = 'configs'
    os.makedirs(dir_name, exist_ok=True)
    for f in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, f))
    for my_optimizer_name in optimizers:
        for fid in fids:
            for iid in iids:
                for dim in dims:
                    for rep in range(reps):
                        experiment_config = {
                                'folder': f'{results_folder}-{cur_config_number}',
                                'opt': my_optimizer_name,
                                'fid': fid,
                                'iid': iid,
                                'dim': dim,
                                'seed': rep,
                                'lb': lb,
                                'ub': ub,
                                }
                        with open(f'{dir_name}/{cur_config_number}.json', 'w') as f:
                            json.dump(experiment_config, f)
                        cur_config_number += 1
    print(f'Generated {cur_config_number} files')


if __name__ == '__main__':
    generate_configs(sys.argv[1])

