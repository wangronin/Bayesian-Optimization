import sys
import os
import json
from run_by_config import validate_optimizers


MY_EXPEREMENT_FOLDER = "TMP"
fids = [21]
iids = [0]
dims = [10]
reps = 5
problem_type = 'BBOB'
optimizers = sys.argv[1:]
lb, ub = -5, 5


def generate_configs():
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
                                'folder': f'{MY_EXPEREMENT_FOLDER}-{cur_config_number}',
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


if __name__ == '__main__':
    generate_configs()

