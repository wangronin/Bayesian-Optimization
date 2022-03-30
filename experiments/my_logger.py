import os
import time
import benchmark.bbobbenchmarks as bn
from ioh import get_problem


class MyIOHFormatOnEveryEvaluationLogger:
    def __init__(self, folder_name='TMP', algorithm_name='UNKNOWN', suite='unkown suite', algorithm_info='algorithm_info'):
        self.folder_name = MyIOHFormatOnEveryEvaluationLogger.__generate_dir_name(folder_name)
        self.algorithm_name = algorithm_name
        self.algorithm_info = algorithm_info
        self.suite = suite
        self.create_time = time.time()

    @staticmethod
    def __generate_dir_name(name, x=0):
        while True:
            dir_name = (name + ('-' + str(x))).strip()
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
                return dir_name
            else:
                x = x + 1

    def watch(self, algorithm, extra_data):
        # self.extra_info_getters = [getattr(algorithm, attr) for attr in extra_data]
        self.algorithm = algorithm
        self.extra_info_getters = extra_data

    def _set_up_logger(self, fid, iid, dim, func_name):
        self.log_info_path = f'{self.folder_name}/IOHprofiler_f{fid}_{func_name}.info'
        with open(self.log_info_path, 'a') as f:
            f.write(f'suite = \"{self.suite}\", funcId = {fid}, funcName = \"{func_name}\", DIM = {dim}, maximization = \"F\", algId = \"{self.algorithm_name}\", algInfo = \"{self.algorithm_info}\"\n')
        self.log_file_path = f'data_f{fid}_{func_name}/IOHprofiler_f{fid}_DIM{dim}.dat'
        self.log_file_full_path = f'{self.folder_name}/{self.log_file_path}'
        os.makedirs(os.path.dirname(self.log_file_full_path), exist_ok=True)
        self.first_line = 0
        self.last_line = 0
        with open(self.log_file_full_path, 'a') as f:
            f.write('\"function evaluation\" \"current f(x)\" \"best-so-far f(x)\" \"current af(x)+b\" \"best af(x)+b\"')
            for extra_info in self.extra_info_getters:
                f.write(f' {extra_info}')
            f.write('\n')

    def log(self, cur_evaluation, cur_fitness, best_so_far):
        with open(self.log_file_full_path, 'a') as f:
            f.write(f'{cur_evaluation} {cur_fitness} {best_so_far} {cur_fitness} {best_so_far}')
            for fu in self.extra_info_getters:
                try:
                    extra_info = getattr(self.algorithm, fu)
                except Exception as e:
                    extra_info = 'None'
                f.write(f' {extra_info}')
            f.write('\n')
            self.last_line += 1

    def finish_logging(self):
        time_taken = time.time() - self.create_time
        with open(self.log_info_path, 'a') as f:
            f.write('%\n')
            f.write(f'{self.log_file_path}, {self.first_line}:{self.last_line}|{time_taken}\n')


class MyObjectiveFunctionWrapper:
    def __init__(self, fid, iid, dim, directed_by='Hao'):
        self.fid = fid
        self.iid = iid
        self.dim = dim
        self.my_loggers = []
        if directed_by == 'Hao':
            self.my_function, self.optimum = bn.instantiate(ifun=fid, iinstance=iid)
            iohf = get_problem(fid, dimension=dim, instance=iid, problem_type = 'Real')
            self.func_name = iohf.meta_data.name
        elif directed_by == 'IOH':
            _, self.optimum = bn.instantiate(ifun=fid, iinstance=iid)
            self.my_function = get_problem(fid, dimension=dim, instance=iid, problem_type = 'Real')
            self.func_name = self.my_function.meta_data.name
        else:
            raise ValueError('Unknown way to create function using', directed_by)
        self.cnt_eval = 0
        self.best_so_far = float('inf')
        self.min_distance = float('inf')

    def __call__(self, x):
        print(x)
        cur_value = self.my_function(x)
        distance = cur_value - self.optimum
        self.best_so_far = min(self.best_so_far, cur_value)
        self.min_distance = min(self.min_distance, distance)
        self.cnt_eval += 1
        for l in self.my_loggers:
            l.log(self.cnt_eval, distance, self.min_distance)
        return cur_value

    def attach_logger(self, logger):
        self.my_loggers.append(logger)
        logger._set_up_logger(self.fid, self.iid, self.dim, self.func_name)


