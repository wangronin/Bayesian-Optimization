import requests, json, os, subprocess, shutil, sys, time
import numpy as np

payload = {
    "search_param" : {
        "emissivity" : {
            "type" : "r",
            "range" : [0, 1],
            "N" : 2,
            "precision" : 2
        },

        "offset" : {
            "type" : "r",
            "range" : [0, 1],
            "N" : 2,
            "precision" : 2
        },

        "power" : {
            "type" : "r",
            "range" : [0, 1],
            "N" : 1,
            "precision" : 2  # 数值精度：小数点后位数
        },
    },

    "bo_param" : {
        "n_job" : 1,         # 服务器上并行进程数
        "n_point" : 1,       # 每次迭代返回参数值个数
        "max_iter" : 20,     # 最大迭代次数
        "DoE_size" : 3,      # 初始（第一代）采样点个数，其一般与`n_point`相等
        "minimize" : True,   # 最大化/最小化
        "n_obj": 1
    }
}

address = 'http://127.0.0.1:7200'

def obj_func_dict_eval2(par):
    """范例目标函数，其输入`par`为一个包含了一组候选参数的字典
    这些参数在字典中的键值是`search_space`中给出的参数名，如'continuous'，'nominal'
    这个函数应该返回由用户自定义的目标值。在这里我们给出了一个很简单的范例，其不蕴含任何实际意义
    """
    x_r = np.asarray([par[k] for k in par.keys() if k.startswith('emissivity')])
    x_r2 = np.asarray([par[k] for k in par.keys() if k.startswith('offset')])
    x_i = par['power']
    return np.sum(x_r ** 2.) + \
        abs(x_i - 3.5) + \
            np.sum(x_r2 ** 2.) + \
                np.random.randn() * np.sqrt(.5)

def test_remote():
    env = os.environ.copy()
    if not 'PYTHONPATH' in env:
        env['PYTHONPATH'] = ''

    env['PYTHONPATH'] = "../" +  env['PYTHONPATH']

    # proc = subprocess.Popen([
    #     'python3', '-m', 'bayes_optim.SimpleHTTPServer', '-w', '7200', '-v'
    # ], env=env)
    # time.sleep(3)

    r = requests.post(address, json=payload)
    job_id = r.json()['job_id']
    print('Job id is %s'%(job_id))

    for i in range(3):
        print('iteration %d'%(i))

        r = requests.get(address, params={'ask' : 'null', 'job_id' : job_id})
        tell_data = r.json()

        y = [obj_func_dict_eval2(_) for _ in tell_data['X']]
        tell_data['y'] = y

        print(tell_data)
        r = requests.post(address, json=tell_data)

    r = requests.get(address, params={'finalize' : 'null', 'job_id' : job_id})

    # proc.kill()
    # shutil.rmtree('7200')

test_remote()