import requests, json
import numpy as np

data = {
    "search_param" : {
        "emissivity" : {     
            "type" : "r",          
            "range" : [0.95, 1],
            "N" : 2          
        },

        "offset" : {       
            "type" : "r",
            "range" : [-10, 10],    
            "N" : 2
        },

        "power" : {           
            "type" : "r",
            "range" : [3.2, 3.8],    
            "N" : 1
        },
    },

    "bo_param" : {
        "n_job" : 1,         # 服务器上并行进程数
        "n_point" : 1,       # 每次迭代返回参数值个数
        "max_iter" : 20,     # 最大迭代次数
        "n_init_sample" : 3, # 初始（第一代）采样点个数，其一般与`n_point`相等
        "minimize" : True,
        "noisy" : True,
        "n_obj": 1
    }
}

address = 'http://207.246.97.250:7200'
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

# 请求创建新的优化任务，其初始化数据由一个json数据`data`给定
r = requests.post(address, json=data)
job_id = r.json()['job_id']
print('Job id is %s'%(job_id))

for i in range(20):  
    print('iteration %d'%(i))

    # 请求一组候选参数值用于测试。请求时必须附加之前初始化时返回的任务id
    r = requests.get(address, params={'ask' : 'null', 'job_id' : job_id})
    tell_data = r.json()
    
    # 运行并评估优化服务器返回的候选参数
    y = [obj_func_dict_eval2(_) for _ in tell_data['X']]
    tell_data['y'] = y 

    print(tell_data)
    
    # 将候选参数的目标值`y`返回给服务器
    r = requests.post(address, json=tell_data)

# 在优化任务结束后，我们向服务器发出请求结束掉这个任务并清除服务器上的相关文件
r = requests.get(address, params={'finalize' : 'null', 'job_id' : job_id})