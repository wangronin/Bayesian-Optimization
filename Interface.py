"""
@author: Hao Wang
@email: wangronin@gmail.com
"""

from pdb import set_trace
import os, sys, json, logging, dill
import numpy as np

import urllib.parse as urlparse
from optparse import OptionParser
from http.server import BaseHTTPRequestHandler, HTTPServer
    
from GaussianProcess import GaussianProcess
from GaussianProcess.trend import constant_trend

from BayesOpt.utils import Daemon
from BayesOpt import ParallelBO, SearchSpace
from BayesOpt.SearchSpace import ContinuousSpace
from BayesOpt.Surrogate import RandomForest
from BayesOpt.misc import random_string

# Configuration request Handler
class RemoteBO(BaseHTTPRequestHandler, object):
    """ 
    Note 
    ----
        This class is instantiated for every incoming HTTP request and it is terminated 
        after hanlding the HTTP request. Any non-static attribute will not be accessible 
        across HTTP requests.
    """
    work_dir, verbose = '', False
    def log_message(self, format, *args):
        msg = "%s - - %s" % (self.address_string(), format%args)
        self.logger.info(msg)

    def _get_job_id(self, info):
        if 'job_id' not in info:
            raise Exception('`job_id` is necessary!')
        
        job_id = info['job_id']
        if isinstance(job_id, list):
            job_id = job_id[0]
        return job_id

    def _get_dump_file(self, job_id, check=True):
        dump_file = os.path.join(self.work_dir, job_id + '.dump')
        if check and not os.path.exists(dump_file):
            raise Exception('The dump file of job %s is not found!'%(job_id))

        return dump_file

    def _send_response(self, rsp_data):
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(rsp_data).encode('utf-8'))

    def do_GET(self):
        parsed = urlparse.urlparse(self.path)
        try:
            info = urlparse.parse_qs(parsed.query)
        except Exception as ex:
            self.logger.error(str(ex))
            self.send_error(500, str(ex))
            return 
        
        rsp_data = {}
        if 'ask' in info:
            try:
                n_point = int(info['ask'])
            except:
                n_point = None

            try:
                job_id = self._get_job_id(info)
                dump_file = self._get_dump_file(job_id)
            except Exception as ex:
                self.logger.error(str(ex))
                self.send_error(500, str(ex))
                self._send_response(rsp_data)
                return 

            try:
                self.logger.info('ask request from job %s'%job_id)
                opt = ParallelBO.load(dump_file)
                X = opt.ask(n_point)

                rsp_data['job_id'] = job_id
                rsp_data['X'] = X
                self.send_response(200)
            except Exception as ex:
                self.logger.error(str(ex))
                self.send_error(500, str(ex))

        elif 'finalize' in info:
            try:
                job_id = self._get_job_id(info)
                self.logger.info('finalize request from job %s'%job_id)
                dump_file = self._get_dump_file(job_id)
                # data_file = os.path.join(self.work_dir, job_id + '.csv')
            except Exception as ex:
                self.logger.error(str(ex))
                self.send_error(500, str(ex))
                self._send_response(rsp_data)
                return
            
            os.remove(dump_file)
            # os.remove(data_file)
            self.send_response(200)

        self._send_response(rsp_data)
    
    def _create_optimizer(self, search_param, bo_param, data_file, log_file):
        search_space = SearchSpace.from_dict(search_param, space_name=False)
        n_obj = bo_param['n_obj']
        max_FEs = bo_param['max_iter'] * bo_param['n_point'] + bo_param['DoE_size']
        del bo_param['max_iter']
        del bo_param['n_obj']

        # TODO: turn this off until the feature importance of GPR is implemented
        if len(search_space.id_N) == 0 and len(search_space.id_O) == 0 and 11 < 2:
            dim = search_space.dim
            lb, ub = np.atleast_2d(search_space.bounds).T 
            # autocorrelation parameters of GPR
            thetaL = 1e-10 * (ub - lb) * np.ones(dim)
            thetaU = 2 * (ub - lb) * np.ones(dim)
            theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

            mean = constant_trend(dim, beta=0) 
            model = GaussianProcess(
                mean=mean, corr='matern',
                theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                nugget=1e-10, noise_estim=True,
                optimizer='BFGS', wait_iter=5, random_start=30 * dim,
                likelihood='concentrated', eval_budget=200 * dim
            )
            optimizer = 'BFGS'
        else :
            model = RandomForest(levels=search_space.levels)
            optimizer = 'MIES'

        if n_obj == 1:   # single-objective Bayesian optimizer
            opt = ParallelBO(
                search_space=search_space, 
                obj_fun=None,
                model=model, 
                eval_type='dict',
                acquisition_fun='MGFI',
                max_FEs=max_FEs,
                acquisition_par={'t' : 2},
                acquisition_optimization={'optimizer' : optimizer},
                logger=log_file,
                data_file=data_file,
                **bo_param
            )
        else: 
            raise NotImplementedError
        
        return opt

    def do_POST(self):
        content_len = int(self.headers.get('content-length'))
        post_body = self.rfile.read(content_len)
        rsp_data = {}

        try:
            data = json.loads(post_body)
        except:
            self.logger.error('Received message: \n{}\nis not JSON!'.format(post_body))
            self.send_error(500, 'file not found')
            return 
        
        if 'search_param' in data:  
            try:
                job_id = random_string()
                rsp_data = {'job_id' : job_id}
                dump_file = self._get_dump_file(job_id, check=False) 
                data_file = os.path.join(self.work_dir, job_id + '.csv')
                log_file = os.path.join(self.work_dir, job_id + '.log')

                opt = self._create_optimizer(
                    data['search_param'], data['bo_param'], data_file, log_file
                )

                if os.path.exists(dump_file):
                    self.logger.info('overwritting the dump file!')

                opt.save(dump_file)
                self.logger.info('create job %s'%job_id)
                self.send_response(200)
            except Exception as ex:
                self.logger.error(str(ex))
                self.send_error(500, str(ex))

        elif 'check_job' in data:
            job_ids = data['check_job']
            for job_id in job_ids:
                try:
                    dump_file = self._get_dump_file(job_id)
                    opt = ParallelBO.load(dump_file)
                except Exception as ex:
                    pass
                
                rsp_data[job_id] = {
                    'dim' : opt.search_space.dim,
                    'max_FEs' : opt.max_FEs,
                    'step' : opt.iter_count,
                    'fopt' : opt.fopt if hasattr(opt, 'fopt') else None
                }
            self.send_response(200)
        
        elif 'get_history' in data:
            job_ids = data['get_history']
            for job_id in job_ids:
                try:
                    dump_file = self._get_dump_file(job_id)
                    opt = ParallelBO.load(dump_file)
                except Exception as ex:
                    pass
                
                rsp_data[job_id] = {
                    'hist_f' : opt.hist_f if hasattr(opt, 'hist_f') else None
                }
            self.send_response(200)
        
        elif 'get_feature_importance' in data:
            job_ids = data['get_feature_importance']
            for job_id in job_ids:
                try:
                    dump_file = self._get_dump_file(job_id)
                    opt = ParallelBO.load(dump_file)
                    model = opt.model
                except Exception as ex:
                    rsp_data[job_id] = {}
                    continue
                
                
                feature_name = opt.search_space.var_name
                levels = opt.search_space.levels
                imp = model.feature_importances_ if hasattr(model, 'feature_importances_')\
                    else None
                
                if levels is not None:
                    imp_ = np.zeros(len(feature_name))
                    cat_idx = model._cat_idx
                    _idx = list(set(range(len(imp_))) - set(cat_idx))
                    _n = len(feature_name) - len(levels)
                    idx = np.cumsum([0] + [len(_) for _ in levels.values()]) + _n
                    _imp = [np.sum(imp[idx[i]:idx[i + 1]]) for i in range(len(idx) - 1)]

                    imp_[cat_idx] = _imp
                    imp_[_idx] = imp[0:_n]
                else:
                    imp_ = imp
                    
                rsp_data[job_id] = {
                    'feature' : feature_name,
                    'imp' : imp_.tolist()
                }
                
            self.send_response(200)

        elif 'X' in data and 'y' in data:
            try:
                job_id = self._get_job_id(data)
                dump_file = self._get_dump_file(job_id)
            except Exception as ex:
                self.logger.error(str(ex))
                self.send_error(500, str(ex))
                self._send_response(rsp_data)
                return

            try:
                self.logger.info('tell request from job %s'%job_id)
                opt = ParallelBO.load(dump_file)

                opt.tell(data['X'], data['y'])
                opt.save(dump_file)

                rsp_data['xopt'] = opt.xopt
                rsp_data['fopt'] = opt.fopt

                self.send_response(200)
            except Exception as ex:
                self.logger.error(str(ex))
                self.send_error(500, str(ex))

        self._send_response(rsp_data)


class RemoteBODaemon(Daemon):
    def set_logger(self, log_file, verbose):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] -- %(message)s')

        if verbose != 0:
            # create console handler and set level to warning
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

        # create file handler and set level to debug
        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
        self.logger.propagate = False

    def run(self, host, port, verbose, work_dir, log_file):
        print('initialize the remote BO server...')

        work_dir = str(port) if len(work_dir) == 0 else work_dir
        work_dir = os.path.join(os.path.expanduser('~'), work_dir)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        log_file = os.path.join(work_dir, log_file)
        self.set_logger(log_file, verbose)

        RemoteBO.logger = self.logger
        RemoteBO.verbose = verbose
        RemoteBO.work_dir = work_dir
        httpd = HTTPServer((host, port), RemoteBO)

        print('runnning...')
        httpd.serve_forever()
  
if __name__ == '__main__':
    opt_parser = OptionParser(usage=('usage: %prog -h hostname -p port'))

    opt_parser.add_option('-p', '--port', action='store', dest='port', 
                          help='port number', default=7200, type='int')
    
    opt_parser.add_option('-H', '--host', action='store', dest='host', 
                          help='host address', default='127.0.0.1', type='string')
    
    opt_parser.add_option('-l', '--log', action='store', dest='log_file', 
                          help='log file name', default='log', type='string')
    
    opt_parser.add_option('-w', '--working-dir', action='store', dest='work_dir', 
                          help='directory of temporary files', default='', type='string')
    
    opt_parser.add_option('-v', '--verbose', action='store_true', dest='verbose', 
                          help='foreground/background mode', default=False)

    options, args = opt_parser.parse_args()
    host, port = options.host, options.port

    # start the server in the daemon mode
    dm = RemoteBODaemon('.bo.pid')
    if options.verbose:
        dm.run(**options.__dict__)
    else:
        dm.start(**options.__dict__)