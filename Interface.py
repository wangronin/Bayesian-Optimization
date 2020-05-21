# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:42:52 2017

@author: Hao Wang
@email: wangronin@gmail.com
"""

from pdb import set_trace
import os, sys, json, logging, dill

import urllib.parse as urlparse
from optparse import OptionParser
from http.server import BaseHTTPRequestHandler, HTTPServer
    
from BayesOpt.utils import Daemon
from BayesOpt import BO
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import from_dict
from BayesOpt.misc import random_string

# Configuration request Handler
class RemoteBO(BaseHTTPRequestHandler, object):
    """ Note 
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
                opt = BO.load(dump_file)
                X = opt.ask(n_point)
                X = [x.to_dict('var') for x in X]

                rsp_data['job_id'] = job_id
                rsp_data['X'] = X
                self.logger('ask request from job %s'%job_id)
                self.send_response(200)
            except Exception as ex:
                self.logger.error(str(ex))
                self.send_error(500, str(ex))

        elif 'finalize' in info:
            try:
                job_id = self._get_job_id(info)
                dump_file = self._get_dump_file(job_id)
                data_file = os.path.join(self.work_dir, job_id + '.csv')
            except Exception as ex:
                self.logger.error(str(ex))
                self.send_error(500, str(ex))
                self._send_response(rsp_data)
                return
            
            os.remove(dump_file)
            os.remove(data_file)
            self.logger('finalize request from job %s'%job_id)
            self.send_response(200)

        self._send_response(rsp_data)
        
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

                search_param, bo_param = data['search_param'], data['bo_param']
                search_space = from_dict(search_param, space_name=False)
                n_obj = bo_param['n_obj']
                del bo_param['n_obj']

                if n_obj == 1:   # single-objective Bayesian optimizer
                    opt = BO(search_space=search_space, obj_func=None, 
                             surrogate=RandomForest(levels=search_space.levels), 
                             logger=self.logger, data_file=data_file,
                             eval_type='dict', optimizer='MIES', **bo_param)
                
                # TODO: to test this part
                elif n_obj > 1:  # multi-objective Bayesian optimizer
                    pass
                
                if os.path.exists(dump_file):
                    self.logger.info('overwritting the dump file!')

                opt.save(dump_file)
                self.logger('create job %s'%job_id)
                self.send_response(200)
            except Exception as ex:
                self.logger.error(str(ex))
                self.send_error(500, str(ex))

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
                opt = BO.load(dump_file)

                X = [list(x.values()) for x in data['X']]
                opt.tell(X, data['y'])
                opt.save(dump_file)

                rsp_data['xopt'] = opt.xopt.to_dict()
                rsp_data['fopt'] = opt.fopt

                self.logger('tell request from job %s'%job_id)
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