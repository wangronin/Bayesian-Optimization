# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:15:50 2017

@author: wangronin
"""
import logging, random, string, re, os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Custom formatter
# TODO: use relative path for %(pathname)s
class MyFormatter(logging.Formatter):
    default_time_format = '%m/%d/%Y %H:%M:%S'
    default_msec_format = '%s,%02d'

    FORMATS = {
        logging.DEBUG : '%(asctime)s - [%(levelname)s] {%(pathname)s:%(lineno)d} -- %(message)s',
        logging.INFO : '%(asctime)s - [%(levelname)s] -- %(message)s',
        logging.WARNING : '%(asctime)s - [%(levelname)s] {%(name)s} -- %(message)s',
        logging.ERROR : '%(asctime)s - [%(levelname)s] {%(name)s} -- %(message)s',
        'DEFAULT' : '%(asctime)s - %(levelname)s -- %(message)s'}
    
    def __init__(self, fmt='%(asctime)s - %(levelname)s -- %(message)s'):
        MyFormatter.FORMATS['DEFAULT'] = fmt
        super().__init__(fmt=fmt, datefmt=None, style='%') 
    
    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        _fmt = self._style._fmt

        # Replace the original format with one customized by logging level
        self._style._fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])

        # Call the original formatter class to do the grunt work
        fmt = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = _fmt
        return fmt


def random_string(k=15):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=15))

def expand_replace(s):
    m = re.match(r'${.*}', s)
    for _ in m.group():
        s.replace(_, os.path.expandvars(_))
    return s


