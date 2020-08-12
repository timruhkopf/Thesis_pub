import os
import sys
from pip._internal.operations import freeze
from datetime import datetime
from subprocess import check_output

import pandas as pd


class Grid:
    def __init__(self, root):
        print('------ progressing with: {name} --------------------- '.format(
            name=self.__class__.__name__))
        self.root = root
        self.pathresults = root + '/results/'
        if not os.path.isdir(self.pathresults[:-1]):
            os.mkdir(self.pathresults)

        # critical section: reference in bash different compared to
        # debug call from subclass' module
        if not os.path.isfile(self.pathresults +'run_log.csv'):
            df = pd.DataFrame(columns=['index', 'id', 'success', 'config'])
            df.to_csv(self.pathresults + 'run_log.csv')

        self.hash = self._create_hash()
        self._pip_freeze()
        self._log_main_function()

        self.main = self.try_main(self.main)

    def _log_main_function(self):
        import inspect
        source = inspect.getsource(self.main)
        with open(self.pathresults + '{}.log'.format(self.hash), 'a') as file:
            file.write('\n' + source)


    def try_main(self, func):
        """decorate main to ensure the main function runs smoothly"""
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
                self.write_to_table(success=True, config_str=str(args)+ str(kwargs))

            except Exception as error:
                self.write_to_table(success=False, config_str=str(args)+ str(kwargs))
                import sys
                import traceback

                with open(self.pathresults + '{}.log'.format(self.hash), 'a') as file:
                    # catch the entire error message:
                    # exc_type, exc_value, exc_traceback = sys.exc_info()
                    file.write('\n' + traceback.format_exc() + '\n')
                    file.write('ERROR triggered by main(args:{}, kwargs:{}) \n'.format(
                        str(args), str(kwargs)))
        return wrapper

    def main(self, model_config, sampler_config):
        raise NotImplementedError('Create a main function for your grid class')
        # try:
        #     print()
        # except:
        #     print()

    def _create_hash(self):
        return '{model_file}_{commit}_{timestamp}'.format(
            commit=check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip(),
            model_file= self.__class__.__name__, # os.path.basename(__file__)[:-3],
            timestamp=  # strftime("%Y%m%d_%H%M%S%", gmtime())
            datetime.now().strftime("%H%M%S%f")
        )

    def _pip_freeze(self):
        pipfreeze = freeze.freeze()
        with open(self.pathresults + '/{}_pip_freeze.txt'.format(self.hash), 'w') as file:
            file.write('Torch ' + sys.version + '\n')
            for line in pipfreeze:
                file.write(line + '\n')
            file.close()

    def write_to_table(self, success, config_str):
        df = pd.DataFrame({'id': [self.hash], 'success':[success], 'config':[config_str]})
        df.to_csv(self.pathresults + 'run_log.csv', mode='a', header=False)
