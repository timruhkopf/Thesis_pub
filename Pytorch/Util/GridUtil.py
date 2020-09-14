import os
import sys
from pip._internal.operations import freeze
from datetime import datetime
from subprocess import check_output

import matplotlib

matplotlib.use('Agg')

import pandas as pd


class Grid:
    def __init__(self, root):
        """
        Grid is a delegation class, that ensures that multiple calls (of potentially
        different configs) to a main function are being tracked and not interrupted
        by a single config messing up.
        unless root is an existing directory, it instantiates this directory with
        a single result_table, that is added information on the "main" run,
        tracking which config is run & whether or not it executed without error.
        Each run is attributed a unique hash for identifiability.
        In addition, it provides a pip_freeze.

        When subclassing, to define a specific execution script please define main function

        class Execution(Grid):
            def main(param):
                # ... do something

        :param root: string: root directory path

        :example:
        gam_unittest = GAM_Grid(root=os.getcwd() if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Pytorch/Experiments')

        # presuming, that the grid_exec_SGNHT returns a generator, which yields two
        # dicts; model_param, sampler_param. each such yielded config are part of a grid
        prelim_configs = gam_unittest.grid_exec_SGNHT(steps=1000,
                        epsilons=np.arange(0.001, 0.05, 0.003),
                        hmc_traj_lengths=[1, 2, 3, 5, 10, 15, 20, 25])
        for prelim_config in prelim_configs:
            gam_unittest.main(n=1000, n_val=100, sampler_name='SGNHT', **prelim_config)
        """
        print('--------------------- progressing with: {name} --------------------- '.format(
            name=self.__class__.__name__))
        self.root = root
        self.pathresults = root
        # Result folder
        result = '/'.join(self.pathresults.split('/')[:-2])
        self.runfolder = self.pathresults.split('/')[-2]
        if not os.path.isdir(result):
            try:
                os.mkdir(result)
            except FileExistsError:
                print('file result existed, still continuing')
        if not os.path.isdir(self.pathresults):
            os.mkdir(self.pathresults)

        self.hash = None
        self._pip_freeze()
        self.main = self.try_main(self.main)

    def try_main(self, func):
        """decorate main to ensure the main function runs smoothly"""

        def wrapper(*args, **kwargs):
            # Ensure, the execution is propperly logged & individual calls are hashed
            self.hash = self._create_hash()
            print('running: {hash}'.format(hash=self.hash))
            # self._log_main_function(func)

            try:
                metrics = func(*args, **kwargs)
                success = True
                self.write_to_table(success=success, config=kwargs, metrics=metrics)

            except Exception as error:
                success = False
                self.write_to_table(success=success, config=kwargs)
                import sys
                import traceback

                with open(self.pathresults + '{}_{}.log'.format(self.runfolder, self.hash), 'a') as file:
                    # catch the entire error message:
                    # exc_type, exc_value, exc_traceback = sys.exc_info()
                    file.write('\n' + traceback.format_exc() + '\n')
                    file.write('ERROR triggered by main(args:{}, kwargs:{}) \n'.format(
                        str(args), str(kwargs)))

            matplotlib.pyplot.close('all')
            # print to console if run was a success
            print({'id': [self.hash], 'success': [success], 'config': [str(args) + str(kwargs)]})

        return wrapper

    def main(self, model_config, sampler_config):
        raise NotImplementedError('Create a main function for your grid class')

    def grid_exec(self):
        """build up the parameter configs, that are to be executed on main function"""
        raise NotImplementedError('')
        # yield config

    def _create_hash(self):
        return '{commit}_{timestamp}'.format(
            commit=check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip(),
            timestamp=  # strftime("%Y%m%d_%H%M%S%", gmtime())
            datetime.now().strftime("%H%M%S%f")
        )

    def _pip_freeze(self):
        pipfreeze = freeze.freeze()
        with open(self.pathresults + '/pip_freeze.txt', 'w') as file:
            file.write('Torch ' + sys.version + '\n')
            for line in pipfreeze:
                file.write(line + '\n')
            file.close()

    def write_to_table(self, success, config, metrics={}, order=None):
        config['model_class'] = config['model_class'].__name__
        config1 = pd.io.json.json_normalize(config, sep='_')  # pd.json_normalize()

        df = pd.DataFrame({**{'id': self.hash, 'success': success}, **config1, **metrics}, index=None)

        # if order is not None:
        #     df = df[order + [col for col in df.columns if col not in order]]

        # critical section: reference in bash different compared to
        # debug call from subclass' module
        if not os.path.isfile(self.pathresults + '{}_run_log.csv'.format(self.runfolder)):
            df.to_csv(self.pathresults + '{}_run_log.csv'.format(self.runfolder), header=True)
        else:
            df.to_csv(self.pathresults + '{}_run_log.csv'.format(self.runfolder), mode='a', header=False)

    # def _log_main_function(self, func):
    #     """
    #     write out the source code of the executed function
    #     :param func:
    #     :return:
    #     """
    #     import inspect
    #     source = inspect.getsource(func)
    #     with open(self.pathresults + '{}.log'.format(self.hash), 'a') as file:
    #         file.write('\n' + source)


if __name__ == '__main__':
    path = '/home/tim/PycharmProjects/Thesis/Pytorch/Experiments/Results/BNN_RHMC/BNN_RHMC_run_log.csv'
    df = pd.read_csv(path)
