import pandas as pd

# hash
from datetime import datetime
from subprocess import check_output

# storing results / logs / pip
import shelve
import os
import sys
import traceback
from pip._internal.operations import freeze


class Grid:
    """Grid Class is intended to work as a sceduler"""
    def __init__(self, root, model, configs):
        """
        Grid Object, taking care of all the OS ops and tracking the Configs
        respective stdout & Tracebacks, printing them into correct files,
        shelving the instances (which did not raise errors).
        :param model: model object, must have
        :param configs: list of dicts of parameters to run on model
        """

        # file structure (redundant, but easily accessible on server)
        self.pathroot = root
        self.pathresults = self.pathroot + 'results'  # consider only for plots
        self.pathlogs = self.pathroot + 'logs/'
        self.pathtf = self.pathroot + 'tf/'
        os.mkdir(self.pathroot)
        os.mkdir(self.pathlogs)
        os.mkdir(self.pathtf)

        # current configuration
        self._pip_freeze()
        self.git_hash = self._get_git_revision_hash()

        self.model = model
        self.configs = configs
        self.result_table = pd.DataFrame(columns=['hash', 'config', 'success'],
                                         index=range(len(configs)))


        for attr in self.__dict__:
            print(attr, self.__dict__[attr] , '\n')

    def run_model(self):
        # TODO parallel?
        for i, config in enumerate(self.configs):

            hash = self._create_hash(model_name=self.model.__name__)
            print('trying {}\n'.format(hash))

            # cache all stdout print statements of an instance
            temp = sys.stdout
            sys.stdout = open(self.pathlogs + hash + '.log', 'a+')

            try:
                # run object config
                instance = self.model(**config)
                # FIXME instance.sample_chain()
                instance.hash = hash

            except:

                with open(self.pathlogs + hash + '.log', 'a') as logfile:
                    logfile.write('Config ' + str(config) + ' failed\n')

                self.result_table.loc[i] = [hash, str(config), False]

                # FLUSH & restore standard output stream
                sys.stdout.close()
                sys.stdout = temp

                traceback.print_exc(file=open(self.pathlogs + hash + '.log', 'a'))

            else:  # passed try

                # ToDo get static, non-TB metrics to table
                # metric = instance.metric_fn()
                # self.result_table.append(metric)

                # ToDo get & store plots

                # ToDo be more verbose
                # instance.metric = metric
                # instance.stdout = stdout

                self._store_instance(instance)

                # FLUSH & restore standard output stream
                sys.stdout.close()
                sys.stdout = temp

                self.result_table.loc[i] = [hash, str(config), True]

        self.result_table.to_csv(self.pathresults)

    def _get_git_revision_hash(self):
        return check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

    def _create_hash(self, model_name):
        return '{hash}_{model}{timestamp}'.format(
            hash=self.git_hash,
            model=model_name,
            timestamp= # strftime("%Y%m%d_%H%M%S%", gmtime())
            datetime.now().strftime("%H:%M:%S.%f")
        )

    def _pip_freeze(self):
        # write pip to file
        self.pipfreeze = freeze.freeze()
        with open(self.pathroot + 'pip_freeze.txt', 'w') as file:
            file.write('Tensorflow ' + sys.version + '\n')
            for line in self.pipfreeze:
                file.write(line + '\n')
            file.close()

    def _store_instance(self, model_object):
        """
        shelving the instance, such that all objects are available
        :param model_object: an executed model instance
        :param
        """
        hash = model_object.hash  # model_object.__class__.__name__

        # HMC does not want to get stored:
        # (1) FIXME: HMC.model_save()
        # ToDo store tensorflow models also elsewhere -
        # so TB can access all instances from one dir
        self.pathtf

        # (2) remove hmc & shelve
        # FIXME FIXME! does this prevent model to be recoverable? - class def init requires adaptiveHMC
        model_object.__delattr__('adaptive_hmc')
        print(self.pathroot)
        with shelve.open(self.pathroot + 'shelve', flag='c') as db:
            db[hash] = model_object

    def recover_instance(self, hash, filepath):
        """Relevant ONLY, if Plots cannot be executed during runtime"""

        # (1) recover shelved without TF
        with shelve.open(filepath + 'shelve', flag='c') as db:
            model_object = db[hash]

        # (2) ToDo restore TF model (saved adaptive HMC)
        model_object.adaptive_HMC = None

        return model_object


# if __name__ == '__main__':
#     from Tensorflow import Xbeta
#     pathroot = os.getcwd()
#
#     # ToDo write actual execution directives for Remote server here, commit, push
#     # pull & execute .sh script
#
#     # (MINIMAL SHELVE TF EXAMPLE) ----------------------------------------------
#     xgrid = (0, 10, 0.5)
#     G = Grid(root='{root}/Grid/'.format(root=os.getcwd()),
#              model=Xbeta, configs=[{'xgrid': xgrid}, {'xgrid': xgrid, 'Fail': True}])
#
#     G.run_model()
#
#     xbeta = G.recover_instance(hash='d2e6849_Xbeta13:28:26.883481', filepath=G.pathroot)
# print('')
