import os
import torch
import pickle
import pandas as pd
from tqdm import tqdm

from Pytorch.Layer import GAM, Hidden, Group_HorseShoe
from Pytorch.Models import BNN, ShrinkageBNN, StructuredBNN


class Continuation:
    # FIXME THIS CLASS IS SHITCODE: rewrite it
    # FIXME: SPLIT ROOTING AND CONTINUE path, to make repetitions explicit!
    # FIXME: ensure, that each continuation run has a new git hash! (but same time identifyer! to the old model)
    def find_successfull(self, path, model, n=2):
        """
        stratified selection of successfull models (based on the results.csv) and on MSE
        :param path:
        :param model: str, specifies the model (e.g. 'Hidden'), of which
        you want to continue all 'successfull' samplers
        :param n: select the n model-sampler inits with the n smallest avg_MSE_diff per unique
        model-sampler config
        :return: generator of models
        """

        L = {(model + '_' + k): None for k in ['RHMC', 'SGLD', 'SGNHT', 'SGRHMC', 'SGRLD', 'MALA']}

        if model == 'Group_HorseShoe':
            # FIXME: WHY is this a special case? should not be like this
            model = 'Grouped_HorseShoe'

        for model_sampler in [name for name in os.listdir(path) if name.startswith(model)]:
            print(model_sampler)

            try:
                # locally, i dont have all folders!

                df = pd.read_csv(path + '/' + model_sampler + '/{}'.format(
                    *[name for name in os.listdir(path + '/' + model_sampler) if name.endswith('.csv')]))
            except:
                # for manually added succeeded models (those that i had on my local machine during testing)
                manual = [m.split('.')[0] for m in os.listdir(path + '/' + model_sampler) if m.endswith('.model')]

                if model_sampler.endswith('_'):
                    model_sampler = model_sampler[:-1]

                L[model_sampler] = manual if len(manual) >= 1 else None
                continue

            df.drop(columns='Unnamed: 0', inplace=True)

            # MEASURE SUCCESS ACCROSS REPETITIONS FOR CONFIG
            success_ratio = df.success.sum() / len(df)

            uniques = ['id', 'ess_min', 'avg_MSE_diff', 'true_MSE', 'avg_log_prob_diff', 'true_log_prob']
            config_id = [name for name in df.columns if name not in uniques]
            successes = pd.concat([group[config_id].iloc[0] for name, group in df.groupby(config_id)],
                                  axis=1).transpose()
            successes.reset_index(drop=True, inplace=True)
            successes['success_ratio'] = pd.Series(
                [group.success.sum() / len(group) for name, group in df.groupby(config_id)])
            # TODO: write out this success story to the original folder???

            # FIND THOSE INITS that are to be conitinued:
            df = df[df['success'] == True]
            if model_sampler.endswith('_'):
                model_sampler = model_sampler[:-1]
                # find the n most successfull initialisations per configurations
            try:
                new_runs = pd.concat([group.nsmallest(columns='avg_MSE_diff', n=n) for name, group in
                                      df.groupby([name for name in df.columns if name not in uniques])])

                L[model_sampler] = [model_sampler + '_' + ids for ids in new_runs['id']]  # selected_models

            except ValueError as error:
                print(model_sampler, 'raised error: ', error)

                L[model_sampler] = None

        return L

    def find_semi_successful(self, path, model):
        """
        successfull models that exhibit config.pkl, .model, .sampler_pkl chain.pkl files,
        but are not necessarily listed in .csv as successfull, since
        :param path:
        :param model: model name e.g. 'Hidden', that are surched for in a folder
        :return:
        """
        sampler = ['RHMC', 'SGLD', 'SGNHT', 'SGRHMC', 'SGRLD', 'MALA']
        L = {(model + '_' + k): None for k in ['RHMC', 'SGLD', 'SGNHT', 'SGRHMC', 'SGRLD', 'MALA']}

        for s in sampler:
            subpath = path + '_' + s
            # has config.pkl, .model, .sampler_pkl chain.pkl
            dir = os.listdir(subpath)  # assumes path to be a single folder 'StructuredBNN_SGRHMC'
            [file for file in dir if file.endswith('.model')]
            [file for file in dir if file.endswith('.config')]
            [file for file in dir if file.endswith('chain.pkl')]
            [file for file in dir if file.endswith('.model')]

    def continue_sampling_successfull(self, n, n_val, n_samples, burn_in, models=None, path=None):
        """

        :param n:
        :param n_val:
        :param n_samples:
        :param burn_in:
        :param models: list of str. if None, defaults to all the available ".models" in path folder
        :param path: either dir containing '.model' files or an entire path to a single .model file
        :return:
        """
        self.n = n
        self.n_val = n_val

        # set up the model and sample some new data
        # self.basename = self.pathresults + '{}_{}_'.format(model_class.__name__, sampler_name) + self.hash
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)
        # self.set_up_model(model_class, model_param, seperated)

        # base = path.split('/')[-2]
        # import pandas as pd
        # df = pd.read_csv(path + '{}_run_log.csv'.format(base))
        # df = df[df['success'] == True]
        # df.drop(['Unnamed: 0', 'success'], axis=1, inplace =True)
        # # df.set_index('id', inplace=True)
        # df.to_dict(orient='records')

        # '/HIDDEN_RHMC1/' + path.split('/')[-2] + '_run_log.csv')

        if path is None:
            path = self.pathresults

        # wrap again every execution with try except & tracking
        # FIXME: make a new basepath for continued
        self._continue = self.try_main(self._continue)
        self.oldpathresults = self.pathresults
        self.newpathresults = path + '_continued/'
        self.result = '/'.join(self.newpathresults.split('/')[:-2])
        self.runfolder = self.newpathresults.split('/')[-2]
        if not os.path.isdir(self.result):
            try:
                os.mkdir(self.result)
            except FileExistsError:
                print('file result existed, still continuing')

        if not os.path.isdir(self.newpathresults):
            os.mkdir(self.newpathresults)

        # CONTINUE ALL .MOdel files in path (i.e. all models that did not fail
        # in the sense of self.model.check_chain

        if isinstance(models, dict):
            for sampler_models, modellist in models.items():
                if modellist is not None:
                    print('\nstarting modellist: {}'.format(modellist))
                    for m in tqdm(modellist, ):
                        print('\ncontinuing with: {}'.format(m))
                        if not os.path.isdir(self.newpathresults + sampler_models + '/'):
                            os.mkdir(self.newpathresults + sampler_models + '/')

                        self.pathresults = self.newpathresults + sampler_models + '/'
                        # FIXME: model_class is here just to prevent crash in wrapper

                        sampler_models1 = sampler_models.split('_')[0]
                        if sampler_models1 == 'Grouped':
                            # FIXME: this will fail in Grouped_lasso case
                            sampler_models1 += '_HorseShoe'

                        model_class = dict(Hidden=Hidden, Group_HorseShoe=Group_HorseShoe,
                                           GAM=GAM, BNN=BNN, ShrinkageBNN=ShrinkageBNN,
                                           StructuredBNN=StructuredBNN)[sampler_models1]

                        self._continue(m=m, model_class=model_class, path=self.pathresults, n_samples=n_samples,
                                       burn_in=burn_in)

            return None

        elif models is None and os.path.isdir(path):
            models = [file for file in os.listdir(path) if file.endswith('.model')]
        elif os.path.isfile(path) and path.endswith('.model'):  # user specified a specific model
            models = [path.split('/')[-2]]

        for m in models:
            # FIXME: MODEL_CLASS ARGUMENT IS here to temporary fix the connection
            #  to Grid_Tracker.write_to_table, that captures and expects a model class argument
            self._continue(m=m, model_class=model_class, path=path, n_samples=n_samples, burn_in=burn_in)

    def _continue(self, m, model_class, path, n_samples, burn_in):

        # naming conventions & parsing
        model_name_base = m
        hashi = m.split('_')[-2:]
        self.hash = '_'.join([hashi[0], hashi[-1].split('.')[0]])
        model_name = '_'.join(m.split('_')[:-2])
        if self.oldpathresults + '/' + model_name + '/' + model_name_base + '.model' == \
                '/home/tim/PycharmProjects/Thesis/Pytorch/Experiment/Result_a83b999/ShrinkageBNN_SGRHMC' \
                '/ShrinkageBNN_SGRHMC_a83b999_150523488677.model':
            print()
        try:
            with open(self.oldpathresults + '/' + model_name + '/' + model_name_base + '_config.pkl', 'rb') as handle:
                config = pickle.load(handle)
        except:
            #  DEPREC: REMOVE EXCEPTION (ONLY TROUBLE SHOOTING FOR PREVIOUS ERROR)
            # ony error comes from renaming with '_'
            with open(self.oldpathresults + '/' + model_name + '_/' + model_name_base + '_config.pkl', 'rb') as handle:
                config = pickle.load(handle)
        model_class, model_param, seperated = config['model_class'], \
                                              config['model_param'], \
                                              config['seperated']
        true_model, sampler_param, sampler_name = config['true_model'], \
                                                  config['sampler_param'], \
                                                  config['sampler_name']

        self.basename = self.pathresults + '{}_{}_'.format(model_class.__name__, sampler_name) + self.hash

        if 'batch_size' in sampler_param.keys():
            batch_size = sampler_param.pop('batch_size')
        else:
            batch_size = self.n

        sampler_param['n_samples'] = n_samples
        sampler_param['burn_in'] = burn_in
        self.model_param = model_param

        self.set_up_model(model_class, model_param, seperated)

        # generate some new data with the true model
        self.model.init_model = config['init_model']
        self.model.true_model = config['true_model']
        self.model.load_state_dict(true_model)
        if hasattr(self.model, 'vec'):
            self.model.true_vec = self.model.vec
        self.set_up_data(self.n, self.n_val, model_param, batch_size)
        try:
            self.model.load_state_dict(
                torch.load(self.oldpathresults + '/' + model_name + '/' + model_name_base + '.model'))
        except:
            # DEPREC: REMOVE THE TRY EXCEPT
            self.model.load_state_dict(
                torch.load(self.oldpathresults + '/' + model_name + '_/' + model_name_base + '.model'))
        self.model.plot(*self.data_plot, path=self.basename + '_initmodel', title='')

        self.set_up_sampler(sampler_name, sampler_param)
        # import random
        # self.sampler.model.plot(*self.data_plot, random.sample(self.sampler.chain, plot_subsample),
        #                         path=self.basename + '_datamodel_random', title='')
        # self.sampler.model.plot(*self.data_plot, self.sampler.chain[-plot_subsample:],
        #                         path=self.basename + '_datamodel_last', title='')

        metrics = self.evaluate_model()  # *self.data_val FIXME

        # for restoration of the model, it needs to be reinstantiated
        # and loaded the state dict upon

        with open(self.basename + '_config.pkl', 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.basename + '_chain.pkl', 'wb') as handle:
            pickle.dump(self.sampler.chain, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return metrics


if __name__ == '__main__':
    from Pytorch.Grid.Grid_Layout import GRID_Layout
    from Pytorch.Layer import *
    from Pytorch.Models import *

    Grid = GRID_Layout(root='/home/tim/PycharmProjects/Thesis/Pytorch/Experiment/Result_0365244')
    a = Grid.find_successfull(path='/home/tim/PycharmProjects/Thesis/Pytorch/Experiment/Result_0365244',
                              model='BNN')

    b = Grid.find_successfull(path='/home/tim/PycharmProjects/Thesis/Pytorch/Experiment/Result_0365244',
                              model='Shrinkage')

    c = Grid.find_successfull(path='/home/tim/PycharmProjects/Thesis/Pytorch/Experiment/Result_0365244',
                              model='Hidden')

    d = Grid.find_successfull(path='/home/tim/PycharmProjects/Thesis/Pytorch/Experiment/Result_0365244',
                              model='Group_HorseShoe')
    Grid.continue_sampling_successfull(n=100, n_val=100, n_samples=100, burn_in=100, models=d)

    print()
