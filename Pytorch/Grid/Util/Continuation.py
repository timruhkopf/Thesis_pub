import os
import torch
import pickle


class Continuation:
    def continue_sampling_successfull(self, n, n_val, n_samples, burn_in, path=None):
        self.n = n
        self.n_val = n_val

        # set up the model and sample some new data
        # self.basename = self.pathresults + '{}_{}_'.format(model_class.__name__, sampler_name) + self.hash
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)
        # self.set_up_model(model_class, model_param, seperated)

        if os.path.isdir(path):
            models = [file for file in os.listdir(path) if file.endswith('.model')]
        elif os.path.isfile(path):  # user specified a specific model
            models = [path.split('/')[-2]]

        base = path.split('/')[-2]
        # import pandas as pd
        # df = pd.read_csv(path + '{}_run_log.csv'.format(base))
        # df = df[df['success'] == True]
        # df.drop(['Unnamed: 0', 'success'], axis=1, inplace =True)
        # # df.set_index('id', inplace=True)
        # df.to_dict(orient='records')

        # '/HIDDEN_RHMC1/' + path.split('/')[-2] + '_run_log.csv')

        # wrap again every execution with try except & tracking
        # FIXME: make a new basepath for continued
        self.pathresults = path[:-1] + '_continued/'
        result = '/'.join(self.pathresults.split('/')[:-2])
        self.runfolder = self.pathresults.split('/')[-2]
        if not os.path.isdir(result):
            try:
                os.mkdir(result)
            except FileExistsError:
                print('file result existed, still continuing')
        if not os.path.isdir(self.pathresults):
            os.mkdir(self.pathresults)

        self.try_main(self._continue)

        for m in models:
            self._continue(m, path, n_samples, burn_in)

    def _continue(self, m, path, n_samples, burn_in):

        model_name_base, _ = m.split('.')
        model_name = model_name_base + '_continued'
        self.basename = path + model_name_base
        with open(path + model_name_base + '.pkl', 'rb') as handle:
            config = pickle.load(handle)
        model_class, model_param, seperated = config['model_class'], \
                                              config['model_param'], \
                                              config['seperated']
        true_model, sampler_param, sampler_name = config['true_model'], \
                                                  config['sampler_param'], \
                                                  config['sampler_name']

        if 'batch_size' in sampler_param.keys():
            batch_size = sampler_param.pop('batch_size')
        else:
            batch_size = n

        sampler_param['n_samples'] = n_samples
        sampler_param['burn_in'] = burn_in
        self.model_param = model_param
        self.set_up_model(model_class, model_param, seperated)

        # generate some new data with the true model
        self.model.load_state_dict(true_model)
        self.set_up_data(n, n_val, model_param, batch_size)
        self.model.load_state_dict(torch.load(path + m))

        self.set_up_sampler(sampler_name, sampler_param)
