from Pytorch.Layer.Hidden import Hidden


class Group_lasso(Hidden):
    # inherritance allows to keep dense method & "activ" activation functions in one place!
    # IDEALLY MAKE IT AUTOGRAD READY IF NECESSARY BY SAMPLER - INHERRIT nn.Module?
    def __init__(self):
        self.bias = False  # to meet the Hidden default standard!
