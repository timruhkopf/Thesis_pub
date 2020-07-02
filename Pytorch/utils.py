from torch.nn.utils import parameters_to_vector, vector_to_parameters

def flatten(model):
    """return a single vector with all stacked weights"""
    return parameters_to_vector(model.parameters())

def unflatten(vec, model):
    """unpacking vec in the model's parameters -- INPLACE!"""
    return vector_to_parameters(vec, model.parameters())

