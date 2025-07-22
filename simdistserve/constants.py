class ModelTypes:
    opt_13b = 'OPT-13B'
    opt_66b = 'OPT-66B'
    opt_175b = 'OPT-175B'
    llama_3_1_8b = 'Llama-3.1-8B'

    @staticmethod
    def formalize_model_name(x):
        if x == ModelTypes.opt_13b:
            return 'facebook/opt-13b'
        if x == ModelTypes.opt_66b:
            return 'facebook/opt-66b'
        if x == ModelTypes.opt_175b:
            return 'facebook/opt-175b'
        if x == ModelTypes.llama_3_1_8b:
            return 'meta-llama/Llama-3.1-8B'
        raise ValueError(x)

    @staticmethod
    def model_str_to_object(model):
        if model == 'opt_13b' or model == "facebook/opt-13b":
            return ModelTypes.opt_13b
        if model == 'opt_66b' or model == "facebook/opt-66b":
            return ModelTypes.opt_66b
        if model == 'opt_175b' or model == "facebook/opt-175b":
            return ModelTypes.opt_175b
        if model == 'llama_3_1_8b' or model == "meta-llama/Llama-3.1-8B":
            return ModelTypes.llama_3_1_8b
        raise ValueError(model)
