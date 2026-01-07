import importlib


def create_model(model_path, **kwargs):
    try:
        module_name, class_name = model_path.rsplit('.', 1)
        
        module = importlib.import_module(f'.{module_name}', package='models')
        
        model_class = getattr(module, class_name)
        
        return model_class(**kwargs)
    
    except (ValueError, ModuleNotFoundError, AttributeError) as e:
        print(f"Error creating model: {e}")
        print(f"Make sure model_path is in format 'module.ClassName'")
        print(f"Available models: shallowNetwork.ShallowNetwork, deepNetwork.DeepNetwork")
        raise
