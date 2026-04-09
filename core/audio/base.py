from abc import ABC , abstractmethod 
# we are importing the above abc as we need a contract
import torch 

class BaseAudioModel(ABC):
    """So the purpose of this class is to make sure that no matter 
    what model is used it could be easily integrated in the pipeline."""
    def __init__(self,model_id:str, device:str = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None 
        self.processor = None 

    @abstractmethod
    def load_model(self):
        """The purpose of this method is to load the model"""
        pass 

    @abstractmethod
    def unload_model(self):
        """The purpose of this method is to remove the model
        from the memory after its work has been done so that it 
        does not consume unnecessary resources """
        pass

    #The below method is being provided with the assumption that the model is going to have the feature of taking the prompt as the input and is going to generate the output music.


    @abstractmethod
    def generate(self,prompt:str , duration:int = 10):
        """
        The purpose of this method is to generate the music"""

        pass
    


    
