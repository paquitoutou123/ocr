import torch
import torch.nn



class OCR():
    """
    it is the architecture of our ocr
    """


    def __init__(self, layout_analysis, ViT, MLPconnector, LLM):
        """
        
        """
        self.layout_analysis = layout_analysis
        self.ViT = ViT
        self.MLPconnector = MLPconnector
        self.LLM = LLM



