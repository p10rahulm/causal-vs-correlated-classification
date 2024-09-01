from abc import ABC, abstractmethod


class BaseDataModule(ABC):
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass
