# DATA MODELS

# Business
class Business:
    def __init__(self, json):
        self.__dict__ = json

# Review
class Review:
    def __init__(self, json):
        self.__dict__ = json