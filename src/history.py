class History:
    def __init__(self) -> None:
        self.history = {}
        
    def store(self, key: str, value: object) -> dict:
        self.history[key] = value
        
    def store_multiple(self, keys: list[str], values: list[object]) -> dict:
        for key, value in zip(keys, values):
            self.history[key] = value
