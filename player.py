from abc import abstractmethod


class Player:
    def __init__(self):
        self.player: int = -1

    def set_player_ind(self, p):
        self.player = p

    @abstractmethod
    def get_action(self, board) -> int:
        pass

