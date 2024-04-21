# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
import sys

import pygame

from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
from player import Player
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras
import pygame


class Human(Player):
    """
    human player
    """

    def get_action(self, board: Board):
        margin = board.board_margin
        cell_size = board.cell_size
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # 检查是否是鼠标左键
                    if event.button == 1:  # 1 表示鼠标左键
                        x, y = event.pos
                        row = round((y - margin + cell_size // 2) / cell_size) - 1
                        col = round((x - margin + cell_size // 2) / cell_size) - 1
                        move = board.location_to_move((row, col))
                        if move in board.availables:
                            return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 8, 8
    model_path = 'best_policy_8_8_5.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_path = model_path)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        try:
            policy_param = pickle.load(open(model_path, 'rb'))
        except:
            policy_param = pickle.load(open(model_path, 'rb'), encoding='bytes')  # To support python3
        best_policy = PolicyValueNetNumpy(width, height, policy_param)

        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # set start_player=0 for human first
        game.start_play(Human(), mcts_player, start_player=1, display=1)
    except KeyboardInterrupt:
        pygame.quit()


if __name__ == '__main__':
    run()
