import numpy as np
import sys
import os
import gym
from gym import spaces

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from freecell import Freecell
from environment.reward import calculate_reward

face_map = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13}
suit_map = {'c': 0, 'h': 1, 's': 2, 'd': 3}
color_map = {'B': 0, 'R': 1}

def encode_card(card_str):
    if card_str == "None":
        return [0, 0, 0]
    face = face_map[card_str[0]]
    suit = suit_map[card_str[1]]
    color = color_map[card_str[-1]]
    return [face, suit, color]

def encode_state_to_array(state):
    arr = []
    for zone in ['pile', 'foundation', 'cell']:
        for stack in state[zone]:
            for card in stack:
                arr.append(encode_card(card))
    return np.array(arr, dtype=np.int32).flatten()

class FreecellEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super().__init__()
        self.game = Freecell()
        self.done = False
        self.steps = 0
        self.uncovered_cards_registry = set()
        self.state_history_hashes = []

        # Ustal rozmiar obserwacji (zakładam stałą)
        sample_state = self.game.get_state()
        sample_obs = encode_state_to_array(sample_state)
        self.observation_space = spaces.Box(low=0, high=13, shape=sample_obs.shape, dtype=np.int32)

        # Maksymalna liczba możliwych ruchów naraz jest trudna do ustalenia
        # więc na początek załóżmy arbitralną liczbę, np 200.  
        # Możesz potem dopracować mapowanie akcji.
        self.action_space = spaces.Discrete(200)

    def reset(self):
        self.game.reset()
        self.done = False
        self.steps = 0
        self.uncovered_cards_registry.clear()
        self.state_history_hashes.clear()

        state = self.game.get_state()
        for pile in state["pile"]:
            if pile:
                self.uncovered_cards_registry.add(pile[-1])

        return encode_state_to_array(state)

    def step(self, action_idx):
        if self.done:
            return self.reset(), 0.0, True, {}

        legal_actions = self.get_legal_actions()

        if action_idx >= len(legal_actions):
            # Nieprawidłowa akcja
            self.done = True
            return encode_state_to_array(self.game.get_state()), -1.0, True, {"invalid_action": True}

        action_str = legal_actions[action_idx]

        prev_state = self.game.get_state()
        self.steps += 1

        try:
            parts = action_str.split()
            if parts[0] == 'p2f':
                self.game.p2f(int(parts[1]), int(parts[2]))
            elif parts[0] == 'p2p':
                self.game.p2p(int(parts[1]), int(parts[2]))
            elif parts[0] == 'p2c':
                self.game.p2c(int(parts[1]), int(parts[2]))
            elif parts[0] == 'c2p':
                self.game.c2p(int(parts[1]), int(parts[2]))
            elif parts[0] == 'c2f':
                self.game.c2f(int(parts[1]), int(parts[2]))
        except Exception:
            pass

        post_state = self.game.get_state()
        reward, done, breakdown = calculate_reward(prev_state, post_state, action_str, self.uncovered_cards_registry)

        state_hash = hash(str(post_state))
        self.state_history_hashes.append(state_hash)

        if self.state_history_hashes.count(state_hash) >= 3:
            done = True
            breakdown["loop_detected"] = 0.0

        self.done = done
        return encode_state_to_array(post_state), reward, done, breakdown

    def render(self, mode='human'):
        self.game.print_game()

    def get_legal_actions(self):
        actions = []
        for p in range(1, 9):
            for f in range(1, 5):
                if self.game.move_to_foundation(p, f):
                    actions.append(f"p2f {p} {f}")
        for p1 in range(1, 9):
            for p2 in range(1, 9):
                if p1 != p2 and self.game.move_in_pile(p1, p2):
                    actions.append(f"p2p {p1} {p2}")
        for p in range(1, 9):
            for c in range(1, 5):
                if self.game.move_to_cell(p, c):
                    actions.append(f"p2c {p} {c}")
        for c in range(1, 5):
            for p in range(1, 9):
                if self.game.move_to_pile(c, p):
                    actions.append(f"c2p {c} {p}")
        for c in range(1, 5):
            for f in range(1, 5):
                if self.game.cell_to_foundation(c, f):
                    actions.append(f"c2f {c} {f}")
        return actions
    
    def get_state(self):
        return self.game.get_state()
