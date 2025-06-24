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

        self.max_actions = 200  # Górny limit liczby możliwych akcji

        # Przygotuj stałą listę wszystkich możliwych akcji
        self.all_actions = self._generate_all_actions()

        # Rozmiar obserwacji
        sample_obs = self.reset()["obs"]  # ← gwarantuje spójność

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=13, shape=sample_obs.shape, dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.max_actions,), dtype=np.uint8)
        })

        self.action_space = spaces.Discrete(self.max_actions)

    def _generate_all_actions(self):
        actions = []
        # p2f
        for p in range(1, 9):
            for f in range(1, 5):
                actions.append(f"p2f {p} {f}")
        # p2p
        for p1 in range(1, 9):
            for p2 in range(1, 9):
                if p1 != p2:
                    actions.append(f"p2p {p1} {p2}")
        # p2c
        for p in range(1, 9):
            for c in range(1, 5):
                actions.append(f"p2c {p} {c}")
        # c2p
        for c in range(1, 5):
            for p in range(1, 9):
                actions.append(f"c2p {c} {p}")
        # c2f
        for c in range(1, 5):
            for f in range(1, 5):
                actions.append(f"c2f {c} {f}")

        # Upewnij się, że nie przekraczasz max_actions
        if len(actions) > self.max_actions:
            raise ValueError(f"Generated more actions ({len(actions)}) than max_actions ({self.max_actions})")

        # Jeśli mniej, dopełnij pustymi akcjami, żeby mieć stałą długość
        while len(actions) < self.max_actions:
            actions.append("noop")  # akcja nic nie robiąca

        return actions

    def _get_action_mask(self):
        mask = np.zeros(self.max_actions, dtype=np.uint8)
        legal_actions = self.get_legal_actions()
        legal_set = set(legal_actions)

        for idx, action_str in enumerate(self.all_actions):
            if action_str in legal_set:
                mask[idx] = 1

        return mask

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

        obs = encode_state_to_array(state).astype(np.float32)
        action_mask = self._get_action_mask()

        return {
            "obs": obs,
            "action_mask": action_mask
        }
    
    def step(self, action_idx, return_action_str=False):
        action_mask = self._get_action_mask()

        # ✅ Jeśli brak legalnych akcji — zakończ epizod
        if action_mask.sum() == 0:
            self.done = True
            obs_data = {
                "obs": encode_state_to_array(self.game.get_state()).astype(np.float32),
                "action_mask": action_mask
            }
            if return_action_str:
                return obs_data, -1.0, True, {"reason": "no_legal_actions"}, None
            else:
                return obs_data, -1.0, True, {"reason": "no_legal_actions"}

        # ❌ Jeśli wybrana akcja jest nielegalna — zakończ epizod z karą
        if action_idx >= self.max_actions or action_mask[action_idx] == 0:
            self.done = True
            print(f"[ERROR] Invalid action index or masked out: {action_idx}")
            obs_data = {
                "obs": encode_state_to_array(self.game.get_state()).astype(np.float32),
                "action_mask": action_mask
            }
            if return_action_str:
                return obs_data, -1.0, True, {"invalid_action": True}, None
            else:
                return obs_data, -1.0, True, {"invalid_action": True}

        action_str = self.all_actions[action_idx]

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
            # noop nie robi nic
        except Exception as e:
            print(f"[WARNING] Exception while performing action '{action_str}': {e}")
            # Ignorujemy wyjątki — akcja mogła być legalna, ale nie wykonała się poprawnie

        post_state = self.game.get_state()
        reward, done, breakdown = calculate_reward(prev_state, post_state, action_str, self.uncovered_cards_registry)

        state_hash = hash(str(post_state))
        self.state_history_hashes.append(state_hash)
        if self.state_history_hashes.count(state_hash) >= 3:
            done = True
            breakdown["loop_detected"] = 0.0

        self.done = done

        obs = encode_state_to_array(post_state).astype(np.float32)
        action_mask = self._get_action_mask()
        obs_data = {
            "obs": obs,
            "action_mask": action_mask
        }

        if return_action_str:
            return obs_data, reward, done, breakdown, action_str
        else:
            return obs_data, reward, done, breakdown


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
