import numpy as np
from freecell import Freecell
from environment.reward import calculate_reward  # dodaj to u góry

# Pomocnicza mapa: z tekstu karty do liczby (np. "Qd R" -> 12,3,1)
face_map = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13}
suit_map = {'c': 0, 'h': 1, 's': 2, 'd': 3}
color_map = {'B': 0, 'R': 1}

def encode_card(card_str):
    if card_str == "None":
        return [0, 0, 0]  # brak karty
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

class FreecellEnv:
    def __init__(self):
        self.game = Freecell()
        self.done = False
        self.steps = 0
        self.uncovered_cards_registry = set()
        self.state_history_hashes = []

    def reset(self):
        self.game.reset()
        self.done = False
        self.steps = 0
        self.uncovered_cards_registry = set()
        self.state_history_hashes = []

        # ZAREJESTRUJ POCZĄTKOWO ODKRYTE KARTY – ostatnie w kolumnach (czyli widoczne)
        state = self.game.get_state()
        for pile in state["pile"]:
            if pile:
                self.uncovered_cards_registry.add(pile[-1])  # top karta
        return self.get_observation()
    
    def get_state(self):
        """Zwróć surowy stan gry jako słownik (do porównań, nie dla modelu ML)."""
        return self.game.get_state()

    def get_observation(self):
        state = self.game.get_state()
        return encode_state_to_array(state)

    def get_legal_actions(self):
        # Na poczatek: lista stringow jak 'p2f 1 1', 'p2p 3 6' itd.
        # W przyszlosci mozesz zakodowac je jako liczby lub wektory one-hot
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

    def step(self, action_str):
        if self.done:
            return self.get_observation(), 0.0, True, {}

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
        except:
            pass

        post_state = self.game.get_state()
        reward, done, breakdown = calculate_reward(prev_state, post_state, action_str, self.uncovered_cards_registry)

        # 🔁 DETEKTOR PĘTLI STANÓW (3x ten sam hash = pętla)
        state_hash = hash(str(post_state))
        self.state_history_hashes.append(state_hash)

        if self.state_history_hashes.count(state_hash) >= 3:
            done = True
            breakdown["loop_detected"] = 0.0  # nie dodawaj nagrody, tylko sygnalizuj

        self.done = done
        return self.get_observation(), reward, done, breakdown

    def render(self):
        self.game.print_game()
