import math

def calculate_reward(state_before, state_after, action, total_freecells=4):
    reward = 0.0
    done = False
    breakdown = {}

    # --- Nagrody pozytywne ---

    if moved_ace_to_foundation(state_before, state_after, action):
        reward += 2.0
        breakdown['moved_ace_to_foundation'] = 2.0

    elif moved_card_to_foundation(state_before, state_after, action):
        reward += 0.5
        breakdown['moved_card_to_foundation'] = 0.5

    uncovered = count_uncovered_cards(state_before, state_after)
    if uncovered > 0:
        bonus = 0.3 * uncovered
        reward += bonus
        breakdown['uncovered_cards'] = bonus

    if moved_card_from_cell_to_pile(state_before, state_after, action):
        reward += 0.2
        breakdown['moved_cell_to_pile'] = 0.2

    for i, (before, after) in enumerate(zip(state_before["pile"], state_after["pile"])):
        if len(after) == 0 and len(before) > 0:
            reward += 1.0
            breakdown[f'emptied_column_{i+1}'] = 1.0
        elif len(after) < len(before):
            reward += 0.2
            breakdown[f'shortened_column_{i+1}'] = 0.2

    # --- Kary ---

    cells_before = sum(1 for c in state_before["cell"] if c)
    cells_after = sum(1 for c in state_after["cell"] if c)
    newly_occupied = cells_after - cells_before
    if newly_occupied > 0:
        penalty = sum(0.05 * math.exp((cells_before + i + 1) / total_freecells * 3)
                      for i in range(newly_occupied))
        reward -= penalty
        breakdown['penalty_cell_occupancy'] = -penalty

    if state_before == state_after:
        reward -= 0.5
        breakdown['no_effect_move'] = -0.5

    # --- Zakończenie epizodu ---

    if all(len(stack) == 13 for stack in state_after["foundation"]):
        reward += 10.0
        breakdown['game_won'] = 10.0
        done = True
    elif no_moves_possible(state_after):
        done = True

    return reward, done, breakdown


# --- Funkcje pomocnicze ---

def moved_ace_to_foundation(before, after, action):
    for b, a in zip(before["foundation"], after["foundation"]):
        if len(a) > len(b):
            return a[-1].startswith("A")
    return False

def moved_card_to_foundation(before, after, action):
    for b, a in zip(before["foundation"], after["foundation"]):
        if len(a) > len(b):
            return True
    return False

def count_uncovered_cards(before, after):
    total = 0
    for b, a in zip(before["pile"], after["pile"]):
        if len(a) > 0 and len(a) < len(b):
            total += 1
    return total

def moved_card_from_cell_to_pile(before, after, action):
    for i in range(len(before["cell"])):
        if len(before["cell"][i]) == 1 and len(after["cell"][i]) == 0:
            return True
    return False

def no_moves_possible(state):
    # Placeholder — tu można użyć logiki env.get_legal_actions()
    return False
