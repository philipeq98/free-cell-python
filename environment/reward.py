import math

def calculate_reward(state_before, state_after, action, uncovered_cards_registry, total_freecells=4):
    reward = 0.0
    done = False
    breakdown = {}

    # 1. Przeniesienie asa do foundation
    if moved_ace_to_foundation(state_before, state_after):
        reward += 2.0
        breakdown['moved_ace_to_foundation'] = 2.0

    # 2. Przeniesienie innej karty do foundation
    elif moved_card_to_foundation(state_before, state_after):
        reward += 0.5
        breakdown['moved_card_to_foundation'] = 0.5

    # 3. Odkrycie nowej karty (po raz pierwszy)
    newly_uncovered = count_newly_uncovered_cards(state_before, state_after, uncovered_cards_registry)
    if newly_uncovered > 0:
        bonus = 0.5 * newly_uncovered
        reward += bonus
        breakdown['newly_uncovered_cards'] = bonus

    # 4. Zwolnienie freecell - nagroda ważona (spójna z karą)
    # Wspólna wartość
    occupied_before = sum(1 for c in state_before["cell"] if c)
    occupied_after = sum(1 for c in state_after["cell"] if c)
    delta = occupied_after - occupied_before
    if delta < 0:
        freed = -delta
        bonus = sum(0.05 * math.exp((occupied_before - i) / total_freecells * 3)
                    for i in range(freed))
        reward += bonus
        breakdown['freed_cells'] = bonus

    # 5. Kara za zajęcie nowego freecell - ważona, rosnąca
    elif delta > 0:
        penalty = sum(0.05 * math.exp((occupied_before + i + 1) / total_freecells * 3)
                    for i in range(delta))
        reward -= penalty
        breakdown['penalty_cell_occupancy'] = -penalty

    # 6. Kara za ruch bez zmiany stanu
    if state_before == state_after:
        reward -= 0.3
        breakdown['no_effect_move'] = -0.3

    # 7. Kara za cofnięcie ruchu foundation do tableau/cell
    if moved_card_from_foundation(state_before, state_after):
        reward -= 1.0
        breakdown['moved_back_from_foundation'] = -1.0

    # --- Zakończenie epizodu ---
    if all(len(stack) == 13 for stack in state_after["foundation"]):
        reward += 10.0
        breakdown['game_won'] = 10.0
        done = True
    elif no_moves_possible(state_after):
        done = True

    return reward, done, breakdown


# --- Funkcje pomocnicze ---

def moved_ace_to_foundation(before, after):
    for b, a in zip(before["foundation"], after["foundation"]):
        if len(a) > len(b):
            return a[-1].startswith("A")
    return False

def moved_card_to_foundation(before, after):
    for b, a in zip(before["foundation"], after["foundation"]):
        if len(a) > len(b):
            return True
    return False

def count_newly_uncovered_cards(before, after, uncovered_cards_registry: set):
    newly_uncovered = 0
    for b_stack, a_stack in zip(before["pile"], after["pile"]):
        if len(a_stack) > 0:
            top_card = a_stack[-1]
            if top_card not in uncovered_cards_registry:
                newly_uncovered += 1
                uncovered_cards_registry.add(top_card)
    return newly_uncovered

def moved_card_from_cell_to_pile(before, after):
    for i in range(len(before["cell"])):
        if len(before["cell"][i]) == 1 and len(after["cell"][i]) == 0:
            return True
    return False

def moved_card_from_foundation(before, after):
    # Sprawdza czy jakaś karta została wyciągnięta z foundation (cofnięcie ruchu)
    for b, a in zip(before["foundation"], after["foundation"]):
        if len(a) < len(b):
            return True
    return False

def no_moves_possible(state):
    # Tu w praktyce najlepiej skorzystać z env.get_legal_actions()
    return False
