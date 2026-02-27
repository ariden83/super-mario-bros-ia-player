#!/usr/bin/env python3
"""
Test des distances de saut corrigées
"""

# Vitesses corrigées
movement_speeds = {
    0: 0,     # NOOP - pas de mouvement
    1: 1.0,   # RIGHT - marche normale
    2: 1.5,   # JUMP - saut avec momentum horizontal
    3: 2.5,   # RUN - course rapide
    4: 3.0,   # RUN_JUMP - course + saut (vitesse maximale)
    5: 0,     # JUMP_ONLY - saut purement vertical
    6: -1.0   # LEFT - reculer
}

# Actions de saut avec nouvelles descriptions
jump_actions = {
    'short_jump': {'base_action': 2, 'duration': 10, 'expected': 15},
    'short_jump_precise': {'base_action': 2, 'duration': 6, 'expected': 9},
    'short_jump_wide': {'base_action': 2, 'duration': 14, 'expected': 21},
    'long_jump': {'base_action': 4, 'duration': 12, 'expected': 36},
    'long_jump_short': {'base_action': 4, 'duration': 8, 'expected': 24},
    'long_jump_far': {'base_action': 4, 'duration': 18, 'expected': 54},
    'pixel_perfect_jump': {'base_action': 2, 'duration': 4, 'expected': 6},
    'gap_jump_medium': {'base_action': 4, 'duration': 9, 'expected': 27},
    'stomp_enemy': {'base_action': 2, 'duration': 8, 'expected': 12},
    'small_hop_right': {'base_action': 2, 'duration': 6, 'expected': 9},
    'big_jump_right': {'base_action': 4, 'duration': 15, 'expected': 45},
}

def calculate_distance(base_action, duration):
    """Calculer la distance parcourue"""
    speed = movement_speeds[base_action]
    return speed * duration

print("🧮 Test des distances de saut corrigées")
print("="*60)
print(f"{'Action':<20} {'Base':<5} {'Dur.':<5} {'Calculé':<8} {'Attendu':<8} {'Status'}")
print("="*60)

all_correct = True

for action_name, config in jump_actions.items():
    base_action = config['base_action']
    duration = config['duration']
    expected = config['expected']
    
    calculated = calculate_distance(base_action, duration)
    
    status = "✅ OK" if calculated == expected else "❌ ERREUR"
    if calculated != expected:
        all_correct = False
    
    print(f"{action_name:<20} {base_action:<5} {duration:<5} {calculated:<8.0f} {expected:<8} {status}")

print("="*60)

if all_correct:
    print("🎉 Toutes les distances sont correctes !")
else:
    print("⚠️  Certaines distances ne correspondent pas aux attentes")

print("\nComparaison avant/après correction:")
print("Avant : JUMP (action 2) = 1.0px/step")
print("Après : JUMP (action 2) = 1.5px/step (+50%)")
print("Avant : RUN_JUMP (action 4) = 2.5px/step") 
print("Après : RUN_JUMP (action 4) = 3.0px/step (+20%)")

print(f"\nExemples concrets:")
print(f"- short_jump (10 frames) : {calculate_distance(2, 10):.0f}px (était 10px)")
print(f"- long_jump (12 frames) : {calculate_distance(4, 12):.0f}px (était 30px)")
print(f"- gap_jump_extreme (15 frames) : {calculate_distance(4, 15):.0f}px (était 37px)")