#!/usr/bin/env python3
"""
Test rapide de la correction d'échelle
"""

# Simulation de la correction d'échelle comme dans le code principal
original_game_width = 256
original_game_height = 240
screenshot_width = 256
screenshot_height = 192

horizontal_scale_factor = original_game_width / screenshot_width  # 1.0  
vertical_scale_factor = original_game_height / screenshot_height  # 1.25

def correct_claude_distance(claude_distance, direction='horizontal'):
    """Corriger les distances données par Claude selon l'échelle du screenshot"""
    if direction == 'horizontal':
        corrected = claude_distance * horizontal_scale_factor
    else:  # vertical
        corrected = claude_distance * vertical_scale_factor
    
    return int(round(corrected))

print("🔧 Test de correction d'échelle")
print(f"Facteurs: Horizontal={horizontal_scale_factor:.2f}, Vertical={vertical_scale_factor:.2f}")
print()

# Tests avec les distances que Claude pourrait donner
test_distances = [10, 15, 18, 20, 30]

print("Distance Claude → Distance Réelle")
print("Horizontal:")
for dist in test_distances:
    corrected = correct_claude_distance(dist, 'horizontal')
    print(f"  {dist}px → {corrected}px")

print("\nVertical:")
for dist in test_distances:
    corrected = correct_claude_distance(dist, 'vertical')
    print(f"  {dist}px → {corrected}px")

print("\nExemples concrets:")
print(f"Claude dit 'avancer 18px' → Mario avance {correct_claude_distance(18)}px")
print(f"Claude dit 'sauter 12px haut' → Mario saute {correct_claude_distance(12, 'vertical')}px")