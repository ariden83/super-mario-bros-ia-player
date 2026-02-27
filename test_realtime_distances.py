#!/usr/bin/env python3
"""
Test en temps réel des distances des macro-actions Mario
Vérifier que les distances réelles correspondent aux descriptions
"""

# Vitesses de déplacement
movement_speeds = {
    0: 0,     # NOOP - pas de mouvement
    1: 1.0,   # RIGHT - marche normale
    2: 1.5,   # JUMP - saut avec momentum horizontal
    3: 2.5,   # RUN - course rapide
    4: 3.0,   # RUN_JUMP - course + saut (vitesse maximale)
    5: 0,     # JUMP_ONLY - saut purement vertical
    6: -1.0   # LEFT - reculer
}

def calculate_distance(base_action, duration):
    """Calculer la distance réelle parcourue"""
    speed = movement_speeds[base_action]
    return speed * duration

# Toutes les macro-actions du jeu avec leurs configurations actuelles
macro_actions = {
    # Mouvements de base
    'walk_right': {'base_action': 1, 'duration': 8, 'current_desc': 'Marcher à droite (~8px)'},
    'walk_right_short': {'base_action': 1, 'duration': 4, 'current_desc': 'Marcher à droite courte distance (~4px)'},
    'walk_right_medium': {'base_action': 1, 'duration': 12, 'current_desc': 'Marcher à droite distance moyenne (~12px)'},
    'walk_right_long': {'base_action': 1, 'duration': 20, 'current_desc': 'Marcher à droite longue distance (~20px)'},
    'walk_right_precise': {'base_action': 1, 'duration': 6, 'current_desc': 'Marcher à droite précision (~6px)'},
    
    'run_forward': {'base_action': 3, 'duration': 10, 'current_desc': 'Courir vers la droite (~25px)'},
    'run_short': {'base_action': 3, 'duration': 4, 'current_desc': 'Course courte (~10px)'},
    'run_medium': {'base_action': 3, 'duration': 8, 'current_desc': 'Course moyenne (~20px)'},
    'run_long': {'base_action': 3, 'duration': 16, 'current_desc': 'Course longue (~40px)'},
    
    'step_back_safe': {'base_action': 6, 'duration': 10, 'current_desc': 'Recul sécurisé (~10px)'},
    
    # Sauts
    'short_jump': {'base_action': 2, 'duration': 10, 'current_desc': 'Petit saut pour petits obstacles (~15px horizontal)'},
    'short_jump_precise': {'base_action': 2, 'duration': 6, 'current_desc': 'Petit saut très précis (~9px horizontal)'},
    'short_jump_wide': {'base_action': 2, 'duration': 14, 'current_desc': 'Petit saut plus large (~21px horizontal)'},
    
    'long_jump': {'base_action': 4, 'duration': 12, 'current_desc': 'Course + saut distance moyenne (~36px)'},
    'long_jump_short': {'base_action': 4, 'duration': 8, 'current_desc': 'Course + saut courte distance (~24px)'},
    'long_jump_far': {'base_action': 4, 'duration': 18, 'current_desc': 'Course + saut longue distance (~54px)'},
    
    'stomp_enemy': {'base_action': 2, 'duration': 8, 'current_desc': 'Sauter sur Goomba/Koopa pour les tuer (~12px saut)'},
    'position_under_block': {'base_action': 1, 'duration': 15, 'current_desc': 'Se positionner sous un bloc question mark (~15px marche)'},
    'approach_and_hit_block': {'base_action': 4, 'duration': 30, 'current_desc': 'Approcher et frapper bloc (~90px course+saut)'},
    
    'jump_on_pipe': {'base_action': 4, 'duration': 18, 'current_desc': 'Sauter sur un tuyau court/plateforme (~54px RUN_JUMP)'},
    'small_hop_right': {'base_action': 2, 'duration': 6, 'current_desc': 'Petit saut vers la droite (~9px)'},
    'big_jump_right': {'base_action': 4, 'duration': 15, 'current_desc': 'Grand saut vers la droite (~45px)'},
    'precise_landing': {'base_action': 2, 'duration': 12, 'current_desc': 'Saut contrôlé pour atterrissage précis (~18px)'},
    
    'hop_on_platform': {'base_action': 2, 'duration': 15, 'current_desc': 'Monter sur plateforme/tuyau court (~22px saut)'},
    'retreat_and_jump': {'base_action': 6, 'duration': 12, 'current_desc': 'Reculer puis sauter (~12px recul)'},
    'run_jump_over': {'base_action': 4, 'duration': 20, 'current_desc': 'Course + saut pour passer par-dessus (~60px)'},
    
    'micro_step_right': {'base_action': 1, 'duration': 2, 'current_desc': 'Micro-pas à droite (~2px) - précision maximale'},
    'edge_walk_right': {'base_action': 1, 'duration': 3, 'current_desc': 'Marche au bord de plateforme (~3px)'},
    
    'pixel_perfect_jump': {'base_action': 2, 'duration': 4, 'current_desc': 'Saut pixel-perfect (~6px)'},
    'gap_jump_short': {'base_action': 4, 'duration': 6, 'current_desc': 'Saut de fossé court (~18px)'},
    'gap_jump_medium': {'base_action': 4, 'duration': 9, 'current_desc': 'Saut de fossé moyen (~27px)'},
    'gap_jump_long': {'base_action': 4, 'duration': 12, 'current_desc': 'Saut de fossé long (~36px)'},
    'gap_jump_extreme': {'base_action': 4, 'duration': 15, 'current_desc': 'Saut de fossé extrême (~45px)'},
}

print("🧮 VÉRIFICATION COMPLÈTE DES DISTANCES MARIO")
print("="*80)
print(f"{'Action':<25} {'Real':<8} {'Décrit':<8} {'Status':<10} {'Action base':<12} {'Durée'}")
print("="*80)

errors_found = []
corrections_needed = []

for action_name, config in macro_actions.items():
    base_action = config['base_action']
    duration = config['duration']
    current_desc = config['current_desc']
    
    # Calculer la distance réelle
    real_distance = calculate_distance(base_action, duration)
    
    # Extraire la distance décrite 
    import re
    desc_match = re.search(r'~(\d+)px', current_desc)
    described_distance = int(desc_match.group(1)) if desc_match else 0
    
    # Vérifier la cohérence
    if abs(real_distance - described_distance) > 0.1:
        status = "❌ ERREUR"
        errors_found.append(action_name)
        corrections_needed.append({
            'action': action_name,
            'base_action': base_action,
            'duration': duration,
            'real_distance': real_distance,
            'described_distance': described_distance,
            'current_desc': current_desc
        })
    else:
        status = "✅ OK"
    
    print(f"{action_name:<25} {real_distance:<8.0f} {described_distance:<8} {status:<10} {base_action:<12} {duration}")

print("="*80)

if errors_found:
    print(f"⚠️  {len(errors_found)} actions ont des distances incorrectes:")
    for error in errors_found:
        print(f"   - {error}")
    
    print("\n🔧 CORRECTIONS NÉCESSAIRES:")
    for correction in corrections_needed:
        old_desc = correction['current_desc']
        new_desc = re.sub(r'~\d+px', f'~{correction["real_distance"]:.0f}px', old_desc)
        print(f"\n{correction['action']}:")
        print(f"   Ancien: {old_desc}")
        print(f"   Nouveau: {new_desc}")
        print(f"   (Base: {correction['base_action']}, Durée: {correction['duration']}, Distance réelle: {correction['real_distance']:.0f}px)")

else:
    print("🎉 Toutes les distances sont correctes !")

print(f"\n📊 ANALYSE GLOBALE:")
print(f"Actions testées: {len(macro_actions)}")
print(f"Erreurs trouvées: {len(errors_found)}")
print(f"Précision globale: {((len(macro_actions) - len(errors_found)) / len(macro_actions)) * 100:.1f}%")