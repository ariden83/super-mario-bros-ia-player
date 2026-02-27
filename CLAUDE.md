# Règles Mario IA Project

## Règles systématiques à respecter

### 🎯 SYSTÈME ANTI-MORT : Tracking du mouvement des ennemis
**Problème résolu**: Le LLM voyait des images statiques et ne comprenait pas que les ennemis BOUGENT vers Mario. Il attendait alors que l'ennemi passe, alors que celui-ci se dirigeait vers lui.

**Solution implémentée**:
- Tracking automatique de la direction et vitesse des ennemis entre chaque frame
- Calcul du niveau de danger en fonction du mouvement (s'approche vs s'éloigne)
- Recommandations d'actions adaptées au mouvement réel des ennemis

**Comportement attendu**:
- Si ennemi AVANCE vers Mario (→ VERS MARIO) + distance < 50px → `stomp_enemy` IMMÉDIATEMENT
- Si ennemi S'ÉLOIGNE (← ou →) → SAFE pour avancer/collecter items
- JAMAIS `wait_for_enemy` si l'ennemi se rapproche !
- En cas de doute → TOUJOURS attaquer plutôt qu'attendre

**Exemples**:
```
Goomba à 30px, direction: → VERS MARIO, vitesse=2px/frame
→ Danger: 🔴 DANGER IMMÉDIAT! (urgence 10/10)
→ Action recommandée: stomp_enemy

Goomba à 80px, direction: ← S'ÉLOIGNE, vitesse=1px/frame
→ Danger: 🟢 SAFE (urgence 2/10)
→ Action recommandée: run_forward ou collecte items
```

### Règles techniques existantes
- Mario doit être rapide et agressif
- Toujours privilégier `run_forward` à `walk_right`
- Attaquer les Goombas au lieu de reculer
- Économiser les appels API Claude
- **Priorité absolue: SURVIE > Collecte de blocs/items**

## Commandes utiles
- Tests: `python mario_fluid_llm.py`
- Logs: vérifier dans `logs/`