# Mario IA Project — Guide Claude Code

## Règles générales
- Ne jamais commiter ni pousser sur GitHub sans demande explicite
- Ne pas utiliser d'emojis sauf si explicitement demandé

## Commandes
```bash
source .venv/bin/activate && python3 mario_fluid_llm.py   # lancer le jeu
ls -lt logs/ | head -10                                    # derniers logs
```

## Fichiers clés
| Fichier | Rôle |
|---------|------|
| `mario_fluid_llm.py` | Jeu principal + LLM (fichier unique ~3900 lignes) |
| `mario_segment_memory.py` | Mémoire des segments de niveau |
| `mario_auto_improver.py` | Boucle méta-apprentissage (patches search/replace) |
| `mario_config_override.json` | Paramètres auto-ajustés par l'auto-improver |
| `logs/` | Sessions de jeu (actions, claude responses, summary) |
| `backups/` | Backups horodatés avant chaque patch auto-improver |

## Architecture du jeu

### Boucle principale (`play_fluid_mario`)
```
env.step(current_action)  ← NE PAS appeler si current_action is None (PAUSE)
→ obs, reward, done, real_info
→ analyze_situation()      ← situation dict envoyé à Claude
→ call_claude_async()      ← thread async, remplit action_queue
→ reflexes (stomp/trou)    ← synchrones, priorité absolue
→ get_current_action()     ← dépile action_queue → current_macro
```

### Macro actions (durées en frames NES)
- `run_forward` : base_action=3 (right+B), 25f
- `stomp_enemy` : base_action=2 (right+A), 20f — **géré par réflexe, Claude ne doit PAS l'utiliser**
- `pipe_jump`   : 2 phases — phase1: right 40f, phase2: right+A+B 40f
- `obstacle_jump`: 2 phases similaires
- `max_jump`    : base_action=4 (right+A+B), 30f

### Système réflexe (synchrone, priorité max)
- **Stomp réflexe OAM** : lit l'OAM NES (RAM 0x0200+) → cherche sprites palette=3 (ennemis) à 15-70px devant Mario (sprite 1 X) → injecte `run_jump_over`. Cooldown : **25 frames**
  - NB: Goombas sont RGB(228,92,16), Mario RGB(248,56,0) — couleurs DIFFÉRENTES, détection couleur ne marche pas
  - L'OAM fournit les positions écran exactes avec le bit palette qui distingue Mario/ennemis
- **Trou réflexe** : détecte absence de sol devant Mario → injecte `max_jump`. Cooldown : **15 frames**
- Les réflexes ne se déclenchent PAS si current_macro est un saut (_JUMP_MACROS)

### Règles critiques — NE PAS CASSER
1. `stomp_enemy` planifié par Claude → **converti en `run_jump_over`** (saut plus fiable que stomp, couvre ~140px)
2. `env.step()` non appelé si `current_action is None` → jeu gelé (PAUSE pendant que Claude pense)
3. `inject_known_solution` : ne re-injecte pas si position n'a pas avancé de ≥40px (`_last_known_solution_x`)
4. `inject_known_solution` : ne clear pas la queue si un jump macro est en cours ou planifié
5. Stuck mode → reset `last_reflex_step = step_count` (force cooldown 25f)

### Système de phases (vies)
- Phase 1 (1ère vie) : IA pure
- Phase 2 (2ème vie) : 50% replay mémoire + IA
- Phase 3 (3ème vie) : replay jusqu'à frontière max_x, puis IA
- Actuellement configuré à **1 vie** (`deaths_count >= 1` → game over)

### PAUSE IA (quand queue vide)
- `current_action = None` → `env.step()` non appelé → ennemis gelés
- Affichage overlay "PAUSE Claude reflechit..."
- En mode replay/Phase3 safe zone : fallback `run_forward` (pas de pause)

## API NES — Save State (pour rewind)
```python
# Sauvegarder l'état NES (2048 bytes RAM)
ram_snapshot = env.unwrapped._ram_buffer().copy()

# Restaurer l'état NES
np.copyto(env.unwrapped._ram_buffer(), ram_snapshot)
obs, _, _, info = env.step(0)  # NOOP pour actualiser l'écran
# info['x_pos'] revient à la valeur sauvegardée ✓
```
Confirmé fonctionnel. `_ram_buffer()` retourne un `np.ndarray` shape (2048,).

## Bugs connus / tentatives échouées
- **stomp_enemy planifié par Claude** → toujours `stomp+run_forward` même après changement d'exemples dans le prompt. Fix : filtrage côté code (converti en run_forward à la réception)
- **Cooldown réflexe 80f trop long** → Mario court 320px sans protection après un stomp. Fix : réduit à 25f
- **inject_known_solution boucle infinie** : réinjectait `run_forward` en boucle. Fix : `_last_known_solution_x` cooldown 40px
- **pipe_jump sabotage** : stomp reflex interrompait pipe_jump Phase 1. Fix : `_JUMP_MACROS` liste

## Évolutions en cours (voir tâches)
1. **Fix réflexe stomp** : Mario court encore dans les ennemis — vérifier si le réflexe se déclenche réellement (chercher "⚡ RÉFLEXE v3" dans console)
2. **Rewind sur mort** : restaurer RAM 60f avant mort → appel Claude correctif → rejouer

## Format logs
```
logs/mario_session_{ts}_actions.log     ← actions exécutées (format pipe-séparé)
logs/mario_session_{ts}_claude.log      ← appels/réponses Claude (résumé)
logs/mario_session_{ts}_responses_full.txt ← JSON complet des réponses Claude
logs/mario_session_{ts}_summary.json    ← stats finales
```
Actions log format : `timestamp | INFO | mario.actions | ACTION - Step N | AI | macro_name | Pos(x,y) | Score:s | reason`
