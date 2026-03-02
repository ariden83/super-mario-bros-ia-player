#!/usr/bin/env python3
"""
Super Mario Bros avec Claude LLM - Version Fluide
Claude donne des macro-actions, Mario les exécute en temps réel
"""

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import time
import json
import os
import cv2
import re
from collections import deque
import anthropic
import threading
import base64
import random
from PIL import Image
import io
import tempfile
from typing import Dict, Any, List, Optional
from mario_history_manager import MarioHistoryManager
from mario_logger import MarioLogger
from mario_level_database import MarioLevelDatabase
from distance_converter import DistanceConverter
from mario_segment_memory import MarioSegmentMemory
from mario_auto_improver import MarioAutoImprover, CONFIG_FILE, TUNABLE_PARAMS

class MarioFluidLLM:
    def __init__(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        
        # Actions de base
        self.actions = {
            0: 'NOOP', 1: 'RIGHT', 2: 'JUMP', 3: 'RUN', 
            4: 'RUN_JUMP', 5: 'JUMP_ONLY', 6: 'LEFT'
        }
        
        # Actions étendues que Claude peut commander (basées sur la recherche Super Mario Bros NES)
        self.macro_actions = {
            # Mouvements de base
            'walk_right': {'base_action': 1, 'duration': 15, 'description': 'Marcher à droite'},
            'run_forward': {'base_action': 3, 'duration': 25, 'description': 'Courir vers la droite (plus rapide)'},
            'step_back': {'base_action': 6, 'duration': 6, 'description': 'Reculer/éviter danger'},
            'wait': {'base_action': 0, 'duration': 3, 'description': 'Attendre/observer'},
            
            # Sauts tactiques
            'short_jump': {'base_action': 2, 'duration': 10, 'description': 'Petit saut pour petits obstacles'},
            'high_jump': {'base_action': 5, 'duration': 8, 'description': 'Saut vertical haut'},
            'long_jump': {'base_action': 4, 'duration': 12, 'description': 'Course + saut pour longues distances'},
            'precise_jump': {'base_action': 2, 'duration': 10, 'description': 'Saut précis sur ennemis/blocs'},
            
            # Actions spéciales Mario Bros
            'stomp_enemy': {'base_action': 2, 'duration': 20, 'description': 'Sauter sur Goomba/Koopa pour les tuer (right+A, 20 frames = saut fiable vers la droite, portée ~40-55px, hauteur suffisante pour passer au-dessus)'},
            'hit_block': {'base_action': 5, 'duration': 20, 'description': 'Frapper bloc ? par dessous pour items (saut vertical long)'},
            'position_under_block': {'base_action': 1, 'duration': 15, 'description': 'Se positionner sous un bloc question mark (mouvement long)'},
            'approach_and_hit_block': {'base_action': 4, 'duration': 30, 'description': 'Approcher et frapper bloc en un mouvement (RUN_JUMP)'},
            
            # Actions granulaires pour situations complexes
            'jump_on_pipe': {'base_action': 4, 'duration': 18, 'description': 'Sauter sur un tuyau court/plateforme (RUN_JUMP)'},
            'small_hop_right': {'base_action': 2, 'duration': 6, 'description': 'Petit saut vers la droite (éviter obstacles bas)'},
            'small_hop_left': {'base_action': 6, 'duration': 6, 'description': 'Petit saut vers la gauche (reculer avec saut)'},
            'big_jump_right': {'base_action': 4, 'duration': 15, 'description': 'Grand saut vers la droite (franchir obstacles)'},
            'precise_landing': {'base_action': 2, 'duration': 12, 'description': 'Saut contrôlé pour atterrissage précis'},
            'duck_and_move': {'base_action': 1, 'duration': 8, 'description': 'Se baisser et avancer (éviter projectiles)'},

            # ═══ SAUTS COMPOSÉS (multi-phases) ═══════════════════════════
            # pipe_jump : séquence garantie pour le premier tuyau de World 1-1
            #   Phase 1 : marcher vers le tuyau (40 frames) → Mario s'arrête automatiquement au pied
            #   Phase 2 : saut maximum depuis la base (40 frames) → franchit le tuyau
            'pipe_jump': {
                'phases': [
                    {'base_action': 1, 'duration': 40},   # right → approche lente jusqu'au pied
                    {'base_action': 4, 'duration': 40},   # right+A+B → saut max depuis la base
                ],
                'description': 'APPROCHE + SAUT pour franchir un tuyau haut : marche 40f jusqu\'au pied, puis saut max 40f (séquence obligatoire NES)'
            },
            # obstacle_jump : course puis saut max pour obstacles larges ou plateforme élevée
            #   Phase 1 : prendre de la vitesse (20 frames right+B)
            #   Phase 2 : saut max avec élan (40 frames right+A+B)
            'obstacle_jump': {
                'phases': [
                    {'base_action': 3, 'duration': 20},   # right+B → élan
                    {'base_action': 4, 'duration': 40},   # right+A+B → saut max avec vitesse
                ],
                'description': 'ÉLAN + SAUT MAX pour obstacles larges ou plateformes élevées (course 20f + saut max 40f)'
            },

            # Actions tactiques spécifiques
            'wait_for_enemy': {'base_action': 0, 'duration': 5, 'description': 'Attendre que l\'ennemi passe (timing)'},
            'retreat_and_jump': {'base_action': 6, 'duration': 12, 'description': 'Reculer puis sauter (éviter puis attaquer)'},
            'run_jump_over': {'base_action': 4, 'duration': 35, 'description': 'Course + saut pour passer par-dessus obstacle (35 frames = minimum confirmé pour le premier tuyau)'},
            'max_jump': {'base_action': 4, 'duration': 40, 'description': 'Saut MAXIMUM : right+A+B maintenu 40 frames = hauteur absolue max NES (obstacles très hauts, tuyaux hauts, plateformes élevées)'},
            'hop_on_platform': {'base_action': 2, 'duration': 15, 'description': 'Monter sur plateforme/tuyau court avec précision'},
            'kick_shell': {'base_action': 1, 'duration': 6, 'description': 'Donner coup de pied à carapace Koopa'},
            'collect_powerup': {'base_action': 1, 'duration': 8, 'description': 'Récupérer champignon/fleur de feu'},
            'avoid_piranha': {'base_action': 0, 'duration': 6, 'description': 'Attendre que Piranha Plant rentre'},
            'quick_run': {'base_action': 3, 'duration': 6, 'description': 'Course rapide sous Piranha Plant'},
            'pipe_down': {'base_action': 0, 'duration': 8, 'description': 'Aller vers le bas sur tuyau (zones bonus)'}
        }
        
        # Configuration Claude
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("❌ ANTHROPIC_API_KEY non définie!")
        
        self.claude_client = anthropic.Anthropic(api_key=self.api_key)
        
        # État du jeu
        self.position_history = deque(maxlen=20)
        self.macro_history = deque(maxlen=8)
        self.current_macro = None
        self.macro_frames_left = 0
        
        # Système non-bloquant
        self.action_queue = deque(maxlen=10)  # Queue d'actions de Claude
        self.claude_thinking = False
        self.claude_thread = None
        self.last_situation = None
        
        # Système de screenshots
        self.last_screenshot_step = 0
        self.last_screenshot_x = 0  # coord monde NES au moment du dernier screenshot
        self.screenshot_frequency = 15  # Prendre un screenshot toutes les 15 steps (meilleure anticipation)
        self.use_visual_analysis = True  # Utiliser l'analyse d'image Claude
        self.screenshot_cost_limit = 1.00  # Limite de coût pour les screenshots ($1.00) - augmentée pour meilleure anticipation
        self.screenshot_costs = 0.0  # Coût cumulé des screenshots
        self.ultra_low_cost_mode = False  # Mode très économique désactivé pour améliorer la vision
        
        # Système d'échelle pour conversion pixels screenshot ↔ jeu
        self.current_scale_factor = 1.0  # Facteur de conversion screenshot → jeu
        self.screenshot_dimensions = (256, 192)  # Dimensions du screenshot envoyé
        self.original_game_dimensions = (256, 204)  # Dimensions réelles de la zone de jeu
        self.distance_converter = DistanceConverter(self.current_scale_factor)
        
        # Historique des réponses LLM pour l'encart scrollable
        self.llm_responses = deque(maxlen=50)  # Garder les 50 dernières réponses
        self.llm_scroll_position = 0  # Position de scroll dans l'encart
        
        # Statistiques
        self.api_calls = 0
        self.total_cost = 0.0
        self.successful_macros = 0
        self._life_macro_count = 0   # Séquences exécutées dans la vie courante (reset à chaque respawn)
        self.deaths_count = 0
        self.lives_used = 0
        self.mario_lives_remaining = 3  # Mario commence avec 3 vies
        
        # Système d'historique d'apprentissage
        self.action_history = deque(maxlen=50)  # Historique détaillé des actions
        self.failure_patterns = []  # Situations qui mènent à la mort
        self.death_locations = []  # Où Mario meurt souvent
        self.successful_strategies = []  # Actions qui marchent bien
        self.last_actions_before_death = deque(maxlen=10)  # Actions juste avant de mourir
        self.repeated_failures = {}  # Compteur d'échecs répétés
        
        # Gestionnaire d'historique persistant
        self.history_manager = MarioHistoryManager()
        self.current_run_started = False
        
        # Système de logging
        self.logger = MarioLogger()
        
        # Base de données des niveaux
        self.level_db = MarioLevelDatabase()
        self.current_world = 1
        self.current_level = 1
        self.level_detection_confidence = 0
        
        # Système de replay
        self.replay_mode = False
        self.replay_actions = []
        self.replay_index = 0
        self.replay_ai_takeover_point = 0  # Point où l'IA reprend la main
        
        # Mémoire persistante par segments
        self.segment_memory = MarioSegmentMemory()
        self._prev_score = 0       # Pour détecter gains de points
        self._prev_coins = 0       # Pour détecter pièces collectées
        self._prev_lives = 3       # Pour détecter les morts
        # Tracking du déblocage réussi
        self._unstick_start_x = None   # Position où Mario était bloqué
        self._unstick_sequence = None  # Séquence tentée pour débloquer
        self._unstick_step = 0         # Step du début du déblocage

        # Système de phases d'apprentissage (cycle 3 runs)
        self._run_phase = 1             # 1=IA pure, 2=mixte, 3=mémoire→IA danger
        self._segment_in_replay = False # Segment courant en replay mémoire
        self._phase3_ai_mode = False    # Phase 3: basculé en mode IA (zone danger)
        self._last_seg_key = None       # Pour détecter les transitions de segment
        self._phase3_last_x: int = 0   # Anti-blocage Phase 3 : dernière x de référence
        self._phase3_last_x_step: int = 0  # Step de la dernière progression détectée

        # Couche réflexe pixel
        self.last_reflex_step = -25      # Cooldown réflexe ennemi (25 frames)
        self.last_hole_reflex_step = -60 # Cooldown réflexe trou (séparé)
        self._hole_reflex_count = 0      # Nb de déclenchements consécutifs sans progression
        self._hole_reflex_last_x = 0    # x_pos lors du dernier déclenchement réflexe trou

        # Système de rewind sur mort
        self.rewind_buffer = deque(maxlen=3)  # 3 checkpoints (60 frames d'écart)
        self.rewind_count = 0                  # Rewinds utilisés cette partie
        self.max_rewinds = float('inf')        # Rewinds illimités
        self._rewind_active = False            # Pour overlay visuel
        self._rewind_correction_msg = None     # Message injecté dans le prochain prompt Claude
        self._raw_action_history = []          # Historique brut des actions NES (pour replay PPU)
        self._final_action_history = []        # Historique "propre" : tronqué aux checkpoints sur rewind
        self._claude_generation = 0            # Incrémenté à chaque rewind pour invalider threads en cours
        self._death_positions = []             # Historique des positions de mort pour le message rewind
        self._danger_zone_x = None             # Position X à éviter après rewind (filet de sécurité)
        self._rewind_real_info = None          # real_info capturé après replay PPU (évite info périmée)

        # Système anti-blocage
        self.stuck_counter = 0           # Nombre de checks consécutifs sans progression
        self.last_stuck_check_step = 0   # Dernier step où on a vérifié le blocage
        self.stuck_check_frequency = 60   # Vérifier toutes les 60 steps (pipe_jump = 80 frames, ok car inject_known_solution cooldown protège)
        self.last_stuck_position = None  # Position au dernier check
        self.stuck_search_done = set()   # Positions déjà cherchées (évite doublons)

        # Système hybride optimisé screenshot + positions
        self.level_context_established = False  # Si Claude a la carte du niveau
        self.last_positions_update = 0  # Dernière mise à jour des positions
        self.positions_update_frequency = 5  # Updates toutes les 5 steps (très fréquent)
        self.context_recalibration_frequency = 120  # Screenshot de recalibrage tous les 120 steps
        self.tracked_elements = {  # Positions des éléments mobiles
            'mario': {'x': 0, 'y': 0, 'direction': 'right', 'speed': 0},
            'enemies': [],  # Liste des ennemis avec positions
            'collectibles': [],  # Power-ups, pièces qui apparaissent
            'blocks_hit': []  # Blocs déjà frappés
        }

        # Système de tracking du mouvement des ennemis (ANTI-MORT)
        self.previous_enemies = []  # Positions des ennemis à la frame précédente
        self.enemy_movement_history = deque(maxlen=3)  # Historique sur 3 frames

        # Ajouts aux prompts (valeur par défaut, écrasée par _load_config_override si config existe)
        self._prompt_additions: dict = {"stuck_mode": [], "main_context": []}

        # Charger les overrides de config (paramètres ajustés automatiquement par l'auto-improver)
        # NB: doit venir APRÈS l'initialisation de _prompt_additions
        self._load_config_override()

        print("✅ Mario Fluide LLM initialisé!")

    def _load_config_override(self):
        """Charge mario_config_override.json et applique les overrides de paramètres."""
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            params = cfg.get("parameters", {})
            # Appliquer chaque paramètre tunable connu
            if "stuck_check_frequency" in params:
                self.stuck_check_frequency = int(params["stuck_check_frequency"])
            if "positions_update_frequency" in params:
                self.positions_update_frequency = int(params["positions_update_frequency"])
            if "reflex_cooldown_frames" in params:
                self.last_reflex_step = -int(params["reflex_cooldown_frames"])
            if "hole_reflex_cooldown_frames" in params:
                self.last_hole_reflex_step = -int(params["hole_reflex_cooldown_frames"])
            # Stocker le cooldown inject_known_solution pour utilisation dynamique
            self._known_solution_cooldown_px = int(
                params.get("known_solution_cooldown_px", 40)
            )
            # Ajouts aux prompts
            additions = cfg.get("prompt_additions", {})
            self._prompt_additions = {
                "stuck_mode": list(additions.get("stuck_mode", [])),
                "main_context": list(additions.get("main_context", [])),
            }
            v = cfg.get("version", 0)
            if v > 0:
                print(f"⚙️  Config override v{v} chargée: {list(params.keys())}")
        except Exception as e:
            print(f"⚠️  Impossible de charger config override: {e}")

    def analyze_situation(self, obs, info, step_count):
        """Analyser la situation pour Claude"""
        
        mario_x = info.get('x_pos', 0)
        mario_y = info.get('y_pos', 0)
        score = info.get('score', 0)
        
        self.position_history.append(mario_x)
        
        # Analyser la progression
        progress_analysis = self.analyze_progression()

        # 🎯 TRACKING MOUVEMENT ENNEMIS (ANTI-MORT)
        enemy_movements = self.track_enemy_movement(mario_x)

        # Analyser l'écran - Utiliser les screenshots Claude ou fallback
        if self.use_visual_analysis:
            # Mode screenshot : pas d'analyse RGB, Claude analyse directement
            screen_analysis = {
                'immediate_obstacles': False,
                'ground_stable': True,
                'environment_type': 'screenshot_mode',
                'question_blocks': False,
                'enemies_nearby': False,
                'power_ups': False,
                'pipes': False,
                'gaps': False,
                'visual_mode': 'Claude Screenshots',
                'status': 'En attente de l\'analyse Claude...'
            }
        else:
            # Mode classique RGB (si screenshots désactivés)
            screen_analysis = self.analyze_visual_context(obs, step_count)

        # Détecter le niveau actuel si confiance faible
        if self.level_detection_confidence < 80:
            self.detect_current_level(mario_x, step_count, screen_analysis)

        # ⛰️ Détection des trous (couche pixel, indépendante du mode screenshot)
        hole_info = self.detect_holes_ahead(obs) if obs is not None else {
            'detected': False, 'nearest': 999, 'width': 0, 'critical': False
        }
        screen_analysis['gaps'] = hole_info['detected']
        screen_analysis['holes'] = hole_info

        return {
            'mario': {'x': mario_x, 'y': mario_y, 'score': score},
            'progress': progress_analysis,
            'screen': screen_analysis,
            'holes': hole_info,
            'enemy_tracking': enemy_movements,
            'history': {
                'positions': list(self.position_history)[-8:],
                'recent_macros': list(self.macro_history)[-4:]
            },
            'step': step_count,
            'lives': info.get('life', 3)
        }
    
    def analyze_progression(self):
        """Analyser si Mario progresse bien"""
        
        if len(self.position_history) < 5:
            return {'status': 'starting', 'effectiveness': 'unknown', 'trend': 0}
        
        positions = list(self.position_history)
        recent_10 = positions[-10:] if len(positions) >= 10 else positions
        recent_5 = positions[-5:]
        
        # Tendance générale
        long_term_progress = recent_10[-1] - recent_10[0] if len(recent_10) >= 2 else 0
        short_term_progress = recent_5[-1] - recent_5[0] if len(recent_5) >= 2 else 0
        
        # Classification
        if short_term_progress <= 0:
            status = 'stuck_or_retreating'
            effectiveness = 'poor'
        elif short_term_progress < 10:
            status = 'slow_progress' 
            effectiveness = 'moderate'
        else:
            status = 'good_progress'
            effectiveness = 'good'
        
        return {
            'status': status,
            'effectiveness': effectiveness,
            'trend': short_term_progress,
            'long_term': long_term_progress
        }

    def track_enemy_movement(self, mario_x):
        """
        🎯 SYSTÈME ANTI-MORT : Tracker le mouvement des ennemis entre les frames
        Calcule direction, vitesse et niveau de danger de chaque ennemi
        """
        current_enemies = self.tracked_elements.get('enemies', [])

        if not current_enemies:
            self.previous_enemies = []
            return []

        enemy_movements = []

        for current_enemy in current_enemies:
            enemy_info = {
                'x': current_enemy.get('x', 0),
                'y': current_enemy.get('y', 0),
                'type': current_enemy.get('type', 'Goomba'),
                'distance_to_mario': abs(current_enemy.get('x', 0) - mario_x),
                'direction': '?',
                'speed': 0,
                'movement_pattern': 'INCONNU',
                'danger_level': '⚪ INCONNU'
            }

            # Trouver l'ennemi correspondant dans la frame précédente
            matched = False
            for prev_enemy in self.previous_enemies:
                # Même ennemi si position proche (tolérance 30px)
                if abs(current_enemy.get('x', 0) - prev_enemy.get('x', 0)) < 30 and \
                   abs(current_enemy.get('y', 0) - prev_enemy.get('y', 0)) < 20:

                    # Calculer le déplacement
                    dx = current_enemy.get('x', 0) - prev_enemy.get('x', 0)
                    dy = current_enemy.get('y', 0) - prev_enemy.get('y', 0)
                    speed = abs(dx)

                    # Déterminer la direction par rapport à Mario
                    if dx < -2:  # Se déplace vers la gauche
                        if current_enemy.get('x', 0) > mario_x:
                            enemy_info['direction'] = '← S\'ÉLOIGNE'
                            enemy_info['movement_pattern'] = 'SAFE'
                        else:
                            enemy_info['direction'] = '← VERS MARIO (PAR DERRIÈRE)'
                            enemy_info['movement_pattern'] = 'DANGER_ARRIERE'
                    elif dx > 2:  # Se déplace vers la droite
                        if current_enemy.get('x', 0) < mario_x:
                            enemy_info['direction'] = '→ S\'ÉLOIGNE'
                            enemy_info['movement_pattern'] = 'SAFE'
                        else:
                            enemy_info['direction'] = '→ VERS MARIO'
                            enemy_info['movement_pattern'] = 'DANGER_FRONTAL'
                    else:
                        enemy_info['direction'] = '⊙ IMMOBILE'
                        enemy_info['movement_pattern'] = 'STATIONNAIRE'

                    enemy_info['speed'] = speed
                    matched = True
                    break

            # Calculer le niveau de danger
            distance = enemy_info['distance_to_mario']
            pattern = enemy_info['movement_pattern']

            if pattern == 'DANGER_FRONTAL' and distance < 50:
                enemy_info['danger_level'] = '🔴 DANGER IMMÉDIAT!'
                enemy_info['urgency'] = 10
                enemy_info['recommended_action'] = 'stomp_enemy'
            elif pattern == 'DANGER_FRONTAL' and distance < 100:
                enemy_info['danger_level'] = '🟠 DANGER PROCHE'
                enemy_info['urgency'] = 8
                enemy_info['recommended_action'] = 'stomp_enemy ou run_jump_over'
            elif pattern == 'DANGER_ARRIERE' and distance < 40:
                enemy_info['danger_level'] = '🟡 DANGER ARRIÈRE'
                enemy_info['urgency'] = 7
                enemy_info['recommended_action'] = 'run_forward (fuir)'
            elif pattern == 'SAFE':
                enemy_info['danger_level'] = '🟢 SAFE (s\'éloigne)'
                enemy_info['urgency'] = 2
                enemy_info['recommended_action'] = 'run_forward ou collecte items'
            elif pattern == 'STATIONNAIRE' and distance < 60:
                enemy_info['danger_level'] = '🟡 PRUDENCE (immobile)'
                enemy_info['urgency'] = 5
                enemy_info['recommended_action'] = 'stomp_enemy ou wait_for_enemy'
            else:
                enemy_info['danger_level'] = '⚪ LOIN'
                enemy_info['urgency'] = 3
                enemy_info['recommended_action'] = 'continuer progression'

            enemy_movements.append(enemy_info)

        # Sauvegarder pour la prochaine frame
        self.previous_enemies = current_enemies.copy()

        # Ajouter à l'historique
        self.enemy_movement_history.append({
            'enemies': enemy_movements,
            'max_urgency': max([e.get('urgency', 0) for e in enemy_movements]) if enemy_movements else 0
        })

        return enemy_movements

    def analyze_visual_context(self, obs, step_count=0):
        """Analyser TOUT l'écran de jeu pour une vue d'ensemble complète"""
        
        try:
            height, width = obs.shape[:2]
            
            # Analyser TOUT l'écran au lieu de petites zones
            game_area = obs  # Tout l'écran
            
            # Zone de jeu principale (exclure l'interface en haut)
            game_height_start = int(height * 0.2)  # Ignorer les 20% du haut (score, etc)
            game_main = obs[game_height_start:, :]
            
            # Analyser les couleurs réelles du screenshot
            elements_detected = self.scan_full_screen(game_main, step_count)
            
            context = {
                'immediate_obstacles': elements_detected['obstacles_near'],
                'medium_obstacles': elements_detected['obstacles_far'],
                'ground_stable': elements_detected['ground_stable'],
                'environment_type': elements_detected['env_type'],
                
                # Éléments spécifiques Mario Bros avec positions
                'question_blocks': elements_detected['question_blocks']['detected'],
                'question_blocks_positions': elements_detected['question_blocks']['positions'],
                'enemies_nearby': elements_detected['enemies']['detected'],
                'enemies_positions': elements_detected['enemies']['positions'], 
                'power_ups': elements_detected['powerups']['detected'],
                'pipes': elements_detected['pipes']['detected'],
                'pipes_positions': elements_detected['pipes']['positions'],
                'gaps': elements_detected['gaps']['detected'],
                'coins': elements_detected['coins']['detected'],
                'underground': elements_detected['underground'],
                
                # Nouvelles infos
                'bricks': elements_detected['bricks']['detected'],
                'level_map': elements_detected['level_map']
            }
            
            return context
            
        except Exception as e:
            print(f"❌ Erreur analyse visuelle: {e}")
            return {
                'immediate_obstacles': False,
                'medium_obstacles': False, 
                'ground_stable': True,
                'environment_type': 'analysis_error',
                'question_blocks': False,
                'enemies_nearby': False,
                'power_ups': False,
                'pipes': False,
                'gaps': False
            }
    
    def scan_full_screen(self, game_area, step_count):
        """Scanner tout l'écran de jeu avec détection spatiale précise"""
        
        height, width = game_area.shape[:2]
        
        # Initialiser les résultats
        results = {
            'question_blocks': {'detected': False, 'positions': []},
            'enemies': {'detected': False, 'positions': []},
            'powerups': {'detected': False, 'positions': []},
            'pipes': {'detected': False, 'positions': []},
            'bricks': {'detected': False, 'positions': []},
            'coins': {'detected': False, 'positions': []},
            'gaps': {'detected': False, 'positions': []},
            'obstacles_near': False,
            'obstacles_far': False,
            'ground_stable': True,
            'underground': False,
            'env_type': 'normal',
            'level_map': [],
            'mario_position': None,
            'spatial_map': {}
        }
        
        # ÉTAPE 1: Trouver Mario d'abord
        mario_pos = self.find_mario_position(game_area)
        results['mario_position'] = mario_pos
        
        try:
            if mario_pos and mario_pos['found']:
                mario_x, mario_y = mario_pos['x'], mario_pos['y']
                
                # DÉTECTER GOOMBAS - SEUILS TRÈS PERMISSIFS
                # Toute zone sombre/brune au niveau du sol
                goomba_mask = (
                    (game_area[:, :, 0] > 50) &      # Un peu de rouge
                    (game_area[:, :, 1] > 30) &      # Un peu de vert
                    (game_area[:, :, 2] < 100) &     # Pas trop de bleu
                    (np.sum(game_area, axis=2) < 400)  # Couleur globalement sombre
                )
                
                # Chercher les Goombas uniquement dans la partie basse (niveau du sol)
                ground_level = int(height * 0.7)  # 70% vers le bas
                ground_area = game_area[ground_level:, :]
                goomba_ground_mask = goomba_mask[ground_level:, :]
                
                if np.any(goomba_ground_mask):
                    # Trouver les positions X des Goombas
                    goomba_columns = np.any(goomba_ground_mask, axis=0)
                    goomba_x_positions = np.where(goomba_columns)[0]
                    
                    if len(goomba_x_positions) > 0:
                        results['enemies']['detected'] = True
                        results['enemies']['positions'] = goomba_x_positions.tolist()
                        
                        # Vérifier si des ennemis sont proches de Mario
                        distances = np.abs(goomba_x_positions - mario_x)
                        if np.any(distances < 100):  # Moins de 100 pixels
                            results['enemies']['nearby'] = True
                
                # DÉTECTER BLOCS ? BLEUS - SEUILS TRÈS PERMISSIFS
                # Toute zone avec du bleu dominant
                question_blue_mask = (
                    (game_area[:, :, 2] > game_area[:, :, 0]) &  # Plus de bleu que de rouge
                    (game_area[:, :, 2] > game_area[:, :, 1]) &  # Plus de bleu que de vert
                    (game_area[:, :, 2] > 100)                   # Un minimum de bleu
                )
                
                # Chercher seulement dans la partie aérienne (pas au sol)
                air_level = int(height * 0.3)  # 30% vers le haut
                air_area = game_area[:air_level, :]
                question_air_mask = question_blue_mask[:air_level, :]
                
                if np.any(question_air_mask):
                    # Trouver les positions des blocs ?
                    question_columns = np.any(question_air_mask, axis=0)
                    question_x_positions = np.where(question_columns)[0]
                    
                    if len(question_x_positions) > 0:
                        results['question_blocks']['detected'] = True
                        results['question_blocks']['positions'] = question_x_positions.tolist()
                
                # IGNORER LES COLLINES VERTES (décor d'arrière-plan)
                # Ne pas détecter comme des tuyaux
                results['pipes']['detected'] = False
                
                # DÉTECTER VRAIS OBSTACLES (briques bleues au niveau de Mario)
                obstacle_mask = question_blue_mask  # Utiliser le même masque bleu
                mario_level_area = game_area[max(0, mario_y-50):mario_y+50, :]
                obstacle_level_mask = obstacle_mask[max(0, mario_y-50):mario_y+50, :]
                
                if np.any(obstacle_level_mask):
                    results['obstacles_near'] = True
                
                # CALCULER DISTANCES ET DIRECTIONS pour le LLM
                spatial_info = []
                
                # Ennemis
                if results['enemies']['detected']:
                    for enemy_x in results['enemies']['positions']:
                        distance = abs(enemy_x - mario_x)
                        direction = "droite" if enemy_x > mario_x else "gauche"
                        spatial_info.append(f"Goomba à {distance}px vers la {direction}")
                
                # Blocs ?
                if results['question_blocks']['detected']:
                    for block_x in results['question_blocks']['positions']:
                        distance = abs(block_x - mario_x)
                        direction = "droite" if block_x > mario_x else "gauche"
                        spatial_info.append(f"Bloc ? à {distance}px vers la {direction}")
                
                results['spatial_map'] = spatial_info
                
                # Log debug TRÈS DÉTAILLÉ
                if step_count % 30 == 0:  # Plus fréquent
                    print(f"🎯 Mario détecté: {mario_pos['found']} en ({mario_x}, {mario_y})")
                    
                    # Analyser quelques pixels spécifiques
                    if height > 100 and width > 100:
                        # Pixel au centre (probablement Mario ou fond)
                        center_pixel = game_area[height//2, width//2]
                        print(f"   Pixel centre RGB: {center_pixel}")
                        
                        # Pixel en bas à droite (probablement Goomba)
                        bottom_right = game_area[int(height*0.8), int(width*0.7)]
                        print(f"   Pixel bas-droite RGB: {bottom_right}")
                        
                        # Pixel des blocs ? (milieu-haut)
                        mid_top = game_area[int(height*0.4), int(width*0.6)]
                        print(f"   Pixel milieu-haut RGB: {mid_top}")
                    
                    # Stats des masques
                    goomba_density = np.mean(goomba_ground_mask) if 'goomba_ground_mask' in locals() else 0
                    question_density = np.mean(question_air_mask) if 'question_air_mask' in locals() else 0
                    
                    print(f"   Densité Goomba: {goomba_density:.4f}, Question: {question_density:.4f}")
                    print(f"   Résultats - Ennemis: {results['enemies']['detected']}, Blocs?: {results['question_blocks']['detected']}")
                    
                    if spatial_info:
                        print(f"   Carte spatiale: {spatial_info[:3]}")  # Afficher les 3 premiers
                
                # DÉTECTION D'URGENCE - Si rien n'est détecté, utiliser des méthodes brutes
                if not results['enemies']['detected'] and not results['question_blocks']['detected']:
                    # Chercher tout ce qui a beaucoup de bleu (blocs ?)
                    blue_anywhere = np.mean(game_area[:, :, 2] > 150)
                    if blue_anywhere > 0.05:  # 5% de l'écran a du bleu
                        results['question_blocks']['detected'] = True
                        print(f"🚨 Détection d'urgence - Bleu détecté: {blue_anywhere:.3f}")
                    
                    # Chercher tout ce qui est sombre (ennemis potentiels)
                    dark_areas = np.mean(np.sum(game_area, axis=2) < 300)
                    if dark_areas > 0.1:  # 10% de l'écran est sombre
                        results['enemies']['detected'] = True
                        print(f"🚨 Détection d'urgence - Zones sombres: {dark_areas:.3f}")
        
        except Exception as e:
            print(f"❌ Erreur scan_full_screen: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def detect_current_level(self, mario_x: int, step_count: int, screen_analysis: Dict) -> None:
        """Détecter le niveau actuel basé sur les données de jeu"""
        try:
            # Heuristiques de détection du niveau
            confidence_points = 0
            detected_world = 1
            detected_level = 1
            
            # Détection basée sur le nom de l'environnement
            env_name = str(self.env.spec.id) if hasattr(self.env, 'spec') else ""
            if "1-1" in env_name:
                detected_world, detected_level = 1, 1
                confidence_points += 50
            elif "1-2" in env_name:
                detected_world, detected_level = 1, 2
                confidence_points += 50
            
            # Détection basée sur les éléments visuels
            level_type = "OVERWORLD"  # Par défaut
            
            # Détection souterraine (couleurs plus sombres)
            if screen_analysis.get('underground', False):
                level_type = "UNDERGROUND"
                confidence_points += 20
                # Probablement 1-2 si monde 1
                if detected_world == 1:
                    detected_level = 2
                    confidence_points += 20
            
            # Détection château (Fire Bars, Bowser)
            if any('fire' in feature.lower() for feature in screen_analysis.get('level_map', [])):
                level_type = "CASTLE"
                confidence_points += 30
                # Probablement x-4
                detected_level = 4
                confidence_points += 20
            
            # Détection aquatique (si implémenté)
            if screen_analysis.get('environment_type') == 'underwater':
                level_type = "UNDERWATER"
                confidence_points += 30
                detected_level = 2  # Généralement x-2
                confidence_points += 15
            
            # Détection basée sur la progression Mario
            if mario_x > 3000:  # Mario a beaucoup progressé
                confidence_points += 10
            
            # Mettre à jour si confiance suffisante
            if confidence_points > self.level_detection_confidence:
                self.current_world = detected_world
                self.current_level = detected_level
                self.level_detection_confidence = confidence_points
                
                # Logger la détection
                self.logger.log_game_event("LEVEL_DETECTED", step_count, {
                    "world": detected_world,
                    "level": detected_level,
                    "confidence": confidence_points,
                    "level_type": level_type
                })
                
                print(f"🗺️ Niveau détecté: World {detected_world}-{detected_level} (confiance: {confidence_points}%)")
        
        except Exception as e:
            self.logger.log_error("LEVEL_DETECTION", str(e), step_count)
    
    def get_level_specific_context(self) -> str:
        """Générer le contexte spécifique au niveau actuel"""
        level_data = self.level_db.get_level_data(self.current_world, self.current_level)
        
        if not level_data:
            return "Niveau générique - informations limitées disponibles."
        
        context_parts = []
        
        # Informations générales du niveau
        context_parts.append(f"🗺️ NIVEAU: World {level_data.world}-{level_data.level} ({level_data.level_type})")
        context_parts.append(f"⏱️ Temps limite: {level_data.time_limit} secondes")
        context_parts.append(f"🎵 Musique: {level_data.background_music}")
        
        # Ennemis spécifiques à ce niveau
        if level_data.enemies:
            context_parts.append(f"\n👾 ENNEMIS CONFIRMÉS DANS CE NIVEAU:")
            for enemy in level_data.enemies:
                threat_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "💀"}[enemy.threat_level]
                context_parts.append(f"   {threat_emoji} {enemy.name}: {enemy.behavior} (Vitesse: {enemy.speed}px/step)")
                context_parts.append(f"      Élimination: {', '.join(enemy.defeat_methods)} | Points: {enemy.points}")
                if enemy.special_notes:
                    context_parts.append(f"      ⚠️ {enemy.special_notes}")
        
        # Blocs et éléments interactifs
        if level_data.blocks:
            context_parts.append(f"\n🧱 BLOCS ET ÉLÉMENTS INTERACTIFS:")
            for block in level_data.blocks:
                context_parts.append(f"   📦 {block.name}: {block.contents}")
                context_parts.append(f"      Comportement: {block.behavior}")
                if block.special_notes:
                    context_parts.append(f"      💡 {block.special_notes}")
        
        # Power-ups disponibles
        if level_data.power_ups:
            context_parts.append(f"\n⭐ POWER-UPS DISPONIBLES:")
            for powerup in level_data.power_ups:
                rarity_emoji = {"COMMON": "🟢", "RARE": "🟡", "VERY_RARE": "🔴"}[powerup.rarity]
                context_parts.append(f"   {rarity_emoji} {powerup.name}: {powerup.effect}")
                if powerup.special_notes:
                    context_parts.append(f"      💡 {powerup.special_notes}")
        
        # Obstacles spécifiques
        if level_data.obstacles:
            context_parts.append(f"\n⚠️ OBSTACLES SPÉCIFIQUES:")
            for obstacle in level_data.obstacles:
                threat_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "💀"}[obstacle.threat_level]
                context_parts.append(f"   {threat_emoji} {obstacle.name}: {obstacle.avoidance_strategy}")
                if obstacle.special_notes:
                    context_parts.append(f"      ⚠️ {obstacle.special_notes}")
        
        # Fonctionnalités spéciales
        if level_data.special_features:
            context_parts.append(f"\n🌟 CARACTÉRISTIQUES SPÉCIALES:")
            for feature in level_data.special_features:
                context_parts.append(f"   ✨ {feature}")
        
        # Stratégie recommandée
        context_parts.append(f"\n🎯 STRATÉGIE RECOMMANDÉE:")
        context_parts.append(f"   📋 {level_data.completion_strategy}")
        
        # Analyse des menaces
        threat_analysis = self.level_db.get_threat_analysis(self.current_world, self.current_level)
        context_parts.append(f"\n📊 ANALYSE DES MENACES:")
        context_parts.append(f"   Niveau max: {threat_analysis['max_threat_level']}")
        if threat_analysis['high_value_targets']:
            context_parts.append(f"   🎯 Cibles prioritaires: {', '.join(threat_analysis['high_value_targets'])}")
        
        # Power-ups recommandés
        recommended = self.level_db.get_recommended_powerups(self.current_world, self.current_level)
        context_parts.append(f"\n💊 POWER-UPS RECOMMANDÉS: {', '.join(recommended)}")
        
        return "\n".join(context_parts)
    
    def find_mario_position(self, game_area):
        """Trouver la position de Mario sur l'écran"""
        try:
            height, width = game_area.shape[:2]
            
            # Mario a généralement des couleurs rouge/brun caractéristiques
            # Chercher des pixels rouges pour sa casquette/shirt
            mario_red_mask = (
                (game_area[:, :, 0] > 180) &     # Beaucoup de rouge
                (game_area[:, :, 1] < 100) &     # Peu de vert
                (game_area[:, :, 2] < 100)       # Peu de bleu
            )
            
            # Chercher des pixels bruns pour son corps
            mario_brown_mask = (
                (game_area[:, :, 0] > 120) &     # Rouge moyen
                (game_area[:, :, 0] < 200) &
                (game_area[:, :, 1] > 80) &      # Vert moyen
                (game_area[:, :, 1] < 150) &
                (game_area[:, :, 2] < 80)        # Peu de bleu
            )
            
            # Combiner les masques
            mario_mask = mario_red_mask | mario_brown_mask
            
            # Trouver le centre de masse
            if np.any(mario_mask):
                y_coords, x_coords = np.where(mario_mask)
                mario_x = int(np.mean(x_coords))
                mario_y = int(np.mean(y_coords))
                return {'x': mario_x, 'y': mario_y, 'found': True}
            else:
                # Fallback: supposer que Mario est au centre
                return {'x': width // 2, 'y': height // 2, 'found': False}
                
        except Exception as e:
            return {'x': width // 2, 'y': height // 2, 'found': False, 'error': str(e)}
    
    def capture_game_screenshot(self, obs):
        """Capturer et optimiser l'écran de jeu pour Claude (coûts réduits)"""
        try:
            print(f"Début capture screenshot - obs shape: {obs.shape if obs is not None else 'None'}")
            # obs est déjà en RGB (standard OpenAI Gym) — pas de conversion nécessaire
            rgb_image = obs.copy()

            # Convertir en PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # OPTIMISATION 1: Cropper pour ne garder que la zone de jeu (enlever interface)
            width, height = pil_image.size
            game_area_crop = pil_image.crop((0, int(height * 0.15), width, height))  # Enlever 15% du haut
            
            # OPTIMISATION 2: Résolution améliorée pour meilleure vision 
            # Mario NES original = 256x240, on garde une résolution décente
            if self.ultra_low_cost_mode:
                optimized_size = (128, 96)  # Résolution minimale
                print("Mode ultra-économique activé")
            else:
                optimized_size = (256, 192)  # Résolution améliorée pour meilleure vision
            
            try:
                # Essayer avec la nouvelle API (Pillow 10+)
                resized_image = game_area_crop.resize(optimized_size, Image.Resampling.LANCZOS)
            except AttributeError:
                # Fallback pour les versions anciennes de Pillow
                resized_image = game_area_crop.resize(optimized_size, Image.LANCZOS)

            # Calculer le masque mystery sur l'image ORIGINALE (pixels NES purs, avant crop/resize)
            # Le resize LANCZOS mélange les pixels → couleurs NES diluées → détection instable
            orig_crop_top = int(height * 0.15)
            rgb_cropped_orig = rgb_image[orig_crop_top:, :]  # même crop, résolution originale
            mystery_mask_orig = self._compute_mystery_mask(rgb_cropped_orig)
            # Scaler avec NEAREST pour garder des bords nets sans mélange de pixels
            self._precomputed_mystery_mask = cv2.resize(
                mystery_mask_orig.astype(np.uint8),
                optimized_size,  # (W, H) convention cv2
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            # OPTIMISATION 3: Appliquer des filtres pour améliorer la détection
            enhanced_image = self.apply_detection_filters(resized_image)

            # ANNOTATION: Boîtes colorées Mario (VERT) + ennemis (ROUGE) pour le LLM
            enhanced_image = self._annotate_entities_for_llm(enhanced_image, obs)

            # OPTIMISATION 4: Format et qualité améliorée pour meilleure vision
            img_buffer = io.BytesIO()
            if self.ultra_low_cost_mode:
                # Convertir en noir et blanc pour réduire encore plus
                grayscale = enhanced_image.convert('L')
                grayscale.save(img_buffer, format='JPEG', quality=50, optimize=True)
            else:
                enhanced_image.save(img_buffer, format='JPEG', quality=85, optimize=True)
            
            img_bytes = img_buffer.getvalue()
            
            # Sauvegarder le screenshot pour débogage
            debug_filename = f"debug_screenshot_{self.api_calls+1}.jpg"
            with open(debug_filename, 'wb') as f:
                f.write(img_bytes)
            print(f"Screenshot sauvé: {debug_filename}")
            
            # Encoder en base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            original_size = len(base64.b64encode(rgb_image.tobytes()).decode('utf-8'))
            optimized_size_bytes = len(img_base64)
            compression_ratio = (1 - optimized_size_bytes / original_size) * 100
            
            print(f"Image optimisée: {compression_ratio:.1f}% plus petite ({optimized_size_bytes//1000}KB)")
            
            # CALCUL FACTEUR D'ÉCHELLE pour conversion pixels screenshot → pixels jeu
            original_game_height = height * 0.85  # Hauteur après crop (85% de l'original)
            screenshot_height = optimized_size[1]  # Hauteur du screenshot envoyé à Claude
            scale_factor = original_game_height / screenshot_height
            
            print(f"📐 Facteur d'échelle: 1px screenshot = {scale_factor:.2f}px jeu")
            
            # Stocker le facteur d'échelle pour utilisation dans les prompts
            self.current_scale_factor = scale_factor
            self.screenshot_dimensions = optimized_size
            self.original_game_dimensions = (width, int(original_game_height))
            
            # Mettre à jour le convertisseur de distances
            self.distance_converter.update_scale_factor(scale_factor)
            
            return img_base64
            
        except Exception as e:
            print(f"❌ Erreur capture screenshot: {e}")
            return None
    
    def _compute_mystery_mask(self, raw_array):
        """Détecter les mystery blocks par correspondance EXACTE de couleur NES.
        RGB(68, 160, 252) est une entrée fixe de la palette hardware NES :
        elle est identique à chaque frame, indépendante de l'animation du sprite."""
        # Correspondance pixel-exact : même NES palette entry → même RGB, toujours
        mystery_color = np.array([68, 160, 252], dtype=np.uint8)
        mask = np.all(raw_array == mystery_color, axis=2)
        if np.any(mask):
            print(f"🟣 Mystery blocks détectés ({np.sum(mask)} pixels)")
        return mask

    def apply_detection_filters(self, image):
        """Appliquer des filtres pour améliorer la détection de Claude.
        NOTE: le CLAHE a été supprimé — il écrasait les couleurs vives NES (orange-rouge 248,56,0
        → sombre ~67,0,0) rendant sprites et sol indiscernables. Les couleurs NES natives
        ont déjà un contraste naturel élevé (ciel bleu foncé vs sprites orange-rouge)."""
        try:
            img_array = np.array(image)

            # Masque mystery blocks calculé sur pixels NES originaux
            mystery_mask = getattr(self, '_precomputed_mystery_mask', None)
            if mystery_mask is None or mystery_mask.shape[:2] != img_array.shape[:2]:
                mystery_mask = self._compute_mystery_mask(img_array)  # fallback

            # Netteté standard : kernel [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]] sans scale
            # → zones uniformes préservées (9p-8p=p), bords rehaussés
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
            sharpened = cv2.filter2D(img_array.astype(np.float32), -1, kernel)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

            # Appliquer couleur dédiée MAGENTA sur les mystery blocks
            if np.any(mystery_mask):
                sharpened[mystery_mask] = [255, 0, 255]

            return Image.fromarray(sharpened)

        except Exception as e:
            print(f"⚠️ Erreur filtres, image originale utilisée: {e}")
            return image

    def _annotate_entities_for_llm(self, screenshot_pil, obs):
        """Peint les sprites ennemis en JAUNE VIF en lisant l'OAM NES directement.
        L'OAM (Object Attribute Memory) contient les coordonnées ÉCRAN de chaque sprite.
        Palette 0 = Mario, Palette 3 = ennemis → distinction fiable indépendante des couleurs.
        """
        try:
            img = np.array(screenshot_pil)
            img_h, img_w = img.shape[:2]
            obs_h = obs.shape[0]  # 240

            crop_top = int(obs_h * 0.15)       # 36px supprimés en haut (HUD)
            scale_y  = img_h / (obs_h - crop_top)  # ≈ 0.941

            # Lire le buffer OAM depuis la RAM NES (0x0200-0x027F, 64 sprites × 4 bytes)
            # Format OAM: [Y_écran, tile_num, attributs, X_écran]
            # Attribut bits 0-1 = index palette: 0=Mario, 1=HUD/objets, 2=?, 3=ennemis
            try:
                ram = self.env.unwrapped.ram
            except AttributeError:
                return screenshot_pil

            OAM_BASE   = 0x0200
            HUD_Y_MAX  = 32   # ignorer sprites HUD dans la barre du haut (y<32)
            ENEMY_PAL  = 0x03 # palette 3 = ennemis NES (Goombas, Koopas, etc.)

            # 1er passage : collecter tous les sprites ennemis
            raw_sprites = []
            for i in range(64):
                y_nes = int(ram[OAM_BASE + i*4 + 0])
                attr  = int(ram[OAM_BASE + i*4 + 2])
                x_nes = int(ram[OAM_BASE + i*4 + 3])
                if y_nes < HUD_Y_MAX or y_nes >= 240:
                    continue
                if (attr & 0x03) != ENEMY_PAL:
                    continue
                raw_sprites.append((x_nes, y_nes))

            # 2e passage : grouper les sprites proches en une seule entité
            # (un Goomba = 6 sprites OAM 8×8 en 2 col × 3 rangées)
            entities = []  # liste de (x_min, x_max, y_min, y_max) en NES coords
            for (sx, sy) in raw_sprites:
                merged = False
                for e in entities:
                    # Fusionner si le sprite est à ≤16px x/y d'une entité existante
                    if sx <= e[1] + 16 and sx + 8 >= e[0] - 16 and sy <= e[3] + 16 and sy + 8 >= e[2] - 16:
                        e[0] = min(e[0], sx)
                        e[1] = max(e[1], sx + 8)
                        e[2] = min(e[2], sy)
                        e[3] = max(e[3], sy + 8)
                        merged = True
                        break
                if not merged:
                    entities.append([sx, sx + 8, sy, sy + 8])

            # 3e passage : peindre la bounding box de chaque entité en JAUNE VIF
            n_painted = 0
            for (x1_nes, x2_nes, y1_nes, y2_nes) in entities:
                # Si l'entité dépasse 16px de haut (3 rangées OAM), sauter la rangée
                # du haut (souvent une tuile auxiliaire vide, tile=0xFC) pour garder
                # une taille correcte de ~16px (2 rangées = sprite standard Goomba)
                h_nes = y2_nes - y1_nes
                y1_draw = y1_nes + 8 if h_nes > 16 else y1_nes
                sy1 = int((y1_draw - crop_top) * scale_y)
                sy2 = int((y2_nes  - crop_top) * scale_y)
                if sy2 < 0 or sy1 >= img_h:
                    continue
                y1 = max(0, sy1)
                y2 = min(img_h, sy2)
                x1 = max(0, x1_nes)
                x2 = min(img_w, x2_nes)
                img[y1:y2, x1:x2] = [255, 255, 0]
                n_painted += 1

            if n_painted > 0:
                print(f"🎨 OAM: {n_painted} ennemi(s) peints en jaune ({len(raw_sprites)} sprites groupés)")
            else:
                print(f"🔍 OAM: aucun sprite ennemi (palette 3) visible à l'écran")

            return Image.fromarray(img)

        except Exception as e:
            print(f"⚠️ OAM annotation erreur: {e}")
            return screenshot_pil

    def analyze_screenshot_with_claude(self, screenshot_b64, situation, step_count):
        """Envoyer le screenshot à Claude pour analyse visuelle"""
        
        try:
            mario = situation['mario']
            progress = situation['progress']
            
            # Calculer la vitesse de Mario
            mario_speed = progress['trend'] / 30 if progress['trend'] != 0 else 0  # pixels par step
            
            # Estimer les distances des éléments visibles
            mario_x, mario_y = mario['x'], mario['y']
            screen_width = 256  # Largeur écran NES
            
            # Bloc correction rewind : injecté en tête si Mario vient d'être rembobiné
            _rewind_block = ""
            if self._rewind_correction_msg:
                _rewind_block = self._rewind_correction_msg + "\n\n"
                self._rewind_correction_msg = None  # Consommé, ne pas répéter

            prompt = _rewind_block + f"""Tu es Claude, expert Mario Bros ! Analyse cette capture d'écran du jeu en temps réel.

CONTEXTE DÉTAILLÉ MARIO:
🔹 Position monde: X={mario_x}px (coord NES absolue depuis début niveau), Y={mario_y}px
🔹 Ecran NES: {screen_width}x240px — Mario apparaît visuellement vers le tiers gauche de l'écran (~80px) quand la caméra scroll
🔹 Vitesse: {mario_speed:.1f} pixels/step ({'vers droite' if mario_speed > 0 else 'stationnaire' if mario_speed == 0 else 'vers gauche'})
🔹 Score: {mario['score']} | Step: {step_count}
🔹 Progression: {progress['status']} (tendance: {progress['trend']}px sur 30 steps)
🔹 Morts: {self.deaths_count} | Vies utilisées: {self.lives_used}

📐 ÉCHELLE SCREENSHOT → JEU:
🔹 Dimensions screenshot: {self.screenshot_dimensions[0]}x{self.screenshot_dimensions[1]}px
🔹 LARGEUR: 1:1 exact — 1px horizontal sur screenshot = 1px NES. Mario apparaît ~{self.screenshot_dimensions[0]//3}px depuis la gauche.
🔹 Pour "px" dans run_forward: mesure directement en pixels horizontaux sur le screenshot, PAS de facteur à appliquer.

📚 HISTORIQUE D'APPRENTISSAGE - APPRENDS DE TES ERREURS:
{self.get_learning_context()}

{self.segment_memory.get_context_for_position(mario_x)}
{self._get_phase1_optimization_hint(mario_x)}{self._get_phase3_frontier_context(mario_x)}
🗺️ INFORMATIONS SPÉCIFIQUES DU NIVEAU ACTUEL:
{self.get_level_specific_context()}

VITESSES DE RÉFÉRENCE (pour calculs de timing):
- Goomba: ~0.5 pixels/step vers la gauche
- Mario marche: ~1-2 pixels/step vers droite  
- Mario court: ~3-4 pixels/step vers droite
- Collision dans ~{abs(mario_x-200)//2:.0f} steps si ennemi à droite et vitesses normales

🔍 ÉVALUATION DE SÉCURITÉ PRIORITAIRE:
Regarde attentivement cette image et identifie EN PRIORITÉ ABSOLUE:

1. ENNEMIS ET DANGERS MORTELS: 
   - QUELS ennemis vois-tu (Goombas bruns, Koopas verts)?
   - DISTANCE CRITIQUE: À quelle distance EXACTE de Mario en pixels horizontaux du screenshot (largeur 1:1 NES). Seuils: <15px = DANGER IMMÉDIAT, 15-30px = ATTENTION, >30px = SÛR)?
   - DIRECTION ET VITESSE: L'ennemi se déplace-t-il VERS Mario (DANGER) ou s'éloigne-t-il (SÛR)?
   - CALCUL COLLISION: Combien de steps avant collision si Mario continue à droite?
   - ÉCHAPPATOIRES: Y a-t-il des plateformes/tuyaux pour fuir?

2. BLOCS QUESTION MARKS SPATIAUX:
   - Y a-t-il des blocs ? (carrés bleus) visibles?
   - Sont-ils AU-DESSUS de Mario, à sa GAUCHE, à sa DROITE?
   - Mario est-il déjà SOUS un bloc ou doit-il se DÉPLACER?
   - Distance approximative pour atteindre le bloc le plus proche?

3. POSITIONNEMENT SPATIAL MARIO:
   - Mario est-il au sol, en l'air, en train de sauter?
   - Y a-t-il des obstacles (trous, tuyaux) devant lui?
   - TUYAUX HAUTS (World 1-1): Utilise 'pipe_jump' (séquence automatique : approche 40f + saut max 40f)
   - OBSTACLES LARGES/PLATEFORMES: Utilise 'obstacle_jump' (élan 20f + saut max 40f)
   - BLOCAGE: Si Mario semble coincé (vitesse négative), utilise 'step_back' puis 'pipe_jump'
   - Espace libre devant Mario pour avancer?

ACTIONS MARIO DISPONIBLES:
{chr(10).join([f"'{key}': {action['description']}" for key, action in self.macro_actions.items()])}

🔥 RÈGLES ABSOLUES (dans cet ordre):
1. SURVIE IMMÉDIATE: Ennemi <50px → 'stomp_enemy' (saut avec élan, ~50px de portée). Ennemi 50-100px → 'run_forward' pour approcher PUIS stomp quand < 50px. Goombas bougent vers Mario — anticipe.
   ❌ INTERDIT: 'walk_right'/'run_forward' directement sur un ennemi < 50px (Mario se ferait tuer).
   ❌ INTERDIT: stomp_enemy si ennemi > 80px (saut trop court, n'atteindra pas l'ennemi).
2. BLOCS ? (si zone sûre, ennemi >50px): 'approach_and_hit_block', 'hit_block' sous le bloc, ou 'hop_on_platform'+'hit_block'. Abandonne si ennemi s'approche < 50px.
3. PROGRESSION: 'run_forward' si voie libre.

🦘 3 TYPES DE SAUT — UTILISE LE BON:
  1. 'stomp_enemy'     → TUER un ennemi  (right+A, 12 frames, portée ~40px)
  2. 'pipe_jump'       → FRANCHIR un tuyau haut / obstacle fixe
                         (séquence auto : marche 40f jusqu'au pied + saut max 40f depuis la base)
                         ✅ TOUJOURS utiliser pour le premier tuyau de World 1-1 (x≈174)
  3. 'obstacle_jump'   → FRANCHIR obstacles larges / plateformes élevées avec élan
                         (course 20f + saut max 40f)

⚠️ PHYSIQUE NES — INTERDICTIONS:
  ❌ NE JAMAIS sauter avant d'atteindre le pied du tuyau (Mario se coince à mi-hauteur)
  ❌ NE PAS utiliser 'max_jump' seul sans approche préalable (trop loin du tuyau)
  → 'pipe_jump' gère tout ça automatiquement

- TUYAU COURT: utilise 'jump_on_pipe' ou 'hop_on_platform'
- TUYAU (entrée zone bonus): 'pipe_down'
- PLATEFORME + ENNEMI: 'wait_for_enemy' puis 'pipe_jump' quand ennemi passé

📐 DISTANCES OBLIGATOIRES — format JSON:
- run_forward DOIT avoir "px" = pixels à parcourir (≈2px/frame). MAX 60px par action (ex: zone libre → 3x run_forward px=60).
  Exemples: ennemi à 80px → px=55 (s'arrêter à 25px, réflexe gère le stomp)
            tuyau à 120px → px=90 (s'arrêter devant), puis pipe_jump
- max_jump DOIT avoir "px" = pixels d'approche avant le saut quand le trou n'est pas immédiat.
  Formule: px = distance_trou - 20 (ex: trou à 85px → px=65, trou à 30px → px=10, trou à 15px → sans px).
  Sans px : saute immédiatement depuis position actuelle (seulement si trou < 20px).
- Autres sauts (pipe_jump, stomp_enemy, obstacle_jump, run_jump_over) : PAS de px (durée fixe)

Donne 3-5 actions. ⚠️ RÉPONDS EN JSON UNIQUEMENT — ZÉRO TEXTE, ZÉRO EXPLICATION, ZÉRO COMMENTAIRE:
{{"actions":[{{"macro_action":"run_forward","px":60}},{{"macro_action":"pipe_jump"}}],"urgency":<1-10>}}"""

            self.api_calls += 1
            print(f"📸 Envoi screenshot à Claude (appel #{self.api_calls})...")
            
            # Logger le prompt avec le screenshot associé
            self.logger.log_claude_prompt("SCREENSHOT", prompt, step_count,
                                          screenshot_b64=screenshot_b64)
            
            print("="*80)
            print("🔍 PROMPT ENVOYÉ À CLAUDE:")
            print(prompt)
            print("="*80)
            
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=400,  # Augmenté pour permettre des réponses JSON complètes avec plusieurs actions
                temperature=0.1,
                system="Tu es un contrôleur de jeu Mario. Réponds UNIQUEMENT en JSON valide, sans aucun texte avant ou après. Aucune explication, aucun commentaire.",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",  # Changé de PNG vers JPEG
                            "data": screenshot_b64
                        }}
                    ]
                }]
            )
            
            response_text = response.content[0].text
            print(f"✅ Claude analyse reçue ({len(response_text)} chars)")
            
            # Calculer le coût d'abord
            image_cost = min(0.01, len(screenshot_b64) * 0.000001)  # Coût proportionnel à la taille
            text_cost = len(prompt) * 0.25 / 1000000 + len(response_text) * 1.25 / 1000000
            cost = text_cost + image_cost
            
            # Logger la réponse avec coût calculé
            self.logger.log_claude_response(response_text, step_count, cost)
            
            print("="*80)
            print("💭 RÉPONSE DE CLAUDE:")
            print(response_text)
            print("="*80)
            
            self.total_cost += cost
            self.screenshot_costs += image_cost
            
            # Ajuster la fréquence si on dépasse le budget
            if self.screenshot_costs > self.screenshot_cost_limit:
                self.screenshot_frequency = min(100, self.screenshot_frequency + 10)  # Réduire la fréquence
                print(f"💰 Budget screenshots dépassé, fréquence réduite à {self.screenshot_frequency} steps")
            
            print(f"💰 Coût screenshot: ${image_cost:.4f} (total screenshots: ${self.screenshot_costs:.3f})")
            
            # Convertir les distances dans la réponse
            converted_response = self.distance_converter.process_claude_response(response_text)
            converted_response = self.distance_converter.add_scale_info_to_response(converted_response)
            
            print("🔧 RÉPONSE AVEC DISTANCES CONVERTIES:")
            print(converted_response)
            print("="*80)
            
            # Ajouter la réponse à l'historique pour l'encart
            self.add_llm_response("SCREENSHOT", converted_response, step_count)
            
            return converted_response
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Erreur analyse screenshot: {e}")
            print(f"🔍 Détails de l'erreur:")
            print(error_details)
            
            # Logger l'erreur avec détails
            self.logger.log_error("CLAUDE_API_FAILURE", f"{str(e)} | Traceback: {error_details}", step_count)
            
            return None
    
    def create_claude_prompt(self, situation):
        """Créer un prompt pour que Claude choisisse une macro-action"""

        mario = situation['mario']
        progress = situation['progress']
        screen = situation['screen']
        history = situation['history']
        enemy_tracking = situation.get('enemy_tracking', [])

        # Construire la liste des macro-actions
        macro_list = []
        for key, macro in self.macro_actions.items():
            macro_list.append(f"'{key}': {macro['description']} ({macro['duration']} frames)")

        # 🎯 SYSTÈME ANTI-MORT : Construire l'analyse des ennemis en mouvement
        enemy_movement_section = ""
        if enemy_tracking:
            enemy_movement_section = "\n🚨 ENNEMIS EN MOUVEMENT (TRACKING TEMPS RÉEL):"
            for enemy in enemy_tracking:
                enemy_movement_section += f"\n  • {enemy['type']} à {enemy['distance_to_mario']}px: {enemy['direction']} (vitesse={enemy['speed']}px/frame)"
                enemy_movement_section += f"\n    └─ {enemy['danger_level']} | URGENCE: {enemy['urgency']}/10 | ACTION: {enemy['recommended_action']}"

            # Ajouter un résumé de l'urgence maximale
            max_urgency = max([e.get('urgency', 0) for e in enemy_tracking])
            if max_urgency >= 8:
                enemy_movement_section += f"\n\n⚠️ ALERTE MAXIMALE: Urgence {max_urgency}/10 détectée!"
        else:
            enemy_movement_section = "\n✅ Aucun ennemi détecté dans la zone visible"

        # ⛰️ Section trou pour le prompt
        hole_info = situation.get('holes', screen.get('holes', {}))
        if hole_info.get('detected'):
            _h_near = hole_info['nearest']
            _h_w    = hole_info['width']
            # Calcul de l'approche optimale : s'arrêter ~20px avant le bord pour avoir de l'élan
            _approach_px = max(0, _h_near - 20)
            if hole_info.get('critical'):
                if _approach_px <= 5:
                    _hole_json = f'{{"macro_action":"max_jump"}}'
                    _hole_hint = "saut immédiat sans approche"
                else:
                    _hole_json = f'{{"macro_action":"max_jump","px":{_approach_px}}}'
                    _hole_hint = f"avance {_approach_px}px puis saute"
                hole_section = (f"\n⛰️ TROU CRITIQUE: sol absent à {_h_near}px (largeur {_h_w}px)"
                                f"\n→ {_hole_hint}: {{{_hole_json}}}")
            elif hole_info.get('urgent'):
                _hole_json = f'{{"macro_action":"max_jump","px":{_approach_px}}}'
                hole_section = (f"\n⛰️ TROU URGENT à {_h_near}px (largeur {_h_w}px)"
                                f"\n→ avance {_approach_px}px puis saute max: {{{_hole_json}}}")
            else:
                _approach_px2 = max(0, _h_near - 30)
                _hole_json = f'{{"macro_action":"max_jump","px":{_approach_px2}}}'
                hole_section = (f"\n⛰️ TROU DÉTECTÉ à {_h_near}px (largeur {_h_w}px)"
                                f"\n→ prépare: run_forward px={_approach_px2-40} puis {_hole_json}")
        else:
            hole_section = ""

        # Bloc correction rewind (chemin texte-seul, même logique que screenshot)
        _rewind_block = ""
        if self._rewind_correction_msg:
            _rewind_block = self._rewind_correction_msg + "\n\n"
            self._rewind_correction_msg = None  # Consommé

        prompt = _rewind_block + f"""🍄 Tu es Claude, EXPERT MARIO BROS NES ! Mario a besoin de 2-3 actions RAPIDES car le jeu est dangereux !

📍 SITUATION MARIO:
• Position: X={mario['x']}, Y={mario['y']} | Score: {mario['score']} | Step: {situation['step']}
• Progression: {progress['status']} (tendance: {progress['trend']}px)

🔍 ANALYSE VISUELLE MARIO BROS:
• Obstacles: {'OUI' if screen['immediate_obstacles'] else 'NON'} | Sol stable: {'OUI' if screen['ground_stable'] else 'NON'}
• Blocs ?: {'OUI - FRAPPE-LES!' if screen.get('question_blocks') else 'NON'}
• Power-ups: {'OUI - RÉCUPÈRE!' if screen.get('power_ups') else 'NON'}
• Trous devant: {'⛰️ OUI - ' + str(hole_info['nearest']) + 'px' if hole_info.get('detected') else 'NON'}
• Environnement: {screen['environment_type']}
{enemy_movement_section}{hole_section}

🗺️ CARTE SPATIALE (distances/directions):
{chr(10).join(screen.get('spatial_map', ['Aucun élément détecté'])[:5])}

🎮 ACTIONS MARIO BROS DISPONIBLES:
{chr(10).join(macro_list)}

🗺️ CONTEXTE SPÉCIFIQUE AU NIVEAU:
{self.get_level_specific_context()}

🧠 STRATÉGIES MARIO BROS GÉNÉRALES:
• Stomp enemies (Goomba/Koopa) en sautant dessus pour les tuer et gagner points
• Hit blocks ? par dessous pour obtenir coins/power-ups/champignons
• Collect power-ups pour devenir Super Mario ou Fire Mario
• Avoid Piranha Plants en attendant qu'elles rentrent ou en courant vite
• Use pipes pour accéder aux zones bonus souterraines
• Kick shells de Koopa pour tuer autres ennemis

🚨 RÈGLES DE SURVIE CRITIQUES (ANTI-MORT):
1. 🏃 TOUJOURS run_forward par défaut — ne JAMAIS utiliser 'wait' (les ennemis avancent vers toi pendant que tu attends!)
2. ❌ JAMAIS 'step_back' sauf si Mario est BLOQUÉ contre un mur (vitesse négative détectée)
3. ✅ Ennemi < 50px → stomp_enemy IMMÉDIATEMENT (le saut avec élan peut l'atteindre)
   ✅ Ennemi 50-100px → run_forward D'ABORD pour approcher, le réflexe auto gère le saut à < 55px
   ❌ NE JAMAIS mettre run_forward après stomp_enemy (si le saut rate, Mario court vers l'ennemi!)
4. ✅ Si ennemi "S'ÉLOIGNE" → run_forward pour le rattraper et le stomp (ou passer s'il est trop loin)
5. ⚡ EN CAS DE DOUTE → run_forward pour approcher, puis stomp_enemy quand < 50px
6. 🎯 Priorité absolue: SURVIE > Collecte de blocs/items
7. ⛰️ TROU DÉTECTÉ → max_jump IMMÉDIATEMENT, jamais run_forward vers un trou!
8. 🎮 Séquence correcte: run_forward (approcher) → stomp_enemy (< 50px) → stomp_enemy si autre ennemi, sinon run_forward SEULEMENT si zone dégagée
9. ⚠️ APRÈS stomp_enemy : NE PAS mettre run_forward si ennemi encore visible ! Mario atterrit et court dedans.

🎯 DONNE 2-3 ACTIONS ADAPTÉES À LA SITUATION!

📐 run_forward "px" ≤ 60 (≈2px/frame). max_jump accepte "px" = approche avant saut.
⚠️ JSON UNIQUEMENT — ZÉRO TEXTE, ZÉRO EXPLICATION:
{{"actions":[{{"macro_action":"run_forward","px":60}},{{"macro_action":"<saut_ou_action>"}}],"urgency":<1-10>}}

Exemples avec distances réelles:
Ennemi à 30px: {{"actions":[{{"macro_action":"stomp_enemy"}}],"urgency":10}}
Ennemi à 80px: {{"actions":[{{"macro_action":"run_forward","px":55}},{{"macro_action":"stomp_enemy"}}],"urgency":8}}
Tuyau à 100px: {{"actions":[{{"macro_action":"run_forward","px":70}},{{"macro_action":"pipe_jump"}}],"urgency":6}}
Zone libre 200px: {{"actions":[{{"macro_action":"run_forward","px":200}}],"urgency":4}}
Trou à 85px: {{"actions":[{{"macro_action":"max_jump","px":65}}],"urgency":10}}
Trou à 20px: {{"actions":[{{"macro_action":"max_jump"}}],"urgency":10}}
Après obstacle: {{"actions":[{{"macro_action":"run_forward","px":100}}],"urgency":5}}"""

        return prompt
    
    def detect_block_hit_success(self, obs_before, obs_after):
        """Détecter si Mario a réussi à frapper un bloc (changement visuel)"""
        try:
            # Comparer les images avant/après pour détecter des changements
            diff = cv2.absdiff(obs_before, obs_after)
            # Si il y a des changements significatifs dans la zone supérieure (blocs), c'est probablement réussi
            upper_region = diff[:120, :]  # Zone supérieure de l'écran
            total_diff = np.sum(diff)
            upper_diff = np.sum(upper_region)
            
            # Si les changements sont concentrés dans la zone supérieure, c'est probablement un bloc frappé
            if upper_diff > total_diff * 0.3:
                return True
            return False
        except:
            return False
    
    def call_claude_for_macro(self, prompt):
        """Demander à Claude quelle macro-action utiliser"""
        
        try:
            self.api_calls += 1
            print(f"🧠 Claude réfléchit... (appel #{self.api_calls})")
            
            # Logger le prompt
            self.logger.log_claude_prompt("TEXT", prompt, 0)
            
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,  # Suffisant pour 4-6 actions JSON sans reasoning
                temperature=0.1,  # Plus déterministe
                system="Tu es un contrôleur de jeu Mario. Réponds UNIQUEMENT en JSON valide, sans aucun texte avant ou après. Aucune explication, aucun commentaire.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            
            # Coût estimé
            cost = len(prompt) * 0.25 / 1000000 + len(response_text) * 1.25 / 1000000
            self.total_cost += cost
            
            # Logger la réponse
            self.logger.log_claude_response(response_text, 0, cost)
            
            return response_text
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Erreur Claude: {e}")
            print(f"🔍 Détails de l'erreur:")
            print(error_details)
            
            # Logger l'erreur avec détails
            self.logger.log_error("CLAUDE_MACRO_API_FAILURE", f"{str(e)} | Traceback: {error_details}", 0)
            
            return None
    
    def call_claude_async(self, situation, obs=None, step_count=0):
        """Appeler Claude en arrière-plan avec système hybride optimisé"""

        def claude_worker(generation=self._claude_generation):
            try:
                # 🧠 DÉCISION HYBRIDE: Screenshot complet vs Mise à jour positionnelle
                use_screenshot, reason = self.should_use_screenshot_vs_positions(step_count)
                
                print(f"🤖 Mode hybride: {'📸 Screenshot' if use_screenshot else '📍 Positions'} - {reason}")
                
                if use_screenshot:
                    # MODE SCREENSHOT COMPLET (établir contexte ou recalibrage)
                    if obs is not None:
                        screenshot = self.capture_game_screenshot(obs)
                        if screenshot:
                            claude_response = self.analyze_screenshot_with_claude(screenshot, situation, step_count)
                            self.last_screenshot_step = step_count
                            self.last_screenshot_x = situation.get('mario', {}).get('x', self.last_screenshot_x)
                            self.last_positions_update = step_count  # sync pour cooldown trigger
                            # Marquer le contexte comme établi après le premier screenshot
                            if not self.level_context_established:
                                self.level_context_established = True
                                print("✅ Contexte du niveau établi pour Claude!")
                        else:
                            # Fallback textuel
                            prompt = self.create_claude_prompt(situation)
                            claude_response = self.call_claude_for_macro(prompt)
                    else:
                        # Pas d'observation, fallback textuel
                        prompt = self.create_claude_prompt(situation)
                        claude_response = self.call_claude_for_macro(prompt)
                    
                    analysis_type = "🖼️ Visuelle"
                    
                else:
                    # MODE MISE À JOUR POSITIONNELLE (plus fréquent, moins cher)
                    if obs is not None:
                        # Extraire les positions précises
                        info = {'x_pos': situation.get('mario', {}).get('x', 0), 
                               'y_pos': situation.get('mario', {}).get('y', 0)}
                        positions_data = self.extract_precise_positions(obs, info, step_count)
                        
                        if positions_data:
                            # Mettre à jour les positions trackées
                            self.last_positions_update = step_count
                            claude_response = self.call_claude_for_positions_update(positions_data, step_count)
                        else:
                            # Fallback textuel si extraction échoue
                            prompt = self.create_claude_prompt(situation)
                            claude_response = self.call_claude_for_macro(prompt)
                    else:
                        # Pas d'observation, fallback textuel
                        prompt = self.create_claude_prompt(situation)
                        claude_response = self.call_claude_for_macro(prompt)
                    
                    analysis_type = "📍 Positionnelle"
                
                # Parser les actions (même format JSON pour les deux modes)
                actions = self.parse_claude_actions(claude_response)
                
                # Ajouter les actions à la queue
                # Vérifier que le rewind n'a pas eu lieu pendant l'appel API.
                # Si la génération a changé, les actions sont périmées → les ignorer.
                if generation != self._claude_generation:
                    print(f"⚠️ Thread Claude périmé (gen {generation} → {self._claude_generation}), actions ignorées")
                    return

                # Règles de filtrage :
                #   'wait'        → 'run_forward'
                #   'step_back'   → 'run_forward' si Mario avance
                #   'stomp_enemy' → 'run_forward' : le réflexe pixel gère le stomp automatiquement
                #                   au bon moment (<55px). Claude ne doit PAS planifier les stomps.
                _mario_speed = (self.last_situation or {}).get('mario', {}).get('speed', 1)
                for action in actions:
                    if len(self.action_queue) < 5:  # Éviter l'overflow (max 4 actions planifiées)
                        mname = action.get('macro_name', '')
                        if mname == 'wait':
                            action = dict(action,
                                          macro_name='run_forward',
                                          reasoning='[wait→run] Courir plutôt qu\'attendre')
                        elif mname == 'step_back' and _mario_speed >= 0:
                            action = dict(action,
                                          macro_name='run_forward',
                                          reasoning='[step_back→run] Avancer plutôt que reculer')
                        elif mname == 'stomp_enemy':
                            # stomp_enemy → run_jump_over : saut par-dessus l'ennemi.
                            # run_jump_over (right+A+B, 35f, ~140px) est plus fiable que
                            # stomp_enemy (right+A, 20f, ~55px) car couvre plus de distance.
                            action = dict(action,
                                          macro_name='run_jump_over',
                                          reasoning='[stomp→jump] Saut par-dessus l\'ennemi')
                        self.action_queue.append(action)
                
                print(f"✅ Claude ({analysis_type}) a fourni {len(actions)} actions")
                
            except Exception as e:
                print(f"❌ Erreur thread Claude hybride: {e}")
                # Action de secours
                fallback = self.get_fallback_macro()
                self.action_queue.append(fallback)
            
            finally:
                self.claude_thinking = False
        
        if not self.claude_thinking:
            self.claude_thinking = True
            self.claude_thread = threading.Thread(target=claude_worker)
            self.claude_thread.daemon = True
            self.claude_thread.start()
    
    def parse_claude_actions(self, response_text):
        """Parser la réponse de Claude pour extraire plusieurs actions"""
        
        if not response_text:
            return [self.get_fallback_macro()]
        
        try:
            # Extraire le JSON outermost en suivant le niveau de nesting
            data = None
            start = response_text.find('{')
            if start != -1:
                depth = 0
                end = start
                for i, ch in enumerate(response_text[start:], start):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                json_text = response_text[start:end + 1]
                json_text = self.fix_broken_json(json_text)
                try:
                    data = json.loads(json_text)
                except json.JSONDecodeError:
                    pass
            
            if data:
                actions_list = []
                actions_data = data.get('actions', [])
                strategy = data.get('strategy', 'Stratégie Claude')
                urgency = int(data.get('urgency', 5))
                
                for action_data in actions_data:
                    if isinstance(action_data, dict):
                        macro_name = action_data.get('macro_action', 'walk_right')
                        reasoning = action_data.get('reasoning', 'Action Claude')
                        px = action_data.get('px')  # distance en pixels (optionnel)

                        # Valider la macro-action
                        if macro_name not in self.macro_actions:
                            print(f"⚠️ Macro inconnue '{macro_name}', utilisation de walk_right")
                            macro_name = 'walk_right'

                        action_dict = {
                            'macro_name': macro_name,
                            'reasoning': reasoning,
                            'strategy': strategy,
                            'urgency': urgency,
                            'confidence': 80
                        }
                        if px is not None:
                            try:
                                action_dict['px'] = int(px)
                            except (ValueError, TypeError):
                                pass
                        actions_list.append(action_dict)
                
                if actions_list:
                    visual_analysis = data.get('visual_analysis', '')
                    if visual_analysis:
                        print(f"👁️ Claude voit: {visual_analysis[:80]}...")
                    print(f"📋 Claude: {len(actions_list)} actions - {strategy}")
                    return actions_list
            
            # Fallback : parser le texte pour deviner des actions
            return [self.parse_text_for_single_macro(response_text)]
                
        except Exception as e:
            print(f"⚠️ Erreur parsing: {e}")
            return [self.get_fallback_macro()]
    
    def fix_broken_json(self, json_text):
        """Réparer un JSON cassé/tronqué"""
        
        # Compter les accolades et crochets
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        open_brackets = json_text.count('[')
        close_brackets = json_text.count(']')
        
        # Fermer les structures ouvertes
        if open_brackets > close_brackets:
            json_text += ']' * (open_brackets - close_brackets)
        
        if open_braces > close_braces:
            json_text += '}' * (open_braces - close_braces)
        
        # Réparer les guillemets non fermés
        if json_text.count('"') % 2 != 0:
            json_text += '"'
        
        return json_text
    
    def parse_claude_macro(self, response_text):
        """Parser la réponse de Claude pour extraire la macro-action"""
        
        if not response_text:
            return self.get_fallback_macro()
        
        try:
            # Chercher du JSON
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                macro_name = data.get('macro_action', 'walk_right')
                reasoning = data.get('reasoning', 'Raisonnement non fourni')
                outcome = data.get('expected_outcome', 'Résultat non spécifié')
                confidence = int(data.get('confidence', 50))
                
                # Valider la macro-action
                if macro_name not in self.macro_actions:
                    print(f"⚠️ Macro inconnue '{macro_name}', utilisation de walk_right")
                    macro_name = 'walk_right'
                
                return {
                    'macro_name': macro_name,
                    'reasoning': reasoning,
                    'expected_outcome': outcome,
                    'confidence': confidence
                }
            
            else:
                # Parser le texte
                return self.parse_text_for_macro(response_text)
                
        except json.JSONDecodeError:
            return self.parse_text_for_macro(response_text)
    
    def parse_text_for_single_macro(self, text):
        """Parser une réponse textuelle pour deviner une seule macro"""
        
        text_lower = text.lower()
        
        if 'long' in text_lower and 'jump' in text_lower:
            macro = 'long_jump'
        elif 'short' in text_lower and 'jump' in text_lower:
            macro = 'short_jump'
        elif 'high' in text_lower and 'jump' in text_lower:
            macro = 'high_jump'
        elif 'precise' in text_lower and 'jump' in text_lower:
            macro = 'precise_jump'
        elif 'run' in text_lower:
            macro = 'run_forward'
        elif 'jump' in text_lower:
            macro = 'short_jump'
        elif 'wait' in text_lower:
            macro = 'wait'
        elif 'back' in text_lower:
            macro = 'step_back'
        else:
            macro = 'walk_right'
        
        return {
            'macro_name': macro,
            'reasoning': f"Interprétation de: {text[:50]}...",
            'strategy': 'Analyse textuelle',
            'urgency': 5,
            'confidence': 60
        }
    
    def parse_text_for_macro(self, text):
        """Parser une réponse textuelle pour deviner la macro (compatibilité)"""
        return self.parse_text_for_single_macro(text)
    
    def get_contextual_fallback_action(self):
        """Action de fallback basée sur le dernier état connu - évite de courir aveuglément"""
        # Avant que Claude ait analysé le niveau : marcher prudemment, pas courir
        if not self.level_context_established:
            return 1  # WALK - on ne sait pas encore où sont les ennemis

        if self.last_situation is None:
            return 1  # WALK par sécurité

        enemy_tracking = self.last_situation.get('enemy_tracking', [])

        if not enemy_tracking:
            return 3  # Aucun ennemi connu → courir librement

        # Chercher l'ennemi le plus dangereux
        most_urgent = max(enemy_tracking, key=lambda e: e.get('urgency', 0))
        urgency = most_urgent.get('urgency', 0)
        pattern = most_urgent.get('movement_pattern', 'INCONNU')
        distance = most_urgent.get('distance_to_mario', 999)

        if pattern == 'DANGER_FRONTAL' and distance < 100:
            # Ennemi en face et proche → sauter dessus
            return 2  # JUMP (stomp_enemy)
        elif pattern == 'DANGER_ARRIERE' and distance < 40:
            # Ennemi par derrière et proche → fuir en courant
            return 3  # RUN
        elif pattern == 'STATIONNAIRE' and distance < 60:
            # Ennemi immobile proche → sauter par-dessus
            return 4  # RUN_JUMP
        else:
            # Zone libre → courir
            return 3  # RUN

    def check_immediate_threat(self, obs):
        """⚡ COUCHE RÉFLEXE OAM : détection ennemis via RAM NES (OAM sprites).
        Les Goombas sont RGB(228,92,16) ≠ Mario RGB(248,56,0) → la détection couleur
        ne fonctionnait pas. L'OAM donne les positions écran exactes avec palette=3 (ennemis).
        """
        try:
            ram = self.env.unwrapped.ram
            OAM_BASE  = 0x0200
            HUD_Y_MIN = 32    # ignorer HUD
            ENEMY_PAL = 0x03  # palette 3 = ennemis NES

            # Lire la position écran X de Mario depuis l'OAM (sprite 1 = corps haut gauche)
            # Sprite 0 est le compteur HUD, sprite 1 est le premier sprite du corps de Mario
            mario_x_oam = int(ram[OAM_BASE + 1*4 + 3])  # X du sprite 1
            # Zone de menace : 15 à 70px devant Mario
            threat_x_min = mario_x_oam + 15
            threat_x_max = mario_x_oam + 70
            # Zone verticale : sol uniquement (y NES > 60% de 240 = 144)
            threat_y_min = int(obs.shape[0] * 0.60)  # 144

            for i in range(64):
                y_nes = int(ram[OAM_BASE + i*4 + 0])
                attr  = int(ram[OAM_BASE + i*4 + 2])
                x_nes = int(ram[OAM_BASE + i*4 + 3])
                if y_nes < HUD_Y_MIN or y_nes >= 240:
                    continue
                if (attr & 0x03) != ENEMY_PAL:
                    continue
                if y_nes < threat_y_min:
                    continue
                # Ennemi dans la zone de menace ?
                if threat_x_min <= x_nes <= threat_x_max:
                    return True

            return False

        except Exception:
            return False

    def inject_emergency_jump(self):
        """⚡ Injecte un saut long en TÊTE de queue et interrompt l'action en cours.
        Utilise run_jump_over (right+A+B, 12f) plutôt que stomp_enemy (right+A, 20f) :
        - Plus de vitesse horizontale (B = run) → ~150-200px de portée
        - Dégager une paire de Goombas (x=295+x=305) en un seul saut
        - stomp_enemy (right+A seulement) était trop court et Mario atterrissait sur le 2ème Goomba
        """
        emergency = {
            'macro_name': 'run_jump_over',
            'reasoning': '⚡ RÉFLEXE PIXEL: ennemi détecté → saut long par-dessus',
            'strategy': 'Saut urgence long',
            'urgency': 10,
            'confidence': 95
        }
        self.action_queue.appendleft(emergency)
        self.current_macro = None  # Interrompre l'action en cours

    def detect_holes_ahead(self, obs):
        """⛰️ Détecte les trous dans le sol devant Mario via analyse pixel.

        Principe : dans la bande de sol au bas de l'écran, les tuiles ont des
        couleurs chaudes (rouge/brun, R >> B). Là où le sol est absent (trou),
        seul le ciel bleu ou le noir est visible (B >= R ou tout noir).
        On cherche des groupes de colonnes consécutives sans tuile devant Mario.

        Retourne :
            detected  : bool — trou confirmé (>= 8 colonnes sans sol)
            nearest   : int  — distance en pixels écran depuis Mario
            width     : int  — largeur du trou en pixels écran
            critical  : bool — True si trou à < 70px (saut immédiat)
            urgent    : bool — True si trou à < 100px (préparer le saut)
        """
        _empty = {'detected': False, 'nearest': 999, 'width': 0,
                  'critical': False, 'urgent': False}
        try:
            height, width = obs.shape[:2]
            # HUD en haut (~20%) → zone de jeu commence à game_top
            game_top = int(height * 0.20)
            game_h = height - game_top

            # Bande de sol élargie : les 20% inférieurs de la zone de jeu
            # (plus large = moins de faux négatifs sur rebords de tuiles)
            floor_start = game_top + int(game_h * 0.80)
            floor_band = obs[floor_start:height, :]

            r = floor_band[:, :, 0].astype(np.int16)
            b = floor_band[:, :, 2].astype(np.int16)

            # Pixel "sol" = couleur chaude : rouge nettement dominant sur le bleu
            # Ciel SMB1-1 ≈ (92, 148, 252) → r-b < 0 → exclu
            # Sol brique ≈ (228, 156, 84)  → r-b ≈ +144 → inclus
            ground_mask = (r - b > 30) & (r > 80)

            # Pour chaque colonne : sol présent ?
            has_ground = np.any(ground_mask, axis=0)  # shape: (width,)

            # Mario est dans le tiers gauche de l'écran
            mario_screen_x = width // 3

            # Zone d'anticipation : 8 à 160px devant Mario
            # (8px pour ne pas capter les pieds de Mario, 160px pour anticiper tôt)
            ahead_start = mario_screen_x + 8
            ahead_end = min(mario_screen_x + 160, width - 1)

            # Colonnes sans sol dans la zone d'anticipation
            indices = np.arange(ahead_start, ahead_end)
            no_ground_mask = ~has_ground[ahead_start:ahead_end]
            no_ground_idx = indices[no_ground_mask]

            if len(no_ground_idx) < 3:
                return _empty

            # Regrouper les colonnes sans sol en gaps consécutifs
            # On ignore les micro-discontinuités (≤ 2 colonnes de sol entre deux vides)
            gaps = []
            gap_start = int(no_ground_idx[0])
            gap_end = int(no_ground_idx[0])
            for idx in no_ground_idx[1:]:
                if idx - gap_end <= 3:   # continuité tolérante
                    gap_end = int(idx)
                else:
                    if gap_end - gap_start >= 8:  # trou >= 8px = réel
                        gaps.append((gap_start, gap_end))
                    gap_start = int(idx)
                    gap_end = int(idx)
            if gap_end - gap_start >= 8:
                gaps.append((gap_start, gap_end))

            if not gaps:
                return _empty

            nearest_col = gaps[0][0]
            far_col = gaps[0][1]
            nearest_dist = nearest_col - mario_screen_x
            hole_width = far_col - nearest_col

            return {
                'detected': True,
                'nearest': nearest_dist,
                'width': hole_width,
                'critical': nearest_dist < 70,   # Saut immédiat obligatoire
                'urgent': nearest_dist < 100,    # Préparer le saut
            }
        except Exception:
            return _empty

    def inject_hole_jump(self, hole_info: dict):
        """⚡ Injecte un max_jump pour franchir un trou détecté par pixel.
        Vide la queue et annule la macro en cours pour un saut IMMÉDIAT."""
        dist = hole_info.get('nearest', '?')
        w = hole_info.get('width', '?')
        # Après le saut, reprendre la course pour maintenir l'élan
        run_after = {
            'macro_name': 'run_forward',
            'reasoning': 'Reprendre la course après franchissement du trou',
            'strategy': 'Après trou', 'urgency': 7, 'confidence': 90,
        }
        jump = {
            'macro_name': 'max_jump',
            'reasoning': f'⚡ RÉFLEXE TROU: sol absent à {dist}px (larg={w}px) → saut max',
            'strategy': 'Franchir trou', 'urgency': 10, 'confidence': 95,
        }
        # Vider la queue et annuler la macro en cours pour forcer le saut immédiatement
        self.action_queue.clear()
        self.action_queue.append(jump)
        self.action_queue.append(run_after)
        self.current_macro = None

    def _inject_segment_replay(self, sequence: List[tuple], x: int):
        """Injecte une séquence mémorisée dans la queue et passe en mode replay.
        Ne coupe PAS la macro en cours : si Mario est en plein saut lors d'une
        transition de segment, le saut se termine avant que la nouvelle séquence
        commence. Évite les atterrissages dans les tuyaux/trous."""
        self.action_queue.clear()
        # current_macro intentionnellement NON réinitialisé ici
        for macro_name, count in sequence:
            for _ in range(count):
                self.action_queue.append({
                    'macro_name': macro_name,
                    'reasoning': f'Replay optimal segment x={x}',
                    'strategy': 'Replay mémoire', 'urgency': 5, 'confidence': 90
                })
        self._segment_in_replay = True

    def _on_segment_enter(self, seg_key: str, x: int, step_count: int):
        """
        Appelé à chaque transition vers un nouveau segment (uniquement en avançant).
        Réinitialise l'enregistrement du segment pour ne capturer que le passage actuel,
        puis applique la logique de phase : replay, mixte, ou IA.
        """
        self._segment_in_replay = False  # Reset au début de chaque segment
        # Toujours repartir d'un enregistrement vierge pour ce segment :
        # évite l'accumulation de macros lors de rebonds ou re-entrées.
        self.segment_memory.reset_run_recording(x)
        # Reset du tracker anti-blocage replay : chaque segment repart de zéro
        self._phase3_last_x = int(x)
        self._phase3_last_x_step = step_count

        if self._run_phase == 1:
            return  # Phase 1 : IA pure, rien à faire

        if self._run_phase == 3 and self._phase3_ai_mode:
            return  # Phase 3 déjà en mode IA : laisser Claude continuer

        seq = self.segment_memory.get_stage_sequence(seg_key)

        if self._run_phase == 2:
            # Phase 2 : tirage aléatoire par segment
            if seq and random.random() < 0.5:
                self._inject_segment_replay(seq, x)
                print(f"🎲 Phase 2: REPLAY segment {seg_key} ({len(seq)} macros)")
            else:
                print(f"🎲 Phase 2: IA libre segment {seg_key}")

        elif self._run_phase == 3:
            # Phase 3 : replay jusqu'au segment frontière (furthest_x), puis IA
            danger_x = self.segment_memory.get_stage_danger_frontier()
            if x >= danger_x:
                self._phase3_ai_mode = True
                print(f"⚠️ Phase 3: frontière atteinte x={x} (record={self.segment_memory.furthest_x}) → IA")
            elif seq:
                self._inject_segment_replay(seq, x)
                print(f"📼 Phase 3: replay sûr segment {seg_key} ({len(seq)} macros)")
            else:
                # Pas de séquence mémorisée : injecter run_forward par défaut pour traverser la zone
                self._inject_segment_replay([('run_forward', 3)], x)
                print(f"📼 Phase 3: segment {seg_key} sans mémoire → run_forward x3 par défaut")

    def inject_known_solution(self, current_x: int, step_count: int) -> bool:
        """
        Si une solution confirmée (≥2 runs) existe dans les 120px devant Mario,
        l'injecte directement dans la queue sans passer par Claude.
        Retourne True si une solution a été injectée.
        """
        # Ne pas surcharger une queue déjà remplie
        if len(self.action_queue) >= 3:
            return False

        # Ne pas écraser un saut déjà planifié ou en cours
        _jump_macros = {'stomp_enemy', 'pipe_jump', 'obstacle_jump',
                        'max_jump', 'run_jump_over', 'big_jump_right'}
        if any(a.get('macro_name') in _jump_macros for a in self.action_queue):
            return False
        if self.current_macro and self.current_macro.get('name') in _jump_macros:
            return False

        # Ne pas ré-injecter si la position n'a pas avancé depuis la dernière injection
        # (évite la boucle infinie si la solution mémorisée ne fonctionne pas à cette position)
        last_inject_x = getattr(self, '_last_known_solution_x', -999)
        cooldown_px = getattr(self, '_known_solution_cooldown_px', 40)
        if abs(current_x - last_inject_x) < cooldown_px:
            return False

        best = None
        for seg in self.segment_memory.segments.values():
            for s in seg.successes:
                if s.count >= 2 and current_x < s.x <= current_x + 120:
                    if best is None or s.count > best.count:
                        best = s

        if best is None:
            return False

        print(f"🧠 MÉMOIRE: solution x={best.x} confirmée {best.count}x → injection directe: "
              f"{' → '.join(best.winning_sequence)}")
        self.action_queue.clear()
        self.current_macro = None
        for macro_name in best.winning_sequence:
            self.action_queue.append({
                'macro_name': macro_name,
                'reasoning': f'Solution mémorisée x={best.x} ({best.count}x confirmée)',
                'strategy': 'Mémoire segment', 'urgency': 8, 'confidence': 95
            })
        # Mémoriser la position d'injection pour éviter la boucle
        self._last_known_solution_x = current_x
        # Marquer comme tentative de déblocage pour confirmer le succès
        self._unstick_start_x = current_x
        self._unstick_sequence = best.winning_sequence
        self._unstick_step = step_count
        return True

    def _get_phase1_optimization_hint(self, mario_x: int) -> str:
        """
        Phase 1 uniquement : si une meilleure séquence est connue pour ce segment,
        demande à Claude de l'améliorer (vitesse + blocs ?) plutôt que de repartir de zéro.
        Retourne une chaîne vide si pas en Phase 1 ou aucune mémoire disponible.
        """
        if self._run_phase != 1:
            return ""
        seg_key = self.segment_memory._key(int(mario_x))
        seq = self.segment_memory.stage.sequences.get(seg_key)
        if not seq:
            return ""
        seq_str = " → ".join(f"{name}×{count}" for name, count in seq)
        return f"""
🏆 OPTIMISATION PHASE 1 — AMÉLIORE LE MEILLEUR PARCOURS CONNU:
Lors d'un run précédent, cette zone ({seg_key}) a été franchie avec:
  {seq_str}

🎯 TON OBJECTIF: PROPOSER UNE SÉQUENCE MEILLEURE QUE CELLE-CI
Critères d'amélioration (par ordre de priorité):
1. VITESSE: Remplace 'walk_right' par 'run_forward', supprime les 'wait' inutiles
2. BLOCS ? (boîtes à interrogation): Ces blocs jaunes avec "?" contiennent pièces/champignons/power-ups.
   Si tu en vois un au-dessus de Mario → saute dessous pour le frapper (action 'short_jump' ou 'max_jump' selon hauteur)
   Ne passe PAS sous un bloc ? sans le frapper — c'est un item gratuit !
3. SCORE: Écraser les Goombas rapporte des points et sécurise le passage

⚠️ IMPORTANT: Ta séquence ne remplacera la sauvegarde QUE si Mario va plus loin ou plus vite.
Si la séquence actuelle est déjà optimale, utilise-la mais essaie quand même d'y intégrer la collecte des blocs ?.
"""

    def _get_phase3_frontier_context(self, mario_x: int) -> str:
        """
        Phase 3 uniquement, mode IA actif : fournit le contexte d'élan à Claude.
        Indique que Mario vient de rejouer la mémoire et aborde la zone frontière.
        Retourne une chaîne vide si pas applicable.
        """
        if self._run_phase != 3 or not self._phase3_ai_mode:
            return ""
        frontier_x = self.segment_memory.get_stage_danger_frontier()
        furthest_x = self.segment_memory.furthest_x

        # Segments rejoués juste avant (les 3 segments précédant la frontière)
        from mario_segment_memory import SEGMENT_SIZE
        replayed_segs = []
        for i in range(3, 0, -1):
            seg_start = frontier_x - i * SEGMENT_SIZE
            if seg_start < 0:
                continue
            key = f"{seg_start}-{seg_start + SEGMENT_SIZE}"
            seq = self.segment_memory.stage.sequences.get(key)
            if seq:
                seq_str = " → ".join(f"{n}×{c}" for n, c in seq)
                replayed_segs.append(f"  x={seg_start}: {seq_str}")

        # Morts enregistrées dans la zone frontière
        frontier_key = f"{frontier_x}-{frontier_x + SEGMENT_SIZE}"
        frontier_seg = self.segment_memory.segments.get(frontier_key)
        death_lines = frontier_seg.death_summary() if frontier_seg else []

        lines = [
            f"\n🚀 CONTEXTE PHASE 3 — ZONE FRONTIÈRE (record actuel: x={furthest_x}px):",
            f"Mario vient de rejouer automatiquement les segments mémorisés jusqu'à x={frontier_x}px.",
            "Il a de l'élan — PRIORITÉ: maintenir la vitesse et ne pas s'arrêter !",
        ]
        if replayed_segs:
            lines.append("Segments rejoués juste avant (pour contexte de vitesse):")
            lines.extend(replayed_segs)
        if death_lines:
            lines.append(f"⚠️ ZONE DANGEREUSE x={frontier_x}-{frontier_x + SEGMENT_SIZE} — morts précédentes:")
            lines.extend(death_lines)
        lines.append("🎯 OBJECTIF: Dépasser x={} — utilise run_forward + sauts anticipés !".format(furthest_x))
        return "\n".join(lines) + "\n"

    def detect_stuck(self, current_x, step_count):
        """Détecte si Mario est bloqué : position stagnante + même action répétée.
        Retourne le niveau de blocage : 0=libre, 1=légèrement bloqué, 2=vraiment bloqué.
        """
        if step_count - self.last_stuck_check_step < self.stuck_check_frequency:
            return 0

        self.last_stuck_check_step = step_count

        # --- Critère 1 : position n'a pas bougé depuis le dernier check ---
        position_stuck = (
            self.last_stuck_position is not None and
            abs(current_x - self.last_stuck_position) < 8  # moins de 8px en 40 steps
        )
        self.last_stuck_position = current_x

        # --- Critère 2 : même action dominante dans l'historique récent ---
        recent = list(self.macro_history)[-6:]
        action_stuck = False
        if len(recent) >= 4:
            names = [m['name'] for m in recent]
            most_common = max(set(names), key=names.count)
            action_stuck = names.count(most_common) >= 4  # 4 fois la même sur 6

        if position_stuck and action_stuck:
            self.stuck_counter += 1
            print(f"🔁 Blocage détecté (niveau {self.stuck_counter}) à x={current_x:.0f} - action répétée: {most_common}")
        elif not position_stuck:
            self.stuck_counter = 0  # Mario a avancé, réinitialiser

        return self.stuck_counter

    def search_mario_strategy(self, position, failed_actions):
        """Recherche web via DuckDuckGo Instant Answers (sans clé API).
        Retourne un texte de stratégie ou None si la recherche échoue.
        """
        import urllib.request, urllib.parse, json

        zone = int(position // 100) * 100  # Arrondir à la centaine (zone du niveau)
        search_key = f"{zone}_{','.join(failed_actions[:2])}"
        if search_key in self.stuck_search_done:
            return None  # Déjà cherché pour cette zone/action

        query = f"Super Mario Bros World 1-1 stuck obstacle position {zone} strategy walkthrough"
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"

        try:
            print(f"🌐 Recherche web : {query}")
            req = urllib.request.Request(url, headers={'User-Agent': 'MarioAI/1.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            results = []
            if data.get('AbstractText'):
                results.append(data['AbstractText'])
            for topic in data.get('RelatedTopics', [])[:3]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append(topic['Text'])

            self.stuck_search_done.add(search_key)

            if results:
                summary = ' | '.join(results[:2])[:400]
                print(f"🌐 Résultat: {summary[:100]}...")
                return summary
        except Exception as e:
            print(f"🌐 Recherche web échouée: {e}")

        return None

    def call_claude_stuck_mode(self, situation, position, failed_actions, obs, step_count, web_results=None):
        """Appel Claude spécial mode déblocage : contexte enrichi + résultats web."""
        failed_str = ', '.join(failed_actions) if failed_actions else 'inconnues'
        web_section = f"\n🌐 INFOS WEB:\n{web_results}" if web_results else ""

        # Ajouts au prompt stuck mode générés par l'auto-improver
        _additions = getattr(self, '_prompt_additions', {}).get('stuck_mode', [])
        _additions_str = ("\n⚙️  RÈGLES APPRIS DES SESSIONS PRÉCÉDENTES:\n" +
                          "\n".join(f"- {a}" for a in _additions) + "\n") if _additions else ""

        prompt = f"""🚨 MARIO EST BLOQUÉ depuis plusieurs secondes!

Position Mario: x={position:.0f}px (World 1-1)
Actions répétées sans succès: {failed_str}
Progression: {situation.get('progress', {}).get('trend', 0):.1f}px/check
{web_section}{_additions_str}
ANALYSE REQUISE:
- Quel obstacle bloque Mario à cette position ?
- Si c'est un TUYAU HAUT → utilise 'pipe_jump' (séquence automatique : approche 40f + saut max 40f)
- Si c'est une PLATEFORME ÉLEVÉE → utilise 'obstacle_jump' (élan 20f + saut max 40f)
- Autres options: hop_on_platform, approach_and_hit_block, step_back + pipe_jump

ACTIONS INTERDITES (déjà échouées): {failed_str}

🎯 DONNE 3-4 ACTIONS DIFFÉRENTES pour débloquer Mario!
⚠️ JSON UNIQUEMENT — ZÉRO TEXTE, ZÉRO EXPLICATION:
{{"actions":[{{"macro_action":"run_forward","px":80}},{{"macro_action":"pipe_jump"}}],"urgency":9}}
(run_forward DOIT avoir px = pixels à parcourir)"""

        actions = self.parse_claude_actions(
            self.call_claude_for_macro(prompt)
        )
        if actions:
            print(f"🔓 Mode déblocage: {len(actions)} nouvelles actions injectées")
            self.action_queue.clear()
            self.current_macro = None
            for action in actions:
                self.action_queue.append(action)

    def get_fallback_macro(self):
        """Macro-action par défaut en cas d'erreur - continuer à avancer"""
        return {
            'macro_name': 'run_forward',
            'reasoning': 'Continuer à courir en attendant les instructions Claude',
            'strategy': 'Avancer par défaut',
            'urgency': 1,
            'confidence': 90
        }
    
    def execute_macro_action(self, macro_decision):
        """Démarrer l'exécution d'une macro-action"""

        macro_name = macro_decision['macro_name']
        macro_config = self.macro_actions[macro_name]

        px = macro_decision.get('px')

        if macro_name == 'max_jump' and px is not None:
            # max_jump avec px = approche N pixels vers la droite puis saut max
            # Transformé en 2 phases : phase1=run_right(approach_frames), phase2=max_jump(40f)
            approach_frames = max(3, min(80, round(int(px) / 2)))
            self.current_macro = {
                'name': macro_name,
                'phases': [
                    {'base_action': 3, 'duration': approach_frames},  # run right (B held)
                    {'base_action': 4, 'duration': 40}  # right+A+B max jump
                ],
                'current_phase': 0,
                'base_action': 3,
                'frames_left': approach_frames,
                'decision': macro_decision
            }
        elif 'phases' in macro_config:
            # Macro multi-phases : phase 1 d'abord (durée fixe, px ignoré)
            self.current_macro = {
                'name': macro_name,
                'phases': macro_config['phases'],
                'current_phase': 0,
                'base_action': macro_config['phases'][0]['base_action'],
                'frames_left': macro_config['phases'][0]['duration'],
                'decision': macro_decision
            }
        else:
            # Pour les macros de mouvement (run_forward, walk_right...), Claude peut
            # fournir un champ 'px' = distance en pixels NES à parcourir.
            # Vitesse run_forward ≈ 2 px/frame → frames = px / 2.
            # Sans 'px', on utilise la durée par défaut de la macro.
            if px is not None:
                frames = max(5, min(300, round(int(px) / 2)))
            else:
                frames = macro_config['duration']
            self.current_macro = {
                'name': macro_name,
                'base_action': macro_config['base_action'],
                'frames_left': frames,
                'decision': macro_decision
            }
        
        self._life_macro_count += 1
        self.macro_history.append({
            'name': macro_name,
            'reasoning': macro_decision['reasoning'][:30]
        })
        
        # Enregistrer l'action pour apprentissage
        if hasattr(self, 'last_situation') and self.last_situation is not None:
            self.record_action(
                macro_name, 
                self.last_situation, 
                getattr(self, 'current_step', 0),
                macro_decision.get('reasoning', '')
            )
        
        if 'phases' in macro_config:
            total_frames = sum(p['duration'] for p in macro_config['phases'])
            print(f"🎮 Exécution: {macro_name} ({len(macro_config['phases'])} phases, {total_frames} frames) - {macro_decision.get('reasoning', '')[:50]}")
        else:
            print(f"🎮 Exécution: {macro_name} ({macro_config['duration']} frames) - {macro_decision.get('reasoning', '')[:50]}")
    
    def get_current_action(self):
        """Obtenir l'action à exécuter cette frame"""

        if self.current_macro and self.current_macro['frames_left'] > 0:
            # Continuer la macro en cours
            self.current_macro['frames_left'] -= 1
            return self.current_macro['base_action']
        else:
            # Phase terminée — vérifier s'il y a une phase suivante (multi-phases)
            if self.current_macro and 'phases' in self.current_macro:
                next_phase = self.current_macro['current_phase'] + 1
                if next_phase < len(self.current_macro['phases']):
                    ph = self.current_macro['phases'][next_phase]
                    self.current_macro['current_phase'] = next_phase
                    self.current_macro['base_action'] = ph['base_action']
                    self.current_macro['frames_left'] = ph['duration']
                    return self.get_current_action()

            # Macro (ou dernière phase) terminée
            if self.current_macro:
                self.successful_macros += 1
                self.current_macro = None
            
            # Vérifier s'il y a des actions en attente
            if self.action_queue:
                next_action = self.action_queue.popleft()
                self.execute_macro_action(next_action)
                return self.get_current_action()  # Récursion pour obtenir l'action
            
            return None  # Pas d'action, temps de demander à Claude
    
    def add_llm_response(self, response_type, content, step_count):
        """Ajouter une réponse LLM à l'historique"""
        timestamp = time.strftime("%H:%M:%S")
        
        # Formater le contenu pour un meilleur affichage
        formatted_content = self.format_llm_content(content)
        
        entry = {
            'type': response_type,
            'content': formatted_content,
            'step': step_count,
            'timestamp': timestamp
        }
        self.llm_responses.append(entry)
        # Auto-scroll vers le bas quand une nouvelle réponse arrive
        self.llm_scroll_position = max(0, len(self.llm_responses) - 3)
    
    def format_llm_content(self, content):
        """Formater le contenu LLM pour un meilleur affichage"""
        if not content:
            return "Contenu vide"
            
        # Si c'est du JSON, essayer de le formater
        try:
            import json
            import re
            
            # Chercher du JSON dans le contenu
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                parsed_json = json.loads(json_text)
                
                # Formater les éléments importants du JSON
                formatted_parts = []
                
                if 'actions' in parsed_json:
                    for action in parsed_json['actions']:
                        if isinstance(action, dict):
                            action_name = action.get('macro_action', 'Unknown')
                            reasoning = action.get('reasoning', 'No reasoning')
                            formatted_parts.append(f"ACTION: {action_name} → {reasoning}")
                
                if 'spatial_analysis' in parsed_json:
                    formatted_parts.append(f"SPATIAL: {parsed_json['spatial_analysis']}")
                    
                if 'immediate_danger' in parsed_json:
                    formatted_parts.append(f"DANGER: {parsed_json['immediate_danger']}")
                    
                if 'strategy' in parsed_json:
                    formatted_parts.append(f"STRATEGY: {parsed_json['strategy']}")
                    
                if formatted_parts:
                    return " | ".join(formatted_parts)
            
            # Si pas de JSON ou JSON non parsable, retourner le contenu original tronqué
            return content[:500] + "..." if len(content) > 500 else content
            
        except (json.JSONDecodeError, Exception):
            # Fallback: retourner le contenu original tronqué
            return content[:500] + "..." if len(content) > 500 else content
    
    def draw_llm_panel(self, canvas, x_start, y_start, width, height):
        """Dessiner l'encart scrollable des réponses LLM"""
        # Fond de l'encart
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GRAY = (80, 80, 80)
        GREEN = (0, 255, 0)
        CYAN = (255, 255, 0)
        YELLOW = (0, 255, 255)
        
        # Bordure de l'encart
        cv2.rectangle(canvas, (x_start, y_start), (x_start + width, y_start + height), WHITE, 2)
        
        # Titre de l'encart
        title_y = y_start + 25
        cv2.putText(canvas, "🤖 RÉPONSES LLM (W/S:↑↓ U/D:Page H/E:Home/End)", (x_start + 10, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 2)
        
        # Zone scrollable
        content_start_y = title_y + 15
        content_height = height - 40
        line_height = 14
        max_lines_visible = content_height // line_height
        
        if not self.llm_responses:
            cv2.putText(canvas, "Aucune réponse LLM encore...", (x_start + 10, content_start_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
            return
        
        # Calculer les lignes à afficher selon le scroll
        start_idx = max(0, self.llm_scroll_position)
        end_idx = min(len(self.llm_responses), start_idx + max_lines_visible // 8)  # ~8 lignes par réponse maintenant
        
        current_y = content_start_y
        
        for i in range(start_idx, end_idx):
            if current_y > y_start + height - 20:
                break
                
            response = self.llm_responses[i]
            
            # En-tête de la réponse
            header_color = CYAN if response['type'] == 'SCREENSHOT' else GREEN
            header_text = f"[{response['timestamp']}] {response['type']} (Step {response['step']})"
            cv2.putText(canvas, header_text, (x_start + 10, current_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, header_color, 1)
            current_y += line_height
            
            # Contenu de la réponse (adapté à la largeur de l'encart)
            # Calculer la largeur disponible en caractères (900px / ~6px par caractère)
            chars_per_line = (width - 30) // 6  # ~145 caractères pour 900px de largeur
            content_lines = self.wrap_text(response['content'], chars_per_line)
            
            # Afficher plus de lignes par réponse (jusqu'à 6 lignes)
            for j, line in enumerate(content_lines[:6]):
                if current_y > y_start + height - 20:
                    break
                cv2.putText(canvas, line, (x_start + 15, current_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1)
                current_y += line_height
            
            # Séparateur
            current_y += 5
        
        # Indicateur de scroll si nécessaire
        if len(self.llm_responses) > max_lines_visible // 8:
            scroll_indicator = f"Réponses {start_idx + 1}-{end_idx}/{len(self.llm_responses)}"
            cv2.putText(canvas, scroll_indicator, (x_start + width - 150, y_start + height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, GRAY, 1)
    
    def handle_scroll_keys(self, key):
        """Gérer les touches de défilement pour l'encart LLM"""
        scroll_changed = False
        
        # Flèche haut ou W (différentes variantes)
        if key in [ord('w'), ord('W'), 119, 87, 82, 2490368]:  # W, w, Up arrow
            self.llm_scroll_position = max(0, self.llm_scroll_position - 1)
            scroll_changed = True
            print(f"📜 Scroll UP → Position: {self.llm_scroll_position}")
            
        # Flèche bas ou S (différentes variantes)  
        elif key in [ord('s'), ord('S'), 115, 83, 84, 2621440]:  # S, s, Down arrow
            max_scroll = max(0, len(self.llm_responses) - 5)
            self.llm_scroll_position = min(max_scroll, self.llm_scroll_position + 1)
            scroll_changed = True
            print(f"📜 Scroll DOWN → Position: {self.llm_scroll_position}/{max_scroll}")
            
        # Page Up / Page Down
        elif key in [2162688, ord('u'), ord('U')]:  # Page Up ou U
            self.llm_scroll_position = max(0, self.llm_scroll_position - 5)
            scroll_changed = True
            print(f"📜 PAGE UP → Position: {self.llm_scroll_position}")
            
        elif key in [2228224, ord('d'), ord('D')]:  # Page Down ou D
            max_scroll = max(0, len(self.llm_responses) - 5)
            self.llm_scroll_position = min(max_scroll, self.llm_scroll_position + 5)
            scroll_changed = True
            print(f"📜 PAGE DOWN → Position: {self.llm_scroll_position}/{max_scroll}")
            
        # Début/fin
        elif key in [ord('h'), ord('H')]:  # Home
            self.llm_scroll_position = 0
            scroll_changed = True
            print(f"📜 HOME → Position: 0")
            
        elif key in [ord('e'), ord('E')]:  # End
            max_scroll = max(0, len(self.llm_responses) - 5)
            self.llm_scroll_position = max_scroll
            scroll_changed = True
            print(f"📜 END → Position: {max_scroll}")
        
        # Debug: afficher le code de la touche pressée si pas reconnue
        elif key != 255:  # 255 = pas de touche
            print(f"🔍 Touche non reconnue: {key} (chr: {chr(key) if 32 <= key <= 126 else 'non-printable'})")
            
        return scroll_changed
    
    def create_display(self, frame, situation, mario_decision, total_reward, step_count):
        """Créer l'affichage avec informations"""
        
        # obs (gym) est en RGB, mais cv2.imshow attend du BGR → conversion obligatoire
        # Sans ça, rouge↔bleu sont inversés : ciel orange, Mario bleu, etc.
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # INTER_NEAREST = pixel art NES authentique (chaque pixel NES → carré net, pas de flou)
        # Le screenshot envoyé à Claude garde son propre pipeline (LANCZOS PIL + filtres + annotations)
        display_frame = cv2.resize(frame_bgr, (600, 480), interpolation=cv2.INTER_NEAREST)
        # Agrandir le canvas pour inclure l'encart LLM en bas
        canvas = np.zeros((900, 1000, 3), dtype=np.uint8)
        
        # Placer le jeu
        canvas[80:560, 50:650] = display_frame

        # Badge mode (REPLAY vs IA) — affiché sur la zone de jeu
        # En Phase 3 zone safe : toujours REPLAY (même entre segments)
        _in_replay_display = self._segment_in_replay or (self._run_phase == 3 and not self._phase3_ai_mode)
        if _in_replay_display:
            _badge_color = (200, 120, 0)   # Orange foncé
            _badge_text  = ">> REPLAY"
        elif self._run_phase == 3 and self._phase3_ai_mode:
            _badge_color = (0, 180, 255)   # Jaune/cyan
            _badge_text  = "IA  FRONTIERE"
        else:
            _badge_color = (30, 140, 30)   # Vert
            _badge_text  = "IA"
        # Rectangle de fond + texte
        cv2.rectangle(canvas, (54, 84), (54 + 160, 84 + 22), _badge_color, -1)
        cv2.putText(canvas, _badge_text, (60, 101),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Phase courante (petit label en dessous)
        _phase_colors = {1: (80, 80, 80), 2: (100, 60, 140), 3: (0, 100, 180)}
        _phase_labels = {1: "Phase 1: IA pure", 2: "Phase 2: mixte", 3: "Phase 3: mem->IA"}
        _pc = _phase_colors.get(self._run_phase, (80, 80, 80))
        _pl = _phase_labels.get(self._run_phase, "")
        cv2.rectangle(canvas, (54, 107), (54 + 160, 107 + 18), _pc, -1)
        cv2.putText(canvas, _pl, (60, 121),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        # Segment courant + macro en cours — affiché en bas à droite de la zone de jeu
        _cur_macro = self.current_macro['name'] if self.current_macro else "-"
        _seg_label = self._last_seg_key if self._last_seg_key else "?"
        _mode_label = "REPLAY" if self._segment_in_replay else "IA"
        _seq_text = f"Seg {_seg_label} | {_cur_macro} [{_mode_label}]"
        _step_tw, _step_th = cv2.getTextSize(_seq_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        _step_x = 648 - _step_tw  # aligné à droite (bord droit du jeu = x≈650)
        _step_y = 555             # juste au-dessus du bord bas du jeu (canvas[80:560])
        cv2.rectangle(canvas, (_step_x - 4, _step_y - _step_th - 2),
                      (_step_x + _step_tw + 4, _step_y + 4), (20, 20, 20), -1)
        cv2.putText(canvas, _seq_text, (_step_x, _step_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        # Couleurs
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        YELLOW = (0, 255, 255)
        CYAN = (255, 255, 0)
        ORANGE = (0, 165, 255)
        
        # Titre
        cv2.putText(canvas, "🎮 MARIO FLUIDE - CLAUDE LLM", (680, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        
        # Stats
        y_pos = 70
        mario = situation['mario']
        # Récupérer les vies depuis les infos du jeu
        lives = situation.get('lives', 3)  # Valeur par défaut
        
        stats = [
            f"Seg: {_seg_label} [{_mode_label}] {_cur_macro}",
            f"Position: X={mario['x']}",
            f"Score: {mario['score']}",
            f"Vies: {lives}",
            f"Morts: {self.deaths_count}",
            f"Appels Claude: {self.api_calls}",
            f"Macros réussies: {self.successful_macros}",
            f"Coût: ${self.total_cost:.3f}"
        ]
        
        for stat in stats:
            cv2.putText(canvas, stat, (680, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)
            y_pos += 18
        
        # Mode de jeu et action actuelle
        y_pos += 20
        if self.replay_mode:
            cv2.putText(canvas, "🔄 MODE REPLAY:", (680, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
            y_pos += 25
            replay_progress = f"{self.replay_index}/{len(self.replay_actions)}"
            cv2.putText(canvas, f"Progression: {replay_progress}", (680, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
            y_pos += 18
            if self.replay_index < len(self.replay_actions):
                current_replay_action = self.replay_actions[self.replay_index]['action_name'] if self.replay_index < len(self.replay_actions) else "Terminé"
                cv2.putText(canvas, f"Action: {current_replay_action}", (680, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, ORANGE, 1)
            else:
                cv2.putText(canvas, "🤖 IA prend la main", (680, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
        else:
            cv2.putText(canvas, "🎯 ACTION ACTUELLE:", (680, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
            
            y_pos += 25
            if self.current_macro:
                macro_name = self.current_macro['name']
                frames_left = self.current_macro['frames_left']
                cv2.putText(canvas, f"Macro: {macro_name}", (680, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
                y_pos += 18
                cv2.putText(canvas, f"Frames restantes: {frames_left}", (680, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)
            else:
                thinking_status = "🧠 Réfléchit..." if self.claude_thinking else "⚡ Prêt"
                cv2.putText(canvas, thinking_status, (680, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)
        
        y_pos += 20
        cv2.putText(canvas, f"Queue: {len(self.action_queue)} actions", (680, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
        
        # Décision de Claude
        if mario_decision:
            y_pos += 40
            cv2.putText(canvas, "🧠 CLAUDE DÉCIDE:", (680, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
            
            y_pos += 25
            reasoning_lines = self.wrap_text(mario_decision['reasoning'], 35)
            for line in reasoning_lines[:2]:
                cv2.putText(canvas, line, (680, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
                y_pos += 16
        
        # Éléments Mario Bros détectés
        y_pos += 30
        screen = situation['screen']
        
        if screen.get('environment_type') == 'screenshot_mode':
            # Mode screenshots Claude
            cv2.putText(canvas, "📸 CLAUDE VISION:", (680, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
            
            y_pos += 25
            elements = [
                f"Mode: Screenshots Claude",
                f"Freq: Toutes les {self.screenshot_frequency} steps", 
                f"Cout: ${self.screenshot_costs:.3f}",
                f"Status: {screen.get('status', 'Actif')}"
            ]
            
            for i, element in enumerate(elements):
                cv2.putText(canvas, element, (680, y_pos + i * 16), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)
        else:
            # Mode classique RGB
            cv2.putText(canvas, "🍄 ÉLÉMENTS MARIO:", (680, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
            
            y_pos += 25
            elements = [
                f"Blocs ?: {'OUI' if screen.get('question_blocks') else 'NON'}",
                f"Ennemis: {'OUI' if screen.get('enemies_nearby') else 'NON'}",
                f"Power-ups: {'OUI' if screen.get('power_ups') else 'NON'}",
                f"Tuyaux: {'OUI' if screen.get('pipes') else 'NON'}",
                f"Vides: {'OUI' if screen.get('gaps') else 'NON'}",
                f"Env: {screen['environment_type']}"
            ]
            
            for i, element in enumerate(elements[:4]):  # Afficher les 4 premiers
                color = GREEN if 'OUI' in element else WHITE
                cv2.putText(canvas, element, (680, y_pos + i * 16), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Progression
        y_pos += 80
        progress = situation['progress']
        cv2.putText(canvas, "📊 PROGRESSION:", (680, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2)
        
        y_pos += 20
        cv2.putText(canvas, f"Statut: {progress['status']}", (680, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        y_pos += 16
        cv2.putText(canvas, f"Tendance: {progress['trend']} px", (680, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        
        # Encart LLM en bas
        llm_panel_y = 580  # Commencer sous les informations principales
        llm_panel_height = 280  # Hauteur de l'encart
        llm_panel_width = 900   # Largeur de l'encart
        
        self.draw_llm_panel(canvas, 50, llm_panel_y, llm_panel_width, llm_panel_height)
        
        # Contrôles (repositionnés en bas)
        cv2.putText(canvas, "ESC: Quitter | ESPACE: Pause | W/S:↑↓ U/D:Page H/E:Home/End | Mario joue en FLUIDE!", (50, 880), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)
        
        return canvas
    
    def wrap_text(self, text, max_length):
        """Diviser le texte"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + word) <= max_length:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        return lines
    
    def record_action(self, action_name, situation, step_count, reasoning=""):
        """Enregistrer une action dans l'historique pour apprentissage"""
        if situation is None:
            print("⚠️ Situation None, pas d'enregistrement d'action")
            return
            
        mario = situation.get('mario', {})
        progress = situation.get('progress', {})
        
        action_record = {
            'timestamp': step_count,
            'action': action_name,
            'mario_position': mario.get('x', 0),
            'mario_y': mario.get('y', 0),
            'progress_status': progress.get('status', 'unknown'),
            'reasoning': reasoning,
            'lives_remaining': situation.get('lives', 3)
        }
        self.action_history.append(action_record)
        self.last_actions_before_death.append(action_record)
        
        # Enregistrer aussi dans l'historique persistant
        mario_speed = progress.get('trend', 0) / 30.0 if progress.get('trend') else 0  # pixels/step en moyenne
        self.history_manager.record_action(
            step_count=step_count,
            position_x=mario.get('x', 0),
            position_y=mario.get('y', 0),
            action_name=action_name,
            reasoning=reasoning,
            mario_speed=mario_speed,
            score=mario.get('score', 0)
        )
        
        # Logger l'action
        source = "REPLAY" if "REPLAY:" in reasoning else "AI" if reasoning else "EMERGENCY"
        self.logger.log_action(
            step_count=step_count,
            action_name=action_name,
            reasoning=reasoning,
            position_x=mario.get('x', 0),
            position_y=mario.get('y', 0),
            score=mario.get('score', 0),
            source=source
        )
    
    def record_death(self, death_step, death_position):
        """Enregistrer une mort pour analyse des patterns d'échec"""
        self.deaths_count += 1
        self.death_locations.append(death_position)
        
        # Analyser les actions qui ont mené à la mort
        recent_actions = list(self.last_actions_before_death)
        if recent_actions:
            failure_pattern = {
                'death_number': self.deaths_count,
                'death_position': death_position,
                'death_step': death_step,
                'actions_before_death': recent_actions[-5:],  # 5 dernières actions
                'pattern_key': f"{death_position//50}_death"  # Zone de mort approximative
            }
            self.failure_patterns.append(failure_pattern)
            
            # Compter les échecs répétés dans cette zone
            pattern_key = failure_pattern['pattern_key']
            self.repeated_failures[pattern_key] = self.repeated_failures.get(pattern_key, 0) + 1
            
            if self.repeated_failures[pattern_key] > 2:
                print(f"⚠️ PATTERN D'ÉCHEC DÉTECTÉ: Mort #{self.repeated_failures[pattern_key]} dans la zone {death_position//50}")
        
        # Vider les actions avant mort pour le prochain cycle
        self.last_actions_before_death.clear()
    
    def record_successful_strategy(self, actions_sequence, progress_made):
        """Enregistrer une séquence d'actions qui a bien fonctionné"""
        if progress_made > 20:  # Progrès significatif
            success_record = {
                'actions': actions_sequence,
                'progress': progress_made,
                'timestamp': len(self.action_history)
            }
            self.successful_strategies.append(success_record)
            
            # Garder seulement les 10 meilleures stratégies
            if len(self.successful_strategies) > 10:
                self.successful_strategies = sorted(
                    self.successful_strategies, 
                    key=lambda x: x['progress'], 
                    reverse=True
                )[:10]
    
    def get_learning_context(self):
        """Générer un contexte d'apprentissage basé sur l'historique"""
        context = []
        
        # Analyses des échecs répétés
        if self.repeated_failures:
            context.append("🚨 ZONES DE DANGER IDENTIFIÉES:")
            for pattern, count in self.repeated_failures.items():
                if count > 1:
                    zone = pattern.replace('_death', '')
                    context.append(f"   - Zone {zone}: {count} morts répétées - ÉVITER ces actions!")
        
        # Patterns d'échec récents
        if len(self.failure_patterns) >= 2:
            recent_failures = self.failure_patterns[-2:]
            context.append("❌ ERREURS RÉCENTES À NE PAS RÉPÉTER:")
            for i, failure in enumerate(recent_failures, 1):
                failed_actions = [a['action'] for a in failure['actions_before_death'][-3:]]
                context.append(f"   Échec #{i}: actions {' → '.join(failed_actions)} à la position {failure['death_position']}")
        
        # Stratégies qui marchent
        if self.successful_strategies:
            best_strategy = max(self.successful_strategies, key=lambda x: x['progress'])
            successful_actions = [a for a in best_strategy['actions'] if isinstance(a, str)][:3]
            context.append(f"✅ STRATÉGIE QUI MARCHE: {' → '.join(successful_actions)} (progrès: {best_strategy['progress']}px)")
        
        # Actions récentes pour éviter les boucles
        if len(self.action_history) >= 5:
            recent_actions = [a['action'] for a in list(self.action_history)[-5:]]
            if len(set(recent_actions)) <= 2:  # Trop d'actions répétitives
                context.append(f"⚠️ ACTIONS RÉPÉTITIVES DÉTECTÉES: {' → '.join(recent_actions[-3:])} - VARIER LES ACTIONS!")
        
        return "\n".join(context) if context else "Première tentative - pas d'historique d'apprentissage disponible."
    
    def extract_precise_positions(self, obs, info, step_count):
        """Extraire les positions précises de tous les éléments mobiles"""
        try:
            height, width = obs.shape[:2]
            mario_x = info.get('x_pos', 0)
            mario_y = info.get('y_pos', 0)
            
            # Zone de jeu (exclure interface)
            game_height_start = int(height * 0.2)
            game_area = obs[game_height_start:, :]
            
            # Calculer vitesse et direction de Mario
            speed = 0
            direction = 'right'
            if len(self.position_history) >= 2:
                positions = list(self.position_history)
                speed = (positions[-1] - positions[-2]) / 1  # pixels par step
                direction = 'right' if speed > 0 else 'left' if speed < 0 else 'stationary'
            
            # Mettre à jour Mario
            self.tracked_elements['mario'] = {
                'x': mario_x, 'y': mario_y, 
                'direction': direction, 'speed': speed
            }
            
            # DÉTECTER ENNEMIS (Goombas bruns) - même logique que scan_full_screen
            enemies_list = []
            goomba_brown_mask = (
                (game_area[:, :, 0] > 100) & (game_area[:, :, 0] < 160) &  # Rouge moyen
                (game_area[:, :, 1] > 50) & (game_area[:, :, 1] < 120) &   # Vert faible
                (game_area[:, :, 2] < 80)   # Peu de bleu
            )
            
            if np.any(goomba_brown_mask):
                goomba_columns = np.any(goomba_brown_mask, axis=0)
                goomba_x_positions = np.where(goomba_columns)[0]
                
                mario_screen_x = width // 3  # Mario est visuellement ~1/3 gauche de l'écran quand la caméra scroll
                for goomba_x in goomba_x_positions:
                    # distance en pixels écran (goomba_x et mario_screen_x sont tous deux en coords écran)
                    distance_from_mario = goomba_x - mario_screen_x  # positif = devant Mario
                    enemies_list.append({
                        'type': 'Goomba',
                        'x': goomba_x,
                        'y': height - 50,  # Goombas sont au sol
                        'distance_from_mario': distance_from_mario,
                        'threat_level': 'HIGH' if abs(distance_from_mario) < 30 else 'MEDIUM' if abs(distance_from_mario) < 60 else 'LOW'
                    })
            
            self.tracked_elements['enemies'] = enemies_list
            
            # DÉTECTER BLOCS QUESTION MARKS
            question_blocks = []
            question_blue_mask = (
                (game_area[:, :, 2] > game_area[:, :, 0]) &  # Plus de bleu que de rouge
                (game_area[:, :, 2] > game_area[:, :, 1]) &  # Plus de bleu que de vert
                (game_area[:, :, 2] > 100)                   # Un minimum de bleu
            )
            
            # Chercher dans la partie aérienne
            air_level = int(height * 0.3)
            question_air_mask = question_blue_mask[:air_level, :]
            
            if np.any(question_air_mask):
                question_columns = np.any(question_air_mask, axis=0)
                question_x_positions = np.where(question_columns)[0]
                
                mario_screen_x = width // 3
                for block_x in question_x_positions:
                    distance_from_mario = block_x - mario_screen_x  # positif = devant Mario
                    question_blocks.append({
                        'type': 'QuestionBlock',
                        'x': block_x,
                        'y': air_level,
                        'distance_from_mario': distance_from_mario,
                        'collectible': True
                    })
            
            # Retourner un résumé des changements depuis la dernière update
            changes = self.detect_position_changes(step_count)

            # ⛰️ Détection des trous pour le prompt positionnel
            hole_info = self.detect_holes_ahead(obs)

            return {
                'mario': self.tracked_elements['mario'],
                'enemies': enemies_list,
                'question_blocks': question_blocks,
                'changes': changes,
                'holes': hole_info,
                'step': step_count
            }
            
        except Exception as e:
            print(f"❌ Erreur extraction positions: {e}")
            return None
    
    def detect_position_changes(self, step_count):
        """Détecter les changements depuis la dernière update"""
        changes = []
        
        # Mario a-t-il beaucoup bougé ?
        mario = self.tracked_elements['mario']
        if abs(mario['speed']) > 2:
            changes.append(f"Mario se déplace rapidement vers la {mario['direction']} ({mario['speed']:.1f}px/step)")
        elif mario['speed'] == 0:
            changes.append("Mario est stationnaire")
            
        # Nouveaux ennemis détectés ?
        current_enemies = len(self.tracked_elements['enemies'])
        if hasattr(self, 'previous_enemy_count'):
            if current_enemies > self.previous_enemy_count:
                changes.append(f"Nouvel ennemi détecté! Total: {current_enemies}")
            elif current_enemies < self.previous_enemy_count:
                changes.append(f"Ennemi éliminé! Total: {current_enemies}")
        
        self.previous_enemy_count = current_enemies
        return changes
    
    def create_positional_update_prompt(self, positions_data, step_count):
        """Créer un prompt optimisé avec seulement les positions des éléments mobiles"""
        mario = positions_data['mario']
        enemies = positions_data['enemies']
        blocks = positions_data['question_blocks']
        changes = positions_data['changes']
        
        # Historique d'apprentissage
        learning_context = self.get_learning_context()
        
        prompt = f"""🎮 MISE À JOUR POSITIONNELLE - STEP {step_count}

CONSERVE LE CONTEXTE DU NIVEAU ! Tu as déjà la carte complète du niveau Super Mario Bros 1-1.
Utilise SEULEMENT ces nouvelles positions pour mettre à jour ta stratégie.

📍 POSITIONS ACTUELLES:

MARIO:
• Position: X={mario['x']}, Y={mario['y']}
• Vitesse: {mario['speed']:.1f} pixels/step ({mario['direction']})
• État: {'En mouvement' if mario['speed'] != 0 else 'Stationnaire'}

ENNEMIS DÉTECTÉS: {len(enemies)}"""
        
        if enemies:
            for i, enemy in enumerate(enemies, 1):
                threat_emoji = "🔴" if enemy['threat_level'] == 'HIGH' else "🟠" if enemy['threat_level'] == 'MEDIUM' else "🟡"
                prompt += f"""
• {threat_emoji} {enemy['type']} #{i}: X={enemy['x']}, distance={enemy['distance_from_mario']}px de Mario ({enemy['threat_level']} threat)"""
        else:
            prompt += "\n• ✅ Aucun ennemi détecté dans la zone visible"
            
        prompt += f"\n\nBLOCS QUESTION MARKS: {len(blocks)}"
        if blocks:
            for i, block in enumerate(blocks, 1):
                prompt += f"""
• 💎 Bloc ? #{i}: X={block['x']}, distance={block['distance_from_mario']}px de Mario"""
        else:
            prompt += "\n• ❌ Aucun bloc ? visible dans la zone"
            
        if changes:
            prompt += f"\n\n🔄 CHANGEMENTS DÉTECTÉS:"
            for change in changes:
                prompt += f"\n• {change}"

        # ⛰️ Section trou
        hole_info = positions_data.get('holes', {})
        if hole_info.get('detected'):
            if hole_info.get('critical'):
                prompt += (f"\n\n⛰️ TROU CRITIQUE: sol absent à {hole_info['nearest']}px devant Mario"
                           f" (largeur {hole_info['width']}px) → max_jump MAINTENANT, PAS run_forward!")
            elif hole_info.get('urgent'):
                prompt += (f"\n\n⛰️ TROU URGENT à {hole_info['nearest']}px"
                           f" (largeur {hole_info['width']}px) → max_jump EN PREMIER!")
            else:
                prompt += (f"\n\n⛰️ TROU DÉTECTÉ à {hole_info['nearest']}px"
                           f" (largeur {hole_info['width']}px) → préparer max_jump!")

        prompt += f"""

📚 HISTORIQUE D'APPRENTISSAGE:
{learning_context}

{self.segment_memory.get_context_for_position(mario['x'])}
{self._get_phase1_optimization_hint(mario['x'])}{self._get_phase3_frontier_context(mario['x'])}
🗺️ CONTEXTE DU NIVEAU ACTUEL:
{self.get_level_specific_context()}

🎯 DÉCISION RAPIDE (2-3 actions). Priorités: Survie > Trous⛰️ > Blocs ? > Progression.
⛰️ RAPPEL: Si TROU DÉTECTÉ → max_jump avant d'atteindre le bord!
📐 run_forward DOIT avoir "px" ≤ 60 (1px écran = 1px NES, ≈2px/frame). max_jump accepte "px" = approche avant saut. Autres sauts = pas de px.

⚠️ JSON UNIQUEMENT — ZÉRO TEXTE, ZÉRO EXPLICATION:
{{"actions":[{{"macro_action":"run_forward","px":80}},{{"macro_action":"<saut>"}}],"urgency":<1-10>}}"""

        return prompt
    
    def call_claude_for_positions_update(self, positions_data, step_count):
        """Appeler Claude avec une mise à jour positionnelle (texte seulement)"""
        try:
            prompt = self.create_positional_update_prompt(positions_data, step_count)
            
            self.api_calls += 1
            print(f"📍 Envoi mise à jour positionnelle à Claude (appel #{self.api_calls})...")
            
            # Logger le prompt
            self.logger.log_claude_prompt("POSITIONS", prompt, step_count)
            
            print("="*50)
            print("🔍 PROMPT POSITIONS ENVOYÉ À CLAUDE:")
            print(prompt)
            print("="*50)
            
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,  # Plus petit que screenshot (seulement JSON)
                temperature=0.1,
                system="Tu es un contrôleur de jeu Mario. Réponds UNIQUEMENT en JSON valide, sans aucun texte avant ou après. Aucune explication, aucun commentaire.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            response_text = response.content[0].text if response.content else ""
            
            # Coût estimé pour mise à jour textuelle (beaucoup moins cher qu'une image)
            estimated_cost = 0.001  # $0.001 vs $0.01 pour screenshot
            self.total_cost += estimated_cost
            
            # Logger la réponse
            self.logger.log_claude_response(response_text, step_count, estimated_cost)
            
            print("✅ Claude analyse reçue (texte)", f"({len(response_text)} chars)")
            print("="*50)
            print("💭 RÉPONSE DE CLAUDE:")
            print(response_text)
            print("="*50)
            
            print(f"💰 Coût mise à jour: ${estimated_cost:.4f} (total: ${self.total_cost:.3f})")
            
            # Ajouter la réponse à l'historique pour l'encart
            self.add_llm_response("POSITIONS", response_text, step_count)
            
            return response_text
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Erreur mise à jour positionnelle: {e}")
            print(f"🔍 Détails de l'erreur:")
            print(error_details)
            
            # Logger l'erreur avec détails
            self.logger.log_error("CLAUDE_POSITION_API_FAILURE", f"{str(e)} | Traceback: {error_details}", step_count)
            
            return None
    
    def should_use_screenshot_vs_positions(self, step_count):
        """Toujours utiliser un screenshot : l'IA doit voir l'écran pour détecter ennemis et obstacles."""
        return True, "Screenshot systématique - vision indispensable pour détecter ennemis/obstacles"
    
    def show_game_menu(self):
        """Afficher le menu de sélection du mode de jeu"""
        print("\n" + "="*60)
        print("🎮 MARIO FLUIDE - CLAUDE LLM")
        print("="*60)
        
        # Afficher l'état de la mémoire segments
        seg_mem = self.segment_memory
        if seg_mem.stage.sequences:
            print(f"🧠 Mémoire segments: {seg_mem.total_runs} runs, "
                  f"record={seg_mem.furthest_x}px, "
                  f"safe jusqu'à {seg_mem.stage.safe_max_x}px "
                  f"({len(seg_mem.stage.sequences)} segments mémorisés)")
        else:
            print("🧠 Mémoire segments: vide (première partie)")

        # Vérifier s'il existe un run parfait sauvegardé
        import glob as _glob
        _perfect_files = _glob.glob(os.path.join(os.path.dirname(__file__) or '.', 'logs', 'perfect_run_*.json'))
        _has_perfect = bool(_perfect_files)

        print("\n🎯 CHOISISSEZ VOTRE MODE:")
        print("   1️⃣  Nouvelle partie (IA pure, sans mémoire)")
        print("   2️⃣  Mémoire automatique (rejoue les segments connus → IA à la frontière)")
        if _has_perfect:
            print("   3️⃣  ▶️  Rejouer le run parfait (sans pauses, sans IA)")
            print("   4️⃣  Effacer la mémoire des segments")
            print("   5️⃣  Quitter")
        else:
            print("   3️⃣  Effacer la mémoire des segments")
            print("   4️⃣  Quitter")

        _max_choice = "5" if _has_perfect else "4"
        try:
            while True:
                try:
                    choice = input(f"\n👉 Votre choix (1-{_max_choice}): ").strip()
                except EOFError:
                    print("Mode non-interactif détecté, sélection automatique: nouvelle partie")
                    choice = "1"

                if choice == "1":
                    self.logger.log_menu_choice("new_game")
                    return "new_game", None
                elif choice == "2":
                    if not seg_mem.stage.sequences:
                        print("⚠️  Mémoire vide — démarrage en IA pure (les segments seront mémorisés).")
                    self.logger.log_menu_choice("memory_first")
                    return "memory_first", None
                elif choice == "3" and _has_perfect:
                    self.play_perfect_replay()
                    return self.show_game_menu()
                elif choice == "3" and not _has_perfect:
                    confirm = input("⚠️  Confirmer l'effacement de la mémoire ? (o/N): ").strip().lower()
                    if confirm == "o":
                        self.segment_memory.clear_memory()
                        print("✅ Mémoire effacée.")
                        self.logger.log_menu_choice("clear_memory")
                    else:
                        print("Annulé.")
                    return self.show_game_menu()
                elif choice == "4" and _has_perfect:
                    confirm = input("⚠️  Confirmer l'effacement de la mémoire ? (o/N): ").strip().lower()
                    if confirm == "o":
                        self.segment_memory.clear_memory()
                        print("✅ Mémoire effacée.")
                        self.logger.log_menu_choice("clear_memory")
                    else:
                        print("Annulé.")
                    return self.show_game_menu()
                elif (choice == "4" and not _has_perfect) or (choice == "5" and _has_perfect):
                    self.logger.log_menu_choice("quit")
                    return "quit", None
                else:
                    print(f"❌ Choix invalide! Veuillez entrer 1-{_max_choice}.")
        except KeyboardInterrupt:
            return "quit", None
    
    def select_replay_run(self):
        """Sélectionner un run pour le replay"""
        available_runs = self.history_manager.get_available_runs_for_replay()
        
        if not available_runs:
            print("❌ Aucun run disponible pour replay!")
            return "new_game", None
        
        print(f"\n🔄 SÉLECTION DU RUN À REJOUER:")
        for i, run in enumerate(available_runs, 1):
            status_emoji = "🏆" if run.completion_status == "victory" else "💀" if run.completion_status == "death" else "⏸️"
            print(f"   {i}. {status_emoji} {run.run_id}")
            print(f"      📍 Distance: {run.max_position_x}px | Actions: {run.actions_count} | Durée: {run.duration:.1f}s")
        
        print(f"   0. 🔙 Retour au menu principal")
        
        while True:
            try:
                choice = input(f"\n👉 Sélectionnez un run (0-{len(available_runs)}): ").strip()
                
                if choice == "0":
                    return self.show_game_menu()  # Retour au menu principal
                
                run_index = int(choice) - 1
                if 0 <= run_index < len(available_runs):
                    selected_run = available_runs[run_index]
                    print(f"✅ Run sélectionné: {selected_run.run_id}")
                    self.logger.log_menu_choice("replay", selected_run.run_id)
                    return "replay", selected_run.run_id
                else:
                    print(f"❌ Choix invalide! Veuillez entrer un nombre entre 0 et {len(available_runs)}.")
            
            except (ValueError, KeyboardInterrupt):
                print("❌ Entrée invalide!")
    
    def setup_replay_mode(self, run_id: str):
        """Configurer le mode replay à partir d'un run existant"""
        # Charger les actions du run sélectionné
        actions = self.history_manager.load_run_actions(run_id)
        
        if not actions:
            print(f"❌ Impossible de charger les actions du run {run_id}")
            return False
        
        # Convertir les ActionRecord en format utilisable
        self.replay_actions = []
        for action in actions:
            # Déterminer l'action de base à partir du nom
            base_action = self.get_base_action_from_name(action.action_name)
            self.replay_actions.append({
                'step_count': action.step_count,
                'action_name': action.action_name,
                'base_action': base_action,
                'position_x': action.position_x,
                'reasoning': action.reasoning
            })
        
        # Définir le point de takeover (3 actions avant la fin)
        self.replay_ai_takeover_point = max(0, len(self.replay_actions) - 3)
        
        self.replay_mode = True
        self.replay_index = 0
        
        print(f"🔄 Mode replay configuré:")
        print(f"   📽️  Actions à rejouer: {len(self.replay_actions)}")
        print(f"   🤖 IA reprend à l'action: {self.replay_ai_takeover_point + 1}")
        print(f"   🎯 Actions replay: {self.replay_ai_takeover_point}")
        
        # Créer un nouveau run pour le replay
        new_run_id = self.history_manager.create_replay_run(run_id)
        self.current_run_started = True
        
        # Logger la configuration du replay
        self.logger.log_replay_event("SETUP", {
            "original_run": run_id,
            "new_run": new_run_id,
            "actions_count": len(self.replay_actions),
            "takeover_point": self.replay_ai_takeover_point
        })
        
        return True
    
    def get_base_action_from_name(self, action_name: str) -> int:
        """Convertir le nom d'action en action de base du jeu"""
        # Mapper les noms d'actions vers les actions de base
        if action_name in self.macro_actions:
            return self.macro_actions[action_name]['base_action']
        
        # Actions par défaut selon le nom
        action_mapping = {
            'walk_right': 1, 'run_forward': 3, 'short_jump': 2, 'high_jump': 5,
            'long_jump': 4, 'precise_jump': 2, 'step_back': 6, 'wait': 0,
            'stomp_enemy': 2, 'hit_block': 5, 'approach_and_hit_block': 4,
            'jump_on_pipe': 4, 'retreat_and_jump': 6
        }
        
        return action_mapping.get(action_name, 1)  # Par défaut: walk_right
    
    def get_replay_action(self, current_step):
        """Obtenir l'action suivante en mode replay"""
        if not self.replay_mode or self.replay_index >= len(self.replay_actions):
            return None
        
        # Vérifier si on a atteint le point de takeover de l'IA
        if self.replay_index >= self.replay_ai_takeover_point:
            progress_info = f"{self.replay_index}/{len(self.replay_actions)}"
            print(f"🤖 IA reprend la main à l'action {self.replay_index + 1}/{len(self.replay_actions)}")
            
            # Logger la transition
            self.logger.log_ai_takeover(current_step, progress_info)
            
            self.replay_mode = False  # Désactiver le replay
            return None  # L'IA reprend
        
        # Obtenir l'action actuelle
        current_action = self.replay_actions[self.replay_index]
        
        print(f"🔄 Replay {self.replay_index + 1}/{len(self.replay_actions)}: {current_action['action_name']} (pos: {current_action['position_x']})")
        
        self.replay_index += 1
        return current_action['base_action']
    
    def _save_perfect_run(self):
        """Sauvegarde _final_action_history dans logs/perfect_run_{ts}.json.
        C'est le run sans les segments ratés (tronqué aux checkpoints à chaque rewind)."""
        if not self._final_action_history:
            return
        import glob
        logs_dir = os.path.join(os.path.dirname(__file__) or '.', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        # Garder seulement le dernier perfect_run (supprimer les anciens)
        for old in glob.glob(os.path.join(logs_dir, 'perfect_run_*.json')):
            os.remove(old)
        ts = int(time.time())
        path = os.path.join(logs_dir, f'perfect_run_{ts}.json')
        with open(path, 'w') as f:
            json.dump({'actions': self._final_action_history,
                       'total': len(self._final_action_history)}, f)
        print(f"💾 Run parfait sauvegardé : {len(self._final_action_history)} actions → {path}")

    def play_perfect_replay(self):
        """Rejoue _final_action_history à pleine vitesse, sans pause, sans Claude.
        Affiche les graphismes NES bruts avec couleurs corrigées (RGB→BGR).
        Le replay s'arrête à la fin des actions ou si l'utilisateur appuie sur ESC."""
        import glob
        logs_dir = os.path.join(os.path.dirname(__file__) or '.', 'logs')
        # Chercher le dernier perfect_run sauvegardé
        files = sorted(glob.glob(os.path.join(logs_dir, 'perfect_run_*.json')))
        if not files:
            print("⚠️  Aucun run parfait sauvegardé.")
            return
        with open(files[-1]) as f:
            data = json.load(f)
        actions = data['actions']
        print(f"▶️  Replay parfait : {len(actions)} actions (ESC pour arrêter)")

        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        obs = env.reset()

        cv2.namedWindow('Mario — Replay Parfait', cv2.WINDOW_AUTOSIZE)
        for i, action in enumerate(actions):
            obs, _, done, info = env.step(action)
            # Affichage NES couleurs réelles (RGB→BGR)
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            display = cv2.resize(frame, (600, 480), interpolation=cv2.INTER_NEAREST)
            # Bandeau de progression
            pct = int(100 * i / len(actions))
            cv2.rectangle(display, (0, 0), (int(6 * pct), 6), (0, 255, 100), -1)
            cv2.putText(display, f"REPLAY PARFAIT  {pct}%  x={info.get('x_pos',0)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('Mario — Replay Parfait', display)
            key = cv2.waitKey(16) & 0xFF  # ~60 FPS
            if key == 27:  # ESC
                break
            if done:
                # Laisser voir la frame finale 1 seconde
                cv2.waitKey(1000)
                break

        cv2.destroyWindow('Mario — Replay Parfait')
        env.close()
        print("✅ Replay terminé.")

    def _cleanup_screenshots(self):
        """Supprime tous les screenshots de débogage générés pendant la session."""
        import glob
        pattern = os.path.join(os.path.dirname(__file__) or '.', 'debug_screenshot_*.jpg')
        files = glob.glob(pattern)
        if files:
            for f in files:
                try:
                    os.remove(f)
                except OSError:
                    pass
            print(f"🗑️  {len(files)} screenshot(s) supprimé(s).")

    def _cleanup_old_logs(self):
        """Supprime les logs des sessions précédentes, ne conserve que la session courante."""
        import glob
        log_dir = self.logger.log_dir
        current_session = self.logger.session_id
        pattern = os.path.join(log_dir, 'mario_session_*')
        all_files = glob.glob(pattern)
        old_files = [f for f in all_files if not os.path.basename(f).startswith(current_session)]
        deleted = 0
        for f in old_files:
            try:
                os.remove(f)
                deleted += 1
            except OSError:
                pass
        if deleted:
            print(f"🗑️  {deleted} log(s) d'anciennes sessions supprimés.")

    def _cleanup_old_historic(self):
        """Supprime les fichiers historic des runs antérieurs à cette session,
        mais conserve toujours le run avec la meilleure distance (max_position_x)."""
        import glob
        historic_dir = self.history_manager.history_dir
        pattern = os.path.join(historic_dir, 'mario_run_*')
        all_files = glob.glob(pattern)

        # Trouver le run_id avec la meilleure distance parmi tous les summaries
        best_run_id = None
        best_x = -1
        for f in all_files:
            if not f.endswith('_summary.json'):
                continue
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                x = data.get('max_position_x', 0)
                if x > best_x:
                    best_x = x
                    best_run_id = data.get('run_id')
            except (OSError, json.JSONDecodeError, KeyError):
                pass

        # Supprimer les fichiers antérieurs à cette session, sauf le meilleur run
        session_start = getattr(self, '_game_session_start', 0)
        old_files = []
        for f in all_files:
            parts = os.path.basename(f).split('_')
            try:
                ts = int(parts[2])  # mario_run_{ts}_...
                if ts < session_start:
                    old_files.append(f)
            except (IndexError, ValueError):
                pass

        deleted = 0
        for f in old_files:
            # Ne jamais supprimer les fichiers du meilleur run
            if best_run_id and os.path.basename(f).startswith(best_run_id):
                continue
            try:
                os.remove(f)
                deleted += 1
            except OSError:
                pass

        if deleted:
            print(f"🗑️  {deleted} fichier(s) historic d'anciens runs supprimés.")
        if best_run_id:
            print(f"🏆 Meilleur run conservé: {best_run_id} ({best_x}px)")

    def play_fluid_mario(self, max_steps=None, forced_game_mode=None):
        """Jouer avec Mario fluide et Claude intelligent.
        forced_game_mode: si fourni, bypasse le menu (ex: 'new_game' en mode auto-amélioration).
        """

        # Timestamp de démarrage de session (pour le nettoyage des anciens historics)
        self._game_session_start = int(time.time())

        if forced_game_mode:
            game_mode, selected_run_id = forced_game_mode, None
        else:
            # Afficher le menu de sélection
            game_mode, selected_run_id = self.show_game_menu()

        if game_mode == "quit":
            self._exit_reason = "user_quit"
            print("👋 Au revoir!")
            return

        # Configuration selon le mode
        if game_mode in ("new_game", "memory_first"):
            # Démarrer un nouveau run dans l'historique
            run_id = self.history_manager.start_new_run()
            self.current_run_started = True
            self.segment_memory.start_run(run_id)

            if game_mode == "memory_first":
                # Phase 3 dès le départ : rejoue les segments mémorisés, IA à la frontière.
                # Subtilités :
                #   - inject_known_solution injecte les macros des segments déjà réussis
                #   - Zone safe (_phase3_ai_mode=False) : run_forward sans pause (terrain connu)
                #   - Au-delà de safe_max_x : _phase3_ai_mode=True → IA reprend la main
                #   - Anti-blocage replay : si bloqué 40f sans avancer → IA immédiatement
                #   - Rewinds illimités pour corriger les erreurs à la frontière
                self._run_phase = 3
                self._phase3_ai_mode = False
                self._last_seg_key = None
                self._segment_in_replay = False
                seg_count = len(self.segment_memory.stage.sequences)
                safe_x = self.segment_memory.stage.safe_max_x
                print(f"🧠 Mode mémoire automatique — {seg_count} segments, safe jusqu'à x={safe_x}px")
                print(f"🔄 Phase 3 : replay mémoire → IA frontière")
            else:
                # Phase 1 : IA pure
                self._run_phase = 1
                self._phase3_ai_mode = False
                self._last_seg_key = None
                self._segment_in_replay = False
                print(f"🔄 Vie 1 → Phase 1: IA pure")

            print(f"🆕 Nouvelle partie - Run: {run_id}")
            self.logger.log_session_start(game_mode, run_id)
        
        print("\n🎮 MARIO FLUIDE avec CLAUDE LLM")
        print("Claude donne des macro-actions, Mario les exécute fluidement!")
        print("=" * 60)
        
        if self.replay_mode:
            mode_display = "🔄 REPLAY + IA"
        elif game_mode == "memory_first":
            mode_display = "🧠 MÉMOIRE AUTO → IA FRONTIÈRE"
        else:
            mode_display = "🤖 IA PURE"
        print(f"Mode: {mode_display}")
        print("=" * 60)
        
        obs = self.env.reset()
        self._raw_action_history.clear()
        self._final_action_history.clear()
        total_reward = 0
        step_count = 0    # Steps totaux de la session (jamais réinitialisé, utilisé en interne)
        _life_step = 0    # Steps de la vie courante (reset à chaque respawn, pour affichage)
        paused = False
        last_mario_decision = None
        _run_max_x = 0  # Position maximale atteinte dans ce run
        
        cv2.namedWindow('Mario Fluide - Claude LLM', cv2.WINDOW_AUTOSIZE)
        
        self._exit_reason = "unknown"

        try:
            while max_steps is None or step_count < max_steps:
                if not paused:
                    # PRIORITÉ 1: Mode replay
                    if self.replay_mode:
                        current_action = self.get_replay_action(step_count)
                        
                        # Si le replay est terminé, passer en mode IA
                        if current_action is None:
                            print("🔄➡️🤖 Transition: Replay terminé, IA prend la main!")
                            # La logique IA normale prendra le relais
                        
                        # Enregistrer l'action rejouée dans le nouvel historique
                        if current_action is not None and self.replay_index > 0:
                            # Obtenir l'action précédente pour l'enregistrer
                            prev_action_data = self.replay_actions[self.replay_index - 1]
                            
                            # Vérifier que prev_action_data n'est pas None et a les clés nécessaires
                            if prev_action_data and 'action_name' in prev_action_data and 'position_x' in prev_action_data:
                                situation = self.analyze_situation(obs, {
                                    'x_pos': prev_action_data['position_x'],
                                    'y_pos': 200,
                                    'score': total_reward
                                }, step_count)
                                
                                # Enregistrer dans le nouvel historique
                                self.record_action(
                                    prev_action_data['action_name'], 
                                    situation, 
                                    step_count, 
                                    f"REPLAY: {prev_action_data.get('reasoning', 'No reasoning')}"
                                )
                    
                    # PRIORITÉ 2: Mode IA (quand replay désactivé ou pas de current_action)
                    if not self.replay_mode or current_action is None:
                        # Obtenir l'action courante de l'IA
                        current_action = self.get_current_action()
                        
                        if current_action is None:
                            # Pas de macro en cours, besoin de Claude
                            situation = self.analyze_situation(obs, {
                                'x_pos': 40 + step_count * 2,
                                'y_pos': 200,
                                'score': total_reward
                            }, step_count)
                            
                            # Sauvegarder pour l'historique d'apprentissage
                            self.last_situation = situation
                            self.current_step = step_count
                            
                            # Démarrer Claude en arrière-plan si pas déjà en cours
                            # et si on a peu d'actions en réserve
                            # Bloqué en Phase 3 zone safe (replay pur jusqu'à la frontière danger)
                            _phase3_safe_zone = self._run_phase == 3 and not self._phase3_ai_mode
                            if not self.claude_thinking and len(self.action_queue) < 3 and not _phase3_safe_zone:
                                trigger_reason = "Post-replay" if hasattr(self, 'replay_mode') and not self.replay_mode else "Normal"
                                print(f"Déclenchement Claude ({trigger_reason}) - thinking:{self.claude_thinking}, queue:{len(self.action_queue)}, step:{step_count}")
                                self.call_claude_async(situation, obs, step_count)

                            # File vide : pause ou replay selon le contexte
                            if len(self.action_queue) == 0:
                                if _phase3_safe_zone or self._segment_in_replay:
                                    # Replay / Phase 3 zone safe : avancer par défaut
                                    for _ in range(3):
                                        self.action_queue.append({
                                            'macro_name': 'run_forward',
                                            'reasoning': 'Replay: avance en attendant le prochain segment',
                                            'strategy': 'Replay fallback', 'urgency': 5, 'confidence': 70
                                        })
                                    current_action = self.get_current_action()
                                else:
                                    # MODE IA : PAUSE TOTALE
                                    # env.step() ne sera PAS appelé → ennemis gelés, timer gelé
                                    # Mario ne court plus dans les trous pendant que Claude réfléchit
                                    current_action = None
                                    if not getattr(self, '_pause_printed', False):
                                        print("⏸  File vide — en attente d'instructions...")
                                        self._pause_printed = True
                            else:
                                self._pause_printed = False
                                # Utiliser l'action en queue
                                current_action = self.get_current_action()
                    
                    # ⚡ COUCHE RÉFLEXE v3 : détection pixel fiable, cooldown 80 frames
                    # Ne s'active PAS si une séquence de saut est en cours ou planifiée :
                    # pipe_jump/obstacle_jump ont besoin de s'exécuter sans interruption
                    current_macro_name = self.current_macro['name'] if self.current_macro else None
                    _JUMP_MACROS = {'stomp_enemy', 'pipe_jump', 'obstacle_jump',
                                    'max_jump', 'run_jump_over', 'big_jump_right'}
                    _jump_in_queue = any(
                        a.get('macro_name') in _JUMP_MACROS
                        for a in self.action_queue
                    )
                    reflex_ready = (
                        self.level_context_established and
                        step_count - self.last_reflex_step >= 25 and
                        current_macro_name in ('run_forward', 'walk_right', None) and
                        current_macro_name not in _JUMP_MACROS and
                        not _jump_in_queue
                    )
                    if reflex_ready and self.check_immediate_threat(obs):
                        self.inject_emergency_jump()
                        current_action = self.get_current_action()
                        self.last_reflex_step = step_count
                        print("⚡ RÉFLEXE v3: ennemi détecté → run_jump_over (35f, A+B) !")

                    # ⛰️ RÉFLEXE TROU : saut max automatique quand le sol est absent devant Mario
                    # Cooldown court (15 frames) — priorité absolue sur toute autre macro
                    # Ne se déclenche PAS si Mario est déjà en plein saut (max_jump, short_jump…)
                    _is_jumping = current_macro_name in (
                        'max_jump', 'short_jump', 'long_jump', 'high_jump',
                        'precise_jump', 'run_jump_over', 'stomp_enemy', 'big_jump_right',
                        'pipe_jump', 'obstacle_jump'
                    )
                    reflex_hole_ready = (
                        self.level_context_established and
                        step_count - self.last_hole_reflex_step >= 15 and
                        not _is_jumping
                    )
                    if reflex_hole_ready:
                        _hole = self.detect_holes_ahead(obs)
                        if _hole['detected'] and _hole['urgent']:
                            _cur_x_hole = real_info.get('x_pos', 0) if 'real_info' in locals() else 0
                            # Compteur de boucle : si Mario n'a pas avancé de ≥10px depuis le dernier réflexe
                            if abs(_cur_x_hole - self._hole_reflex_last_x) < 10:
                                self._hole_reflex_count += 1
                            else:
                                self._hole_reflex_count = 0
                            self._hole_reflex_last_x = _cur_x_hole
                            # Après 3 déclenchements sans progression → cooldown long (150f) pour laisser
                            # Claude choisir une autre stratégie (ex: approach + max_jump avec px=30)
                            if self._hole_reflex_count >= 3:
                                print(f"⛰️ RÉFLEXE TROU boucle détectée ({self._hole_reflex_count}× à x={_cur_x_hole}) "
                                      f"→ cooldown 150f, laisser Claude décider")
                                self.last_hole_reflex_step = step_count + 135  # +135 = cooldown 150f total
                                self._hole_reflex_count = 0
                            else:
                                self.inject_hole_jump(_hole)
                                current_action = self.get_current_action()
                                self.last_hole_reflex_step = step_count
                                severity = "CRITIQUE" if _hole['critical'] else "URGENT"
                                print(f"⛰️ RÉFLEXE TROU [{severity}] #{self._hole_reflex_count}: trou à {_hole['nearest']}px "
                                      f"(larg={_hole['width']}px) → max_jump!")

                    # ⚠️ ZONE DANGER post-rewind : saut automatique si Mario approche de la mort précédente
                    if (self._danger_zone_x is not None and
                            current_macro_name not in _JUMP_MACROS and
                            not _jump_in_queue and
                            'real_info' in locals() and
                            step_count - self.last_reflex_step >= 25):
                        _cur_x = real_info.get('x_pos', 0)
                        # Déclencher 40px avant la zone fatale
                        if self._danger_zone_x - 40 <= _cur_x <= self._danger_zone_x + 10:
                            self.inject_emergency_jump()
                            current_action = self.get_current_action()
                            self.last_reflex_step = step_count
                            print(f"⚠️ ZONE DANGER: saut forcé à x={_cur_x} (mort précédente à x={self._danger_zone_x})")
                            self._danger_zone_x = None  # utilisé, reset

                    # Exécuter l'action dans le jeu
                    done = False  # valeur par défaut quand env.step() n'est pas appelé (pause)
                    if current_action is not None:
                        obs, reward, done, real_info = self.env.step(current_action)
                        self._raw_action_history.append(int(current_action))
                        self._final_action_history.append(int(current_action))
                        total_reward += reward
                        step_count += 1
                        _life_step += 1

                        # 📚 MÉMOIRE SEGMENTS : position + événements de jeu
                        _seg_x = real_info.get('x_pos', 0)
                        if _seg_x > _run_max_x:
                            _run_max_x = _seg_x
                        self.segment_memory.record_position(_seg_x, step_count)

                        # 💾 Checkpoint rewind (toutes les 60 frames)
                        if step_count % 60 == 0 and not self._rewind_active:
                            _ram_snap = self.env.unwrapped._ram_buffer().copy()
                            _recent_macros = [m['name'] for m in list(self.macro_history)[-8:]]
                            self.rewind_buffer.append({
                                'step': step_count,
                                'ram': _ram_snap,
                                'x_pos': int(_seg_x),
                                'macros': _recent_macros,
                                'action_history': list(self._raw_action_history),
                            })
                            self.logger.log_game_event("REWIND_CHECKPOINT", step_count, {
                                "x_pos": int(_seg_x), "buffer_size": len(self.rewind_buffer),
                                "history_len": len(self._raw_action_history)})

                        # Détecter transition de segment (uniquement en avançant)
                        _new_seg_key = self.segment_memory._key(int(_seg_x))
                        if (_new_seg_key != self._last_seg_key and
                                (self._last_seg_key is None or
                                 int(_new_seg_key.split('-')[0]) >
                                 int(self._last_seg_key.split('-')[0]))):
                            self._on_segment_enter(_new_seg_key, int(_seg_x), step_count)
                            self._last_seg_key = _new_seg_key

                        # Enregistrer la macro courante pour la mémoire du meilleur passage
                        if current_macro_name:
                            self.segment_memory.record_macro_in_segment(
                                int(_seg_x), current_macro_name)

                        # ⚠️ ANTI-BLOCAGE REPLAY (Phase 2 ET Phase 3 zone safe)
                        # detect_stuck normal exige 4/6 même action → inefficace sur réflexes alternés
                        # Ce check position-seule couvre Phase 2 ET Phase 3 pendant les segments replay
                        if self._segment_in_replay:
                            if int(_seg_x) > self._phase3_last_x + 10:
                                # Progression : mise à jour des références
                                self._phase3_last_x = int(_seg_x)
                                self._phase3_last_x_step = step_count
                            elif step_count - self._phase3_last_x_step >= 40:
                                # Bloqué 40 frames sans avancer de 10px → sortir du replay
                                print(f"⚠️ Anti-blocage replay (Phase {self._run_phase}): "
                                      f"bloqué à x={_seg_x:.0f} depuis "
                                      f"{step_count - self._phase3_last_x_step} frames → IA reprend la main")
                                self._segment_in_replay = False
                                self.action_queue.clear()
                                if self._run_phase == 3:
                                    self._phase3_ai_mode = True
                                # Effacer la séquence bloquante du run en cours
                                # (le stage ne sera pas écrasé si ce run fait moins bien)
                                _stuck_seg_key = self.segment_memory._key(int(_seg_x))
                                self.segment_memory._run_macros_per_segment.pop(_stuck_seg_key, None)
                                # Démarrer le tracking déblocage pour sauvegarder la solution IA
                                self._unstick_start_x = int(_seg_x)
                                self._unstick_sequence = None
                                self._unstick_step = step_count

                        # 🏆 DÉTECTION DE DÉBLOCAGE RÉUSSI : Mario a franchil'obstacle
                        if (self._unstick_start_x is not None and
                                _seg_x > self._unstick_start_x + 30):
                            _steps_taken = step_count - self._unstick_step
                            if self._unstick_sequence is None:
                                # Séquence déterminée par Claude : capturer les macros exécutées
                                _win_seq = [m['name'] for m in list(self.macro_history)[-4:]]
                            else:
                                _win_seq = self._unstick_sequence
                            if _win_seq:
                                self.segment_memory.record_success(
                                    int(self._unstick_start_x), _win_seq, _steps_taken)
                                print(f"🏆 Déblocage mémorisé x={self._unstick_start_x}: "
                                      f"{' → '.join(_win_seq)} ({_steps_taken} steps)")
                            self._unstick_start_x = None
                            self._unstick_sequence = None
                        # Ennemi écrasé : reward élevé + action de saut
                        if reward > 100 and current_macro_name in (
                                'stomp_enemy', 'short_jump', 'max_jump',
                                'run_jump_over', 'big_jump_right'):
                            self.segment_memory.record_enemy_stomp(_seg_x, 'ennemi', int(reward))
                        # Items collectés : score ou pièces augmentent
                        _new_score = real_info.get('score', 0)
                        _new_coins = real_info.get('coins', 0)
                        _score_delta = _new_score - self._prev_score
                        _is_stomp = current_macro_name in (
                            'stomp_enemy', 'short_jump', 'max_jump',
                            'run_jump_over', 'big_jump_right')
                        if _score_delta >= 1000:
                            self.segment_memory.record_item_collected(
                                _seg_x, 'item', _score_delta)
                            # Power-up issu d'un bloc ? (pas d'un ennemi écrasé)
                            if not _is_stomp:
                                self.segment_memory.record_block_hit_in_run(int(_seg_x))
                        if _new_coins > self._prev_coins:
                            self.segment_memory.record_item_collected(
                                _seg_x, 'coin', (_new_coins - self._prev_coins) * 200)
                            # Pièce(s) issue(s) d'un bloc ? frappé
                            self.segment_memory.record_block_hit_in_run(int(_seg_x))
                        self._prev_score = _new_score
                        self._prev_coins = _new_coins

                    # 🔁 DÉTECTION DE BLOCAGE : position + répétition d'actions
                    if self.level_context_established:
                        current_x = real_info.get('x_pos', 0) if 'real_info' in locals() else 0
                        stuck_level = self.detect_stuck(current_x, step_count)

                        if stuck_level >= 2:
                            # Niveau 2 : recherche web + appel Claude déblocage
                            # Exclure les primitives de base (walk/run/jump) de la liste "échouées"
                            # pour que Claude puisse toujours les utiliser dans de nouvelles combinaisons
                            _always_available = {'walk_right', 'run_forward', 'max_jump',
                                                 'run_jump_over', 'pipe_jump', 'obstacle_jump'}
                            failed = [m['name'] for m in list(self.macro_history)[-6:]
                                      if m['name'] not in _always_available]
                            web = self.search_mario_strategy(current_x, failed)
                            sit = self.last_situation or {}
                            self.call_claude_stuck_mode(sit, current_x, failed, obs, step_count, web)
                            self.stuck_counter = 0  # Réinitialiser après intervention
                            # Forcer cooldown réflexe pour laisser pipe_jump/obstacle_jump s'exécuter
                            self.last_reflex_step = step_count
                            # Mémoriser la tentative (séquence déterminée après succès)
                            self._unstick_start_x = current_x
                            self._unstick_sequence = None  # Sera capturée depuis macro_history
                            self._unstick_step = step_count
                        elif stuck_level == 1:
                            # Niveau 1 : séquence de déblocage NES-validée
                            # walk_right d'abord pour atteindre le pied de l'obstacle,
                            # PUIS max_jump depuis le pied (physique NES confirmée)
                            _unstick_seq = ['walk_right', 'max_jump', 'run_forward']
                            print(f"🔁 Blocage léger: séquence {' → '.join(_unstick_seq)} (physique NES)")
                            self.action_queue.clear()
                            self.current_macro = None
                            for alt in _unstick_seq:
                                self.action_queue.append({
                                    'macro_name': alt,
                                    'reasoning': 'Déblocage NES: marcher jusqu\'au pied puis saut maximum',
                                    'strategy': 'Anti-blocage mur', 'urgency': 7, 'confidence': 85
                                })
                            # Mémoriser la tentative de déblocage
                            self._unstick_start_x = current_x
                            self._unstick_sequence = _unstick_seq
                            self._unstick_step = step_count

                    # Si la queue de replay est épuisée, sortir du mode replay
                    if self._segment_in_replay and len(self.action_queue) == 0:
                        self._segment_in_replay = False

                    # 🧠 INJECTION PROACTIVE : solutions mémorisées dans les 120px devant Mario
                    # Priorité maximale — bypasse Claude si une solution confirmée existe
                    # Bloqué pendant toute la zone safe du Phase 3 (replay pur jusqu'à la frontière)
                    if (self.level_context_established and
                            len(self.action_queue) <= 2 and not self.claude_thinking and
                            not self._segment_in_replay and
                            not (self._run_phase == 3 and not self._phase3_ai_mode)):
                        _mem_x = real_info.get('x_pos', 0) if 'real_info' in locals() else 0
                        self.inject_known_solution(int(_mem_x), step_count)

                    # 🚀 DÉCLENCHEMENT HYBRIDE OPTIMISÉ
                    # Bloqué pendant le replay d'un segment mémorisé ET toute la zone safe du Phase 3
                    _not_replay_zone = (
                        not self._segment_in_replay and
                        not (self._run_phase == 3 and not self._phase3_ai_mode)
                    )
                    # Déclenchement normal : queue presque vide
                    _queue_trigger = (len(self.action_queue) <= 2 and
                                      step_count - self.last_positions_update >= self.positions_update_frequency)
                    # Déclenchement périodique : toutes les 60 frames même avec actions en queue
                    # → Claude voit les nouveaux décors et ennemis apparus depuis le dernier appel
                    _periodic_trigger = (len(self.action_queue) < 4 and
                                         step_count - self.last_screenshot_step >= 60)
                    # Déclenchement position : Mario a avancé >60px depuis dernier screenshot
                    # → vider la queue + annuler call Claude en cours → pause/rescan immédiat
                    # Fonctionne même pendant un saut (current_macro continue, mais queue vidée)
                    _cur_x_for_trigger = real_info.get('x_pos', self.last_screenshot_x) if 'real_info' in locals() else self.last_screenshot_x
                    _advanced_px = _cur_x_for_trigger - self.last_screenshot_x
                    _currently_jumping = current_macro_name in _JUMP_MACROS
                    _position_trigger = (
                        _advanced_px >= 60 and
                        _not_replay_zone
                    )
                    if _position_trigger:
                        # Vider la queue → pause naturelle après le saut actuel
                        self.action_queue.clear()
                        # Si Claude pense encore sur des infos périmées → annuler son call
                        if self.claude_thinking:
                            self._claude_generation += 1
                            self.claude_thinking = False
                            _jump_note = " (en saut, rescan à l'atterrissage)" if _currently_jumping else ""
                            print(f"📸 POSITION TRIGGER: Mario +{_advanced_px}px, call Claude annulé → rescan{_jump_note} (x={_cur_x_for_trigger})")
                        else:
                            _jump_note = " (en saut, rescan à l'atterrissage)" if _currently_jumping else ""
                            print(f"📸 POSITION TRIGGER: Mario +{_advanced_px}px → pause forcée{_jump_note} (x={_cur_x_for_trigger})")

                    should_trigger_claude = (
                        not self.claude_thinking and
                        (_queue_trigger or _periodic_trigger or _position_trigger) and
                        _not_replay_zone
                    )

                    if should_trigger_claude:
                        _why = "queue basse" if _queue_trigger else ("position +60px" if _position_trigger else "périodique 60f")
                        trigger_type = "📸 Initial" if not self.level_context_established else "📸 Scan"
                        print(f"🚀 Déclenchement {trigger_type} [{_why}] - queue:{len(self.action_queue)}, step:{step_count}")
                        
                        # Après un rewind, utiliser real_info capturé pendant le replay
                        # (évite d'envoyer la position de mort à Claude au lieu du checkpoint)
                        _info_for_situation = (
                            self._rewind_real_info or
                            (real_info if 'real_info' in locals() else {
                                'x_pos': 40 + step_count * 2, 'y_pos': 200, 'score': total_reward
                            })
                        )
                        self._rewind_real_info = None  # consommé
                        situation = self.analyze_situation(obs, _info_for_situation, step_count)
                        # Mettre à jour la position de référence pour le prochain trigger position
                        self.last_screenshot_x = situation.get('mario', {}).get('x', self.last_screenshot_x)
                        
                        # Enregistrer les stratégies qui marchent bien
                        if len(self.action_history) >= 3:
                            recent_actions = list(self.action_history)[-3:]
                            progress_made = situation.get('progress', {}).get('trend', 0)
                            if progress_made > 20:  # Bon progrès
                                action_names = [a['action'] for a in recent_actions]
                                self.record_successful_strategy(action_names, progress_made)
                        
                        self.call_claude_async(situation, obs, step_count)
                    
                    # Gérer la mort de Mario
                    if done:
                        mario_lives_env = real_info.get('life', 0)
                        flag_get = real_info.get('flag_get', False)
                        
                        if flag_get:
                            print("🎉 VICTOIRE! Mario a terminé le niveau!")
                            last_mario_decision = {'reasoning': 'VICTOIRE! Niveau terminé!', 'strategy': 'Mission accomplie'}
                            
                            # Logger la victoire
                            self.logger.log_game_event("VICTORY", step_count, {
                                "final_score": total_reward,
                                "steps_taken": step_count,
                                "api_calls": self.api_calls,
                                "total_cost": self.total_cost
                            })
                            
                            # Terminer le run avec victoire
                            self.segment_memory.finalize_stage(_run_max_x, step_count, died=False)
                            if self.current_run_started:
                                summary = self.history_manager.end_run("victory", total_reward)
                                if summary:
                                    self.history_manager.print_run_summary(summary)

                            time.sleep(3)  # Pause pour admirer la victoire
                            self._exit_reason = "victory"
                            break
                        else:
                            # Mario est mort - mettre à jour le compteur interne
                            self.mario_lives_remaining = mario_lives_env
                            mario_x_death = real_info.get('x_pos', situation.get('mario', {}).get('x', 0))
                            self.record_death(step_count, mario_x_death)
                            self.lives_used += 1
                            print(f"💀 Mario est mort! (Mort #{self.deaths_count}) Vies restantes: {self.mario_lives_remaining}")

                            # 📚 Mémoriser la mort par segment
                            _time_left = real_info.get('time', 400)
                            _mario_y = real_info.get('y_pos', 0)
                            if _time_left == 0:
                                _death_cause = 'time_out'
                            elif _mario_y > 210:
                                _death_cause = 'fell_in_hole'
                            else:
                                _death_cause = 'enemy_hit'
                            _approach = [m['name'] for m in list(self.macro_history)[-3:]]
                            _last_macro = _approach[-1] if _approach else 'unknown'
                            # Ne sauvegarder que les morts à la frontière (pas les morts
                            # dans des zones déjà franchies lors de runs précédents)
                            if mario_x_death >= self.segment_memory.furthest_x:
                                self.segment_memory.record_death(
                                    mario_x_death, _death_cause, _last_macro, _approach)
                                # Marquer aussi les 2 segments précédant la mort comme approche fatale
                                self.segment_memory.record_death_approach(mario_x_death, n_approach=2)
                            else:
                                print(f"ℹ️ Mort à x={mario_x_death} ignorée "
                                      f"(déjà dépassé en run précédent, record={self.segment_memory.furthest_x})")
                            
                            # Tentative de rewind avant game over
                            self.logger.log_game_event("DEATH_REWIND_CHECK", step_count, {
                                "deaths": self.deaths_count, "rewind_count": self.rewind_count,
                                "max_rewinds": self.max_rewinds, "buffer_len": len(self.rewind_buffer),
                                "death_x": mario_x_death})
                            if (self.deaths_count >= 1 and
                                    self.rewind_count < self.max_rewinds and
                                    self.rewind_buffer):
                                # Choisir le checkpoint le plus éloigné de la zone de mort
                                # (le plus ancien du buffer = le plus de marge avant le danger)
                                checkpoint = self.rewind_buffer[0]
                                self.logger.log_game_event("REWIND_START", step_count, {
                                    "rewind_num": self.rewind_count + 1,
                                    "checkpoint_x": checkpoint['x_pos'],
                                    "checkpoint_step": checkpoint['step'],
                                    "death_x": mario_x_death,
                                    "cause": _death_cause})
                                self.rewind_count += 1
                                self._rewind_active = True

                                # Restauration RAM sans env.reset() pour préserver le PPU.
                                #
                                # Problème de env.reset() + RAM restore :
                                #   env.reset() réinitialise le PPU (nametables = tiles du début
                                #   du niveau). Après restore RAM à x=1077+, le jeu tente de
                                #   marcher sur des tiles qui n'existent pas → Mario tombe à
                                #   travers le sol.
                                #
                                # Solution : effacer uniquement le flag `done` sur NESEnv
                                # (seul vrai verrou de env.step()), puis restaurer la RAM.
                                # Le PPU VRAM du jeu courant est préservé → collision correcte.
                                checkpoint_ram = checkpoint.get('ram')
                                self._raw_action_history.clear()
                                # Historique "propre" : on revient à l'état du checkpoint
                                # (supprime les actions du segment raté)
                                self._final_action_history = list(checkpoint.get('action_history', []))
                                _last_replay_info = {}
                                if checkpoint_ram is not None:
                                    # 1. Débloquer env.step() sans toucher au PPU
                                    self.env.unwrapped.done = False
                                    # 2. Restaurer l'état NES exact du checkpoint
                                    np.copyto(self.env.unwrapped._ram_buffer(), checkpoint_ram)
                                    # 3. NOOP pour rafraîchir l'écran et récupérer les infos
                                    obs, _, _, _last_replay_info = self.env.step(0)
                                    print(f"⏪ RAM restaurée (PPU intact) → x={checkpoint['x_pos']} "
                                          f"(step={checkpoint['step']})")
                                else:
                                    # Fallback : replay complet depuis frame 0 si pas de snapshot
                                    self.env.reset()
                                    _replay_actions = checkpoint.get('action_history', [])
                                    print(f"⏪ Fallback replay {len(_replay_actions)} actions "
                                          f"(pas de snapshot RAM)")
                                    _replay_died = False
                                    for _ra in _replay_actions:
                                        obs, _, _done, _last_replay_info = self.env.step(_ra)
                                        if _done:
                                            _replay_died = True
                                            break
                                    if _replay_died:
                                        obs = self.env.reset()
                                        _last_replay_info = {'x_pos': 40, 'y_pos': 200,
                                                             'score': 0}
                                        print("⚠️ Fallback replay mort — reset au début")
                                # Stocker real_info du checkpoint (pas de la mort) pour
                                # que le prochain analyze_situation ait la bonne position
                                self._rewind_real_info = _last_replay_info

                                # Préparer le message de correction pour le prochain appel Claude
                                # (injecté en tête du prompt screenshot/texte via _rewind_correction_msg)
                                cause_fr = {'enemy_hit': 'ennemi', 'fell_in_hole': 'trou', 'time_out': 'temps écoulé'}
                                # Utiliser macro_history AU MOMENT DE LA MORT (pas du checkpoint)
                                # pour avoir la vraie séquence fatale
                                _death_macros = [m['name'] for m in list(self.macro_history)[-8:]]
                                macros_str = ' → '.join(_death_macros) or '(aucune)'
                                _danger_dist = mario_x_death - checkpoint['x_pos']
                                # Enregistrer cette mort dans l'historique rewind
                                self._death_positions.append({
                                    'x': mario_x_death,
                                    'y': _mario_y,
                                    'cause': _death_cause,
                                    'rewind': self.rewind_count,
                                })
                                # Construire la section historique si morts précédentes
                                _prev_deaths = self._death_positions[:-1]  # toutes sauf la courante
                                _history_lines = ""
                                if _prev_deaths:
                                    _h = ', '.join(
                                        f"x={d['x']} y={d['y']} ({cause_fr.get(d['cause'], d['cause'])})"
                                        for d in _prev_deaths
                                    )
                                    _history_lines = f"Morts précédentes dans ce run : {_h}\n"
                                self._rewind_correction_msg = (
                                    f"⏪ REWIND #{self.rewind_count}/{self.max_rewinds} — MARIO VIENT DE MOURIR !\n"
                                    f"Position de mort : x={mario_x_death}, y={_mario_y}\n"
                                    f"Cause : {cause_fr.get(_death_cause, _death_cause)}\n"
                                    f"Séquence fatale : {macros_str}\n"
                                    f"Zone dangereuse : à partir de x={mario_x_death - 20} "
                                    f"(mort à {_danger_dist}px du checkpoint)\n"
                                    f"{_history_lines}"
                                    f"Le jeu est REMBOBINÉ à x={checkpoint['x_pos']}.\n"
                                    f"OBLIGATOIRE : propose des actions DIFFÉRENTES — "
                                    f"{'SAUTE par-dessus la zone x=' + str(mario_x_death) if _death_cause == 'enemy_hit' else 'EVITE le trou à x=' + str(mario_x_death) if _death_cause == 'fell_in_hole' else 'Avance vite avant le timer'}."
                                )

                                # Mémoriser la zone dangereuse (filet de sécurité : saut auto)
                                self._danger_zone_x = mario_x_death
                                # Invalider le thread Claude en cours (s'il y en a un)
                                self._claude_generation += 1
                                # Libérer le verrou claude_thinking pour que le prochain cycle
                                # puisse déclencher un nouvel appel immédiatement sans attendre
                                # que l'ancien thread API se termine (il verra generation != et
                                # abandonnera ses résultats dans son finally).
                                self.claude_thinking = False
                                # Vider la queue → PAUSE automatique → Claude sera appelé au prochain cycle
                                self.action_queue.clear()
                                self.current_macro = None

                                # Réinitialiser les compteurs de death pour continuer
                                self.deaths_count -= 1  # Annuler la mort comptée
                                self.rewind_buffer.clear()  # Vider le buffer après rewind
                                # Sauvegarder immédiatement l'état restauré comme nouveau checkpoint.
                                # Sans ça : si Mario re-meurt en 1-2 steps (ennemi à 2px), le buffer
                                # est vide → pas de rewind possible → game over injuste.
                                _post_rewind_ram = self.env.unwrapped._ram_buffer().copy()
                                self.rewind_buffer.append({
                                    'step': step_count,
                                    'ram': _post_rewind_ram,
                                    'x_pos': checkpoint['x_pos'],
                                    'macros': [],
                                    'action_history': list(self._final_action_history),
                                })
                                # Réinitialiser les cooldowns réflexes
                                self.last_reflex_step = step_count - 30  # réflexes actifs immédiatement
                                self.last_hole_reflex_step = step_count - 20
                                self._hole_reflex_count = 0  # Reset compteur boucle trou
                                self.logger.log_game_event("REWIND_OK", step_count, {
                                    "checkpoint_x": checkpoint['x_pos'],
                                    "death_x": mario_x_death,
                                    "deaths_count_after": self.deaths_count})
                                # Ne PAS faire continue : laisser la boucle atteindre le display
                                # pour que l'overlay REWIND soit visible une frame
                                # _rewind_active sera remis à False après l'imshow
                                done = False  # Empêcher le break en fin de bloc

                            # Vérifier si c'est vraiment game over (1 vie = game over immédiat)
                            # Skip si un rewind vient d'être appliqué ce frame
                            if self._rewind_active:
                                pass  # Le rewind gère la suite — aller vers le display
                            elif self.deaths_count >= 1:
                                print("💀 GAME OVER - Mario a perdu sa vie!")
                                
                                # Logger le game over
                                self.logger.log_game_event("GAME_OVER", step_count, {
                                    "deaths": self.deaths_count,
                                    "final_score": total_reward,
                                    "steps_taken": step_count,
                                    "final_position": mario_x_death
                                })
                                
                                # Terminer le run avec mort
                                self.segment_memory.finalize_stage(_run_max_x, step_count, died=True)
                                if self.current_run_started:
                                    summary = self.history_manager.end_run("death", total_reward)
                                    if summary:
                                        self.history_manager.print_run_summary(summary)

                                self._exit_reason = "game_over"
                                break
                            else:
                                print("🔄 Redémarrage automatique...")
                                # Sauvegarder les données de la vie qui vient de se terminer
                                # (chaque vie est un run indépendant pour la mémoire segments)
                                self.segment_memory.finalize_stage(_run_max_x, step_count, died=True)
                                _run_max_x = 0  # Reset pour la prochaine vie
                                _life_step = 0  # Reset du compteur de steps
                                self._life_macro_count = 0  # Reset du compteur de séquences affiché
                                # Réinitialiser l'état pour la nouvelle vie
                                obs = self.env.reset()
                                self._raw_action_history.clear()
                                self.current_macro = None
                                self.position_history.clear()
                                self.action_queue.clear()
                                self._segment_in_replay = False
                                self._last_seg_key = None
                                self._phase3_ai_mode = False
                                self._phase3_last_x = 0
                                self._phase3_last_x_step = 0
                                # deaths_count vient d'être incrémenté :
                                # 1 mort → vie 2 → phase 2, 2 morts → vie 3 → phase 3
                                self._run_phase = min(self.deaths_count + 1, 3)
                                # Démarrer un nouveau tracking de macros pour cette vie
                                _life_run_id = f"life{self._run_phase}_{int(time.time())}"
                                self.segment_memory.start_run(_life_run_id)
                                _labels = {1: "IA pure", 2: "Mixte (50% replay/IA)",
                                           3: "Mémoire → IA frontière"}
                                print(f"🔄 Vie {self._run_phase} → Phase {self._run_phase}: "
                                      f"{_labels[self._run_phase]}")
                                time.sleep(1)  # Pause pour voir le redémarrage
                
                # Affichage fluide
                situation = self.analyze_situation(obs, real_info if 'real_info' in locals() else {
                    'x_pos': 40 + step_count * 2, 'y_pos': 200, 'score': total_reward
                }, step_count)

                display = self.create_display(obs, situation, last_mario_decision, total_reward, step_count)

                # Overlay "PAUSE — Claude réfléchit" quand le jeu est gelé
                _ia_paused = (current_action is None and not self._segment_in_replay
                              and not paused)  # paused = pause ESPACE manuelle
                if _ia_paused:
                    h, w = display.shape[:2]
                    cx, cy = w // 2, h // 2
                    # Fond semi-transparent
                    overlay = display.copy()
                    cv2.rectangle(overlay, (cx - 210, cy - 22), (cx + 210, cy + 22), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
                    # Texte
                    label = "PAUSE  Analyse IA en cours..." if self.claude_thinking else "PAUSE  En attente d'instructions"
                    cv2.putText(display, label, (cx - 200, cy + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)

                # Overlay "⏪ REWIND" quand le système de rewind est actif
                if self._rewind_active:
                    h, w = display.shape[:2]
                    cx, cy = w // 2, h // 2
                    overlay = display.copy()
                    cv2.rectangle(overlay, (cx - 210, cy - 22), (cx + 210, cy + 22), (0, 0, 80), -1)
                    cv2.addWeighted(overlay, 0.70, display, 0.30, 0, display)
                    label = f"REWIND #{self.rewind_count}/{self.max_rewinds}  Claude corrige..."
                    cv2.putText(display, label, (cx - 200, cy + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 255), 2)

                cv2.imshow('Mario Fluide - Claude LLM', display)
                if self._rewind_active:
                    self._rewind_active = False  # Overlay montré, on réinitialise

                # Contrôles (30 FPS pour être plus réactif avec Claude)
                key = cv2.waitKey(33) & 0xFF  # ~30 FPS pour Claude
                if key == 27:  # ESC
                    self._exit_reason = "user_esc"
                    break
                # Fermeture de la fenêtre avec le bouton X
                if cv2.getWindowProperty('Mario Fluide - Claude LLM', cv2.WND_PROP_VISIBLE) < 1:
                    self._exit_reason = "window_closed"
                    break
                elif key == 32:  # ESPACE
                    paused = not paused
                    print("⏸️ Pause" if paused else "▶️ Reprise")
                elif self.handle_scroll_keys(key):
                    # Touche de défilement traitée
                    pass
                    
        except KeyboardInterrupt:
            self._exit_reason = "keyboard_interrupt"
            print("\n⏹️ Arrêt demandé")
            
            # Logger l'interruption
            self.logger.log_game_event("INTERRUPTED", step_count, {
                "reason": "user_keyboard_interrupt",
                "final_score": total_reward,
                "steps_taken": step_count
            })
            
            # Terminer le run avec interruption
            self.segment_memory.finalize_stage(_run_max_x, step_count, died=True)
            if self.current_run_started:
                summary = self.history_manager.end_run("interrupted", total_reward)
                if summary:
                    self.history_manager.print_run_summary(summary)
        
        finally:
            # Logger la fin de session avec statistiques
            final_stats = {
                "steps_total": step_count,
                "final_score": total_reward,
                "deaths": self.deaths_count,
                "lives_used": self.lives_used,
                "api_calls": self.api_calls,
                "successful_macros": self.successful_macros,
                "total_cost": self.total_cost,
                "final_position": real_info.get('x_pos', 0) if 'real_info' in locals() else 0
            }

            self.logger.log_session_end(final_stats)

            # 💾 Sauvegarder le run parfait (historique tronqué aux rewinds)
            self._save_perfect_run()

            cv2.destroyAllWindows()
            self.env.close()

            # Supprimer les screenshots de débogage
            self._cleanup_screenshots()

            # Fermer le logger (finalise l'écriture des fichiers)
            self.logger.close()

            # Supprimer les logs des sessions précédentes
            self._cleanup_old_logs()

            # Supprimer les historics des runs précédents
            self._cleanup_old_historic()

            # Afficher les fichiers de log créés
            log_files = self.logger.get_session_files()
            print(f"\n📝 Fichiers de log créés:")
            for log_file in log_files:
                if os.path.exists(log_file):
                    size_kb = os.path.getsize(log_file) / 1024
                    print(f"   📄 {os.path.basename(log_file)} ({size_kb:.1f} KB)")
        
        # Raison d'arrêt
        _exit_labels = {
            "victory": "🎉 Niveau terminé !",
            "game_over": "💀 GAME OVER (3 morts)",
            "user_esc": "⏹️ Arrêt par l'utilisateur (ESC)",
            "window_closed": "🪟 Fenêtre fermée",
            "keyboard_interrupt": "⌨️ Interruption clavier (Ctrl+C)",
        }
        if max_steps is not None and self._exit_reason == "unknown":
            self._exit_reason = "max_steps"
            _exit_labels["max_steps"] = f"⏱️ Limite de steps atteinte ({max_steps} steps)"
        print(f"\n🔚 Fin de partie : {_exit_labels.get(self._exit_reason, f'Raison inconnue ({self._exit_reason})')}")

        # Statistiques finales
        print(f"\n🏆 RÉSULTATS MARIO FLUIDE:")
        print(f"   🎮 Steps total: {step_count}")
        print(f"   🏆 Score final: {total_reward}")
        print(f"   💀 Morts de Mario: {self.deaths_count}")
        print(f"   ❤️ Vies utilisées: {self.lives_used}")
        print(f"   🧠 Décisions Claude: {self.api_calls}")
        print(f"   ⚡ Macros réussies: {self.successful_macros}")
        print(f"   💰 Coût total: ${self.total_cost:.3f}")
        print(f"   🚀 Distance finale: {real_info.get('x_pos', 0) if 'real_info' in locals() else 0}")
        
        # Taux de réussite
        if self.deaths_count > 0:
            survival_rate = (step_count - self.deaths_count * 20) / step_count * 100  # Approximation
            print(f"   📈 Taux de survie: {survival_rate:.1f}%")
        
        # Afficher les statistiques d'historique
        print(f"\n📊 HISTORIQUE GLOBAL:")
        updated_stats = self.history_manager.get_run_stats()
        print(f"   🏃 Runs totaux: {updated_stats.get('total_runs', 0)}")
        if updated_stats.get('total_runs', 0) > 0:
            print(f"   🏆 Record distance: {updated_stats['best_distance']} pixels")
            print(f"   🚀 Record vitesse: {updated_stats['best_speed']:.2f} px/s")
            completion = updated_stats.get('completion_rates', {})
            print(f"   🎯 Victoires: {completion.get('victory', 0)} | Morts: {completion.get('death', 0)} | Interruptions: {completion.get('interrupted', 0)}")
            
            # Comparer avec le meilleur run
            best_run = self.history_manager.get_best_run()
            if best_run:
                current_distance = real_info.get('x_pos', 0) if 'real_info' in locals() else 0
                if current_distance > best_run.max_position_x:
                    print(f"   🎉 NOUVEAU RECORD! Ancien: {best_run.max_position_x} → Nouveau: {current_distance}")
                else:
                    print(f"   📊 Performance: {current_distance}/{best_run.max_position_x} pixels du record")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mario Bros IA - Claude LLM")
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Nombre maximum de steps avant arrêt automatique (défaut: illimité)"
    )
    args = parser.parse_args()

    print("🚀 Mario Bros FLUIDE - Claude LLM avec Macro-Actions")
    print("Mario exécute les décisions de Claude de façon naturelle!")
    if args.max_steps:
        print(f"⏱️  Arrêt automatique après {args.max_steps} steps")

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY requise!")
        return

    while True:
        mario_fluid = None
        try:
            mario_fluid = MarioFluidLLM()
            mario_fluid.play_fluid_mario(max_steps=args.max_steps)

        except KeyboardInterrupt:
            print("\n⛔ Arrêt demandé (Ctrl+C)")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()

        # Récupérer la raison d'arrêt
        exit_reason = getattr(mario_fluid, '_exit_reason', 'unknown') if mario_fluid else 'error'

        # Arrêts volontaires → quitter
        if exit_reason in ('keyboard_interrupt', 'user_esc', 'window_closed', 'user_quit'):
            print("\n👋 À bientôt !")
            break

        # Fin naturelle (game_over, victoire, max_steps) → retour au menu automatiquement
        if exit_reason in ('game_over', 'victory', 'max_steps'):
            print("\n" + "="*60)
            print("🔄 Retour au menu...")
            continue

        # Autres cas (erreur inconnue) → demander
        print("\n" + "="*60)
        try:
            again = input("🔄 Nouvelle partie ? (o/N) : ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            again = "n"
        if again != "o":
            print("👋 À bientôt !")
            break

if __name__ == "__main__":
    main()