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
from mario_segment_memory import MarioSegmentMemory, get_memory_path
from mario_auto_improver import MarioAutoImprover, CONFIG_FILE, TUNABLE_PARAMS

# Séquence complète des niveaux Super Mario Bros (World 1-1 → World 8-4)
LEVEL_SEQUENCE = [
    (1,1),(1,2),(1,3),(1,4),
    (2,1),(2,2),(2,3),(2,4),
    (3,1),(3,2),(3,3),(3,4),
    (4,1),(4,2),(4,3),(4,4),
    (5,1),(5,2),(5,3),(5,4),
    (6,1),(6,2),(6,3),(6,4),
    (7,1),(7,2),(7,3),(7,4),
    (8,1),(8,2),(8,3),(8,4),
]

def get_next_level(world: int, level: int):
    """Retourne (next_world, next_level) ou None si dernier niveau."""
    try:
        idx = LEVEL_SEQUENCE.index((world, level))
        if idx + 1 < len(LEVEL_SEQUENCE):
            return LEVEL_SEQUENCE[idx + 1]
    except ValueError:
        pass
    return None

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
            'step_back': {'base_action': 6, 'duration': 20, 'description': 'Reculer/éviter danger (~60px par défaut)'},
            'wait': {'base_action': 0, 'duration': 3, 'description': 'Attendre/observer'},

            # Sauts tactiques
            'short_jump': {'base_action': 2, 'duration': 10, 'description': 'Petit saut pour petits obstacles'},
            'high_jump': {'base_action': 5, 'duration': 8, 'description': 'Saut vertical haut'},
            'long_jump': {'base_action': 4, 'duration': 12, 'description': 'Course + saut pour longues distances'},
            'precise_jump': {'base_action': 2, 'duration': 10, 'description': 'Saut précis sur ennemis/blocs'},

            # Actions spéciales Mario Bros
            'stomp_enemy': {'base_action': 2, 'duration': 15, 'description': 'Écraser Goomba/Koopa par-dessus (right+A, arc bas). Accepte px=approche avant saut. Formule: px = distance_ennemi - 10 (ex: ennemi à 50px → px=40). Sans px: saut immédiat si ennemi < 15px.'},
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
                    {'base_action': 3, 'duration': 30},   # right+B → approche en courant jusqu'au pied
                    {'base_action': 4, 'duration': 40},   # right+A+B → saut max depuis la base
                ],
                'description': 'APPROCHE EN COURANT + SAUT MAX pour franchir un tuyau haut : run 30f jusqu\'au pied, puis saut max 40f'
            },
            # obstacle_jump : course puis saut max pour obstacles MOYENS (pas les tuyaux hauts)
            #   Phase 1 : prendre de la vitesse (20 frames right+B)
            #   Phase 2 : saut max avec élan (40 frames right+A+B)
            'obstacle_jump': {
                'phases': [
                    {'base_action': 3, 'duration': 20},   # right+B → élan
                    {'base_action': 4, 'duration': 40},   # right+A+B → saut max avec vitesse
                ],
                'description': 'ÉLAN COURT + SAUT MAX pour obstacles MOYENS (hauteur 20-40px, course 20f + saut 40f). Pour obstacles HAUTS (>40px) utilise high_obstacle_jump ou pipe_jump.'
            },
            # high_obstacle_jump : élan LONG + saut max — pour TUYAUX TRES HAUTS ou plateformes très élevées
            #   Phase 1 : longue accélération (40 frames right+B) → plus de momentum que pipe_jump
            #   Phase 2 : saut max avec élan maximal (40 frames right+A+B)
            'high_obstacle_jump': {
                'phases': [
                    {'base_action': 3, 'duration': 40},   # right+B → élan long
                    {'base_action': 4, 'duration': 40},   # right+A+B → saut max avec vitesse max
                ],
                'description': 'ÉLAN LONG + SAUT MAX pour obstacles TRES HAUTS (>40px). Nécessite de l\'espace devant Mario (>30px). Si Mario est collé au tuyau (<30px), utilise pipe_vertical_jump à la place.'
            },
            # pipe_vertical_jump : saut VERTICAL puis dérive droite — technique NES classique pour tuyaux hauts
            #   Quand Mario est collé au tuyau (< 30px), il ne peut pas prendre d'élan forward.
            #   Phase 1 : saut vertical pur (A seulement, 15 frames) → monte sans avancer → pas de collision
            #   Phase 2 : dérive droite + A (right+A, 30 frames) → glisse sur le dessus du tuyau au sommet du saut
            'pipe_vertical_jump': {
                'phases': [
                    {'base_action': 5, 'duration': 15},   # A only → saut vertical pur, zéro déplacement horizontal
                    {'base_action': 2, 'duration': 30},   # right+A → dérive vers la droite au sommet du saut
                ],
                'description': 'SAUT VERTICAL + dérive droite. Pour tuyaux TRES HAUTS (>50px) quand Mario est COLLÉ au tuyau (<30px). Saute sur place (pas de collision), puis glisse sur le dessus au sommet.'
            },

            # Actions tactiques spécifiques
            'wait_for_enemy': {'base_action': 0, 'duration': 5, 'description': 'Attendre que l\'ennemi passe (timing)'},
            'retreat_and_jump': {'base_action': 6, 'duration': 12, 'description': 'Reculer puis sauter (éviter puis attaquer)'},
            'run_jump_over': {'base_action': 4, 'duration': 50, 'description': 'Course + saut long pour passer par-dessus obstacle ou groupe d\'ennemis (50 frames = ~130px de portée, couvre une grappe de 2-3 Goombas)'},
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
            raise ValueError(" ANTHROPIC_API_KEY non définie!")

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
        self.llm_responses = deque(maxlen=500)  # Garder les 500 dernières réponses
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

        # Mémoire persistante par segments (fichier séparé par niveau)
        self.segment_memory = MarioSegmentMemory(get_memory_path(self.current_world, self.current_level))
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


        self.last_oam_trigger_step = -100  # Cooldown minimal anti-boucle (5 frames)
        self._scene_active = False          # True = scène détectée, en attente que Mario franchisse l'obstacle
        self._last_run_jump_over_x = -999  # Position du dernier run_jump_over (conversion auto si répété)
        self._last_scene_snapshot = None   # Snapshot scène connu par Claude (ennemis + tuyau + trou)
        # Seuils de détection multi-niveaux (px) — chaque seuil déclenche Claude une fois
        # Reset quand l'obstacle disparaît ou que Mario le dépasse
        self._enemy_thresholds_hit = set()   # ex. {180, 100} = déjà signalés
        self._pipe_thresholds_hit = set()
        self._hole_thresholds_hit = set()
        self._last_known_enemy_dist = None   # Pour détecter disparition ennemi
        self._new_enemy_appeared = False     # True quand un nouvel ennemi remplace le précédent
        self._last_known_pipe_dist = None
        self._last_known_hole_dist = None
        self.last_pipe_trigger_step = -60  # déplacé ici (était en double)
        self._threshold_restitution_steps = {}  # {(type, seuil): step} — anti-spam restitution

        # Système de rewind sur mort
        self.rewind_buffer = deque(maxlen=3)  # 3 checkpoints (60 frames d'écart)
        self.rewind_count = 0                  # Rewinds utilisés cette partie
        self.max_rewinds = float('inf')        # Rewinds illimités
        self._rewind_active = False            # Pour overlay visuel
        self._rewind_correction_msg = None     # Message injecté dans le prochain prompt Claude
        self._rewind_death_x = None            # Position de mort — contexte consommé seulement quand Mario s'en approche
        self._raw_action_history = []          # Historique brut des actions NES (pour replay PPU)
        self._final_action_history = []        # Historique complet sans erreurs (x=0→fin, branche morte retirée)
        self._rewind_index = None             # Index dans _final_action_history où le dernier rewind a eu lieu
        self._perfect_start_ram = None         # Snapshot RAM du checkpoint rewind (pour restaurer timing ennemis)
        self._perfect_start_x = 0             # x_pos du checkpoint rewind
        self._rewind_checkpoints = []          # Liste de {index, ram, x} — un par rewind (pour replay multi-rewind)
        self._claude_generation = 0            # Incrémenté à chaque rewind pour invalider threads en cours
        self._death_positions = []             # Historique des positions de mort pour le message rewind
        self._danger_zone_x = None             # Position X à éviter après rewind (filet de sécurité)
        self._rewind_real_info = None          # real_info capturé après replay PPU (évite info périmée)
        self._ppu_warmup_until = 0             # Step jusqu'auquel bloquer Claude (PPU pas encore sync après rewind)
        self._mid_jump_pause_triggered = False  # True si la pause d'urgence en plein saut a déjà été déclenchée
        self._mid_air_called = False            # True si Claude a déjà été consulté pour l'atterrissage ce saut
        self._mid_air_emergency_called = False  # inutilisé, conservé pour compat
        self._mid_air_prev_y = 0               # y_pos frame précédente, pour détecter la descente
        self._jump_aborted = False              # True si le saut a été avorté par la pause (frames_left forcé à 0)
        self._pre_jump_ram = None              # RAM snapshot juste avant une macro de saut
        self._pre_jump_has_full_backup = False  # True si _backup() complet pris au pré-saut
        self._pre_jump_x = 0                   # x_pos au moment du pré-saut
        self._pre_jump_y = 200                 # y_pos au moment du pré-saut (doit être sur le sol)
        self._pre_jump_history_len = 0         # len(_final_action_history) au pré-saut

        # Système anti-blocage
        self.stuck_counter = 0           # Nombre de checks consécutifs sans progression
        self.last_stuck_check_step = 0   # Dernier step où on a vérifié le blocage
        self.stuck_check_frequency = 60   # Vérifier toutes les 60 steps (pipe_jump = 80 frames, ok car inject_known_solution cooldown protège)
        self.last_stuck_position = None  # Position au dernier check
        self.stuck_search_done = set()   # Positions déjà cherchées (évite doublons)
        self._blocked_macros_by_pos = {}  # {bucket_50px: set(macro_names)} — macros échouées par zone

        # Replay brut d'un run parfait sauvegardé (mode "continue_with_replay")
        # Les actions sont des entiers NES bruts, rejoués frame par frame avant de passer en IA.
        self._raw_replay_actions = []   # list[int] — chargé depuis perfect_run_{w}-{l}.json
        self._raw_replay_index = 0      # index courant dans _raw_replay_actions

        # Tracking des tentatives de saut échouées par position
        # Une tentative est "échouée" si le saut se termine avec delta_x < 40px (obstacle non franchi).
        self._failed_jump_attempts = {}  # {x_bucket_30px: [macro_name, ...]}
        self._prev_macro_name = None     # macro_name capturé au frame précédent (détection transition)

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

        print(" Mario Fluide LLM initialisé!")

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
                print(f"  Config override v{v} chargée: {list(params.keys())}")
        except Exception as e:
            print(f"  Impossible de charger config override: {e}")

    def analyze_situation(self, obs, info, step_count):
        """Analyser la situation pour Claude"""

        mario_x = info.get('x_pos', 0)
        mario_y = info.get('y_pos', 0)
        score = info.get('score', 0)

        self.position_history.append(mario_x)

        # Analyser la progression
        progress_analysis = self.analyze_progression()

        #  TRACKING MOUVEMENT ENNEMIS (ANTI-MORT)
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

        #  Détection des trous (couche pixel, indépendante du mode screenshot)
        hole_info = self.detect_holes_ahead(obs) if obs is not None else {
            'detected': False, 'nearest': 999, 'width': 0, 'critical': False
        }
        screen_analysis['gaps'] = hole_info['detected']
        screen_analysis['holes'] = hole_info

        #  Détection des tuyaux/obstacles (couche pixel, indépendante du mode screenshot)
        pipe_info = self.detect_pipe_ahead(obs) if obs is not None else {
            'detected': False, 'distance_px': 999, 'height_px': 0, 'jump_type': 'none', 'urgent': False
        }
        screen_analysis['pipes'] = pipe_info['detected']
        screen_analysis['pipe_ahead'] = pipe_info

        return {
            'mario': {'x': mario_x, 'y': mario_y, 'score': score},
            'progress': progress_analysis,
            'screen': screen_analysis,
            'holes': hole_info,
            'pipe_ahead': pipe_info,
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
         SYSTÈME ANTI-MORT : Tracker le mouvement des ennemis entre les frames
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
                'danger_level': ' INCONNU'
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
                enemy_info['danger_level'] = ' DANGER IMMÉDIAT!'
                enemy_info['urgency'] = 10
                enemy_info['recommended_action'] = 'stomp_enemy'
            elif pattern == 'DANGER_FRONTAL' and distance < 100:
                enemy_info['danger_level'] = ' DANGER PROCHE'
                enemy_info['urgency'] = 8
                enemy_info['recommended_action'] = 'stomp_enemy ou run_jump_over'
            elif pattern == 'DANGER_ARRIERE' and distance < 40:
                enemy_info['danger_level'] = ' DANGER ARRIÈRE'
                enemy_info['urgency'] = 7
                enemy_info['recommended_action'] = 'run_forward (fuir)'
            elif pattern == 'SAFE':
                enemy_info['danger_level'] = ' SAFE (s\'éloigne)'
                enemy_info['urgency'] = 2
                enemy_info['recommended_action'] = 'run_forward ou collecte items'
            elif pattern == 'STATIONNAIRE' and distance < 60:
                enemy_info['danger_level'] = ' PRUDENCE (immobile)'
                enemy_info['urgency'] = 5
                enemy_info['recommended_action'] = 'stomp_enemy ou wait_for_enemy'
            else:
                enemy_info['danger_level'] = ' LOIN'
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
            print(f" Erreur analyse visuelle: {e}")
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
                    print(f" Mario détecté: {mario_pos['found']} en ({mario_x}, {mario_y})")

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
                        print(f" Détection d'urgence - Bleu détecté: {blue_anywhere:.3f}")

                    # Chercher tout ce qui est sombre (ennemis potentiels)
                    dark_areas = np.mean(np.sum(game_area, axis=2) < 300)
                    if dark_areas > 0.1:  # 10% de l'écran est sombre
                        results['enemies']['detected'] = True
                        print(f" Détection d'urgence - Zones sombres: {dark_areas:.3f}")

        except Exception as e:
            print(f" Erreur scan_full_screen: {e}")
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

                print(f" Niveau détecté: World {detected_world}-{detected_level} (confiance: {confidence_points}%)")

        except Exception as e:
            self.logger.log_error("LEVEL_DETECTION", str(e), step_count)

    def get_level_specific_context(self) -> str:
        """Générer le contexte spécifique au niveau actuel"""
        level_data = self.level_db.get_level_data(self.current_world, self.current_level)

        if not level_data:
            return "Niveau générique - informations limitées disponibles."

        context_parts = []

        # Informations générales du niveau
        context_parts.append(f" NIVEAU: World {level_data.world}-{level_data.level} ({level_data.level_type})")
        context_parts.append(f"⏱ Temps limite: {level_data.time_limit} secondes")
        context_parts.append(f" Musique: {level_data.background_music}")

        # Ennemis spécifiques à ce niveau
        if level_data.enemies:
            context_parts.append(f"\n ENNEMIS CONFIRMÉS DANS CE NIVEAU:")
            for enemy in level_data.enemies:
                threat_emoji = {"LOW": "", "MEDIUM": "", "HIGH": "", "CRITICAL": ""}[enemy.threat_level]
                context_parts.append(f"   {threat_emoji} {enemy.name}: {enemy.behavior} (Vitesse: {enemy.speed}px/step)")
                context_parts.append(f"      Élimination: {', '.join(enemy.defeat_methods)} | Points: {enemy.points}")
                if enemy.special_notes:
                    context_parts.append(f"       {enemy.special_notes}")

        # Blocs et éléments interactifs
        if level_data.blocks:
            context_parts.append(f"\n BLOCS ET ÉLÉMENTS INTERACTIFS:")
            for block in level_data.blocks:
                context_parts.append(f"    {block.name}: {block.contents}")
                context_parts.append(f"      Comportement: {block.behavior}")
                if block.special_notes:
                    context_parts.append(f"       {block.special_notes}")

        # Power-ups disponibles
        if level_data.power_ups:
            context_parts.append(f"\n⭐ POWER-UPS DISPONIBLES:")
            for powerup in level_data.power_ups:
                rarity_emoji = {"COMMON": "", "RARE": "", "VERY_RARE": ""}[powerup.rarity]
                context_parts.append(f"   {rarity_emoji} {powerup.name}: {powerup.effect}")
                if powerup.special_notes:
                    context_parts.append(f"       {powerup.special_notes}")

        # Obstacles spécifiques
        if level_data.obstacles:
            context_parts.append(f"\n OBSTACLES SPÉCIFIQUES:")
            for obstacle in level_data.obstacles:
                threat_emoji = {"LOW": "", "MEDIUM": "", "HIGH": "", "CRITICAL": ""}[obstacle.threat_level]
                context_parts.append(f"   {threat_emoji} {obstacle.name}: {obstacle.avoidance_strategy}")
                if obstacle.special_notes:
                    context_parts.append(f"       {obstacle.special_notes}")

        # Fonctionnalités spéciales
        if level_data.special_features:
            context_parts.append(f"\n CARACTÉRISTIQUES SPÉCIALES:")
            for feature in level_data.special_features:
                context_parts.append(f"    {feature}")

        # Stratégie recommandée
        context_parts.append(f"\n STRATÉGIE RECOMMANDÉE:")
        context_parts.append(f"    {level_data.completion_strategy}")

        # Analyse des menaces
        threat_analysis = self.level_db.get_threat_analysis(self.current_world, self.current_level)
        context_parts.append(f"\n ANALYSE DES MENACES:")
        context_parts.append(f"   Niveau max: {threat_analysis['max_threat_level']}")
        if threat_analysis['high_value_targets']:
            context_parts.append(f"    Cibles prioritaires: {', '.join(threat_analysis['high_value_targets'])}")

        # Power-ups recommandés
        recommended = self.level_db.get_recommended_powerups(self.current_world, self.current_level)
        context_parts.append(f"\n POWER-UPS RECOMMANDÉS: {', '.join(recommended)}")

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

            print(f" Facteur d'échelle: 1px screenshot = {scale_factor:.2f}px jeu")

            # Stocker le facteur d'échelle pour utilisation dans les prompts
            self.current_scale_factor = scale_factor
            self.screenshot_dimensions = optimized_size
            self.original_game_dimensions = (width, int(original_game_height))

            # Mettre à jour le convertisseur de distances
            self.distance_converter.update_scale_factor(scale_factor)

            return img_base64

        except Exception as e:
            print(f" Erreur capture screenshot: {e}")
            return None

    def _compute_mystery_mask(self, raw_array):
        """Détecter les mystery blocks par correspondance EXACTE de couleur NES.
        RGB(68, 160, 252) est une entrée fixe de la palette hardware NES :
        elle est identique à chaque frame, indépendante de l'animation du sprite."""
        # Correspondance pixel-exact : même NES palette entry → même RGB, toujours
        mystery_color = np.array([68, 160, 252], dtype=np.uint8)
        mask = np.all(raw_array == mystery_color, axis=2)
        if np.any(mask):
            print(f" Mystery blocks détectés ({np.sum(mask)} pixels)")
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

            # Supprimer les décors d'arrière-plan (ciel + nuages) → gris neutre
            # Ciel SMB1-1 = RGB(92, 148, 252) exacte palette NES, tolérance ±15
            sky = np.array([92, 148, 252], dtype=np.int16)
            sky_mask = np.all(np.abs(sharpened.astype(np.int16) - sky) <= 15, axis=2)
            sharpened[sky_mask] = [80, 80, 80]

            # Nuages = pixels quasi-blancs (R,G,B tous ≥ 230) dans la zone ciel
            # (limitée aux 45% supérieurs pour ne pas toucher les pièces ou Mario blanc)
            cloud_zone_end = int(sharpened.shape[0] * 0.45)
            cloud_mask = np.all(sharpened[:cloud_zone_end] >= 230, axis=2)
            sharpened[:cloud_zone_end][cloud_mask] = [80, 80, 80]

            # Appliquer couleur dédiée MAGENTA sur les mystery blocks
            if np.any(mystery_mask):
                sharpened[mystery_mask] = [255, 0, 255]

            return Image.fromarray(sharpened)

        except Exception as e:
            print(f" Erreur filtres, image originale utilisée: {e}")
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
            MARIO_PAL  = 0x00 # palette 0 = Mario — à exclure
            # Palettes 1, 2, 3 = ennemis/objets (Koopas=1, Koopas shell=2, Goombas=3)
            # On exclut seulement Mario (palette 0) pour attraper tous les ennemis

            # 1er passage : collecter tous les sprites ennemis
            raw_sprites = []
            for i in range(64):
                y_nes = int(ram[OAM_BASE + i*4 + 0])
                attr  = int(ram[OAM_BASE + i*4 + 2])
                x_nes = int(ram[OAM_BASE + i*4 + 3])
                if y_nes < HUD_Y_MAX or y_nes >= 240:
                    continue
                if (attr & 0x03) == MARIO_PAL:  # exclure Mario seulement
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
                print(f" OAM: {n_painted} ennemi(s) peints en jaune ({len(raw_sprites)} sprites groupés)")
            else:
                print(f" OAM: aucun sprite ennemi (palette 1/2/3) visible à l'écran")

            # Recolorer les tuyaux en CYAN avant de neutraliser le vert
            # → permet de distinguer tuyaux (gameplay) des buissons/collines (décor)
            pipe_info = self.detect_pipe_ahead(obs)
            if pipe_info.get('detected'):
                pipe_dist = pipe_info.get('distance_px', 999)
                pipe_h    = max(16, pipe_info.get('height_px', 32))
                scale_x   = img_w / 256.0
                mario_sx  = img_w // 3
                px1 = max(0, int(mario_sx + (pipe_dist - 2) * scale_x))
                px2 = min(img_w, int(mario_sx + (pipe_dist + 18) * scale_x))  # ~16px NES
                # Hauteur tuyau : depuis le bas de l'image en remontant
                pipe_screen_h = int(pipe_h * scale_y)
                py1 = max(0, img_h - pipe_screen_h)
                img[py1:img_h, px1:px2] = [0, 220, 220]  # CYAN

            # Neutraliser les pixels verts restants (buissons, collines = décor)
            # Vert dominant : G nettement > R et G nettement > B
            r = img[:, :, 0].astype(np.int16)
            g = img[:, :, 1].astype(np.int16)
            b = img[:, :, 2].astype(np.int16)
            green_decor = (g - r > 40) & (g - b > 40) & (g > 80)
            img[green_decor] = [90, 90, 90]

            return Image.fromarray(img)

        except Exception as e:
            print(f" OAM annotation erreur: {e}")
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

            # Bloc correction rewind : injecté seulement quand Mario approche la zone de mort (≤80px)
            # Avant d'arriver là, Claude fait la navigation normale sans ce contexte.
            _rewind_block = ""
            if self._rewind_correction_msg:
                _near_death = (self._rewind_death_x is None or mario_x >= self._rewind_death_x - 80)
                if _near_death:
                    _rewind_block = self._rewind_correction_msg + "\n\n"
                    self._rewind_correction_msg = None  # Consommé
                    self._rewind_death_x = None

            # Section macros bloquées à la position courante
            _ss_bucket = int(mario_x // 50) * 50
            _ss_blocked = getattr(self, '_blocked_macros_by_pos', {}).get(_ss_bucket, set())
            if _ss_blocked:
                _all_jumps = {'max_jump', 'pipe_jump', 'obstacle_jump', 'high_obstacle_jump', 'pipe_vertical_jump', 'run_jump_over'}
                _not_tried = sorted(_all_jumps - _ss_blocked)
                _hint = f" ESSAIE PLUTÔT: {', '.join(_not_tried)}" if _not_tried else " Tous les sauts échoués — tente step_back + élan"
                _ss_blocked_block = (f"\n ACTIONS DÉJÀ ESSAYÉES SANS SUCCÈS à x≈{_ss_bucket}: {', '.join(sorted(_ss_blocked))}"
                                     f"\n→ N'utilise PAS ces actions — elles ont déjà échoué à cette position!"
                                     f"\n→{_hint}\n")
            else:
                _ss_blocked_block = ""

            # Section OAM ennemis (positions exactes NES)
            oam_enemies = self.get_enemies_from_oam()
            if oam_enemies:
                _oam_lines = [" ENNEMIS DÉTECTÉS (OAM NES — positions écran exactes):"]
                for e in oam_enemies:
                    dist = e['distance_px']
                    if dist > 0:
                        if dist <= 30:
                            action_hint = "stomp_enemy ou run_jump_over (DANGER IMMÉDIAT)"
                        elif dist <= 80:
                            action_hint = "stomp_enemy ou run_jump_over"
                        else:
                            action_hint = "prépare stomp_enemy ou run_jump_over"
                        _oam_lines.append(
                            f"   Ennemi à {dist}px DEVANT Mario (écran x={e['x_screen']}) → {action_hint}"
                        )
                    else:
                        _oam_lines.append(
                            f"  ℹ Ennemi à {abs(dist)}px DERRIÈRE Mario (écran x={e['x_screen']})"
                        )
                _oam_lines.append(" NE PAS faire run_forward si ennemi devant — utilise stomp_enemy ou run_jump_over D'ABORD")
                _oam_section = "\n".join(_oam_lines)
            else:
                _oam_section = " Aucun ennemi OAM détecté — voie libre"

            # Section mystery blocks (masque couleur NES RGB 68,160,252 → MAGENTA dans screenshot)
            mystery_blocks = self.get_mystery_blocks_from_screenshot()
            closest_enemy_dist = min((abs(e['distance_px']) for e in oam_enemies), default=999)
            blocks_in_front = [b for b in mystery_blocks if 0 < b['distance_px'] < 200]
            if blocks_in_front:
                _block_lines = [" BLOCS MYSTERE DETECTES (pixels MAGENTA dans le screenshot):"]
                for b in blocks_in_front:
                    dist = b['distance_px']
                    if closest_enemy_dist > 80:
                        hint = f"ZONE SURE → position_under_block px={max(0, dist - 20)} puis hit_block"
                    else:
                        hint = f"DANGEREUX — ennemi a {closest_enemy_dist}px (eliminer ennemi d'abord)"
                    _block_lines.append(f"  Bloc ? a {dist}px devant Mario → {hint}")
                _mystery_section = "\n".join(_block_lines)
                print(f" Mystery: {len(blocks_in_front)} bloc(s) devant Mario, ennemi le plus proche a {closest_enemy_dist}px")
            else:
                _mystery_section = ""

            # Détection de trou depuis situation (calculé par analyze_situation → detect_holes_ahead)
            _hole_info = situation.get('holes', {})
            _hole_urgent = _hole_info.get('urgent', False)  # trou < 100px
            _hole_dist = _hole_info.get('nearest', 999)
            _hole_width = _hole_info.get('width', 0)

            # Détection de tuyau/obstacle (calculé par _pipe_trigger → detect_pipe_ahead)
            _pipe_info = situation.get('pipe', {})
            _pipe_urgent = _pipe_info.get('urgent', False)  # obstacle < 90px
            _pipe_dist = _pipe_info.get('distance_px', 999)
            _pipe_height = _pipe_info.get('height_px', 0)
            _pipe_jump_type = _pipe_info.get('jump_type', 'pipe_jump')

            # Priorité de prompt : ennemi > trou > tuyau > voie libre
            enemies_in_front = [e for e in oam_enemies if 0 < e['distance_px'] < 80]

            if enemies_in_front:
                # PROMPT COURT — ennemi devant : décision urgente uniquement
                closest = enemies_in_front[0]
                dist = closest['distance_px']
                # Compter les morts récentes dans cette zone (±80px) pour adapter le conseil
                _nearby_enemy_deaths = sum(
                    1 for d in self._death_positions
                    if d.get('cause') == 'enemy_hit' and abs(d['x'] - mario_x) <= 80
                )
                if _nearby_enemy_deaths >= 2:
                    _enemy_hint = "run_jump_over (stomp a déjà échoué ici — SAUTER par-dessus est plus sûr)"
                    _enemy_default = "run_jump_over"
                else:
                    _enemy_hint = "run_jump_over (recommandé — sauter par-dessus l'ennemi)"
                    _enemy_default = "run_jump_over"
                prompt = _rewind_block + f"""⚠️⚠️⚠️ ENNEMI DÉTECTÉ — DÉCISION URGENTE ⚠️⚠️⚠️
Mario: X={mario_x}px | Step: {step_count} | Score: {mario['score']}
ENNEMI à {dist}px DEVANT Mario (position écran x={closest['x_screen']}).

Ennemi détecté à environ {dist}px. Regarde le screenshot et CHOISIS:
  A) run_jump_over px=<distance_saut> [approach_px=<élan_optionnel>]
     → Sauter PAR-DESSUS l'ennemi.
     → px = distance que le saut doit couvrir (en pixels NES).
     → approach_px = frames de course avant le saut pour prendre de l'élan (0 ou omis = saut immédiat).
  B) stomp_enemy px=<distance_approche>
     → Écraser l'ennemi en sautant dessus (px = distance à courir avant de sauter).

JSON uniquement — 1 seule action:
{{"actions":[{{"macro_action":"run_jump_over","px":<ta_valeur>}}],"urgency":10}}"""
            elif _hole_urgent:
                # PROMPT COURT — trou devant : décision urgente
                #  JAMAIS de px pour max_jump sur trou urgent !
                # L'approche "px" fait courir Mario jusqu'au bord du trou AVANT de sauter
                # → Mario tombe (impossible de sauter depuis le vide). Sauter IMMÉDIATEMENT.
                _hole_action_hint = f'{{"actions":[{{"macro_action":"max_jump"}}],"urgency":10}}'
                prompt = _rewind_block + f"""🕳️🕳️🕳️ TROU DÉTECTÉ — DÉCISION URGENTE 🕳️🕳️🕳️
Mario: X={mario_x}px | Step: {step_count}
TROU à {_hole_dist}px DEVANT Mario (largeur ~{_hole_width}px).

Regarde le screenshot — le sol s'arrête devant Mario.

CHOISIS:
  A) max_jump  → sauter IMMÉDIATEMENT (recommandé — le trou est assez loin pour sauter par-dessus)
  B) pipe_jump → SEULEMENT si tu vois un GRAND TUYAU VERT devant le trou

❌ INTERDIT: run_forward (Mario tombe dans le trou → mort garantie)
❌ INTERDIT: max_jump avec "px" (courir jusqu'au bord = chute garantie)

JSON uniquement — 1 seule action:
{_hole_action_hint}"""
            elif _pipe_urgent:
                # PROMPT COURT — tuyau/obstacle devant : saut anticipé
                # Mario a encore de la distance → on peut choisir le bon saut
                _pipe_action_hint = f'{{"actions":[{{"macro_action":"{_pipe_jump_type}"}}],"urgency":9}}'
                if _pipe_jump_type == 'pipe_jump':
                    _pipe_desc = "pipe_jump (recommandé — approche + grand saut = franchit les tuyaux hauts)"
                    _pipe_alt = "obstacle_jump (si tuyau très haut)"
                elif _pipe_jump_type == 'obstacle_jump':
                    _pipe_desc = "obstacle_jump (obstacle très haut)"
                    _pipe_alt = "pipe_jump (si tuyau standard)"
                else:
                    _pipe_desc = "max_jump (obstacle bas)"
                    _pipe_alt = "pipe_jump (si tuyau standard)"
                prompt = _rewind_block + f"""🟢🟢🟢 TUYAU/OBSTACLE DÉTECTÉ — SAUT ANTICIPÉ 🟢🟢🟢
Mario: X={mario_x}px | Step: {step_count}
OBSTACLE à {_pipe_dist}px DEVANT Mario (hauteur estimée ~{_pipe_height}px).

Regarde le screenshot — il y a un TUYAU VERT ou obstacle qui bloque la route.
Mario est encore à bonne distance pour prendre de l'élan et franchir.

CHOISIS:
  A) {_pipe_desc}
  B) {_pipe_alt}
  C) max_jump → si obstacle bas (<20px)

❌ INTERDIT: run_forward seul (Mario se coince contre le tuyau)
❌ INTERDIT: attendre (Mario s'immobilise contre le tuyau)

JSON uniquement — 1 seule action:
{_pipe_action_hint}"""
            else:
                # PROMPT NORMAL — analyse visuelle obligatoire
                _pipe_ahead_info = situation.get('pipe_ahead', {})
                if _pipe_ahead_info.get('detected'):
                    _pipe_dist_a = _pipe_ahead_info.get('distance_px', '?')
                    _pipe_h_a = _pipe_ahead_info.get('height_px', '?')
                    # Recommandation de saut selon la hauteur ET la distance à l'obstacle
                    try:
                        _h_val = int(_pipe_h_a)
                    except (ValueError, TypeError):
                        _h_val = 0
                    try:
                        _d_val = int(_pipe_dist_a)
                    except (ValueError, TypeError):
                        _d_val = 999
                    if _h_val >= 50:
                        if _d_val < 30:
                            # Obstacle très proche ET très haut : toute approche forward heurte le tuyau.
                            # Technique NES : saut vertical pur (A seulement) puis dérive droite au sommet.
                            _jump_rec = (
                                f"\n→ Obstacle TRES HAUT ({_h_val}px) à seulement {_d_val}px: "
                                f"OBLIGATOIRE — utilise pipe_vertical_jump (saut vertical + dérive droite au sommet). "
                                f"N'utilise PAS high_obstacle_jump, obstacle_jump ni pipe_jump (l'élan forward heurterait le mur du tuyau avant que le saut commence)."
                            )
                        else:
                            _jump_rec = (
                                f"\n→ Obstacle TRES HAUT ({_h_val}px): utilise pipe_jump px=80 (approche + saut max) "
                                f"ou high_obstacle_jump px=80 (élan long). N'utilise PAS obstacle_jump ni max_jump seul."
                            )
                    elif _h_val >= 30:
                        _jump_rec = (
                            f"\n→ Obstacle HAUT ({_h_val}px): utilise pipe_jump px=60 ou obstacle_jump px=50."
                        )
                    else:
                        _jump_rec = (
                            f"\n→ Utilise max_jump px=40 ou run_jump_over px=45."
                        )
                    _obstacle_line = (
                        f"\nALERTE OBSTACLE: tuyau/obstacle détecté à {_pipe_dist_a}px devant Mario "
                        f"(hauteur ~{_pipe_h_a}px). Ne pas faire run_forward — Mario se coince!"
                        f"{_jump_rec}"
                    )
                else:
                    _obstacle_line = ""

                _blocked_note = _ss_blocked_block if _ss_blocked_block else ""
                _mystery_block_note = f"\n{_mystery_section}" if _mystery_section else ""
                prompt = _rewind_block + f"""Mario X={mario_x}px | Score:{mario['score']} | Step:{step_count} | Vitesse:{mario_speed:.1f}px/step{_blocked_note}{_obstacle_line}
{_oam_section}{_mystery_block_note}

{self.segment_memory.get_context_for_position(mario_x)}{self._get_phase1_optimization_hint(mario_x)}{self._get_phase3_frontier_context(mario_x)}
{self.get_learning_context()}
REGARDE LE SCREENSHOT — QUE VOIS-TU DIRECTEMENT A DROITE DU SPRITE DE MARIO ?
  Sol plat libre     => run_forward px=<distance NES jusqu'au prochain obstacle>
  Tuyau vert         => pipe_jump px=<distance NES jusqu'au tuyau>  (px = approche d'élan)
  Sol absent / vide  => max_jump px=<distance d'approche avant bord>
  Ennemi             => run_jump_over px=<distance_saut> approach_px=<élan> ou stomp_enemy px=<distance_ennemi - 10>

"px" = distance en pixels NES entre Mario et l'obstacle dans le screenshot. OBLIGATOIRE sur toutes les actions.
  Bord droit de l'écran ≈ 210px. Mi-écran ≈ 105px. Quart d'écran ≈ 50px.
  Évalue la vraie distance — ne pas toujours mettre la même valeur !

ACTIONS DISPONIBLES (toutes avec px obligatoire):
  run_forward px | pipe_jump px | max_jump px
  obstacle_jump px (obstacles MOYENS 20-40px) | high_obstacle_jump px (obstacles HAUTS >40px, si espace >30px)
  pipe_vertical_jump (tuyau TRES HAUT >50px ET Mario COLLÉ <30px — saut vertical puis dérive droite)
  run_jump_over px approach_px (optionnel) | stomp_enemy px | step_back
  high_jump (saut vertical haut, utile escalier) | big_jump_right (saut max vers droite)
  DOUBLE SAUT possible : donne 2 sauts consécutifs ex: {{"actions":[{{"macro_action":"obstacle_jump"}},{{"macro_action":"max_jump"}}],"urgency":9}}

Donne 1-3 actions MAX. JSON avec UN seul objet actions[] :
{{"actions":[{{"macro_action":"pipe_jump","px":80}}],"urgency":9}}"""

            self.api_calls += 1
            print(f" Envoi screenshot à Claude (appel #{self.api_calls})...")

            # Logger le prompt avec le screenshot associé
            self.logger.log_claude_prompt("SCREENSHOT", prompt, step_count,
                                          screenshot_b64=screenshot_b64)

            print("="*80)
            print(" PROMPT ENVOYÉ À CLAUDE:")
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
            print(f" Claude analyse reçue ({len(response_text)} chars)")

            # Calculer le coût d'abord
            image_cost = min(0.01, len(screenshot_b64) * 0.000001)  # Coût proportionnel à la taille
            text_cost = len(prompt) * 0.25 / 1000000 + len(response_text) * 1.25 / 1000000
            cost = text_cost + image_cost

            # Logger la réponse avec coût calculé
            self.logger.log_claude_response(response_text, step_count, cost)

            print("="*80)
            print(" RÉPONSE DE CLAUDE:")
            print(response_text)
            print("="*80)

            self.total_cost += cost
            self.screenshot_costs += image_cost

            # Ajuster la fréquence si on dépasse le budget
            if self.screenshot_costs > self.screenshot_cost_limit:
                self.screenshot_frequency = min(100, self.screenshot_frequency + 10)  # Réduire la fréquence
                print(f" Budget screenshots dépassé, fréquence réduite à {self.screenshot_frequency} steps")

            print(f" Coût screenshot: ${image_cost:.4f} (total screenshots: ${self.screenshot_costs:.3f})")

            # Convertir les distances dans la réponse
            converted_response = self.distance_converter.process_claude_response(response_text)
            converted_response = self.distance_converter.add_scale_info_to_response(converted_response)

            print(" RÉPONSE AVEC DISTANCES CONVERTIES:")
            print(converted_response)
            print("="*80)

            # Ajouter la réponse à l'historique pour l'encart
            self.add_llm_response("SCREENSHOT", converted_response, step_count)

            return converted_response

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f" Erreur analyse screenshot: {e}")
            print(f" Détails de l'erreur:")
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

        #  ENNEMIS : lecture OAM NES (fiable, palette=3)
        oam_enemies = self.get_enemies_from_oam()
        enemy_movement_section = ""
        if oam_enemies:
            enemy_movement_section = "\n ENNEMIS DÉTECTÉS (OAM NES — positions écran exactes):"
            for e in oam_enemies:
                dist = e['distance_px']
                if dist > 0:
                    action_hint = "stomp_enemy ou max_jump"
                    enemy_movement_section += (
                        f"\n   Ennemi à {dist}px DEVANT Mario (écran x={e['x_screen']}) "
                        f"→ ACTION REQUISE: {action_hint}"
                    )
                else:
                    enemy_movement_section += (
                        f"\n  ℹ Ennemi à {abs(dist)}px DERRIÈRE Mario (écran x={e['x_screen']})"
                    )
            enemy_movement_section += "\n NE PAS faire run_forward si un ennemi est devant — utilise stomp_enemy ou max_jump D'ABORD"
        else:
            enemy_movement_section = "\n Aucun ennemi OAM détecté à l'écran"

        #  Section trou pour le prompt
        hole_info = situation.get('holes', screen.get('holes', {}))
        if hole_info.get('detected'):
            _h_near = hole_info['nearest']
            _h_w    = hole_info['width']
            if hole_info.get('critical'):
                # Saut IMMÉDIAT — jamais d'approche (courir jusqu'au bord = chute)
                _hole_json = f'{{"macro_action":"max_jump"}}'
                hole_section = (f"\n TROU CRITIQUE: sol absent à {_h_near}px (largeur {_h_w}px)"
                                f"\n→ SAUTER IMMÉDIATEMENT: {{{_hole_json}}} —  jamais de px pour max_jump sur trou")
            elif hole_info.get('urgent'):
                # Saut IMMÉDIAT — jamais d'approche (courir jusqu'au bord = chute)
                _hole_json = f'{{"macro_action":"max_jump"}}'
                hole_section = (f"\n TROU URGENT à {_h_near}px (largeur {_h_w}px)"
                                f"\n→ SAUTER IMMÉDIATEMENT: {{{_hole_json}}} —  jamais de px pour max_jump sur trou")
            else:
                _approach_px2 = max(0, _h_near - 30)
                _hole_json = f'{{"macro_action":"max_jump","px":{_approach_px2}}}'
                hole_section = (f"\n TROU DÉTECTÉ à {_h_near}px (largeur {_h_w}px)"
                                f"\n→ prépare: run_forward px={_approach_px2-40} puis {_hole_json}")
        else:
            hole_section = ""

        # Bloc correction rewind (chemin texte-seul, même logique que screenshot)
        _rewind_block = ""
        if self._rewind_correction_msg:
            _near_death = (self._rewind_death_x is None or mario['x'] >= self._rewind_death_x - 80)
            if _near_death:
                _rewind_block = self._rewind_correction_msg + "\n\n"
                self._rewind_correction_msg = None  # Consommé
                self._rewind_death_x = None

        # Section macros bloquées à la position courante
        _mario_bucket = int(mario['x'] // 50) * 50
        _blocked_here = getattr(self, '_blocked_macros_by_pos', {}).get(_mario_bucket, set())
        if _blocked_here:
            _all_jumps = {'max_jump', 'pipe_jump', 'high_obstacle_jump', 'pipe_vertical_jump', 'obstacle_jump', 'run_jump_over'}
            _not_tried_here = sorted(_all_jumps - _blocked_here)
            _hint_here = f" ESSAIE PLUTÔT: {', '.join(_not_tried_here)}" if _not_tried_here else " Tous les sauts échoués — tente step_back + élan"
            _blocked_section = (f"\n ACTIONS DÉJÀ ESSAYÉES SANS SUCCÈS à x≈{_mario_bucket}: {', '.join(sorted(_blocked_here))}"
                                f"\n→ N'utilise PAS ces actions — elles ont déjà échoué à cette position!"
                                f"\n→{_hint_here}")
        else:
            _blocked_section = ""

        prompt = _rewind_block + f"""🍄 Tu es Claude, EXPERT MARIO BROS NES ! Mario a besoin de 2-3 actions RAPIDES car le jeu est dangereux !

📍 SITUATION MARIO:
• Position: X={mario['x']}, Y={mario['y']} | Score: {mario['score']} | Step: {situation['step']}
• Progression: {progress['status']} (tendance: {progress['trend']}px){_blocked_section}

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

🟣 BLOCS MYSTÈRE (MYSTERY BOXES) — OPPORTUNITÉ BONUS:
• Dans le screenshot, les blocs mystère sont colorés en MAGENTA (rose vif) — c'est la couleur de surbrillance du système
• Si un bloc MAGENTA est visible ET accessible (pas d'ennemi entre Mario et le bloc) → utilise obstacle_jump pour sauter dessous et le frapper par en-dessous
• Le bloc mystère donne champignon (power-up), pièce, ou étoile — ça vaut le détour si la zone est sûre
• Ne saute sous un bloc MAGENTA QUE si le chemin vers lui est libre (aucun ennemi < 80px)

🚨 RÈGLES DE SURVIE CRITIQUES (ANTI-MORT):
1. 🏃 TOUJOURS run_forward par défaut — ne JAMAIS utiliser 'wait' (les ennemis avancent vers toi pendant que tu attends!)
2. ❌ JAMAIS 'step_back' sauf si Mario est BLOQUÉ contre un mur (vitesse négative détectée)
3. ✅ Ennemi < 50px → stomp_enemy IMMÉDIATEMENT (le saut avec élan peut l'atteindre)
   ✅ Ennemi 50-100px → run_forward D'ABORD pour approcher, le réflexe auto gère le saut à < 55px
   ❌ NE JAMAIS mettre run_forward après stomp_enemy (si le saut rate, Mario court vers l'ennemi!)
4. ✅ Si ennemi "S'ÉLOIGNE" → run_forward pour le rattraper et le stomp (ou passer s'il est trop loin)
5. ⚡ EN CAS DE DOUTE → run_forward pour approcher, puis stomp_enemy quand < 50px
6. 🎯 Priorité absolue: SURVIE > Collecte de blocs/items
7. ⛰️ TROU DÉTECTÉ → max_jump px=<approche> (trou loin) ou max_jump sans px (trou immédiat <20px). "px" = distance à courir AVANT de sauter, PAS la largeur du trou.
8. ⚠️ APRÈS stomp_enemy : NE PAS mettre run_forward si ennemi encore visible !

🎯 DONNE 2-3 ACTIONS ADAPTÉES À LA SITUATION!

📐 TOUTES les actions DOIVENT avoir "px". stomp_enemy "px" = distance_ennemi - 10. Sauts "px" = approche avant saut. run_forward "px" ≤ 60.
⚠️ JSON UNIQUEMENT — ZÉRO TEXTE, ZÉRO EXPLICATION:
{{"actions":[{{"macro_action":"run_forward","px":60}},{{"macro_action":"<saut_ou_action>","px":<dist>}}],"urgency":<1-10>}}

Exemples avec distances réelles:
Ennemi à 15px: {{"actions":[{{"macro_action":"stomp_enemy","px":5}}],"urgency":10}}
Ennemi à 40px: {{"actions":[{{"macro_action":"stomp_enemy","px":30}}],"urgency":10}}
Ennemi à 80px: {{"actions":[{{"macro_action":"run_jump_over","px":80,"approach_px":20}}],"urgency":8}}
Tuyau à 100px: {{"actions":[{{"macro_action":"run_forward","px":60}},{{"macro_action":"pipe_jump","px":30}}],"urgency":6}}
Zone libre 200px: {{"actions":[{{"macro_action":"run_forward","px":60}}],"urgency":4}}
Trou à 85px: {{"actions":[{{"macro_action":"max_jump","px":55}}],"urgency":10}}
Trou à 20px: {{"actions":[{{"macro_action":"max_jump","px":5}}],"urgency":10}}
Après obstacle: {{"actions":[{{"macro_action":"run_forward","px":60}}],"urgency":5}}"""

        return prompt

    def call_claude_for_macro(self, prompt, obs=None):
        """Demander à Claude quelle macro-action utiliser.
        Si obs est fourni, capture et envoie le screenshot pour donner un contexte visuel."""

        try:
            self.api_calls += 1
            print(f" Claude réfléchit... (appel #{self.api_calls})")

            # Logger le prompt
            self.logger.log_claude_prompt("TEXT", prompt, 0)

            # Construire le contenu du message (texte seul ou texte+image)
            _screenshot_b64 = None
            if obs is not None:
                try:
                    _screenshot_b64 = self.capture_game_screenshot(obs)
                except Exception:
                    _screenshot_b64 = None

            if _screenshot_b64:
                _content = [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _screenshot_b64
                    }}
                ]
            else:
                _content = prompt

            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,  # Suffisant pour 4-6 actions JSON sans reasoning
                temperature=0.1,  # Plus déterministe
                system="Tu es un contrôleur de jeu Mario. Réponds UNIQUEMENT en JSON valide, sans aucun texte avant ou après. Aucune explication, aucun commentaire.",
                messages=[{"role": "user", "content": _content}]
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
            print(f" Erreur Claude: {e}")
            print(f" Détails de l'erreur:")
            print(error_details)

            # Logger l'erreur avec détails
            self.logger.log_error("CLAUDE_MACRO_API_FAILURE", f"{str(e)} | Traceback: {error_details}", 0)

            return None

    def call_claude_async(self, situation, obs=None, step_count=0):
        """Appeler Claude en arrière-plan avec système hybride optimisé"""

        def claude_worker(generation=self._claude_generation):
            try:
                #  DÉCISION HYBRIDE: Screenshot complet vs Mise à jour positionnelle
                use_screenshot, reason = self.should_use_screenshot_vs_positions(step_count)

                _mario_x = situation.get('mario', {}).get('x', '?')
                _mode_label = " Screenshot" if use_screenshot else " Positions"
                self.add_llm_response('APPEL', f"{_mode_label} | x={_mario_x} | {reason}", step_count)
                print(f" Mode hybride: {_mode_label} - {reason}")

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
                                print(" Contexte du niveau établi pour Claude!")
                        else:
                            # Fallback textuel
                            prompt = self.create_claude_prompt(situation)
                            claude_response = self.call_claude_for_macro(prompt)
                    else:
                        # Pas d'observation, fallback textuel
                        prompt = self.create_claude_prompt(situation)
                        claude_response = self.call_claude_for_macro(prompt)

                    analysis_type = " Visuelle"

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

                    analysis_type = " Positionnelle"

                # Parser les actions (même format JSON pour les deux modes)
                actions = self.parse_claude_actions(claude_response)

                # Ajouter les actions à la queue
                # Vérifier que le rewind n'a pas eu lieu pendant l'appel API.
                # Si la génération a changé, les actions sont périmées → les ignorer.
                if generation != self._claude_generation:
                    print(f" Thread Claude périmé (gen {generation} → {self._claude_generation}), actions ignorées")
                    self.add_llm_response('ANNULE', f"Call annulé (rewind/rescan) — gen {generation}→{self._claude_generation}", step_count)
                    self.logger.log_queue_event("STALE ", step_count,
                        f"async annulé gen {generation}→{self._claude_generation}")
                    # Queue vide → PAUSE naturelle → nouvel appel Claude déclenché au prochain cycle
                    return

                # Règles de filtrage : 'wait' → 'run_forward' (wait n'existe pas comme macro NES)
                _mario_speed = (self.last_situation or {}).get('mario', {}).get('speed', 1)
                for action in actions:
                    if len(self.action_queue) < 5:  # Max 5 actions en attente
                        mname = action.get('macro_name', '')
                        if mname == 'wait':
                            action = dict(action,
                                          macro_name='run_forward',
                                          reasoning='[wait→run] Courir plutôt qu\'attendre')
                        self.action_queue.append(action)
                        _mname_log = action.get('macro_name', '?')
                        _px_log = action.get('px')
                        _px_str_log = f" px={_px_log}" if _px_log else ""
                        self.logger.log_queue_event("PUSH  ", step_count,
                            f"async→{_mname_log}{_px_str_log} (queue={len(self.action_queue)})")

                # Construire résumé lisible des actions pour le panel
                _action_parts = []
                for a in actions:
                    _n = a.get('macro_name', '?')
                    _px = a.get('px')
                    _action_parts.append(f"{_n}({_px}px)" if _px else _n)
                _actions_str = " → ".join(_action_parts) if _action_parts else "aucune action"
                self.add_llm_response('ACTIONS', _actions_str, step_count)
                print(f" Claude ({analysis_type}) a fourni {len(actions)} actions")

            except Exception as e:
                print(f" Erreur thread Claude hybride: {e}")
                # Action de secours
                fallback = self.get_fallback_macro()
                self.action_queue.append(fallback)

            finally:
                # Ne reset le flag que si ce thread est encore le thread actif.
                # Si la génération a changé (rewind, position trigger), un nouveau thread
                # est peut-être déjà en cours → ne pas tuer son flag claude_thinking.
                if generation == self._claude_generation:
                    # Incrémenter la génération pour invalider tous les autres calls parallèles
                    # qui ont la même génération. Sans ça, 40 appels simultanés accepteraient
                    # tous leur réponse et relanceraient chacun un nouveau call.
                    self._claude_generation += 1
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
                        px = action_data.get('px')           # distance en pixels (optionnel)
                        approach_px = action_data.get('approach_px')  # run_jump_over : course avant saut

                        # Valider la macro-action
                        if macro_name not in self.macro_actions:
                            print(f" Macro inconnue '{macro_name}', utilisation de walk_right")
                            macro_name = 'walk_right'

                        action_dict = {
                            'macro_name': macro_name,
                            'strategy': strategy,
                            'urgency': urgency,
                            'confidence': 80
                        }
                        if px is not None:
                            try:
                                # +5px offset (espace screenshot) avant conversion → pixels jeu NES.
                                # Claude estime les distances sur un screenshot redimensionné (plus
                                # petit que le jeu natif) → ses valeurs px sont systématiquement
                                # sous-estimées. On corrige : (px_claude + 5) * scale_factor.
                                _raw_px = int(px) + 5
                                action_dict['px'] = int(round(_raw_px * self.current_scale_factor))
                            except (ValueError, TypeError):
                                pass
                        if approach_px is not None:
                            try:
                                action_dict['approach_px'] = int(approach_px)
                            except (ValueError, TypeError):
                                pass
                        actions_list.append(action_dict)

                if actions_list:
                    # Détecter la hallucination des exemples verbatim :
                    # si Claude renvoie exactement run_forward+pipe_jump+stomp+max_jump
                    # avec les valeurs canoniques des exemples du prompt → réponse invalide.
                    _example_sig = {('run_forward', 140), ('pipe_jump', 80), ('stomp_enemy', 35),
                                    ('run_jump_over', 45), ('max_jump', 30)}
                    _resp_sig = {(a.get('macro_name'), a.get('px')) for a in actions_list}
                    if _resp_sig == _example_sig:
                        print(" Claude a copié les exemples verbatim → réponse rejetée, fallback")
                        return [self.get_fallback_macro()]

                    # Limiter à 3 actions max (le décor change après chaque action,
                    # inutile de planifier trop loin)
                    actions_list = actions_list[:3]

                    visual_analysis = data.get('visual_analysis', '')
                    if visual_analysis:
                        print(f" Claude voit: {visual_analysis[:80]}...")
                    print(f" Claude: {len(actions_list)} actions - {strategy}")
                    return actions_list

            # Fallback : parser le texte pour deviner des actions
            return [self.parse_text_for_single_macro(response_text)]

        except Exception as e:
            print(f" Erreur parsing: {e}")
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
            'strategy': 'Analyse textuelle',
            'urgency': 5,
            'confidence': 60
        }

    def parse_text_for_macro(self, text):
        """Parser une réponse textuelle pour deviner la macro (compatibilité)"""
        return self.parse_text_for_single_macro(text)

    def get_enemies_from_oam(self):
        """Lit l'OAM NES et retourne la liste des ennemis visibles avec leur distance à Mario.
        Retourne une liste de dicts : {x_screen, y_screen, distance_px, direction}
        où distance_px est la distance horizontale depuis Mario (positif = devant, négatif = derrière).
        """
        try:
            ram = self.env.unwrapped.ram
            OAM_BASE  = 0x0200
            HUD_Y_MIN = 32
            MARIO_PAL = 0x00  # palette 0 = Mario — à exclure

            mario_x_oam = int(ram[OAM_BASE + 1*4 + 3])

            # Regrouper les sprites OAM par proximité (un Goomba = plusieurs sprites 8×8)
            raw = []
            for i in range(64):
                y = int(ram[OAM_BASE + i*4 + 0])
                attr = int(ram[OAM_BASE + i*4 + 2])
                x = int(ram[OAM_BASE + i*4 + 3])
                if y < HUD_Y_MIN or y >= 240:
                    continue
                if (attr & 0x03) == MARIO_PAL:  # exclure Mario seulement
                    continue
                raw.append((x, y))

            if not raw:
                return []

            # Fusionner les sprites proches en entités (tolérance 20px)
            entities = []
            used = set()
            for i, (x1, y1) in enumerate(raw):
                if i in used:
                    continue
                group = [(x1, y1)]
                used.add(i)
                for j, (x2, y2) in enumerate(raw):
                    if j in used:
                        continue
                    if abs(x1 - x2) <= 20 and abs(y1 - y2) <= 24:
                        group.append((x2, y2))
                        used.add(j)
                cx = sum(g[0] for g in group) // len(group)
                cy = sum(g[1] for g in group) // len(group)
                dist = cx - mario_x_oam
                entities.append({
                    'x_screen': cx,
                    'y_screen': cy,
                    'distance_px': dist,
                    'direction': 'devant' if dist > 0 else 'derrière',
                })

            # Trier par distance absolue
            entities.sort(key=lambda e: abs(e['distance_px']))
            return entities

        except Exception:
            return []

    def get_mystery_blocks_from_screenshot(self):
        """Retourne les mystery blocks détectés via le masque couleur NES (RGB 68,160,252).
        Retourne une liste de dicts: {x_screen, y_screen, distance_px}
        distance_px positif = devant Mario, négatif = derrière.
        """
        try:
            mystery_mask = getattr(self, '_precomputed_mystery_mask', None)
            if mystery_mask is None or not np.any(mystery_mask):
                return []

            h, w = mystery_mask.shape[:2]
            mario_sx = w // 3  # Mario est ~1/3 de l'écran en largeur

            cols_with_mystery = np.where(np.any(mystery_mask, axis=0))[0]
            if len(cols_with_mystery) == 0:
                return []

            # Grouper les colonnes proches en blocs distincts (tolérance 8px)
            blocks = []
            current_group = [int(cols_with_mystery[0])]
            for col in cols_with_mystery[1:]:
                if int(col) - current_group[-1] <= 8:
                    current_group.append(int(col))
                else:
                    cx = sum(current_group) // len(current_group)
                    col_pixels = np.where(mystery_mask[:, cx])[0]
                    cy = int(np.mean(col_pixels)) if len(col_pixels) > 0 else h // 3
                    blocks.append({'x_screen': cx, 'y_screen': cy, 'distance_px': cx - mario_sx})
                    current_group = [int(col)]
            # Dernier groupe
            cx = sum(current_group) // len(current_group)
            col_pixels = np.where(mystery_mask[:, cx])[0]
            cy = int(np.mean(col_pixels)) if len(col_pixels) > 0 else h // 3
            blocks.append({'x_screen': cx, 'y_screen': cy, 'distance_px': cx - mario_sx})

            # Trier par distance (les blocs devant Mario en premier)
            blocks.sort(key=lambda b: b['distance_px'])
            return blocks
        except Exception:
            return []

    def _mario_is_grounded(self, obs):
        """Détecte si Mario est au sol en vérifiant les pixels directement sous ses pieds.
        Utilise l'OAM NES pour la position exacte du sprite Mario (index 1, palette 0).
        Retourne True si sol détecté (pixels non-fond sous les pieds), False si en l'air.

        Logique : si les pixels sous Mario sont du "vide" (fond du ciel ou noir underground),
        Mario est en l'air. Si c'est autre chose (sol, tuile, tuyau...), Mario a atterri.
        """
        try:
            ram = self.env.unwrapped.ram
            # Sprite 1 = Mario principal (palette 0 = MARIO_PAL)
            # OAM format : [Y_screen, tile, attr, X_screen]
            mario_y_oam = int(ram[0x0200 + 1 * 4 + 0])
            mario_x_oam = int(ram[0x0200 + 1 * 4 + 3])

            # Mario sprite = 16px de haut (2 tuiles 8×8 empilées)
            # Les pieds sont à mario_y_oam + 16
            feet_y_nes = mario_y_oam + 16  # coordonnée NES écran (0-239)

            # Détection rapide par position OAM : au sol du niveau principal ≈ y_oam ≥ 190
            # Seuil 190 (et non 178) pour éviter false-positive au décollage
            if mario_y_oam >= 190:
                return True

            # Vérification pixel si Mario est dans la zone proche du sol (y_oam >= 140)
            if obs is None or mario_y_oam < 140:
                return False

            obs_h, obs_w = obs.shape[:2]
            scale_x = obs_w / 256.0
            scale_y = obs_h / 240.0

            px_x = int(mario_x_oam * scale_x)
            px_y_feet = int(feet_y_nes * scale_y)

            if px_y_feet >= obs_h:
                return True  # hors écran en bas = sol

            # Vérifier une bande de 3px sous les pieds, sur 10px de large
            non_bg_count = 0
            total = 0
            for dy in range(1, 4):
                py = px_y_feet + dy
                if py >= obs_h:
                    break
                for dx in range(0, 10):
                    px = px_x + dx
                    if 0 <= px < obs_w:
                        r, g, b = int(obs[py, px, 0]), int(obs[py, px, 1]), int(obs[py, px, 2])
                        total += 1
                        # Fond "vide" : ciel bleu NES ≈ (92,148,252) ou noir underground
                        is_sky  = (b > 180 and b > r + 60 and b > g + 50)
                        is_dark = (r < 20 and g < 20 and b < 20)
                        if not is_sky and not is_dark:
                            non_bg_count += 1

            if total > 0 and non_bg_count / total >= 0.3:
                return True

            return False

        except Exception:
            return False

    def take_scene_snapshot(self, obs):
        """Capture l'état de la scène : ennemis (OAM), tuyau (pixel), trou (pixel).
        Retourne un dict compact utilisé pour détecter les changements entre deux appels Claude."""
        enemies = self.get_enemies_from_oam()
        enemy_front = [e for e in enemies if 0 < e['distance_px'] < 200]
        closest_enemy = enemy_front[0]['distance_px'] if enemy_front else None

        pipe = self.detect_pipe_ahead(obs) if obs is not None else {}
        pipe_dist = pipe.get('distance_px') if pipe.get('detected') else None

        hole = self.detect_holes_ahead(obs) if obs is not None else {}
        # Capturer le trou dès qu'il est détecté (pas seulement quand urgent < 120px)
        hole_dist = hole.get('nearest') if hole.get('detected') else None

        return {
            'enemy_dist': closest_enemy,   # px jusqu'à l'ennemi le plus proche devant, None si aucun
            'pipe_dist': pipe_dist,         # px jusqu'au tuyau, None si absent
            'hole_dist': hole_dist,         # px jusqu'au trou, None si absent
            'pipe': pipe,
            'hole': hole,
            'enemy_front': enemy_front,
        }

    # Seuils de déclenchement par niveau (world, stage) → {enemy, pipe, hole}
    # Chaque valeur déclenche Claude une fois quand l'obstacle passe sous ce seuil de distance.
    # World 1-2 : fond souterrain noir → faux positifs trous fréquents → seuils réduits.
    _LEVEL_THRESHOLDS = {
        (1, 1): {'enemy': [180, 100, 50], 'pipe': [150, 80, 40], 'hole': [200, 130, 70, 35]},
        (1, 2): {'enemy': [150,  80, 40], 'pipe': [150, 80, 40], 'hole': [ 80,  45, 20]},
        (1, 3): {'enemy': [180, 100, 50], 'pipe': [150, 80, 40], 'hole': [150,  80, 40]},
        (1, 4): {'enemy': [150,  80, 40], 'pipe': [150, 80, 40], 'hole': [100,  50, 25]},
    }
    _LEVEL_THRESHOLDS_DEFAULT = {'enemy': [180, 100, 50], 'pipe': [150, 80, 40], 'hole': [200, 130, 70, 35]}

    def _get_level_thresholds(self):
        """Retourne les seuils de détection pour le niveau courant."""
        return self._LEVEL_THRESHOLDS.get(
            (self.current_world, self.current_level),
            self._LEVEL_THRESHOLDS_DEFAULT
        )

    def check_scene_thresholds(self, snap):
        """Vérifie si un obstacle a franchi un nouveau seuil de distance.
        Chaque seuil déclenche Claude une seule fois.
        Retourne (triggered, reason) ou (False, '')."""

        # ── Ennemi ──────────────────────────────────────────────────────────
        ed = snap['enemy_dist']
        if ed is not None:
            _prev_ed = self._last_known_enemy_dist
            # Détection d'un NOUVEL ennemi : le précédent est passé sous Mario
            # (distance était <15px, maintenant >40px → Goomba 1 passé, Goomba 2 apparaît)
            _new_enemy = (_prev_ed is not None and _prev_ed < 15 and ed > 40)
            if _new_enemy:
                self._enemy_thresholds_hit.clear()
                self._new_enemy_appeared = True  # Signal pour le code appelant
            for t in self._get_level_thresholds()['enemy']:
                if ed < t and t not in self._enemy_thresholds_hit:
                    self._enemy_thresholds_hit.add(t)
                    self._last_known_enemy_dist = ed
                    return True, f"ennemi franchi seuil {t}px (dist={ed}px)"
            self._last_known_enemy_dist = ed
        else:
            # Ennemi disparu → reset seuils pour le prochain ennemi
            if self._last_known_enemy_dist is not None:
                self._enemy_thresholds_hit.clear()
            self._last_known_enemy_dist = None

        # ── Tuyau ───────────────────────────────────────────────────────────
        pd = snap['pipe_dist']
        if pd is not None:
            for t in self._get_level_thresholds()['pipe']:
                if pd < t and t not in self._pipe_thresholds_hit:
                    self._pipe_thresholds_hit.add(t)
                    self._last_known_pipe_dist = pd
                    return True, f"tuyau franchi seuil {t}px (dist={pd}px)"
            self._last_known_pipe_dist = pd
        else:
            if self._last_known_pipe_dist is not None:
                self._pipe_thresholds_hit.clear()
            self._last_known_pipe_dist = None

        # ── Trou ────────────────────────────────────────────────────────────
        hd = snap['hole_dist']
        if hd is not None:
            for t in self._get_level_thresholds()['hole']:
                if hd < t and t not in self._hole_thresholds_hit:
                    self._hole_thresholds_hit.add(t)
                    self._last_known_hole_dist = hd
                    return True, f"trou franchi seuil {t}px (dist={hd}px)"
            self._last_known_hole_dist = hd
        else:
            if self._last_known_hole_dist is not None:
                self._hole_thresholds_hit.clear()
            self._last_known_hole_dist = None

        return False, ""

    def detect_holes_ahead(self, obs):
        """ Détecte les trous dans le sol devant Mario via analyse pixel.

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

            # Zone d'anticipation : 8 à 210px devant Mario
            # (8px pour ne pas capter les pieds de Mario, 210px pour anticiper tôt)
            ahead_start = mario_screen_x + 8
            ahead_end = min(mario_screen_x + 210, width - 1)

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
                'urgent': nearest_dist < 120,    # Préparer le saut (augmenté pour réagir plus tôt)
            }
        except Exception:
            return _empty

    def detect_pipe_ahead(self, obs):
        """ Détecte un tuyau/obstacle devant Mario pour saut anticipé.

        Scanne colonne par colonne de 15px à 130px devant Mario et cherche
        le premier obstacle vertical non-sol non-ciel (tuyau vert, mur de briques).
        Permet de déclencher Claude bien avant que Mario soit collé au tuyau.

        Retourne :
            detected    : bool — obstacle confirmé
            distance_px : int  — distance écran depuis Mario (pixels)
            height_px   : int  — hauteur estimée de l'obstacle
            jump_type   : str  — 'none'|'max_jump'|'pipe_jump'|'obstacle_jump'
            urgent      : bool — True si distance < 90px (décision urgente)
        """
        _empty = {'detected': False, 'distance_px': 999, 'height_px': 0,
                  'jump_type': 'none', 'urgent': False}
        try:
            h, w = obs.shape[:2]
            game_top = int(h * 0.20)
            game_h = h - game_top
            # Lire la position écran réelle de Mario depuis OAM (sprite 1 = corps)
            # w//3 était décalé de ~27px → Mario détectait son propre sprite comme "tuyau"
            try:
                mario_screen_x = int(self.env.unwrapped.ram[0x0200 + 1 * 4 + 3])
            except Exception:
                mario_screen_x = w // 3

            # Sol approximatif (même référence que detect_obstacle_height)
            floor_y = game_top + int(game_h * 0.85)

            # Couleur du ciel : mesurée dans le haut de la zone de jeu
            sky_col_start = mario_screen_x + 15
            sky_col_end = min(mario_screen_x + 60, w - 1)
            sky_row = obs[game_top + 5, sky_col_start:sky_col_end]
            if len(sky_row) == 0:
                return _empty
            sky_r = int(np.median(sky_row[:, 0]))
            sky_g = int(np.median(sky_row[:, 1]))
            sky_b = int(np.median(sky_row[:, 2]))

            max_obstacle_h = 60  # hauteur max tuyau W1-1 (~3 tuiles NES)

            # Scanner colonne par colonne de 15px à 130px devant Mario
            obstacle_col = None
            obstacle_height = 0

            for col_offset in range(15, 160):
                col = mario_screen_x + col_offset
                if col >= w:
                    break
                # Pixels dans la bande verticale au-dessus du sol
                col_pixels = obs[floor_y - max_obstacle_h:floor_y, col]
                r = col_pixels[:, 0].astype(np.int16)
                g = col_pixels[:, 1].astype(np.int16)
                b = col_pixels[:, 2].astype(np.int16)
                # Pixel "ciel" = proche de la teinte ciel détectée
                is_sky = ((np.abs(r - sky_r) < 30) &
                          (np.abs(g - sky_g) < 30) &
                          (np.abs(b - sky_b) < 30))
                # Pixel "obstacle" = visible et pas ciel (tuyau vert OU brique)
                is_bright = (r > 50) | (g > 50)
                # Exclure les décors (buissons/collines verts) : vert dominant = décor d'arrière-plan
                # Même logique que l'obfuscation des screenshots envoyés à Claude (ligne ~1201)
                is_decoration = (g - r > 40) & (g - b > 40) & (g > 80)
                is_obstacle = (~is_sky) & is_bright & (~is_decoration)

                n_obs = int(np.sum(is_obstacle))
                if n_obs >= 4:  # ≥4 pixels consécutifs = obstacle réel (pas bruit)
                    obstacle_height = n_obs
                    obstacle_col = col_offset
                    break

            if obstacle_col is None or obstacle_height < 4:
                return _empty

            # Type de saut selon la hauteur
            if obstacle_height <= 20:
                jump_type = 'max_jump'
            elif obstacle_height <= 45:
                jump_type = 'pipe_jump'
            else:
                jump_type = 'obstacle_jump'

            return {
                'detected': True,
                'distance_px': obstacle_col,
                'height_px': obstacle_height,
                'jump_type': jump_type,
                'urgent': obstacle_col < 120,  # 120px = assez d'élan pour pipe_jump
            }
        except Exception:
            return _empty

    def _inject_segment_replay(self, sequence: List[tuple], x: int):
        """Injecte une séquence mémorisée dans la queue et passe en mode replay.
        Ne coupe PAS la macro en cours : si Mario est en plein saut lors d'une
        transition de segment, le saut se termine avant que la nouvelle séquence
        commence. Évite les atterrissages dans les tuyaux/trous."""
        self.action_queue.clear()
        self.logger.log_queue_event("CLEAR", getattr(self, '_current_step', 0), "inject_segment_replay")
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
                print(f" Phase 2: REPLAY segment {seg_key} ({len(seq)} macros)")
            else:
                print(f" Phase 2: IA libre segment {seg_key}")

        elif self._run_phase == 3:
            # Phase 3 : replay jusqu'au segment frontière (furthest_x), puis IA
            danger_x = self.segment_memory.get_stage_danger_frontier()
            if x >= danger_x:
                self._phase3_ai_mode = True
                print(f" Phase 3: frontière atteinte x={x} (record={self.segment_memory.furthest_x}) → IA")
            elif seq:
                self._inject_segment_replay(seq, x)
                print(f" Phase 3: replay sûr segment {seg_key} ({len(seq)} macros)")
            else:
                # Pas de séquence mémorisée : injecter run_forward par défaut pour traverser la zone
                self._inject_segment_replay([('run_forward', 3)], x)
                print(f" Phase 3: segment {seg_key} sans mémoire → run_forward x3 par défaut")

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
 OPTIMISATION PHASE 1 — AMÉLIORE LE MEILLEUR PARCOURS CONNU:
Lors d'un run précédent, cette zone ({seg_key}) a été franchie avec:
  {seq_str}

 TON OBJECTIF: PROPOSER UNE SÉQUENCE MEILLEURE QUE CELLE-CI
Critères d'amélioration (par ordre de priorité):
1. VITESSE: Remplace 'walk_right' par 'run_forward', supprime les 'wait' inutiles
2. BLOCS ? (boîtes à interrogation): Ces blocs jaunes avec "?" contiennent pièces/champignons/power-ups.
   Si tu en vois un au-dessus de Mario → saute dessous pour le frapper (action 'short_jump' ou 'max_jump' selon hauteur)
   Ne passe PAS sous un bloc ? sans le frapper — c'est un item gratuit !
3. SCORE: Écraser les Goombas rapporte des points et sécurise le passage

 IMPORTANT: Ta séquence ne remplacera la sauvegarde QUE si Mario va plus loin ou plus vite.
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
            f"\n CONTEXTE PHASE 3 — ZONE FRONTIÈRE (record actuel: x={furthest_x}px):",
            f"Mario vient de rejouer automatiquement les segments mémorisés jusqu'à x={frontier_x}px.",
            "Il a de l'élan — PRIORITÉ: maintenir la vitesse et ne pas s'arrêter !",
        ]
        if replayed_segs:
            lines.append("Segments rejoués juste avant (pour contexte de vitesse):")
            lines.extend(replayed_segs)
        if death_lines:
            lines.append(f" ZONE DANGEREUSE x={frontier_x}-{frontier_x + SEGMENT_SIZE} — morts précédentes:")
            lines.extend(death_lines)
        lines.append(" OBJECTIF: Dépasser x={} — utilise run_forward + sauts anticipés !".format(furthest_x))
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
            action_stuck = names.count(most_common) >= 3  # 3 fois la même sur 6

        if position_stuck and action_stuck:
            self.stuck_counter += 1
            print(f" Blocage détecté (niveau {self.stuck_counter}) à x={current_x:.0f} - action répétée: {most_common}")
            # Mémoriser l'action échouée dans le bucket 50px
            _bucket = int(current_x // 50) * 50
            if _bucket not in self._blocked_macros_by_pos:
                self._blocked_macros_by_pos[_bucket] = set()
            # Ajouter toutes les macros répétées récentes comme échouées
            for _m in recent:
                self._blocked_macros_by_pos[_bucket].add(_m['name'])
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
            print(f" Recherche web : {query}")
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
                print(f" Résultat: {summary[:100]}...")
                return summary
        except Exception as e:
            print(f" Recherche web échouée: {e}")

        return None

    def _find_completed_and_next_stage(self):
        """Cherche les runs parfaits sauvegardés et retourne le dernier stage terminé
        et le suivant dans LEVEL_SEQUENCE.
        Retourne (last_completed, next_stage, perfect_run_path) ou (None, None, None)."""
        import glob as _g, re as _re
        logs_dir = os.path.join(os.path.dirname(__file__) or '.', 'logs')
        files = _g.glob(os.path.join(logs_dir, 'perfect_run_*-*.json'))
        completed = []
        for f in files:
            m = _re.search(r'perfect_run_(\d+)-(\d+)\.json', os.path.basename(f))
            if m:
                w, l = int(m.group(1)), int(m.group(2))
                if (w, l) in LEVEL_SEQUENCE:
                    completed.append((w, l, f))
        if not completed:
            return None, None, None
        # Trier par ordre dans LEVEL_SEQUENCE
        completed.sort(key=lambda t: LEVEL_SEQUENCE.index((t[0], t[1])))
        last_w, last_l, last_path = completed[-1]
        next_stage = get_next_level(last_w, last_l)
        return (last_w, last_l), next_stage, last_path

    def _transition_to_level(self, world: int, level: int, step_count: int) -> bool:
        """Transition vers un nouveau niveau : recréer l'env, charger la mémoire dédiée.
        Retourne True si la transition a réussi, False en cas d'erreur."""
        try:
            # Fermer l'ancien env
            self.env.close()
            # Créer le nouvel env
            env_name = f'SuperMarioBros-{world}-{level}-v3'
            new_env = gym_super_mario_bros.make(env_name)
            self.env = JoypadSpace(new_env, SIMPLE_MOVEMENT)
            print(f" Env créé : {env_name}")

            # Mettre à jour le niveau courant
            self.current_world = world
            self.current_level = level

            # Charger la mémoire dédiée au nouveau niveau
            self.segment_memory = MarioSegmentMemory(get_memory_path(world, level))
            print(f" Mémoire chargée : {self.segment_memory.furthest_x}px connue pour W{world}-{level}")

            # Réinitialiser l'état par niveau (pas le score/coût de session)
            self.rewind_buffer.clear()
            self.rewind_count = 0
            self.deaths_count = 0
            self.lives_used = 0
            self.stuck_counter = 0
            self.last_stuck_check_step = 0
            self.last_stuck_position = None
            self.stuck_search_done = set()
            self._blocked_macros_by_pos = {}
            self._failed_jump_attempts = {}
            self._prev_macro_name = None
            self._last_run_jump_over_x = -999
            self.last_oam_trigger_step = -100
            self._scene_active = False
            self._mid_air_called = False
            self._run_phase = 1
            self._phase3_ai_mode = False
            self._segment_in_replay = False
            self._last_known_solution_x = -999
            self._unstick_start_x = None
            self._unstick_sequence = None
            self._raw_action_history = []
            self._final_action_history = []
            self._claude_generation += 1  # Invalider les threads en cours
            self.claude_thinking = False
            self.level_context_established = False
            self.last_screenshot_step = -999
            self.last_screenshot_x = 0
            self.last_positions_update = 0
            self._prev_score = 0
            self._prev_coins = 0
            self._prev_lives = 3
            self._death_positions = []
            self._danger_zone_x = None
            self._rewind_correction_msg = None
            self._rewind_death_x = None
            self._rewind_active = False
            self._post_rewind_block_inject = False
            self._pre_jump_ram = None
            self._pre_jump_has_full_backup = False
            self._pre_jump_x = 0
            self._pre_jump_y = 200
            self._pre_jump_history_len = 0

            # Logger la transition
            self.logger.log_game_event("LEVEL_TRANSITION", step_count, {
                "world": world, "level": level,
                "env": env_name,
                "memory_furthest_x": self.segment_memory.furthest_x
            })
            print(f" Transition complète → World {world}-{level} | mémoire: {self.segment_memory.furthest_x}px")
            return True

        except Exception as e:
            print(f" Erreur transition vers W{world}-{level}: {e}")
            import traceback; traceback.print_exc()
            return False

    def call_claude_landing_sync(self, obs, mario_x: int, frames_left: int, step_count: int,
                                 scan_enemy_dist: int = None) -> str:
        """Appel Claude Haiku synchrone pendant un saut pour décider de l'atterrissage.
        Retourne : 'far' | 'short' | 'left' | 'stop'
        - far   → garder right+A+B (atterrir loin)
        - short → relâcher A, réduire frames (atterrir court)
        - left  → virer à gauche
        - stop  → chute verticale (NOOP)
        scan_enemy_dist : distance ennemi vue par le SCAN (plus fiable que OAM en saut).
        """
        enemies = self.get_enemies_from_oam()
        enemy_front = [e for e in enemies if 0 < e['distance_px'] < 200]
        hole = self.detect_holes_ahead(obs) if obs is not None else {}
        pipe = self.detect_pipe_ahead(obs) if obs is not None else {}

        # Utiliser la distance SCAN si OAM ne voit pas l'ennemi (peut être "derrière" en OAM)
        oam_dist = enemy_front[0]['distance_px'] if enemy_front else None
        best_enemy_dist = oam_dist
        if best_enemy_dist is None and scan_enemy_dist is not None and scan_enemy_dist > 0:
            best_enemy_dist = scan_enemy_dist
            enemy_str = f"ennemi à ~{scan_enemy_dist}px (SCAN)"
        elif enemy_front:
            enemy_str = ', '.join(f"ennemi à {e['distance_px']}px" for e in enemy_front)
        else:
            enemy_str = 'aucun'

        hole_str = f"trou à {hole['nearest']}px (larg {hole['width']}px)" if hole.get('detected') else 'aucun'
        pipe_str = f"obstacle à {pipe['distance_px']}px (h={pipe['height_px']}px)" if pipe.get('detected') else 'aucun'

        # Estimation de la vitesse horizontale selon l'action de base du saut.
        # action 4 = right+A+B (max speed) → ~3px/frame
        # action 2 = right+A (jump, no run)  → ~1.5px/frame (saut quasi-vertical, peu d'élan)
        # action 3 = right+B (run only)      → ~2.5px/frame
        _base_act = self.current_macro.get('base_action', 4) if self.current_macro else 4
        if _base_act == 4:
            _px_per_frame = 3.0
        elif _base_act == 2:
            _px_per_frame = 1.5  # run_jump_over : saut lent, faible déplacement horizontal
        else:
            _px_per_frame = 2.5
        _est_travel = int(frames_left * _px_per_frame)
        # Zone de danger = distance d'atterrissage + zone de réaction sol (~60px)
        _danger_zone = _est_travel + 60

        # ── Prédictions d'atterrissage ────────────────────────────────────────────
        # Vitesse enemy ~2px/frame vers Mario (Goomba marchant à gauche)
        _enemy_speed = 2
        # Distance parcourue par Mario si "short" (8 frames NOOP max)
        # NOOP = plus de poussée horizontale → décélération rapide ≈ 1px/frame (friction NES)
        # NE PAS utiliser _px_per_frame ici (qui est la vitesse active right+A+B)
        _short_frames = min(frames_left, 8)
        _short_land_px = int(_short_frames * 1.0)  # ~1px/frame avec NOOP

        def _predict(land_px, fly_frames):
            """Calcule la situation (ennemi, trou) au moment de l'atterrissage."""
            result = {}
            if best_enemy_dist is not None:
                # Position de l'ennemi au moment de l'atterrissage (il marche vers Mario)
                enemy_at_land = best_enemy_dist - land_px - (fly_frames * _enemy_speed)
                if enemy_at_land < -10:
                    result['enemy'] = f"ennemi DÉPASSÉ ({abs(int(enemy_at_land))}px derrière) → sûr"
                elif -10 <= enemy_at_land <= 20:
                    result['enemy'] = f"ennemi à {int(enemy_at_land)}px → STOMP (Mario atterrit dessus)"
                else:
                    result['enemy'] = f"ennemi encore à {int(enemy_at_land)}px devant → danger"
            if hole.get('detected'):
                h_near = hole['nearest']
                h_width = hole.get('width', 32)
                # Mario atterrit à land_px devant lui — est-ce dans le trou ?
                if land_px < h_near:
                    result['hole'] = f"trou à {h_near - land_px}px après l'atterrissage → OK"
                elif land_px <= h_near + h_width:
                    result['hole'] = f"Mario atterrit DANS le trou → mort !"
                else:
                    result['hole'] = f"Mario atterrit {land_px - h_near - h_width}px après le trou → OK"
            return result

        _pred_far   = _predict(_est_travel, frames_left)
        _pred_short = _predict(_short_land_px, _short_frames)

        def _fmt(pred):
            parts = []
            if 'enemy' in pred: parts.append(pred['enemy'])
            if 'hole'  in pred: parts.append(pred['hole'])
            return ' | '.join(parts) if parts else 'voie libre'

        _stomp_window = 20
        # Conseil basé sur les prédictions
        _far_enemy = _pred_far.get('enemy', '')
        _short_enemy = _pred_short.get('enemy', '')
        _short_hole_fatal = 'mort' in _pred_short.get('hole', '')
        _far_hole_fatal = 'mort' in _pred_far.get('hole', '')
        if 'STOMP' in _far_enemy:
            _conseil = f"→ \"far\" = {_fmt(_pred_far)} ✓ STOMP possible | \"short\" = {_fmt(_pred_short)}"
        elif 'DÉPASSÉ' in _far_enemy:
            # Ennemi prédit derrière avec far : est-ce que short est aussi sûr ?
            _short_ok = ('DÉPASSÉ' in _short_enemy or 'danger' not in _short_enemy) and not _short_hole_fatal
            if _short_ok:
                _conseil = f"→ Les 2 atterrissages évitent l'ennemi. Préfère \"short\" (atterrir plus près = moins de risque de tomber dans l'inconnu)."
            else:
                _conseil = f"→ \"far\" = {_fmt(_pred_far)} ✓ | \"short\" = {_fmt(_pred_short)} ✗ danger"
        elif 'danger' in _far_enemy:
            _conseil = f"→ \"far\" = {_fmt(_pred_far)} ✗ | \"short\" = {_fmt(_pred_short)}"
        elif best_enemy_dist is None and not hole.get('detected'):
            _conseil = "→ Voie libre. Préfère \"short\" pour atterrir plus près et rester en terrain connu."
        else:
            _conseil = f"→ \"far\" = {_fmt(_pred_far)} | \"short\" = {_fmt(_pred_short)}"

        prompt = f"""Mario est EN L'AIR. frames_left={frames_left}. X={mario_x}.
Atterrissage prédit : "far"=~{_est_travel}px | "short"=~{_short_land_px}px
Situation actuelle : ennemis=[{enemy_str}] | trou=[{hole_str}] | obstacle=[{pipe_str}]

PRÉDICTIONS AU MOMENT DE L'ATTERRISSAGE :
  "far"   → atterrit à ~{mario_x + _est_travel}px : {_fmt(_pred_far)}
  "short" → atterrit à ~{mario_x + _short_land_px}px : {_fmt(_pred_short)}

{_conseil}

Choisis l'atterrissage:
- "far"   → continuer right+A+B (atterrir loin ~{_est_travel}px)
- "short" → relâcher A (atterrir court ~{_short_land_px}px)
- "left"  → virer à gauche (danger immédiat devant)
- "stop"  → chute verticale (stomp si ennemi pile dessous)

JSON uniquement: {{"landing":"<choix>"}}"""

        try:
            self.api_calls += 1
            response = self.claude_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()
            m = re.search(r'"landing"\s*:\s*"(\w+)"', text)
            if m:
                choice = m.group(1)
                if choice in ('far', 'short', 'left', 'stop'):
                    print(f" MID-AIR: atterrissage → {choice} (frames_left={frames_left})")
                    return choice
        except Exception as e:
            print(f" MID-AIR: erreur Claude ({e}), fallback far")
        return 'far'

    def call_claude_scene_sync(self, situation, obs, step_count):
        """Appel Claude synchrone (bloquant) lors d'un changement de scène urgent.
        Le jeu est gelé pendant cet appel. Les actions reçues sont directement
        ajoutées à la queue.
        """
        try:
            self.claude_thinking = True
            screenshot = self.capture_game_screenshot(obs) if obs is not None else None
            if screenshot:
                response_text = self.analyze_screenshot_with_claude(screenshot, situation, step_count)
                self.last_screenshot_step = step_count
                self.last_screenshot_x = situation.get('mario', {}).get('x', self.last_screenshot_x)
                self.last_positions_update = step_count
                if not self.level_context_established:
                    self.level_context_established = True
            else:
                prompt = self.create_claude_prompt(situation)
                response_text = self.call_claude_for_macro(prompt)

            actions = self.parse_claude_actions(response_text)
            if not actions:
                print(" SCENE SYNC: API sans réponse, pas d'action automatique (Claude décidera au prochain cycle)")
            _mario_speed = situation.get('mario', {}).get('speed', 1)
            for action in actions:
                if len(self.action_queue) < 5:
                    mname = action.get('macro_name', '')
                    if mname == 'wait':
                        action = dict(action, macro_name='run_forward',
                                      reasoning='[wait→run] Courir plutôt qu\'attendre')
                    self.action_queue.append(action)
                    _mname_log = action.get('macro_name', '?')
                    _px_log = action.get('px')
                    _px_str_log = f" px={_px_log}" if _px_log else ""
                    self.logger.log_queue_event("PUSH  ", step_count,
                        f"sync→{_mname_log}{_px_str_log} (queue={len(self.action_queue)})")

            _action_parts = []
            for a in actions:
                _n = a.get('macro_name', '?')
                _px = a.get('px')
                _action_parts.append(f"{_n}({_px}px)" if _px else _n)
            print(f" SCENE SYNC: {' → '.join(_action_parts) if _action_parts else 'aucune action'}")
            self.add_llm_response('ACTIONS', ' → '.join(_action_parts) if _action_parts else 'aucune', step_count)

        except Exception as e:
            print(f" SCENE SYNC erreur: {e}")
            self.action_queue.append(self.get_fallback_macro())
        finally:
            self._claude_generation += 1
            self.claude_thinking = False

    def call_claude_obstacle_retry(self, failed_macros, x_before, x_after, obs, step_count):
        """Appel Claude dédié quand Mario a échoué 2+ fois à franchir le même obstacle.
        Donne le contexte précis des tentatives et laisse Claude décider librement."""
        _macro_counts = {}
        for m in failed_macros:
            _macro_counts[m] = _macro_counts.get(m, 0) + 1
        _summary = ', '.join(f"{m}×{c}" for m, c in _macro_counts.items())
        _jump_tried = set(failed_macros)
        _jump_all = ['max_jump', 'pipe_vertical_jump', 'high_obstacle_jump']
        _jump_not_tried = [j for j in _jump_all if j not in _jump_tried]
        _alternatives = (f"Sauts non encore essayés : {', '.join(_jump_not_tried)}"
                         if _jump_not_tried else
                         "Tous les sauts standards ont échoué. Envisage : step_back (reculer pour élan), "
                         "ou l'obstacle est peut-être infranchissable par le dessus (tuyau de sortie ?)")

        prompt = f"""OBSTACLE NON FRANCHI après {len(failed_macros)} tentatives.

Position Mario : x={x_after:.0f}px (revenait de x={x_before:.0f}px)
Tentatives échouées : {_summary}
Delta x à chaque essai : < 40px (Mario revient au point de départ)

{_alternatives}

ANALYSE : Quel est cet obstacle ? Est-il franchissable par le dessus ?
- Si tuyau haut → utilise step_back PUIS pipe_jump (reculer d'abord pour prendre de l'élan AVANT le tuyau!)
- Si plateforme large → obstacle_jump (élan 20f + saut max)
- Si l'obstacle est une sortie/destination → pipe_down ou autre approche
- STRATÉGIE CLEF si tous les sauts directs ont échoué : step_back (reculer ~60px par défaut, ou step_back px=80 pour plus d'élan) pour s'éloigner du bord, PUIS pipe_jump avec élan complet

ACTIONS DÉJÀ ÉCHOUÉES (à éviter) : {_summary}

Donne 2-3 actions différentes pour passer cet obstacle.
JSON UNIQUEMENT :
{{"actions":[{{"macro_action":"step_back"}},{{"macro_action":"pipe_jump"}}],"urgency":8}}"""

        actions = self.parse_claude_actions(self.call_claude_for_macro(prompt, obs=obs))
        if actions:
            print(f"OBSTACLE RETRY: {len(actions)} nouvelles actions après {len(failed_macros)} échecs à x={x_before:.0f}")
            self.action_queue.clear()
            self.logger.log_queue_event("CLEAR", step_count, f"obstacle_retry échecs={len(failed_macros)} x={x_before:.0f}")
            self.current_macro = None
            for action in actions:
                self.action_queue.append(action)
            # Mémoriser les macros échouées pour ne pas les reproposer
            _jbucket = int(x_before // 50) * 50
            if _jbucket not in self._blocked_macros_by_pos:
                self._blocked_macros_by_pos[_jbucket] = set()
            self._blocked_macros_by_pos[_jbucket].update(_jump_tried)

    def call_claude_stuck_mode(self, situation, position, failed_actions, obs, step_count, web_results=None):
        """Appel Claude spécial mode déblocage : contexte enrichi + résultats web."""
        failed_str = ', '.join(failed_actions) if failed_actions else 'inconnues'
        web_section = f"\n INFOS WEB:\n{web_results}" if web_results else ""

        # Ajouts au prompt stuck mode générés par l'auto-improver
        _additions = getattr(self, '_prompt_additions', {}).get('stuck_mode', [])
        _additions_str = ("\n  RÈGLES APPRIS DES SESSIONS PRÉCÉDENTES:\n" +
                          "\n".join(f"- {a}" for a in _additions) + "\n") if _additions else ""

        # Identifier quels types de sauts ont déjà été essayés
        _failed_set = set(failed_actions)
        _jump_tried = [j for j in ['max_jump', 'pipe_jump', 'pipe_vertical_jump', 'obstacle_jump', 'high_obstacle_jump', 'run_jump_over'] if j in _failed_set]
        _jump_not_tried = [j for j in ['pipe_jump', 'pipe_vertical_jump', 'high_obstacle_jump', 'obstacle_jump', 'max_jump'] if j not in _failed_set]
        _jump_escalation = ""
        if _jump_tried:
            _jump_escalation = f"\n SAUTS DÉJÀ ESSAYÉS SANS SUCCÈS: {', '.join(_jump_tried)}"
            if _jump_not_tried:
                _jump_escalation += f"\n→ ESSAIE MAINTENANT: {', '.join(_jump_not_tried)} (différent!)"
            else:
                _jump_escalation += "\n→ Tous les sauts simples échoués. Essaie step_back + élan + pipe_jump"

        prompt = f"""🚨 MARIO EST BLOQUÉ depuis plusieurs secondes à la même position!

Position Mario: x={position:.0f}px (World 1-1)
Actions répétées sans succès: {failed_str}
Progression: {situation.get('progress', {}).get('trend', 0):.1f}px/check
{_jump_escalation}{web_section}{_additions_str}
ANALYSE REQUISE:
- Quel obstacle bloque Mario à cette position ?
- Si max_jump a échoué → c'est probablement un TUYAU HAUT → utilise 'pipe_jump' (séquence auto : approche 40f + saut max 40f)
- Si max_jump+pipe_jump ont échoué → plateforme très élevée → utilise 'obstacle_jump' (élan 20f + saut max 40f)
- Essaie aussi: step_back (reculer) pour prendre de l'élan, puis relancer un saut plus loin

ACTIONS INTERDITES (déjà essayées sans effet): {failed_str}

🎯 DONNE 3-4 ACTIONS DIFFÉRENTES pour débloquer Mario (pas celles qui ont déjà échoué!)
⚠️ JSON UNIQUEMENT — ZÉRO TEXTE, ZÉRO EXPLICATION:
{{"actions":[{{"macro_action":"step_back"}},{{"macro_action":"pipe_jump"}}],"urgency":9}}
(run_forward DOIT avoir px = pixels à parcourir)"""

        actions = self.parse_claude_actions(
            self.call_claude_for_macro(prompt)
        )
        if actions:
            print(f" Mode déblocage: {len(actions)} nouvelles actions injectées")
            self.action_queue.clear()
            self.logger.log_queue_event("CLEAR", getattr(self, '_current_step', 0), "mode_deblocage")
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

        # Sauvegarder l'état NES juste avant un saut — utilisé pour le rewind fell_in_hole
        # (permet de restaurer Mario au sol, avant le saut fatal, sans replay problématique)
        _PRE_JUMP_MACROS = {'pipe_jump', 'obstacle_jump', 'high_obstacle_jump', 'pipe_vertical_jump',
                            'max_jump', 'run_jump_over', 'big_jump_right', 'short_jump', 'long_jump',
                            'high_jump', 'precise_jump'}

        # Nouveau saut : reset mid-air + seuils de détection pour CE saut.
        if macro_name in _PRE_JUMP_MACROS:
            self._mid_air_called = False
            self._mid_air_emergency_called = False
            # Pré-remplir les seuils ennemis déjà franchis au moment du saut.
            # Ne pas vider complètement : le même ennemi re-déclencherait immédiatement
            # sur la frame suivante (boucle infinie CLEAR+SYNC → run_jump_over jamais exécuté).
            # On marque comme "déjà vus" tous les seuils que la distance actuelle a dépassés.
            # Un NOUVEL ennemi (distance passe de <15 à >40) vide les seuils via check_scene_thresholds.
            _cur_ed = self._last_known_enemy_dist
            if _cur_ed is not None:
                _all_enemy_t = set(self._get_level_thresholds().get('enemy', []))
                self._enemy_thresholds_hit = {t for t in _all_enemy_t if _cur_ed <= t}
            # else : pas d'ennemi connu → laisser les seuils tels quels
            self._scene_active = False  # Permettre re-déclenchement pour UN NOUVEL ennemi
            self._new_enemy_appeared = False  # Reset au départ du saut
        if macro_name in _PRE_JUMP_MACROS:
            try:
                _ram_now = self.env.unwrapped._ram_buffer()
                # Détection "collé au tuyau" (jump→jump sans avancer) :
                # Si le saut précédent n'a pas avancé Mario de 10px, on compte un échec.
                # Seuil 10px (et non 40px) pour éviter les faux positifs sur plateformes en escalier
                # où Mario saute de plateforme en plateforme avec ~20-35px d'avance par saut.
                # Au 2e échec consécutif dans la même zone : step_back injecté avant ce saut.
                _new_jump_x = int(_ram_now[0x6D]) * 256 + int(_ram_now[0x86])
                if self._pre_jump_ram is not None:
                    _pipe_advance = _new_jump_x - self._pre_jump_x
                    if _pipe_advance < 10:
                        _jbucket_pipe = int(max(self._pre_jump_x, 0) // 50) * 50
                        if _jbucket_pipe not in self._failed_jump_attempts:
                            self._failed_jump_attempts[_jbucket_pipe] = []
                        self._failed_jump_attempts[_jbucket_pipe].append(macro_name)
                        _n_pipe_fails = len(self._failed_jump_attempts[_jbucket_pipe])
                        self.logger.log_queue_event("JUMP_FAIL", getattr(self, '_current_step', 0),
                            f"jump→jump advance={_pipe_advance:.0f}px bucket={_jbucket_pipe} n={_n_pipe_fails}")
                        if _n_pipe_fails >= 2:
                            self._failed_jump_attempts.pop(_jbucket_pipe, None)
                            self.action_queue.appendleft(macro_decision)
                            self.action_queue.appendleft({'macro_name': 'step_back'})
                            self.logger.log_queue_event("STUCK ", getattr(self, '_current_step', 0),
                                f"pipe stuck x={_new_jump_x} → step_back + {macro_name}")
                            print(f"Pipe stuck ({_n_pipe_fails}x sans avancer 10px) → step_back + {macro_name}")
                            return  # step_back prend la main, saut rejoué ensuite
                self._pre_jump_ram = _ram_now.copy()
                # Lire x_pos directement depuis la RAM NES (évite la staleness de last_situation)
                # Formule gym-super-mario-bros : RAM[0x6D]*256 + RAM[0x86]
                self._pre_jump_x = int(_ram_now[0x6D]) * 256 + int(_ram_now[0x86])
                self._pre_jump_y = int(_ram_now[0x03])  # RAM 0x03 = y_pos NES
                self._pre_jump_history_len = len(self._final_action_history)
                # Backup complet (CPU+PPU) pour restore exact sans rejouer dans le trou
                self.env.unwrapped._backup()
                self._pre_jump_has_full_backup = True
                # Invalider les checkpoints rewind buffer (slot unique écrasé)
                for _cp in self.rewind_buffer:
                    _cp['has_full_backup'] = False
            except Exception:
                self._pre_jump_has_full_backup = False
                if self.last_situation:
                    self._pre_jump_x = self.last_situation['mario']['x']
                    self._pre_jump_y = self.last_situation['mario']['y']

        px = macro_decision.get('px')

        if macro_name == 'run_jump_over':
            # px = distance du saut (obligatoire), approach_px = course avant saut (optionnel)
            _jump_px = px if px is not None else 40
            _approach_px = macro_decision.get('approach_px', 0) or 0
            approach_frames = max(0, round(int(_approach_px) / 2))
            jump_frames = max(10, min(40, round((int(_jump_px) + 10) / 3)))
            # Guard : si ennemi dans la zone d'approche → saut immédiat
            _enemy_dist_now = getattr(self, '_log_last_enemy_dist', None)
            if approach_frames > 0 and _enemy_dist_now is not None and _enemy_dist_now > 0:
                _max_safe = max(0, int((_enemy_dist_now - 30) / 5))
                if approach_frames > _max_safe:
                    print(f"run_jump_over approche réduite {approach_frames}f→0 (ennemi={_enemy_dist_now}px)")
                    approach_frames = 0
            if approach_frames == 0:
                self.current_macro = {
                    'name': macro_name,
                    'base_action': 4,
                    'frames_left': jump_frames,
                    '_initial_frames': jump_frames,
                    'decision': macro_decision
                }
            else:
                self.current_macro = {
                    'name': macro_name,
                    'phases': [
                        {'base_action': 3, 'duration': approach_frames},
                        {'base_action': 4, 'duration': jump_frames}
                    ],
                    'current_phase': 0,
                    'base_action': 3,
                    'frames_left': approach_frames,
                    'decision': macro_decision
                }

        if macro_name == 'max_jump' and px is None:
            # max_jump SANS px = saut direct court (ennemi proche, pas d'approche)
            # Durée fixe 20 frames = ~60px horizontal. MID-AIR peut raccourcir à 8f si besoin.
            self.current_macro = {
                'name': macro_name,
                'base_action': 4,
                'frames_left': 20,
                '_initial_frames': 20,
                'decision': macro_decision
            }
        elif macro_name == 'max_jump' and px is not None:
            # max_jump avec px = approche N pixels vers la droite puis saut max
            # Transformé en 2 phases : phase1=run_right(approach_frames), phase2=max_jump(40f)
            approach_frames = max(3, min(80, round(int(px) / 2)))
            # Guard : si un ennemi est dans la zone d'approche → saut immédiat sans approche
            _enemy_dist_now = getattr(self, '_log_last_enemy_dist', None)
            if _enemy_dist_now is not None and _enemy_dist_now > 0:
                _max_safe_approach = max(0, int((_enemy_dist_now - 30) / 5))
                if approach_frames > _max_safe_approach:
                    print(f"max_jump approche réduite {approach_frames}f→0 (ennemi={_enemy_dist_now}px, saut immédiat)")
                    approach_frames = 0
            if approach_frames == 0:
                # Saut direct — px = distance à couvrir → durée = px / 3px/frame, min 15f, max 40f
                jump_frames = max(15, min(40, round((int(px) + 10) / 3)))
                print(f"max_jump saut direct {jump_frames}f pour couvrir {px+10}px ({px}px+10 marge)")
                self.current_macro = {
                    'name': macro_name,
                    'base_action': 4,
                    'frames_left': jump_frames,
                    '_initial_frames': jump_frames,
                    'decision': macro_decision
                }
            else:
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
        elif macro_name == 'stomp_enemy':
            # Smart stomp : courir vers l'ennemi, sauter automatiquement quand OAM < seuil.
            # Sans px : approche longue (200f), le smart stomp OAM déclenche le saut.
            # Avec px : on soustrait ~15px pour compenser le mouvement de l'ennemi vers Mario
            #   (Goomba ~0.75px/frame × ~20 frames d'approche ≈ 15px de fermeture supplémentaire)
            #   → Mario déclenche le saut un peu plus tôt, évite d'arriver trop près.
            if px is not None:
                _enemy_movement_offset = 15  # px soustraits pour mouvement ennemi
                approach_frames = max(3, min(200, round(max(0, int(px) - _enemy_movement_offset) / 2)))
            else:
                approach_frames = 200  # Max 400px d'approche — OAM déclenchera le saut
            self.current_macro = {
                'name': macro_name,
                'phases': [
                    {'base_action': 3, 'duration': approach_frames},  # run right (B held)
                    {'base_action': 4, 'duration': 35}                 # right+A+B saut rapide avec élan
                ],
                'current_phase': 0,
                'base_action': 3,
                'frames_left': approach_frames,
                'decision': macro_decision
            }
        elif macro_name in ('pipe_jump', 'obstacle_jump') and px is not None:
            # pipe_jump/obstacle_jump avec px = distance d'approche avant le saut
            # Claude contrôle combien de frames courir avant de sauter
            approach_frames = max(3, min(80, round(int(px) / 2)))
            # Sécurité : si un ennemi est dans la zone d'approche, raccourcir l'élan.
            # Taux de fermeture ~5px/frame (Mario 3px/f + Goomba 2px/f).
            # On veut garder ≥30px de marge au moment du saut → max_safe = (dist-30)/5.
            _enemy_dist_now = getattr(self, '_log_last_enemy_dist', None)
            if _enemy_dist_now is not None and _enemy_dist_now > 0:
                _max_safe_approach = max(0, int((_enemy_dist_now - 30) / 5))
                if approach_frames > _max_safe_approach:
                    print(f"pipe_jump approche réduite {approach_frames}f→0 (ennemi={_enemy_dist_now}px, saut immédiat avec élan actuel)")
                    approach_frames = 0  # Saut immédiat — vitesse de l'action précédente (~3px/f), meilleur que approche partielle
            self.current_macro = {
                'name': macro_name,
                'phases': [
                    {'base_action': 3, 'duration': approach_frames},  # course d'élan
                    {'base_action': 4, 'duration': 40}                 # saut max
                ],
                'current_phase': 0,
                'base_action': 3,
                'frames_left': approach_frames,
                'decision': macro_decision
            }
        elif macro_name == 'step_back':
            # Recul vers la gauche. Vitesse ≈ 3px/frame.
            # Claude peut fournir px = distance souhaitée, sinon durée par défaut (20f = ~60px).
            if px is not None:
                frames = max(5, min(100, round(int(px) / 3)))
            else:
                frames = macro_config['duration']  # 20 frames par défaut
            self.current_macro = {
                'name': macro_name,
                'base_action': macro_config['base_action'],
                'frames_left': frames,
                '_initial_frames': frames,
                'decision': macro_decision
            }
        elif 'phases' in macro_config:
            # Macro multi-phases : phase 1 d'abord (px ignoré → durée config par défaut)
            _ph0_dur = macro_config['phases'][0]['duration']
            self.current_macro = {
                'name': macro_name,
                'phases': macro_config['phases'],
                'current_phase': 0,
                'base_action': macro_config['phases'][0]['base_action'],
                'frames_left': _ph0_dur,
                '_initial_frames': _ph0_dur,  # requis pour _enough_airtime dans LAND detection
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
                '_initial_frames': frames,
                'decision': macro_decision
            }

        self._life_macro_count += 1
        self.macro_history.append({'name': macro_name})

        # Enregistrer l'action pour apprentissage
        if hasattr(self, 'last_situation') and self.last_situation is not None:
            if px is not None:
                _log_name = f"{macro_name}({px}px)"
            elif 'phases' in macro_config:
                _phase_desc = "+".join(f"{p['duration']}f" for p in macro_config['phases'])
                _log_name = f"{macro_name}[{_phase_desc}]"
            else:
                _log_name = f"{macro_name}[{macro_config['duration']}f]"
            self.record_action(
                _log_name,
                self.last_situation,
                getattr(self, 'current_step', 0),
                "AI"
            )

        if 'phases' in macro_config:
            total_frames = sum(p['duration'] for p in macro_config['phases'])
            print(f"Execution: {macro_name} ({len(macro_config['phases'])} phases, {total_frames} frames)")
        else:
            print(f"Execution: {macro_name} ({macro_config['duration']} frames)")

        _step = getattr(self, 'current_step', 0)
        _px_str = f" px={px}" if px is not None else ""
        self.logger.log_queue_event("PLAY  ", _step, f"{macro_name}{_px_str}")

    def get_current_action(self):
        """Obtenir l'action à exécuter cette frame"""

        if self.current_macro and self.current_macro['frames_left'] > 0:
            # Continuer la macro en cours
            self.current_macro['frames_left'] -= 1
            action = self.current_macro['base_action']
            # Dernière frame d'un saut : relâcher A pour garantir le front montant
            # du saut suivant. NES exige release+press ; sans ça, le 2e saut ne part pas.
            _JUMP_MACROS_LOCAL = {'pipe_jump', 'obstacle_jump', 'high_obstacle_jump',
                                  'pipe_vertical_jump', 'max_jump', 'run_jump_over',
                                  'big_jump_right', 'short_jump', 'long_jump',
                                  'high_jump', 'precise_jump'}
            if (self.current_macro['frames_left'] == 0 and
                    self.current_macro.get('name') in _JUMP_MACROS_LOCAL and
                    action in {2, 4}):
                action = 3 if action == 4 else 1  # strip A
                self._a_release_frames = 0  # LAND ne devra pas rajouter de délai
            return action
        else:
            # Phase terminée — vérifier s'il y a une phase suivante (multi-phases)
            if self.current_macro and 'phases' in self.current_macro:
                next_phase = self.current_macro['current_phase'] + 1
                if next_phase < len(self.current_macro['phases']):
                    ph = self.current_macro['phases'][next_phase]
                    self.current_macro['current_phase'] = next_phase
                    self.current_macro['base_action'] = ph['base_action']
                    self.current_macro['frames_left'] = ph['duration']
                    self.current_macro['_initial_frames'] = ph['duration']  # reset pour LAND check
                    return self.get_current_action()

            # Macro (ou dernière phase) terminée
            if self.current_macro:
                self.successful_macros += 1
                _JUMP_MACROS_LOCAL = {'pipe_jump', 'obstacle_jump', 'max_jump',
                                      'run_jump_over', 'stomp_enemy'}
                if self.current_macro.get('name') in _JUMP_MACROS_LOCAL:
                    # Saut terminé → reset des seuils : les obstacles franchis sont derrière
                    self._last_scene_snapshot = None
                    self._enemy_thresholds_hit.clear()
                    self._pipe_thresholds_hit.clear()
                    self._hole_thresholds_hit.clear()
                self.current_macro = None

            # Vérifier s'il y a des actions en attente
            if self.action_queue:
                next_action = self.action_queue.popleft()
                _step_pop = getattr(self, 'current_step', 0)
                _mname_pop = next_action.get('macro_name', '?')
                _px_pop = next_action.get('px')
                _px_str_pop = f" px={_px_pop}" if _px_pop else ""
                self.logger.log_queue_event("POP   ", _step_pop,
                    f"{_mname_pop}{_px_str_pop} (remaining={len(self.action_queue)})")
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
        # Auto-scroll vers le bas uniquement si l'utilisateur n'a pas scrollé vers le haut
        # (si on est dans les 5 dernières entrées, on suit le bas automatiquement)
        items_per_page_est = 5
        near_bottom = self.llm_scroll_position >= len(self.llm_responses) - items_per_page_est - 1
        if near_bottom or len(self.llm_responses) <= items_per_page_est:
            self.llm_scroll_position = max(0, len(self.llm_responses) - items_per_page_est)

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
                            formatted_parts.append(f"ACTION: {action_name}")

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
        cv2.putText(canvas, "HISTORIQUE LLM  [W/S ou fleches: scroll  U/D: page  H/E: debut/fin]", (x_start + 10, title_y),
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

            # En-tête de la réponse — couleur selon le type d'événement
            _rtype = response['type']
            if _rtype == 'APPEL':
                header_color = (255, 180, 0)    # Bleu-cyan : call en cours
            elif _rtype == 'ANNULE':
                header_color = (100, 100, 100)  # Gris : annulé
            elif _rtype == 'ACTIONS':
                header_color = (0, 220, 0)      # Vert vif : réponse reçue
            elif _rtype == 'SCREENSHOT':
                header_color = CYAN
            else:
                header_color = GREEN
            header_text = f"[{response['timestamp']}] {_rtype} step={response['step']}"
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

        # Scrollbar visuelle (barre verticale à droite)
        total = len(self.llm_responses)
        items_per_page = max(1, max_lines_visible // 8)
        sb_x = x_start + width - 10   # bande de 8px à droite
        sb_y_top = content_start_y
        sb_height = content_height - 10
        # Fond de la scrollbar
        cv2.rectangle(canvas, (sb_x, sb_y_top), (sb_x + 8, sb_y_top + sb_height), (50, 50, 50), -1)
        if total > items_per_page:
            # Calcul position et taille du curseur
            thumb_h = max(20, int(sb_height * items_per_page / total))
            max_scroll = max(1, total - items_per_page)
            thumb_y = sb_y_top + int((sb_height - thumb_h) * self.llm_scroll_position / max_scroll)
            cv2.rectangle(canvas, (sb_x, thumb_y), (sb_x + 8, thumb_y + thumb_h), (180, 180, 180), -1)
        # Compteur en bas
        scroll_indicator = f"{start_idx + 1}-{min(end_idx, total)}/{total}"
        cv2.putText(canvas, scroll_indicator, (x_start + width - 70, y_start + height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, GRAY, 1)

    def handle_scroll_keys(self, key):
        """Gérer les touches de défilement pour l'encart LLM"""
        scroll_changed = False

        # Flèche haut ou W (différentes variantes)
        if key in [ord('w'), ord('W'), 119, 87, 82, 2490368]:  # W, w, Up arrow
            self.llm_scroll_position = max(0, self.llm_scroll_position - 1)
            scroll_changed = True
            print(f" Scroll UP → Position: {self.llm_scroll_position}")

        # Flèche bas ou S (différentes variantes)
        elif key in [ord('s'), ord('S'), 115, 83, 84, 2621440]:  # S, s, Down arrow
            max_scroll = max(0, len(self.llm_responses) - 5)
            self.llm_scroll_position = min(max_scroll, self.llm_scroll_position + 1)
            scroll_changed = True
            print(f" Scroll DOWN → Position: {self.llm_scroll_position}/{max_scroll}")

        # Page Up / Page Down
        elif key in [2162688, ord('u'), ord('U')]:  # Page Up ou U
            self.llm_scroll_position = max(0, self.llm_scroll_position - 5)
            scroll_changed = True
            print(f" PAGE UP → Position: {self.llm_scroll_position}")

        elif key in [2228224, ord('d'), ord('D')]:  # Page Down ou D
            max_scroll = max(0, len(self.llm_responses) - 5)
            self.llm_scroll_position = min(max_scroll, self.llm_scroll_position + 5)
            scroll_changed = True
            print(f" PAGE DOWN → Position: {self.llm_scroll_position}/{max_scroll}")

        # Début/fin
        elif key in [ord('h'), ord('H')]:  # Home
            self.llm_scroll_position = 0
            scroll_changed = True
            print(f" HOME → Position: 0")

        elif key in [ord('e'), ord('E')]:  # End
            max_scroll = max(0, len(self.llm_responses) - 5)
            self.llm_scroll_position = max_scroll
            scroll_changed = True
            print(f" END → Position: {max_scroll}")

        # Debug: afficher le code de la touche pressée si pas reconnue
        elif key != 255:  # 255 = pas de touche
            print(f" Touche non reconnue: {key} (chr: {chr(key) if 32 <= key <= 126 else 'non-printable'})")

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
        if self.current_macro:
            _cm_name = self.current_macro['name']
            _cm_px = self.current_macro.get('decision', {}).get('px')
            _cur_macro = f"{_cm_name}({_cm_px}px)" if _cm_px is not None else _cm_name
        else:
            _cur_macro = "-"
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
        cv2.putText(canvas, " MARIO FLUIDE - CLAUDE LLM", (680, 30),
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
            cv2.putText(canvas, " MODE REPLAY:", (680, y_pos),
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
                cv2.putText(canvas, " IA prend la main", (680, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
        else:
            cv2.putText(canvas, " ACTION ACTUELLE:", (680, y_pos),
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
                thinking_status = " Réfléchit..." if self.claude_thinking else " Prêt"
                cv2.putText(canvas, thinking_status, (680, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)

        y_pos += 20
        cv2.putText(canvas, f"Queue: {len(self.action_queue)} actions", (680, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)

        # Décision de Claude
        if mario_decision:
            y_pos += 40
            cv2.putText(canvas, "CLAUDE DECIDE:", (680, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

            strategy = mario_decision.get('strategy', '')
            if strategy:
                y_pos += 25
                strategy_lines = self.wrap_text(strategy, 35)
                for line in strategy_lines[:2]:
                    cv2.putText(canvas, line, (680, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
                    y_pos += 16

        # Éléments Mario Bros détectés
        y_pos += 30
        screen = situation['screen']

        if screen.get('environment_type') == 'screenshot_mode':
            # Mode screenshots Claude
            cv2.putText(canvas, " CLAUDE VISION:", (680, y_pos),
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
            cv2.putText(canvas, " ÉLÉMENTS MARIO:", (680, y_pos),
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
        cv2.putText(canvas, " PROGRESSION:", (680, y_pos),
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
            print(" Situation None, pas d'enregistrement d'action")
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
                print(f" PATTERN D'ÉCHEC DÉTECTÉ: Mort #{self.repeated_failures[pattern_key]} dans la zone {death_position//50}")

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
            context.append(" ZONES DE DANGER IDENTIFIÉES:")
            for pattern, count in self.repeated_failures.items():
                if count > 1:
                    zone = pattern.replace('_death', '')
                    context.append(f"   - Zone {zone}: {count} morts répétées - ÉVITER ces actions!")

        # Patterns d'échec récents
        if len(self.failure_patterns) >= 2:
            recent_failures = self.failure_patterns[-2:]
            context.append(" ERREURS RÉCENTES À NE PAS RÉPÉTER:")
            for i, failure in enumerate(recent_failures, 1):
                failed_actions = [a['action'] for a in failure['actions_before_death'][-3:]]
                context.append(f"   Échec #{i}: actions {' → '.join(failed_actions)} à la position {failure['death_position']}")

        # Stratégies qui marchent
        if self.successful_strategies:
            best_strategy = max(self.successful_strategies, key=lambda x: x['progress'])
            successful_actions = [a for a in best_strategy['actions'] if isinstance(a, str)][:3]
            context.append(f" STRATÉGIE QUI MARCHE: {' → '.join(successful_actions)} (progrès: {best_strategy['progress']}px)")

        # Actions récentes pour éviter les boucles
        if len(self.action_history) >= 5:
            recent_actions = [a['action'] for a in list(self.action_history)[-5:]]
            if len(set(recent_actions)) <= 2:  # Trop d'actions répétitives
                context.append(f" ACTIONS RÉPÉTITIVES DÉTECTÉES: {' → '.join(recent_actions[-3:])} - VARIER LES ACTIONS!")

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

            #  Détection des trous pour le prompt positionnel
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
            print(f" Erreur extraction positions: {e}")
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


📍 POSITIONS ACTUELLES:

MARIO:
• Position: X={mario['x']}, Y={mario['y']}
• Vitesse: {mario['speed']:.1f} pixels/step ({mario['direction']})
• État: {'En mouvement' if mario['speed'] != 0 else 'Stationnaire'}

ENNEMIS DÉTECTÉS: {len(enemies)}"""

        if enemies:
            for i, enemy in enumerate(enemies, 1):
                threat_emoji = "" if enemy['threat_level'] == 'HIGH' else "" if enemy['threat_level'] == 'MEDIUM' else ""
                prompt += f"""
• {threat_emoji} {enemy['type']} #{i}: X={enemy['x']}, distance={enemy['distance_from_mario']}px de Mario ({enemy['threat_level']} threat)"""
        else:
            prompt += "\n•  Aucun ennemi détecté dans la zone visible"

        prompt += f"\n\nBLOCS QUESTION MARKS: {len(blocks)}"
        if blocks:
            closest_enemy_dist_pos = min(
                (abs(e['distance_from_mario']) for e in enemies if e.get('distance_from_mario') is not None),
                default=999
            )
            for i, block in enumerate(blocks, 1):
                dist = block.get('distance_from_mario', '?')
                if isinstance(dist, (int, float)) and dist > 0:
                    if closest_enemy_dist_pos > 80:
                        hint = f" ZONE SURE → position_under_block px={max(0, int(dist) - 20)} puis hit_block"
                    else:
                        hint = f" DANGEREUX — ennemi a {closest_enemy_dist_pos}px (eliminer d'abord)"
                else:
                    hint = ""
                prompt += f"\n• Bloc ? #{i}: X={block['x']}, distance={dist}px de Mario{hint}"
        else:
            prompt += "\n• Aucun bloc ? visible dans la zone"

        if changes:
            prompt += f"\n\n CHANGEMENTS DÉTECTÉS:"
            for change in changes:
                prompt += f"\n• {change}"

        #  Section macros bloquées
        _pu_bucket = int(mario['x'] // 50) * 50
        _pu_blocked = getattr(self, '_blocked_macros_by_pos', {}).get(_pu_bucket, set())
        if _pu_blocked:
            prompt += f"\n\n ACTIONS DÉJÀ ESSAYÉES SANS SUCCÈS à x≈{_pu_bucket}: {', '.join(sorted(_pu_blocked))}\n→ N'utilise PAS ces actions — elles ont déjà échoué!"

        #  Section trou
        hole_info = positions_data.get('holes', {})
        if hole_info.get('detected'):
            if hole_info.get('critical'):
                prompt += (f"\n\n TROU CRITIQUE: sol absent à {hole_info['nearest']}px devant Mario"
                           f" (largeur {hole_info['width']}px) → max_jump MAINTENANT, PAS run_forward!")
            elif hole_info.get('urgent'):
                prompt += (f"\n\n TROU URGENT à {hole_info['nearest']}px"
                           f" (largeur {hole_info['width']}px) → max_jump EN PREMIER!")
            else:
                prompt += (f"\n\n TROU DÉTECTÉ à {hole_info['nearest']}px"
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
📐 TOUTES les actions DOIVENT avoir "px" = distance NES à l'obstacle (1px écran ≈ 1px NES, ≈2px/frame).
  run_forward "px" = distance à courir. Sauts "px" = approche avant saut. stomp_enemy "px" = dist_ennemi - 10.

⚠️ JSON UNIQUEMENT — ZÉRO TEXTE, ZÉRO EXPLICATION:
{{"actions":[{{"macro_action":"run_forward","px":60}},{{"macro_action":"pipe_jump","px":20}}],"urgency":<1-10>}}"""

        return prompt

    def call_claude_for_positions_update(self, positions_data, step_count):
        """Appeler Claude avec une mise à jour positionnelle (texte seulement)"""
        try:
            prompt = self.create_positional_update_prompt(positions_data, step_count)

            self.api_calls += 1
            print(f" Envoi mise à jour positionnelle à Claude (appel #{self.api_calls})...")

            # Logger le prompt
            self.logger.log_claude_prompt("POSITIONS", prompt, step_count)

            print("="*50)
            print(" PROMPT POSITIONS ENVOYÉ À CLAUDE:")
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

            print(" Claude analyse reçue (texte)", f"({len(response_text)} chars)")
            print("="*50)
            print(" RÉPONSE DE CLAUDE:")
            print(response_text)
            print("="*50)

            print(f" Coût mise à jour: ${estimated_cost:.4f} (total: ${self.total_cost:.3f})")

            # Ajouter la réponse à l'historique pour l'encart
            self.add_llm_response("POSITIONS", response_text, step_count)

            return response_text

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f" Erreur mise à jour positionnelle: {e}")
            print(f" Détails de l'erreur:")
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
        print(" MARIO FLUIDE - CLAUDE LLM")
        print("="*60)

        # Afficher l'état de la mémoire segments
        seg_mem = self.segment_memory
        if seg_mem.stage.sequences:
            print(f" Mémoire segments: {seg_mem.total_runs} runs, "
                  f"record={seg_mem.furthest_x}px, "
                  f"safe jusqu'à {seg_mem.stage.safe_max_x}px "
                  f"({len(seg_mem.stage.sequences)} segments mémorisés)")
        else:
            print(" Mémoire segments: vide (première partie)")

        # Vérifier s'il existe un run parfait sauvegardé
        import glob as _glob
        _perfect_files = _glob.glob(os.path.join(os.path.dirname(__file__) or '.', 'logs', 'perfect_run_*.json'))
        _has_perfect = bool(_perfect_files)

        # Identifier le dernier stage terminé et le suivant (pour les options continue)
        _last_stage, _next_stage, _ = self._find_completed_and_next_stage()
        _has_continue = _has_perfect and _next_stage is not None

        print("\n CHOISISSEZ VOTRE MODE:")
        print("   1. Nouvelle partie (IA pure, sans mémoire)")
        print("   2. Mémoire automatique (rejoue les segments connus -> IA à la frontière)")
        if _has_perfect:
            print("   3. Rejouer le run parfait (sans pauses, sans IA)")
        if _has_continue:
            lw, ll = _last_stage
            nw, nl = _next_stage
            print(f"   4. Replay auto World {lw}-{ll} -> IA World {nw}-{nl} (enchaîne les stages)")
            print(f"   5. Commencer directement en World {nw}-{nl} (dernier stage non terminé)")
            print("   6. Effacer la mémoire des segments")
            print("   7. Quitter")
            _max_choice = "7"
        elif _has_perfect:
            print("   4. Effacer la mémoire des segments")
            print("   5. Quitter")
            _max_choice = "5"
        else:
            print("   3. Effacer la mémoire des segments")
            print("   4. Quitter")
            _max_choice = "4"
        try:
            while True:
                try:
                    choice = input(f"\n Votre choix (1-{_max_choice}): ").strip()
                except EOFError:
                    print("Mode non-interactif détecté, sélection automatique: nouvelle partie")
                    choice = "1"

                if choice == "1":
                    self.logger.log_menu_choice("new_game")
                    return "new_game", None
                elif choice == "2":
                    if not seg_mem.stage.sequences:
                        print("  Mémoire vide — démarrage en IA pure (les segments seront mémorisés).")
                    self.logger.log_menu_choice("memory_first")
                    return "memory_first", None
                elif choice == "3" and _has_perfect:
                    self.play_perfect_replay()
                    return self.show_game_menu()
                elif choice == "4" and _has_continue:
                    self.logger.log_menu_choice("continue_with_replay")
                    return "continue_with_replay", None
                elif choice == "5" and _has_continue:
                    self.logger.log_menu_choice("continue_direct")
                    return "continue_direct", None
                elif choice == "3" and not _has_perfect:
                    _choice_clear = "3"
                    confirm = input("  Confirmer l'effacement de la mémoire ? (o/N): ").strip().lower()
                    if confirm == "o":
                        self.segment_memory.clear_memory()
                        print(" Mémoire effacée.")
                        self.logger.log_menu_choice("clear_memory")
                    else:
                        print("Annulé.")
                    return self.show_game_menu()
                elif ((_has_continue and choice == "6") or
                      (_has_perfect and not _has_continue and choice == "4") or
                      (not _has_perfect and choice == "3")):
                    confirm = input("  Confirmer l'effacement de la mémoire ? (o/N): ").strip().lower()
                    if confirm == "o":
                        self.segment_memory.clear_memory()
                        print(" Mémoire effacée.")
                        self.logger.log_menu_choice("clear_memory")
                    else:
                        print("Annulé.")
                    return self.show_game_menu()
                elif ((_has_continue and choice == "7") or
                      (_has_perfect and not _has_continue and choice == "5") or
                      (not _has_perfect and choice == "4")):
                    self.logger.log_menu_choice("quit")
                    return "quit", None
                else:
                    print(f" Choix invalide! Veuillez entrer 1-{_max_choice}.")
        except KeyboardInterrupt:
            return "quit", None

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
            print(f" IA reprend la main à l'action {self.replay_index + 1}/{len(self.replay_actions)}")

            # Logger la transition
            self.logger.log_ai_takeover(current_step, progress_info)

            self.replay_mode = False  # Désactiver le replay
            return None  # L'IA reprend

        # Obtenir l'action actuelle
        current_action = self.replay_actions[self.replay_index]

        print(f" Replay {self.replay_index + 1}/{len(self.replay_actions)}: {current_action['action_name']} (pos: {current_action['position_x']})")

        self.replay_index += 1
        return current_action['base_action']

    def _save_perfect_run(self, world=None, level=None):
        """Sauvegarde _final_action_history dans logs/perfect_run_{world}-{level}.json.
        Chaque stage a son propre fichier — sauvegarder le stage 1 ne détruit plus
        la sauvegarde du stage 2 et vice-versa.
        Si world/level sont None, utilise self.current_world / self.current_level.
        Si un rewind a eu lieu, start_ram contient la RAM NES du checkpoint (base64)
        pour que le replay reparte exactement de ce point avec le même timing ennemi."""
        if not self._final_action_history:
            return
        import glob, base64
        if world is None:
            world = self.current_world
        if level is None:
            level = self.current_level
        logs_dir = os.path.join(os.path.dirname(__file__) or '.', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        # Écraser uniquement le fichier du même stage (pas les autres stages)
        path = os.path.join(logs_dir, f'perfect_run_{world}-{level}.json')
        data = {'actions': self._final_action_history,
                'total': len(self._final_action_history),
                'world': world,
                'level': level}
        if self._rewind_checkpoints:
            data['rewind_checkpoints'] = [
                {'index': c['index'],
                 'ram': base64.b64encode(c['ram'].tobytes()).decode(),
                 'x': c['x']}
                for c in self._rewind_checkpoints
            ]
            # Compat ancien format : dernier checkpoint = start_ram/start_x/rewind_index
            last = self._rewind_checkpoints[-1]
            data['start_ram'] = base64.b64encode(last['ram'].tobytes()).decode()
            data['start_x'] = last['x']
            data['rewind_index'] = last['index']
        with open(path, 'w') as f:
            json.dump(data, f, default=lambda o: int(o) if hasattr(o, 'item') else str(o))
        rewind_info = f" ({len(self._rewind_checkpoints)} rewind(s))" if self._rewind_checkpoints else ""
        print(f" Run parfait {world}-{level} sauvegardé : {len(self._final_action_history)} actions{rewind_info} → {path}")

    def play_perfect_replay(self):
        """Rejoue un run parfait sauvegardé à pleine vitesse, sans pause, sans Claude.
        Chaque stage a son propre fichier perfect_run_{world}-{level}.json.
        Si plusieurs stages sont disponibles, propose un choix à l'utilisateur."""
        import glob, base64
        logs_dir = os.path.join(os.path.dirname(__file__) or '.', 'logs')
        files = sorted(glob.glob(os.path.join(logs_dir, 'perfect_run_*.json')))
        if not files:
            print("  Aucun run parfait sauvegardé.")
            return

        # Sélection du fichier
        if len(files) == 1:
            chosen = files[0]
        else:
            print("\n  Runs parfaits disponibles :")
            for i, f in enumerate(files):
                try:
                    with open(f) as fh:
                        d = json.load(fh)
                    _w, _l = d.get('world'), d.get('level')
                    stage = f"{_w}-{_l}" if _w and _l else os.path.basename(f)
                    n = d.get('total', len(d.get('actions', [])))
                    print(f"    {i+1}. World {stage}  ({n} actions)  [{os.path.basename(f)}]")
                except Exception:
                    print(f"    {i+1}. {os.path.basename(f)}")
            try:
                idx = int(input(f"  Choisir (1-{len(files)}, défaut=1) : ").strip() or "1") - 1
                chosen = files[max(0, min(idx, len(files)-1))]
            except (ValueError, EOFError):
                chosen = files[0]

        with open(chosen) as f:
            data = json.load(f)
        actions = data['actions']

        # Construire la table des checkpoints à restaurer pendant le replay.
        # Nouveau format : rewind_checkpoints (liste triée par index).
        # Ancien format (compat) : rewind_index + start_ram.
        _raw_checkpoints = data.get('rewind_checkpoints')
        if _raw_checkpoints:
            checkpoints = {c['index']: (c['ram'], c['x']) for c in _raw_checkpoints}
        elif data.get('rewind_index') is not None and data.get('start_ram'):
            checkpoints = {data['rewind_index']: (data['start_ram'], data.get('start_x', '?'))}
        else:
            checkpoints = {}

        if checkpoints and not _raw_checkpoints:
            # Ancien format sans snapshot RAM valide déjà vérifié plus haut
            pass
        # Vérifier ancien format sans RAM
        if data.get('rewind_index') is not None and not data.get('start_ram') and not _raw_checkpoints:
            print(f"  Ce replay a été généré avec une version incomplète du code (pas de snapshot RAM).")
            print(f"    → Lancez une nouvelle partie et réessayez option 3.")
            return

        replay_world = data.get('world') or 1
        replay_level = data.get('level') or 1
        rewind_info = f" ({len(checkpoints)} rewind(s))" if checkpoints else ""
        print(f"▶  Replay parfait World {replay_world}-{replay_level} : {len(actions)} actions{rewind_info} (ESC pour arrêter)")

        env_id = f'SuperMarioBros-{replay_world}-{replay_level}-v3'
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        obs = env.reset()

        # Index du dernier checkpoint pour l'affichage
        _last_cp_index = max(checkpoints.keys()) if checkpoints else None

        cv2.namedWindow('Mario — Replay Parfait', cv2.WINDOW_AUTOSIZE)

        def _step_and_show(action, i):
            nonlocal obs
            obs, _, done, info = env.step(action)
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            display = cv2.resize(frame, (600, 480), interpolation=cv2.INTER_NEAREST)
            pct = int(100 * i / len(actions))
            cv2.rectangle(display, (0, 0), (int(6 * pct), 6), (0, 255, 100), -1)
            label = "PRE-REWIND" if (_last_cp_index and i < _last_cp_index) else "REPLAY PARFAIT"
            cv2.putText(display, f"{label}  {pct}%  x={info.get('x_pos', 0)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('Mario — Replay Parfait', display)
            key = cv2.waitKey(16) & 0xFF
            return done, info, key

        interrupted = False
        for i, action in enumerate(actions):
            # Restaurer la RAM NES à chaque checkpoint enregistré
            if i in checkpoints:
                ram_b64, cp_x = checkpoints[i]
                ram_bytes = base64.b64decode(ram_b64)
                ram_array = np.frombuffer(ram_bytes, dtype=np.uint8).copy()
                env.unwrapped.done = False
                np.copyto(env.unwrapped._ram_buffer(), ram_array)
                # 30 NOOPs identiques au jeu original (resynchroniser PPU + timing ennemis)
                for _ in range(30):
                    env.unwrapped.done = False
                    obs, _, _, _ = env.step(0)
                env.unwrapped.done = False
                print(f"⏪ RAM restaurée + 30 NOOPs → checkpoint x={cp_x} (index={i})")

            done, info, key = _step_and_show(action, i)
            if key == 27:  # ESC
                interrupted = True
                break
            if done:
                cv2.waitKey(1000)
                break

        cv2.destroyWindow('Mario — Replay Parfait')
        env.close()
        print(" Replay terminé.")

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
            print(f"  {len(files)} screenshot(s) supprimé(s).")

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
            print(f"  {deleted} log(s) d'anciennes sessions supprimés.")

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
            print(f"  {deleted} fichier(s) historic d'anciens runs supprimés.")
        if best_run_id:
            print(f" Meilleur run conservé: {best_run_id} ({best_x}px)")

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
            print(" Au revoir!")
            return

        # Configuration selon le mode
        if game_mode == "continue_direct":
            # Démarrer directement au dernier stage non terminé (sans replay du stage précédent).
            _, next_stage, _ = self._find_completed_and_next_stage()
            if next_stage is None:
                print("  Aucun stage suivant trouvé — démarrage en IA pure.")
                game_mode = "new_game"
            else:
                nw, nl = next_stage
                print(f" Démarrage direct World {nw}-{nl} (dernier stage non terminé)")
                self._transition_to_level(nw, nl, 0)
                game_mode = "memory_first"  # IA avec mémoire sur ce stage

        if game_mode == "continue_with_replay":
            # Rejouer automatiquement le dernier run parfait, puis IA sur le stage suivant.
            last_stage, next_stage, run_path = self._find_completed_and_next_stage()
            if last_stage is None or next_stage is None or run_path is None:
                print("  Impossible de continuer — aucun run parfait ou plus de stages.")
                game_mode = "new_game"
            else:
                import json as _json
                with open(run_path) as _f:
                    _rdata = _json.load(_f)
                self._raw_replay_actions = [int(a) for a in _rdata['actions']]
                self._raw_replay_index = 0
                lw, ll = last_stage
                nw, nl = next_stage
                print(f" Replay auto World {lw}-{ll} ({len(self._raw_replay_actions)} frames) → IA World {nw}-{nl}")
                # L'env est déjà sur 1-1 ; le replay va jouer toutes les actions de 1-1,
                # puis à la fin (flag_get) la transition normale vers le stage suivant s'active.
                game_mode = "memory_first"  # mode IA après la fin du replay

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
                print(f" Mode mémoire automatique — {seg_count} segments, safe jusqu'à x={safe_x}px")
                print(f" Phase 3 : replay mémoire → IA frontière")
            else:
                # Phase 1 : IA pure
                self._run_phase = 1
                self._phase3_ai_mode = False
                self._last_seg_key = None
                self._segment_in_replay = False
                print(f" Vie 1 → Phase 1: IA pure")

            print(f"🆕 Nouvelle partie - Run: {run_id}")
            self.logger.log_session_start(game_mode, run_id)

        print("\n MARIO FLUIDE avec CLAUDE LLM")
        print("Claude donne des macro-actions, Mario les exécute fluidement!")
        print("=" * 60)

        if self.replay_mode:
            mode_display = " REPLAY + IA"
        elif game_mode == "memory_first":
            mode_display = " MÉMOIRE AUTO → IA FRONTIÈRE"
        else:
            mode_display = " IA PURE"
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
        _seg_x = 0     # Dernière position connue de Mario (persiste entre itérations, y compris en PAUSE)

        cv2.namedWindow('Mario Fluide - Claude LLM', cv2.WINDOW_AUTOSIZE)

        self._exit_reason = "unknown"

        try:
            while max_steps is None or step_count < max_steps:
                if not paused:
                    # PRIORITÉ 0: Replay brut d'un run parfait sauvegardé (mode continue_with_replay)
                    # Rejoue les actions NES frame par frame, sans pause, sans IA.
                    # Quand le stock est épuisé, l'IA prend la main normalement.
                    if self._raw_replay_actions and self._raw_replay_index < len(self._raw_replay_actions):
                        current_action = self._raw_replay_actions[self._raw_replay_index]
                        self._raw_replay_index += 1
                        _pct = int(100 * self._raw_replay_index / len(self._raw_replay_actions))
                        if self._raw_replay_index % 300 == 0:
                            print(f" Replay parfait {_pct}% ({self._raw_replay_index}/{len(self._raw_replay_actions)} frames)")
                        if self._raw_replay_index >= len(self._raw_replay_actions):
                            print(" Replay parfait terminé — passage en mode IA")
                            self._raw_replay_actions = []

                    # PRIORITÉ 1: Mode replay
                    elif self.replay_mode:
                        current_action = self.get_replay_action(step_count)

                        # Si le replay est terminé, passer en mode IA
                        if current_action is None:
                            print(" Transition: Replay terminé, IA prend la main!")
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
                                    f"REPLAY: {prev_action_data.get('action_name', '')}"
                                )

                    # PRIORITÉ 2: Mode IA (quand replay brut fini, replay désactivé, ou pas d'action)
                    _in_raw_replay = bool(self._raw_replay_actions and
                                          self._raw_replay_index <= len(self._raw_replay_actions))
                    if (not _in_raw_replay and not self.replay_mode) or current_action is None:
                        # Obtenir l'action courante de l'IA
                        current_action = self.get_current_action()

                        if current_action is None:
                            # Pas de macro en cours, besoin de Claude
                            situation = self.analyze_situation(obs, {
                                'x_pos': _seg_x,
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
                                self.logger.log_queue_event("CALL  ", step_count,
                                    f"async queue-vide | x={_seg_x} macro={self.current_macro['name'] if self.current_macro else None}")
                                self._last_call_reason = "queue-vide"
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
                                        self.logger.log_queue_event("PAUSE ", step_count,
                                            f"début pause | reason={getattr(self, '_last_call_reason', '?')} | macro={self.current_macro['name'] if self.current_macro else None} thinking={self.claude_thinking}")
                            else:
                                if getattr(self, '_pause_printed', False):
                                    # Reprise après pause
                                    self.logger.log_queue_event("RESUME", step_count,
                                        f"fin pause | macro={self.current_macro['name'] if self.current_macro else None} q={len(self.action_queue)}")
                                self._pause_printed = False
                                # Utiliser l'action en queue
                                current_action = self.get_current_action()

                    # Toutes les décisions (ennemis, trous, obstacles) sont prises par Claude.
                    # Quand la queue est vide → PAUSE (env.step() non appelé, ennemis gelés)
                    # → Claude voit le screenshot et décide l'action appropriée.
                    _prev_frame_macro = self._prev_macro_name  # macro active au frame précédent
                    current_macro_name = self.current_macro['name'] if self.current_macro else None
                    self._prev_macro_name = current_macro_name
                    _JUMP_MACROS = {'stomp_enemy', 'pipe_jump', 'obstacle_jump', 'high_obstacle_jump',
                                    'pipe_vertical_jump', 'max_jump', 'run_jump_over', 'big_jump_right',
                                    'short_jump', 'long_jump', 'high_jump', 'precise_jump'}

                    # Anti-mur : si Mario presse droite SANS A (approche/course) et que x_pos
                    # n'a pas bougé depuis ≥3 frames → collé à un obstacle → recul actif.
                    # Actions sans A : 1 (right), 3 (right+B). Actions avec A : 2, 4 → saut réel,
                    # on ne touche pas (sinon on coupe le saut).
                    # Après détection : 3 frames de LEFT (~2px de recul) pour dégager le tuyau.
                    _RIGHT_NO_JUMP = {1, 3}   # droite sans bouton A
                    _JUMP_ACTIONS  = {2, 4}   # droite + A (phase saut active)
                    _wall_backing = getattr(self, '_wall_backing_frames', 0)
                    if _wall_backing > 0 and current_action not in _JUMP_ACTIONS:
                        # Recul actif en cours — appliquer LEFT sauf si saut en cours
                        current_action = 6  # left
                        self._wall_backing_frames = _wall_backing - 1
                        self._wall_stuck_frames = 0
                    elif current_action in _RIGHT_NO_JUMP and _seg_x > 0:
                        if _seg_x == getattr(self, '_wall_x_prev', -1):
                            self._wall_stuck_frames = getattr(self, '_wall_stuck_frames', 0) + 1
                            if self._wall_stuck_frames >= 3:
                                # Collé → déclencher 3 frames de recul
                                self._wall_backing_frames = 3
                                self._wall_stuck_frames = 0
                                current_action = 6  # left : première frame du recul
                        else:
                            self._wall_stuck_frames = 0
                            self._wall_backing_frames = 0
                    elif current_action not in _JUMP_ACTIONS:
                        self._wall_stuck_frames = 0
                    self._wall_x_prev = _seg_x

                    # NES A-button rising edge : après un LAND, on strip A pendant 2 frames
                    # pour que le saut suivant parte vraiment (relâcher puis presser).
                    if current_action is not None and getattr(self, '_a_release_frames', 0) > 0:
                        if current_action in {2, 4}:
                            current_action = 3 if current_action == 4 else 1
                        self._a_release_frames -= 1

                    # Exécuter l'action dans le jeu
                    done = False  # valeur par défaut quand env.step() n'est pas appelé (pause)
                    if current_action is not None:
                        obs, reward, done, real_info = self.env.step(current_action)
                        self._raw_action_history.append(int(current_action))
                        self._final_action_history.append(int(current_action))
                        total_reward += reward
                        step_count += 1
                        self._current_step = step_count
                        _life_step += 1

                        #  MÉMOIRE SEGMENTS : position + événements de jeu
                        _seg_x = real_info.get('x_pos', 0)
                        if _seg_x > _run_max_x:
                            _run_max_x = _seg_x
                        self.segment_memory.record_position(_seg_x, step_count)

                        #  TRACKING SAUTS ÉCHOUÉS
                        # Détecter quand un saut OBSTACLE vient de se terminer.
                        # On exclut run_jump_over (ennemi, pas obstacle) : il n'échoue pas
                        # face à un tuyau, il passe juste par-dessus un ennemi.
                        # La condition ne requiert plus que la macro suivante soit non-saut :
                        # URGENCE SOL peut enchaîner immédiatement run_jump_over après pipe_jump
                        # (run_jump_over est dans _jump_track_set), ce qui faisait rater le JUMP_FAIL.
                        _obstacle_track_set = {'pipe_jump', 'obstacle_jump', 'high_obstacle_jump',
                                               'pipe_vertical_jump', 'max_jump', 'big_jump_right', 'long_jump'}
                        if (_prev_frame_macro in _obstacle_track_set and
                                current_macro_name != _prev_frame_macro):
                            _jump_x_delta = _seg_x - self._pre_jump_x
                            _jbucket = int(self._pre_jump_x // 50) * 50  # aligné avec _blocked_macros_by_pos
                            if _jump_x_delta < 40:
                                # Saut échoué : obstacle non franchi
                                if _jbucket not in self._failed_jump_attempts:
                                    self._failed_jump_attempts[_jbucket] = []
                                self._failed_jump_attempts[_jbucket].append(_prev_frame_macro)
                                _n_fails = len(self._failed_jump_attempts[_jbucket])
                                self.logger.log_queue_event("JUMP_FAIL", step_count,
                                    f"macro={_prev_frame_macro} x_start={self._pre_jump_x:.0f} delta={_jump_x_delta:.0f}px | "
                                    f"tentatives={_n_fails} bucket={_jbucket}")
                                if _n_fails >= 2:
                                    # Annuler l'appel async en cours (sa réponse sera STALE)
                                    self._claude_generation += 1
                                    self.claude_thinking = False
                                    self.call_claude_obstacle_retry(
                                        self._failed_jump_attempts[_jbucket],
                                        self._pre_jump_x, _seg_x, obs, step_count)
                                    self._failed_jump_attempts.pop(_jbucket, None)
                            elif _seg_x > _jbucket + 100 and _prev_frame_macro != 'run_jump_over':
                                # Saut vraiment réussi : Mario est bien au-delà du bucket de départ.
                                # Seuil 100px ET on exclut run_jump_over (ennemi, pas obstacle) :
                                # un saut ennemi peut déplacer Mario de 60-80px sans franchir le tuyau,
                                # ce qui effacerait les échecs d'obstacle et empêcherait l'obstacle_retry.
                                self._failed_jump_attempts.pop(_jbucket, None)

                        #  Checkpoint rewind (toutes les 60 frames)
                        # Ne pas sauvegarder si done=True (frame de mort) — checkpoint invalide
                        if step_count % 60 == 0 and not self._rewind_active and not done:
                            # Sauvegarde état complet émulateur (CPU + PPU VRAM) — 1 seul slot.
                            # Écrase le backup précédent ; seul le checkpoint le plus récent
                            # bénéficie du restore complet (pas de décor fantôme).
                            self.env.unwrapped._backup()
                            # Le slot est maintenant le checkpoint 60f → le backup pré-saut
                            # n'est plus valide (même slot écrasé).
                            self._pre_jump_has_full_backup = False
                            _ram_snap = self.env.unwrapped._ram_buffer().copy()
                            _recent_macros = [m['name'] for m in list(self.macro_history)[-8:]]
                            _cp_y = real_info.get('y_pos', 200)
                            # Marquer les anciens checkpoints : backup slot écrasé
                            for _old in self.rewind_buffer:
                                _old['has_full_backup'] = False
                            self.rewind_buffer.append({
                                'step': step_count,
                                'ram': _ram_snap,
                                'x_pos': int(_seg_x),
                                'y_pos': int(_cp_y),
                                'macros': _recent_macros,
                                'action_history': list(self._raw_action_history),
                                'perfect_history_len': len(self._final_action_history),
                                'has_full_backup': True,
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

                        #  ANTI-BLOCAGE REPLAY (Phase 2 ET Phase 3 zone safe)
                        # detect_stuck normal exige 4/6 même action → inefficace sur réflexes alternés
                        # Ce check position-seule couvre Phase 2 ET Phase 3 pendant les segments replay
                        if self._segment_in_replay:
                            if int(_seg_x) > self._phase3_last_x + 10:
                                # Progression : mise à jour des références
                                self._phase3_last_x = int(_seg_x)
                                self._phase3_last_x_step = step_count
                            elif step_count - self._phase3_last_x_step >= 40:
                                # Bloqué 40 frames sans avancer de 10px → sortir du replay
                                print(f" Anti-blocage replay (Phase {self._run_phase}): "
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

                        #  DÉTECTION DE DÉBLOCAGE RÉUSSI : Mario a franchil'obstacle
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
                                print(f" Déblocage mémorisé x={self._unstick_start_x}: "
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

                    #  DÉTECTION DE BLOCAGE : position + répétition d'actions
                    if self.level_context_established:
                        current_x = real_info.get('x_pos', 0) if 'real_info' in locals() else 0
                        stuck_level = self.detect_stuck(current_x, step_count)

                        if stuck_level >= 2:
                            # Niveau 2 : recherche web + appel Claude déblocage
                            # Passer TOUTES les macros récentes comme échouées (y compris sauts)
                            # + macros mémorisées comme bloquées pour cette zone
                            _bucket_l2 = int(current_x // 50) * 50
                            _pos_blocked = self._blocked_macros_by_pos.get(_bucket_l2, set())
                            _recent_failed = [m['name'] for m in list(self.macro_history)[-8:]]
                            _all_failed = list(dict.fromkeys(_recent_failed + list(_pos_blocked)))  # dédupliqué
                            web = self.search_mario_strategy(current_x, _all_failed)
                            sit = self.last_situation or {}
                            self.call_claude_stuck_mode(sit, current_x, _all_failed, obs, step_count, web)
                            self.stuck_counter = 0  # Réinitialiser après intervention
                            # Cooldown : ne pas re-déclencher avant 150 steps
                            self.last_stuck_check_step = step_count + 150
                            # Mémoriser la tentative (séquence déterminée après succès)
                            self._unstick_start_x = current_x
                            self._unstick_sequence = None  # Sera capturée depuis macro_history
                            self._unstick_step = step_count
                        elif stuck_level == 1:
                            # Niveau 1 supprimé : on laisse le counter monter naturellement.
                            # Au prochain check (stuck_check_frequency steps), stuck_level >= 2
                            # et Claude sera appelé avec le contexte complet.
                            print(f" Blocage niveau 1 à x={current_x:.0f} — attente niveau 2 pour appel Claude")

                    # Si la queue de replay est épuisée, sortir du mode replay
                    if self._segment_in_replay and len(self.action_queue) == 0:
                        self._segment_in_replay = False

                    #  DÉCLENCHEMENT HYBRIDE OPTIMISÉ
                    # Bloqué pendant le replay d'un segment mémorisé ET toute la zone safe du Phase 3
                    _not_replay_zone = (
                        not self._segment_in_replay and
                        not (self._run_phase == 3 and not self._phase3_ai_mode)
                    )
                    # _currently_jumping : True uniquement pendant le saut ACTIF (A button)
                    # Les phases d'approche (stomp/pipe_jump phase 0 = marche) ne comptent pas :
                    # on doit toujours détecter trous et ennemis pendant une approche.
                    _in_approach_phase = (
                        current_macro_name in ('stomp_enemy', 'pipe_jump', 'obstacle_jump',
                                               'max_jump', 'run_jump_over') and
                        self.current_macro is not None and
                        self.current_macro.get('current_phase', 0) == 0 and
                        len(self.current_macro.get('phases', [])) > 1  # multi-phase = a une approche
                    )
                    # Vérifie si Mario est physiquement en l'air (y_pos < 180).
                    # Évite les faux positifs : rebond sur ennemi, LAND prématuré, etc.
                    # Fallback True si real_info pas encore disponible (début de session).
                    _mario_y_physics = real_info.get('y_pos', 0) if 'real_info' in locals() else 0
                    _actually_airborne = (_mario_y_physics < 180) if _mario_y_physics > 0 else True
                    _currently_jumping = (current_macro_name in _JUMP_MACROS and
                                          not _in_approach_phase and
                                          _actually_airborne)

                    # Détecter atterrissage physique : macro de saut encore active mais Mario au sol.
                    # Double détection : y_pos >= 185 (RAM) ET pixels sous les pieds (OAM+image).
                    # La détection pixel est plus réactive : elle tire dès que Mario touche le sol
                    # sans attendre que y_pos atteigne un seuil fixe (qui peut être raté en 8f).
                    if (_currently_jumping and
                            self.current_macro is not None and
                            'real_info' in locals()):
                        _mario_y_now = real_info.get('y_pos', 0)
                        _fl_remaining = self.current_macro.get('frames_left', 999)
                        # Vérification LAND uniquement après ≥15 frames écoulées depuis le début
                        # de la phase actuelle. Cela s'applique aux deux méthodes (y_pos ET pixels).
                        # Sans garde : _grounded_y >= 185 tirait à fl=39 (elapsed=1) car Mario démarre
                        # à y=194-199 au sol → LAND immédiat → saut annulé → JUMP_FAIL.
                        # 15f laisse Mario s'élever assez pour quitter la zone sol.
                        # Fallback 40f pour macros multi-phases (jump phase toujours ~40f).
                        _initial_fl = self.current_macro.get('_initial_frames', 40)
                        _elapsed_fl = _initial_fl - _fl_remaining
                        _enough_airtime = _elapsed_fl >= 15
                        _grounded_y = _enough_airtime and _mario_y_now >= 185
                        # Delta y_pos : si y ne change plus depuis 1 frame → Mario posé sur surface.
                        # Plus fiable que la détection pixel (décors, tuiles de fond, etc.).
                        _prev_y_land = getattr(self, '_prev_y_for_land', -1)
                        _grounded_delta = (_enough_airtime and
                                           _mario_y_now > 0 and
                                           _mario_y_now == _prev_y_land)
                        self._prev_y_for_land = _mario_y_now
                        if _grounded_y or _grounded_delta:
                            self.current_macro['frames_left'] = 0
                            _currently_jumping = False
                            _method = 'y_pos' if _grounded_y else ('delta_y' if _grounded_delta else 'pixels')
                            self.logger.log_queue_event("LAND  ", step_count,
                                                        f"Atterrissage [{_method}] y={_mario_y_now} fl={_fl_remaining} | macro={current_macro_name} → fin macro")
                            # NES : le bouton A nécessite un front montant (relâcher puis presser).
                            # Si un saut suivant est déjà en queue, ses premières frames presseront A
                            # alors qu'il était encore pressé → Mario ne décolle pas.
                            # On force 2 frames sans A pour garantir le front montant.
                            self._a_release_frames = 1

                    # MID-AIR : consultation synchrone Claude pour décider de l'atterrissage
                    # Déclenché quand frames_left <= 30 (mi-saut, assez tôt pour réagir).
                    # Un second déclenchement d'urgence est possible si un ennemi OAM est
                    # détecté à < 40px pendant la descente (fl < 15), même si déjà appelé.
                    # Le jeu est gelé (current_action = None) pendant l'appel Claude.
                    _mid_air_y_now = real_info.get('y_pos', 255) if 'real_info' in locals() else 255
                    _mid_air_jump_height = self._pre_jump_y - _mid_air_y_now  # positif si Mario a monté

                    if _currently_jumping and self.current_macro is not None:
                        _fl_now = self.current_macro.get('frames_left', 0)
                        # Second déclenchement d'urgence : ennemi très proche pendant la descente.
                        # Une seule fois par saut (_mid_air_emergency_called) pour éviter la boucle infinie.
                        if (self._mid_air_called and not self._mid_air_emergency_called and
                                current_macro_name != 'pipe_vertical_jump' and
                                _fl_now < 15 and _fl_now > 3 and
                                _not_replay_zone and obs is not None):
                            _oam_em = self.get_enemies_from_oam()
                            _close_em = [e for e in _oam_em if 0 < e['distance_px'] < 40]
                            if _close_em:
                                self._mid_air_called = False
                                self._mid_air_emergency_called = True  # Bloque tout second emergency
                                self.logger.log_queue_event("MID-AIR", step_count,
                                                            f"EMERGENCY reset fl={_fl_now} ennemi à {_close_em[0]['distance_px']}px")
                        if _fl_now <= 30 and not self._mid_air_called:
                            if not _not_replay_zone:
                                self.logger.log_queue_event("MID-AIR", step_count, f"SKIP replay_zone macro={current_macro_name} fl={_fl_now}")
                            elif obs is None:
                                self.logger.log_queue_event("MID-AIR", step_count, f"SKIP obs=None macro={current_macro_name} fl={_fl_now}")
                            elif _mid_air_excluded:
                                self.logger.log_queue_event("MID-AIR", step_count, f"SKIP pipe macro={current_macro_name} pipe_in_snap={_pipe_in_snap} fl={_fl_now}")
                            elif _mid_air_jump_height < 50:
                                self.logger.log_queue_event("MID-AIR", step_count, f"SKIP height={_mid_air_jump_height:.0f}px<50 macro={current_macro_name} fl={_fl_now}")
                        elif _fl_now > 30 and not self._mid_air_called:
                            self.logger.log_queue_event("MID-AIR", step_count, f"en attente macro={current_macro_name} fl={_fl_now}")
                    _pipe_in_snap = ('_snap' in locals() and _snap.get('pipe_dist') is not None)
                    _mid_air_excluded = (current_macro_name in ('pipe_vertical_jump', 'pipe_jump') or
                                         (current_macro_name == 'max_jump' and _pipe_in_snap))
                    if (_currently_jumping and
                            not self._mid_air_called and
                            self.current_macro is not None and
                            not _mid_air_excluded and
                            _mid_air_jump_height >= 50 and
                            self.current_macro.get('frames_left', 0) <= 30 and
                            _not_replay_zone and obs is not None):
                        self._mid_air_called = True
                        _fl = self.current_macro['frames_left']
                        _mx = real_info.get('x_pos', _seg_x) if 'real_info' in locals() else _seg_x

                        # Vérifier s'il y a une menace réelle à l'atterrissage.
                        # Si aucun ennemi devant (OAM) et aucun trou détecté (snap),
                        # la réponse optimale est toujours 'far' → pas besoin de pausent Claude.
                        _snap_now = _snap if '_snap' in locals() else {}
                        _oam_check = self.get_enemies_from_oam()
                        _mid_air_enemy = any(0 < e['distance_px'] < 220 for e in _oam_check)
                        _mid_air_hole  = _snap_now.get('hole_dist') is not None
                        _mid_air_has_threat = _mid_air_enemy or _mid_air_hole

                        if not _mid_air_has_threat:
                            self.logger.log_queue_event("MID-AIR", step_count,
                                f"SKIP no-threat → far | macro={current_macro_name} fl={_fl}")
                            _landing = 'far'
                        else:
                            current_action = None  # Gèle le jeu pendant l'appel
                            self.current_macro['frames_left'] += 1  # Compenser le décrément déjà fait
                            self.logger.log_queue_event("MID-AIR", step_count, f"CALL macro={current_macro_name} fl={_fl} x={_mx} enemy={_mid_air_enemy} hole={_mid_air_hole}")
                            _landing = self.call_claude_landing_sync(obs, _mx, _fl, step_count)
                        self.logger.log_queue_event("MID-AIR", step_count, f"LANDING={_landing}")
                        if _landing == 'short':
                            self.current_macro['base_action'] = 3  # right+B, relâche A → chute
                            self.current_macro['frames_left'] = min(_fl, 8)
                            current_action = 3
                        elif _landing == 'left':
                            self.current_macro['base_action'] = 6  # left
                            current_action = 6
                        elif _landing == 'stop':
                            self.current_macro['base_action'] = 0  # NOOP, chute verticale
                            current_action = 0
                        else:  # 'far' → continuer right+A+B
                            current_action = 4
                    elif not _currently_jumping:
                        self._mid_air_called = False           # Reset pour le prochain saut
                        self._mid_air_emergency_called = False

                    # Déclenchement normal : queue presque vide (bloqué pendant saut/stomp)
                    _queue_has_jump_pending = any(
                        a.get('macro_name') in _JUMP_MACROS for a in self.action_queue)
                    _queue_trigger = (len(self.action_queue) <= 1 and
                                      not _currently_jumping and
                                      not _queue_has_jump_pending and
                                      step_count - self.last_positions_update >= self.positions_update_frequency)
                    # Déclenchement périodique : toutes les 60 frames (bloqué pendant saut/stomp)
                    # → Claude voit les nouveaux décors et ennemis apparus depuis le dernier appel
                    _periodic_trigger = (len(self.action_queue) < 2 and
                                         not _currently_jumping and
                                         not _queue_has_jump_pending and
                                         step_count - self.last_screenshot_step >= 60)
                    # Déclenchement position : Mario a avancé >60px depuis dernier screenshot
                    # → vider la queue + annuler call Claude en cours → pause/rescan immédiat
                    # Fonctionne même pendant un saut (current_macro continue, mais queue vidée)
                    _cur_x_for_trigger = real_info.get('x_pos', self.last_screenshot_x) if 'real_info' in locals() else self.last_screenshot_x
                    _advanced_px = _cur_x_for_trigger - self.last_screenshot_x
                    _position_trigger = (
                        _advanced_px >= 60 and
                        not _currently_jumping and  # Ne pas interrompre un saut/stomp
                        _not_replay_zone
                    )
                    if _position_trigger:
                        self.last_screenshot_x = _cur_x_for_trigger
                        self._scene_active = False  # Mario a avancé → scène franchie, détection réactivée
                        if _queue_has_jump_pending:
                            print(f" POSITION TRIGGER: Mario +{_advanced_px}px → saut en queue, conservé (x={_cur_x_for_trigger})")
                        else:
                            self.action_queue.clear()
                            self.logger.log_queue_event("CLEAR", step_count, f"position_trigger +{_advanced_px}px x={_cur_x_for_trigger}")
                            print(f" POSITION TRIGGER: Mario +{_advanced_px}px → queue vidée, Claude continue (x={_cur_x_for_trigger})")

                    # ═══════════════════════════════════════════════════════════
                    #  DÉTECTEUR DE SCÈNE UNIFIÉ
                    #  Règle fondamentale : NE JAMAIS interrompre un saut.
                    #  Au sol : déclenche une seule fois par obstacle (_scene_active).
                    #  _scene_active reste True jusqu'à ce qu'un autre trigger confirme
                    #  que Mario a progressé (queue vide, +60px, ou 60 frames).
                    # ═══════════════════════════════════════════════════════════
                    _scene_trigger = False
                    _scene_sync_needed = False
                    _oam_trigger = False
                    _hole_trigger = False
                    _pipe_trigger = False

                    # Log des déclencheurs actifs ce step
                    if _queue_trigger or _periodic_trigger or _position_trigger:
                        _trig_names = []
                        if _queue_trigger: _trig_names.append("queue")
                        if _periodic_trigger: _trig_names.append("periodic")
                        if _position_trigger: _trig_names.append(f"position+{_advanced_px:.0f}px")
                        self.logger.log_queue_event("TRIG  ", step_count,
                            f"{'+'.join(_trig_names)} | macro={current_macro_name} q={len(self.action_queue)} thinking={self.claude_thinking}")

                    # Reset _scene_active dès qu'un trigger de progression se déclenche
                    if _queue_trigger or _periodic_trigger:
                        self._scene_active = False

                    if (_not_replay_zone and obs is not None):
                        _snap = self.take_scene_snapshot(obs)
                        _changed, _change_reason = self.check_scene_thresholds(_snap)

                        # Log scan résultat (ennemi/trou seulement si présent ou vient d'apparaître/disparaître)
                        _ed_now = _snap['enemy_dist']
                        _hd_now = _snap['hole_dist']
                        _scan_prev_e = getattr(self, '_log_last_enemy_dist', None)
                        _scan_prev_h = getattr(self, '_log_last_hole_dist', None)
                        if (_ed_now != _scan_prev_e or _hd_now != _scan_prev_h or
                                (_ed_now is not None and step_count % 10 == 0)):
                            self.logger.log_queue_event("SCAN  ", step_count,
                                f"ennemi={_ed_now}px trou={_hd_now}px | jump={_currently_jumping} scene_active={self._scene_active} thresholds_hit={self._enemy_thresholds_hit} changed={_changed}")
                            self._log_last_enemy_dist = _ed_now
                            self._log_last_hole_dist = _hd_now

                        # Reset du flag nouvel ennemi (mis par check_scene_thresholds)
                        _new_enemy_mid_jump = getattr(self, '_new_enemy_appeared', False)
                        if _new_enemy_mid_jump:
                            self._new_enemy_appeared = False

                        if _changed:
                            _queue_has_jump = any(
                                a.get('macro_name') in _JUMP_MACROS for a in self.action_queue)
                            _jump_protected = _currently_jumping or _queue_has_jump

                            # Un trou détecté force toujours un nouveau sync, même si _scene_active.
                            # Tomber dans un trou est fatal — priorité absolue sur tout autre état.
                            _is_hole_trigger = 'trou' in _change_reason
                            if _is_hole_trigger and self._scene_active:
                                self._scene_active = False

                            if _jump_protected:
                                # Saut en cours ou planifié.
                                # Si la queue contient déjà une action d'évitement (run_jump_over,
                                # max_jump…), ne pas relancer Claude — l'action sera exécutée
                                # immédiatement à la fin du saut, sans délai supplémentaire.
                                _avoidance_macros = {'run_jump_over', 'max_jump', 'big_jump_right',
                                                     'stomp_enemy', 'obstacle_jump'}
                                _queue_has_avoidance = any(
                                    a.get('macro_name') in _avoidance_macros
                                    for a in self.action_queue)
                                if _queue_has_avoidance and _new_enemy_mid_jump:
                                    # Nouvel ennemi apparu pendant le saut — l'action en queue
                                    # était planifiée pour l'ennemi précédent (déjà passé).
                                    # Transition SEAMLESS : prolonger le saut avec max_jump (même base_action=4)
                                    # Mario continue de sauter sans interruption.
                                    self.action_queue.clear()
                                    self.action_queue.append({
                                        'macro_name': 'run_jump_over',
                                        'reasoning': 'Nouvel ennemi mid-jump — transition seamless',
                                        'urgency': 9
                                    })
                                    # Terminer le macro courant dès le prochain frame
                                    if self.current_macro:
                                        self.current_macro['frames_left'] = 0
                                    self.logger.log_queue_event("SCAN  ", step_count,
                                        f"NOUVEL ENNEMI mid-jump [{_change_reason}] → transition seamless max_jump")
                                    if not self.claude_thinking:
                                        _scene_trigger = True
                                    # Si claude_thinking, le flag seuil reste (déjà ajouté) → re-trigger quand libre
                                elif _queue_has_avoidance:
                                    # Saut planifié dans la queue, mais Mario peut être au sol.
                                    # Exception urgence sol : si Mario n'est PAS en train de sauter
                                    # et que la menace est très proche (ennemi < 60px ou trou < 30px),
                                    # remplacer le saut planifié par la bonne action immédiatement.
                                    _dist_match = re.search(r'dist=(\d+)px', _change_reason)
                                    _threat_dist = int(_dist_match.group(1)) if _dist_match else 999
                                    _ground_urgent = (
                                        not _currently_jumping and
                                        (('ennemi' in _change_reason and _threat_dist < 60) or
                                         (_is_hole_trigger and _threat_dist < 30) or
                                         ('tuyau' in _change_reason and _threat_dist < 80))
                                    )
                                    if _ground_urgent:
                                        # Menace urgente au sol → appel Claude sync (pas d'action automatique)
                                        self.action_queue.clear()
                                        if self.current_macro:
                                            _macro_is_jump = self.current_macro.get('name') in _JUMP_MACROS
                                            if not _macro_is_jump or not _currently_jumping:
                                                # Approche multi-phases (ex: pipe_jump phase 1, jump=False) :
                                                # current_macro=None force le POP immédiat sans avancer à la phase saut.
                                                self.current_macro = None
                                        self._claude_generation += 1
                                        self.claude_thinking = False
                                        _scene_trigger = True
                                        _scene_sync_needed = True
                                        self.logger.log_queue_event("SCAN  ", step_count,
                                            f"URGENCE SOL [{_change_reason}] dist={_threat_dist}px → Claude sync")
                                    else:
                                        # Seuil consommé mais aucune action prise → le restituer,
                                        # sauf si déjà restitué dans les 10 derniers steps (anti-spam).
                                        _rkey = ('avoidance', _change_reason[:40])
                                        _rlast = self._threshold_restitution_steps.get(_rkey, -99)
                                        if step_count - _rlast >= 10:
                                            self._threshold_restitution_steps[_rkey] = step_count
                                            if 'ennemi' in _change_reason:
                                                for t in self._get_level_thresholds()['enemy']:
                                                    if f"seuil {t}px" in _change_reason:
                                                        self._enemy_thresholds_hit.discard(t)
                                                        break
                                            elif 'tuyau' in _change_reason:
                                                for t in self._get_level_thresholds()['pipe']:
                                                    if f"seuil {t}px" in _change_reason:
                                                        self._pipe_thresholds_hit.discard(t)
                                                        break
                                            self.logger.log_queue_event("SCAN  ", step_count,
                                                f"BLOCKED avoidance [{_change_reason}] → seuil restitué")
                                elif not self.claude_thinking:
                                    _scene_trigger = True
                                    print(f" SCÈNE [{_change_reason}] → saut protégé, Claude prépare suite")
                                else:
                                    # Claude occupé — vérifier si la menace est urgente.
                                    # Si l'ennemi/trou est très proche, annuler l'appel en cours
                                    # et relancer le trigger normal (PAUSE + appel Claude urgent).
                                    _bt_match = re.search(r'dist=(\d+)px', _change_reason)
                                    _bt_dist = int(_bt_match.group(1)) if _bt_match else 999
                                    # Ne pas interrompre Claude si Mario est DÉJÀ en train de sauter
                                    # (run_jump_over, max_jump…) : le macro en cours gère la menace.
                                    # Exception : trou détecté en saut → peut nécessiter correction.
                                    _bt_urgent = (
                                        (not _currently_jumping and 'ennemi' in _change_reason and _bt_dist < 50) or
                                        (_is_hole_trigger and _bt_dist < 30) or
                                        (not _currently_jumping and 'tuyau' in _change_reason and _bt_dist < 80)
                                    )
                                    if _bt_urgent:
                                        # Interrompre Claude, relancer avec contexte d'urgence
                                        self._claude_generation += 1
                                        self.claude_thinking = False
                                        _scene_trigger = True
                                        _scene_sync_needed = True
                                        if self.current_macro:
                                            _macro_is_jump2 = self.current_macro.get('name') in _JUMP_MACROS
                                            if not _macro_is_jump2 or not _currently_jumping:
                                                self.current_macro = None
                                        self.action_queue.clear()
                                        self.logger.log_queue_event("SCAN  ", step_count,
                                            f"URGENCE thinking [{_change_reason}] dist={_bt_dist}px → Claude interrompu, relance")
                                    else:
                                        # Menace lointaine : restituer le seuil pour re-trigger quand libre,
                                        # sauf si déjà restitué dans les 10 derniers steps (anti-spam).
                                        _rkey2 = ('thinking', _change_reason[:40])
                                        _rlast2 = self._threshold_restitution_steps.get(_rkey2, -99)
                                        if step_count - _rlast2 >= 10:
                                            self._threshold_restitution_steps[_rkey2] = step_count
                                            if 'ennemi' in _change_reason:
                                                for t in self._get_level_thresholds()['enemy']:
                                                    if f"seuil {t}px" in _change_reason:
                                                        self._enemy_thresholds_hit.discard(t)
                                                        break
                                            elif 'tuyau' in _change_reason:
                                                for t in self._get_level_thresholds()['pipe']:
                                                    if f"seuil {t}px" in _change_reason:
                                                        self._pipe_thresholds_hit.discard(t)
                                                        break
                                            self.logger.log_queue_event("SCAN  ", step_count,
                                                f"BLOCKED thinking [{_change_reason}] → seuil restitué")
                            elif not self._scene_active:
                                self._scene_active = True
                                self.last_oam_trigger_step = step_count

                                if self.claude_thinking:
                                    # Appel async en vol. Vérifier si la menace est urgente.
                                    # Si oui, annuler cet appel et relancer avec contexte d'urgence.
                                    _sc_match = re.search(r'dist=(\d+)px', _change_reason)
                                    _sc_dist = int(_sc_match.group(1)) if _sc_match else 999
                                    _sc_urgent = (
                                        ('ennemi' in _change_reason and _sc_dist < 50) or
                                        (_is_hole_trigger and _sc_dist < 30) or
                                        ('tuyau' in _change_reason and _sc_dist < 80)
                                    )
                                    if _sc_urgent:
                                        # Annuler l'appel en cours, relancer avec prompt urgent
                                        self._claude_generation += 1
                                        self.claude_thinking = False
                                        _scene_trigger = True
                                        _scene_sync_needed = True
                                        _pure_jump_sc = {'run_jump_over', 'max_jump', 'big_jump_right',
                                                         'short_jump', 'long_jump', 'high_jump', 'precise_jump'}
                                        _macro_name_sc = self.current_macro.get('name') if self.current_macro else None
                                        if self.current_macro and _macro_name_sc not in _pure_jump_sc:
                                            self.current_macro['frames_left'] = 0
                                        self.action_queue.clear()
                                        self.logger.log_queue_event("SCENE ", step_count,
                                            f"URGENCE async [{_change_reason}] dist={_sc_dist}px → Claude interrompu, relance sync")
                                        print(f" URGENCE [{_change_reason}] → async annulé, appel synchrone urgent")
                                    else:
                                        # Menace lointaine : attendre la réponse en vol.
                                        self.logger.log_queue_event("SCENE ", step_count,
                                            f"async en vol [{_change_reason}] → attente réponse saut")
                                        print(f" SCÈNE [{_change_reason}] → async en cours, attente réponse saut")
                                else:
                                    # Au sol, pas d'appel en cours : pause + appel synchrone.
                                    _scene_trigger = True
                                    _scene_sync_needed = True

                                    # Tronquer le macro courant sauf si c'est un saut pur
                                    _pure_jump = {'run_jump_over', 'max_jump', 'big_jump_right',
                                                  'short_jump', 'long_jump', 'high_jump', 'precise_jump'}
                                    _macro_name_now = self.current_macro.get('name') if self.current_macro else None
                                    _in_pure_jump = (_macro_name_now in _pure_jump)
                                    if self.current_macro and not _in_pure_jump:
                                        self.current_macro['frames_left'] = 0
                                    self.action_queue.clear()
                                    self.logger.log_queue_event("CLEAR", step_count, f"scene_changed [{_change_reason}]")
                                    print(f" SCÈNE CHANGÉE [{_change_reason}] → PAUSE + appel synchrone")
                            else:
                                # _scene_active = True : obstacle déjà signalé, Mario en train de réagir
                                pass

                        # Enrichir la situation avec les données fraîches du snapshot
                        # (s'applique pour tous les cas : pause urgence, changement scène)
                        if _scene_trigger and '_snap' in locals():
                            if not isinstance(situation, dict):
                                situation = {}
                            if _snap['enemy_front']:
                                situation['oam_enemies'] = _snap['enemy_front']
                                _oam_trigger = True
                            if _snap['pipe_dist'] is not None:
                                situation['pipe'] = _snap['pipe']
                                _pipe_trigger = True
                            if _snap['hole_dist'] is not None:
                                situation['holes'] = _snap['hole']
                                _hole_trigger = True

                    #  RÉFLEXE DANGER ZONE : si Mario approche d'une position de mort connue
                    _cur_x_dz = real_info.get('x_pos', 0) if 'real_info' in locals() else 0
                    if (self._danger_zone_x is not None and
                            not _currently_jumping and
                            not _scene_trigger and
                            _not_replay_zone):
                        _dist_to_danger = self._danger_zone_x - _cur_x_dz
                        if 30 <= _dist_to_danger <= 150:
                            self.action_queue.clear()
                            self.action_queue.append({
                                'macro_name': 'max_jump',
                                'strategy': 'Reflex danger zone',
                                'urgency': 10,
                                'confidence': 95
                            })
                            if self.current_macro and self.current_macro.get('name') not in _JUMP_MACROS:
                                self.current_macro['frames_left'] = 0
                            print(f"DANGER ZONE REFLEX: trou connu a x={self._danger_zone_x}, "
                                  f"Mario a x={_cur_x_dz} ({_dist_to_danger}px) -> max_jump")
                        elif _cur_x_dz > self._danger_zone_x + 50:
                            self._danger_zone_x = None

                    _ppu_warming_up = step_count < self._ppu_warmup_until

                    should_trigger_claude = (
                        not self.claude_thinking and
                        not _ppu_warming_up and
                        (_queue_trigger or _periodic_trigger or _position_trigger
                         or _scene_trigger) and
                        _not_replay_zone
                    )

                    if _ppu_warming_up and _scene_trigger:
                        print(f"⏪ PPU warmup actif (jusqu'au step {self._ppu_warmup_until}) — appel Claude reporté")

                    if should_trigger_claude:
                        # Lever le blocage post-rewind : Claude prend la main
                        if getattr(self, '_post_rewind_block_inject', False):
                            self._post_rewind_block_inject = False
                            print(f"⏪ Post-rewind : inject_known_solution débloqué après appel Claude")
                        _why = (_change_reason if _scene_trigger and '_change_reason' in locals() else
                                "queue basse" if _queue_trigger else
                                "position +60px" if _position_trigger else "périodique 60f")
                        self._last_call_reason = _why
                        trigger_type = " Initial" if not self.level_context_established else " Scan"
                        print(f" Déclenchement {trigger_type} [{_why}] - queue:{len(self.action_queue)}, step:{step_count}")

                        # Après un rewind, utiliser real_info capturé pendant le replay
                        # (évite d'envoyer la position de mort à Claude au lieu du checkpoint)
                        _info_for_situation = (
                            self._rewind_real_info or
                            (real_info if 'real_info' in locals() else {
                                'x_pos': _seg_x, 'y_pos': 200, 'score': total_reward
                            })
                        )
                        self._rewind_real_info = None  # consommé
                        # Sauvegarder les détections avant analyze_situation qui écrase tout
                        _pipe_before = situation.get('pipe') if isinstance(situation, dict) else None
                        _pipe_ahead_before = situation.get('pipe_ahead') if isinstance(situation, dict) else None
                        _holes_before = situation.get('holes') if isinstance(situation, dict) else None
                        situation = self.analyze_situation(obs, _info_for_situation, step_count)
                        # Réinjecter les détections (analyze_situation les efface car elle reconstruit tout)
                        if _pipe_before:
                            situation['pipe'] = _pipe_before
                        if _pipe_ahead_before:
                            situation['pipe_ahead'] = _pipe_ahead_before
                        if _holes_before:
                            situation['holes'] = _holes_before
                        # Mettre à jour la position de référence pour le prochain trigger position
                        self.last_screenshot_x = situation.get('mario', {}).get('x', self.last_screenshot_x)
                        # Enregistrer les stratégies qui marchent bien
                        if len(self.action_history) >= 3:
                            recent_actions = list(self.action_history)[-3:]
                            progress_made = situation.get('progress', {}).get('trend', 0)
                            if progress_made > 20:  # Bon progrès
                                action_names = [a['action'] for a in recent_actions]
                                self.record_successful_strategy(action_names, progress_made)

                        if _scene_sync_needed:
                            # Cas urgent (ennemi/trou/obstacle au sol) : appel synchrone,
                            # le jeu reste gelé (queue vide → current_action = None au
                            # prochain get_current_action) pendant toute la durée de l'appel.
                            self.logger.log_queue_event("CALL  ", step_count,
                                f"SYNC scene | oam={_oam_trigger} trou={_hole_trigger} pipe={_pipe_trigger}")
                            self.call_claude_scene_sync(situation, obs, step_count)
                        elif _currently_jumping and _pipe_trigger and not _oam_trigger and not _hole_trigger:
                            # Saut en cours, seul trigger = tuyau → inutile d'appeler Claude async :
                            # le saut en cours gère déjà l'obstacle. Restituer le seuil pour
                            # re-déclencher après l'atterrissage.
                            for t in self._get_level_thresholds()['pipe']:
                                if f"seuil {t}px" in _change_reason if '_change_reason' in locals() else False:
                                    self._pipe_thresholds_hit.discard(t)
                                    break
                            self.logger.log_queue_event("CALL  ", step_count,
                                f"SKIP async pipe-only jump={_currently_jumping} → seuil restitué post-atterrissage")
                        else:
                            self.logger.log_queue_event("CALL  ", step_count,
                                f"async scene | oam={_oam_trigger} trou={_hole_trigger} pipe={_pipe_trigger} jump={_currently_jumping}")
                            self.call_claude_async(situation, obs, step_count)
                        # Mémoriser le snapshot que Claude vient de recevoir
                        if '_snap' in locals() and _scene_trigger:
                            self._last_scene_snapshot = _snap
                        elif not _scene_trigger:
                            # Appel normal (queue basse, périodique) → reset snapshot
                            # pour que le prochain ennemi/tuyau soit considéré comme nouveau
                            self._last_scene_snapshot = None

                    # Gérer la mort de Mario
                    if done:
                        mario_lives_env = real_info.get('life', 0)
                        flag_get = real_info.get('flag_get', False)

                        if flag_get:
                            print(f" VICTOIRE! Mario a terminé World {self.current_world}-{self.current_level}!")
                            last_mario_decision = {'reasoning': 'VICTOIRE! Niveau terminé!', 'strategy': 'Mission accomplie'}

                            # Logger la victoire
                            self.logger.log_game_event("VICTORY", step_count, {
                                "world": self.current_world, "level": self.current_level,
                                "final_score": total_reward, "steps_taken": step_count,
                                "api_calls": self.api_calls, "total_cost": self.total_cost
                            })

                            # Finaliser la mémoire du niveau terminé
                            self.segment_memory.finalize_stage(_run_max_x, step_count, died=False)
                            if self.current_run_started:
                                summary = self.history_manager.end_run("victory", total_reward)
                                if summary:
                                    self.history_manager.print_run_summary(summary)

                            # Sauvegarder immédiatement le run parfait de CE stage,
                            # avant la transition (évite que le stage suivant écrase l'historique).
                            self._save_perfect_run(self.current_world, self.current_level)
                            # Remettre à zéro l'historique pour que le stage suivant parte proprement.
                            self._final_action_history.clear()
                            self._raw_action_history.clear()
                            self._rewind_index = None
                            self._perfect_start_ram = None
                            self._rewind_checkpoints = []

                            time.sleep(3)  # Pause pour admirer la victoire

                            # Passer au niveau suivant
                            next_lvl = get_next_level(self.current_world, self.current_level)
                            if next_lvl is None:
                                print(" JEU TERMINÉ ! Mario a battu tous les niveaux !")
                                self._exit_reason = "game_complete"
                                break

                            next_world, next_level = next_lvl
                            print(f"  Transition vers World {next_world}-{next_level}...")
                            transitioned = self._transition_to_level(next_world, next_level, step_count)
                            if not transitioned:
                                self._exit_reason = "victory"
                                break
                            # Réinitialiser les variables locales de boucle pour le nouveau niveau
                            obs = self.env.reset()
                            step_count = 0
                            total_reward = 0.0
                            _run_max_x = 0
                            done = False
                            current_action = None
                            current_macro_name = None
                            self.current_macro = None
                            self.action_queue.clear()
                            self.last_situation = None
                            # Forcer un premier appel Claude pour le nouveau niveau
                            self.level_context_established = False
                            continue
                        else:
                            # Mario est mort - mettre à jour le compteur interne
                            self.mario_lives_remaining = mario_lives_env
                            mario_x_death = real_info.get('x_pos', situation.get('mario', {}).get('x', 0))
                            self.record_death(step_count, mario_x_death)
                            self.lives_used += 1
                            print(f" Mario est mort! (Mort #{self.deaths_count}) Vies restantes: {self.mario_lives_remaining}")

                            #  Mémoriser la mort par segment
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
                                print(f"ℹ Mort à x={mario_x_death} ignorée "
                                      f"(déjà dépassé en run précédent, record={self.segment_memory.furthest_x})")

                            # Tentative de rewind avant game over
                            self.logger.log_game_event("DEATH_REWIND_CHECK", step_count, {
                                "deaths": self.deaths_count, "rewind_count": self.rewind_count,
                                "max_rewinds": self.max_rewinds, "buffer_len": len(self.rewind_buffer),
                                "death_x": mario_x_death})
                            # Limite de rewinds par zone de mort (±50px)
                            # Évite la boucle infinie quand l'IA ne sait pas passer un obstacle
                            _max_rewinds_per_zone = 5
                            _deaths_in_zone = sum(
                                1 for d in self._death_positions
                                if abs(d['x'] - mario_x_death) <= 50
                            )
                            _zone_limit_reached = _deaths_in_zone >= _max_rewinds_per_zone
                            if _zone_limit_reached:
                                print(f" Zone de mort x≈{mario_x_death} : {_deaths_in_zone} morts → stop rewind, game over")
                            if (self.deaths_count >= 1 and
                                    self.rewind_count < self.max_rewinds and
                                    self.rewind_buffer and
                                    not _zone_limit_reached):
                                # Choisir le checkpoint :
                                # - fell_in_hole : utiliser le checkpoint pré-saut si disponible
                                #   (Mario est au sol, juste avant le saut fatal → pas de replay)
                                # - Autres causes : le plus ancien (marge maximale)
                                if _death_cause == 'fell_in_hole':
                                    # Pour les trous : chercher le meilleur checkpoint sur le sol,
                                    # suffisamment loin du trou pour que max_jump puisse le franchir.
                                    # Distance minimale : 80px (assez pour prendre de l'élan).
                                    _min_dist = 80
                                    _ground_y_threshold = 170
                                    # 1. Essayer le checkpoint pré-saut s'il est assez loin du trou
                                    if (self._pre_jump_ram is not None and
                                            self._pre_jump_x > 0 and
                                            self._pre_jump_x < mario_x_death - _min_dist):
                                        checkpoint = {
                                            'ram': self._pre_jump_ram,
                                            'x_pos': int(self._pre_jump_x),
                                            'y_pos': int(self._pre_jump_y),
                                            'step': 0,
                                            'perfect_history_len': self._pre_jump_history_len,
                                            'has_full_backup': getattr(self, '_pre_jump_has_full_backup', False),
                                        }
                                        print(f"⏪ Checkpoint pre-saut : x={self._pre_jump_x} (trou x={mario_x_death})")
                                    else:
                                        # 2. Chercher dans le buffer le checkpoint le plus récent
                                        # sur le sol et assez loin du trou
                                        checkpoint = self.rewind_buffer[0]  # fallback : le plus ancien
                                        for _cp in reversed(self.rewind_buffer):
                                            if (_cp.get('y_pos', 200) >= _ground_y_threshold and
                                                    _cp.get('x_pos', 0) < mario_x_death - _min_dist):
                                                checkpoint = _cp
                                                break
                                        print(f"⏪ Checkpoint sol : x={checkpoint['x_pos']} "
                                              f"y={checkpoint.get('y_pos', '?')} (trou x={mario_x_death})")
                                    self._pre_jump_ram = None  # consommer dans tous les cas
                                else:
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
                                _perfect_len = checkpoint.get('perfect_history_len', 0)
                                self._raw_action_history.clear()
                                # Tronquer _final_action_history au niveau du checkpoint
                                # (garde x=0→checkpoint, jette la branche morte checkpoint→mort).
                                del self._final_action_history[_perfect_len:]
                                _last_replay_info = {}
                                _replay_succeeded = False  # Pas de replay — Mario repart du checkpoint
                                if checkpoint_ram is not None:
                                    self.env.unwrapped.done = False
                                    if checkpoint.get('has_full_backup'):
                                        # Restauration complète (CPU + PPU VRAM) via _backup/_restore.
                                        # Visuel = collision dès la frame 1 — aucun décor fantôme.
                                        self.env.unwrapped._restore()
                                        obs, _, _, _last_replay_info = self.env.step(0)  # 1 NOOP pour obs frais
                                        self.env.unwrapped.done = False
                                        print(f"⏪ FULL restore (CPU+PPU) → x={checkpoint['x_pos']} "
                                              f"(step={checkpoint['step']})")
                                    else:
                                        # Fallback RAM uniquement (checkpoint ancien, slot backup écrasé).
                                        # 30 NOOPs pour resynchroniser le PPU (nametable tiles).
                                        # Si Mario meurt pendant les NOOPs (trou), on re-restore la RAM
                                        # immédiatement pour éviter qu'il rejoue la mort.
                                        np.copyto(self.env.unwrapped._ram_buffer(), checkpoint_ram)
                                        _noop_died = False
                                        for _noop_i in range(30):
                                            self.env.unwrapped.done = False
                                            obs, _, _noop_done, _last_replay_info = self.env.step(0)
                                            if _noop_done:
                                                # Mario mort pendant les NOOPs (trou) — re-restaurer
                                                np.copyto(self.env.unwrapped._ram_buffer(), checkpoint_ram)
                                                self.env.unwrapped.done = False
                                                obs, _, _, _last_replay_info = self.env.step(0)
                                                _noop_died = True
                                                print(f"⏪ Mario mort NOOP#{_noop_i} → RAM re-restaurée x={checkpoint['x_pos']}")
                                                break
                                        self.env.unwrapped.done = False
                                        if not _noop_died:
                                            print(f"⏪ RAM fallback + 30 NOOPs → x={checkpoint['x_pos']} "
                                                  f"(step={checkpoint['step']})")
                                    # Bloquer les appels Claude le temps que le PPU soit stable
                                    self._ppu_warmup_until = step_count + 5
                                    # Enregistrer CE checkpoint dans la liste (multi-rewind).
                                    _cp_entry = {'index': _perfect_len, 'ram': checkpoint_ram.copy(), 'x': checkpoint['x_pos']}
                                    self._rewind_checkpoints = [c for c in self._rewind_checkpoints if c['index'] < _perfect_len]
                                    self._rewind_checkpoints.append(_cp_entry)
                                    print(f"   Claude bloqué jusqu'au step {self._ppu_warmup_until}")
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
                                        print(" Fallback replay mort — reset au début")
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
                                # _danger_dist sera calculé après _post_rewind_x (position réelle)
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
                                # Utiliser la position réelle post-NOOP (pas la valeur stale du checkpoint)
                                _post_rewind_x = _last_replay_info.get('x_pos', checkpoint['x_pos'])
                                _danger_dist = mario_x_death - _post_rewind_x
                                _hole_warn = (
                                    f"REWIND PRE-SAUT : Mario est revenu a x={_post_rewind_x}, "
                                    f"avant le saut fatal.\n"
                                    f"   Il y a un TROU vers x={mario_x_death}.\n"
                                    f"   Sauter IMMÉDIATEMENT avec max_jump.\n"
                                ) if _death_cause == 'fell_in_hole' else ""
                                _obligatoire = (
                                    f"SAUTE par-dessus la zone x={mario_x_death}"
                                    if _death_cause == 'enemy_hit'
                                    else "max_jump IMMÉDIATEMENT (trou confirmé)"
                                    if _death_cause == 'fell_in_hole'
                                    else "Avance vite avant le timer"
                                )
                                self._rewind_death_x = mario_x_death
                                self._rewind_correction_msg = (
                                    f"⏪ REWIND #{self.rewind_count}/{self.max_rewinds} — MARIO VIENT DE MOURIR !\n"
                                    f"Position de mort : x={mario_x_death}, y={_mario_y}\n"
                                    f"Cause : {cause_fr.get(_death_cause, _death_cause)}\n"
                                    f"Séquence fatale : {macros_str}\n"
                                    f"Zone dangereuse : à partir de x={mario_x_death - 20} "
                                    f"(mort à {_danger_dist}px du checkpoint)\n"
                                    f"{_history_lines}"
                                    f"Le jeu est REMBOBINÉ à x={_post_rewind_x}.\n"
                                    f"{_hole_warn}"
                                    f"OBLIGATOIRE : propose des actions DIFFÉRENTES — {_obligatoire}."
                                )

                                # Mémoriser la zone dangereuse uniquement pour les trous
                                # (pour les ennemis, l'OAM reflex gère mieux à 15-70px)
                                if _death_cause == 'fell_in_hole':
                                    self._danger_zone_x = mario_x_death
                                else:
                                    self._danger_zone_x = None
                                # Invalider le thread Claude en cours (s'il y en a un)
                                self._claude_generation += 1
                                # Libérer le verrou claude_thinking pour que le prochain cycle
                                # puisse déclencher un nouvel appel immédiatement sans attendre
                                # que l'ancien thread API se termine (il verra generation != et
                                # abandonnera ses résultats dans son finally).
                                self.claude_thinking = False
                                # Vider la queue → Claude sera appelé au prochain cycle
                                self.action_queue.clear()
                                self.current_macro = None
                                # TROU : injecter max_jump depuis le checkpoint pré-saut
                                # (Mario est sur le sol, il peut sauter proprement par-dessus le trou)
                                if _death_cause == 'fell_in_hole':
                                    self.action_queue.append({
                                        'macro_name': 'max_jump',
                                        'strategy': 'Anti-trou post-rewind pre-jump',
                                        'urgency': 10,
                                        'confidence': 95
                                    })
                                    print(f"⏪ max_jump injecte depuis checkpoint pre-saut x={checkpoint['x_pos']}")
                                # Bloquer inject_known_solution jusqu'au prochain appel Claude
                                # (évite que l'ancienne séquence mémorisée soit rejouée avant
                                # que Claude ait eu la chance de proposer une alternative)
                                self._post_rewind_block_inject = True
                                self._last_known_solution_x = _post_rewind_x  # reset cooldown

                                # Réinitialiser les compteurs de death pour continuer
                                self.deaths_count -= 1  # Annuler la mort comptée
                                self.rewind_buffer.clear()  # Vider le buffer après rewind
                                # Sauvegarder immédiatement l'état restauré comme nouveau checkpoint.
                                # Sans ça : si Mario re-meurt en 1-2 steps (ennemi à 2px), le buffer
                                # est vide → pas de rewind possible → game over injuste.
                                _post_rewind_ram = self.env.unwrapped._ram_buffer().copy()
                                _post_rewind_y = _last_replay_info.get('y_pos', 200)
                                self.rewind_buffer.append({
                                    'step': step_count,
                                    'ram': _post_rewind_ram,
                                    'x_pos': _post_rewind_x,
                                    'y_pos': int(_post_rewind_y),
                                    'macros': [],
                                    'action_history': list(self._final_action_history),
                                    'perfect_history_len': len(self._final_action_history),
                                    'has_full_backup': False,
                                })
                                self._last_run_jump_over_x = -999  # Empêche conversion run_jump_over→pipe_jump après rewind
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
                                print(" GAME OVER - Mario a perdu sa vie!")

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
                                print(" Redémarrage automatique...")
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
                                print(f" Vie {self._run_phase} → Phase {self._run_phase}: "
                                      f"{_labels[self._run_phase]}")
                                time.sleep(1)  # Pause pour voir le redémarrage

                # Affichage fluide
                situation = self.analyze_situation(obs, real_info if 'real_info' in locals() else {
                    'x_pos': _seg_x, 'y_pos': 200, 'score': total_reward
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

                # Overlay PAUSE MANUELLE (ESPACE)
                if paused:
                    h, w = display.shape[:2]
                    cx, cy = w // 2, 30
                    overlay = display.copy()
                    cv2.rectangle(overlay, (cx - 220, cy - 18), (cx + 220, cy + 18), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
                    cv2.putText(display, "|| PAUSE MANUELLE  [ESPACE pour reprendre]",
                                (cx - 210, cy + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
                    print("⏸ Pause" if paused else "▶ Reprise")
                elif self.handle_scroll_keys(key):
                    # Touche de défilement traitée
                    pass

        except KeyboardInterrupt:
            self._exit_reason = "keyboard_interrupt"
            print("\n⏹ Arrêt demandé")

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

            #  Sauvegarder le run parfait (historique tronqué aux rewinds)
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
            print(f"\n Fichiers de log créés:")
            for log_file in log_files:
                if os.path.exists(log_file):
                    size_kb = os.path.getsize(log_file) / 1024
                    print(f"    {os.path.basename(log_file)} ({size_kb:.1f} KB)")

        # Raison d'arrêt
        _exit_labels = {
            "victory": " Niveau terminé !",
            "game_over": " GAME OVER (3 morts)",
            "user_esc": "⏹ Arrêt par l'utilisateur (ESC)",
            "window_closed": " Fenêtre fermée",
            "keyboard_interrupt": "⌨ Interruption clavier (Ctrl+C)",
        }
        if max_steps is not None and self._exit_reason == "unknown":
            self._exit_reason = "max_steps"
            _exit_labels["max_steps"] = f"⏱ Limite de steps atteinte ({max_steps} steps)"
        print(f"\nFin de partie : {_exit_labels.get(self._exit_reason, f'Raison inconnue ({self._exit_reason})')}")

        # Statistiques finales
        print(f"\nRÉSULTATS MARIO FLUIDE:")
        print(f"    Steps total: {step_count}")
        print(f"    Score final: {total_reward}")
        print(f"    Morts de Mario: {self.deaths_count}")
        print(f"    Vies utilisées: {self.lives_used}")
        print(f"    Décisions Claude: {self.api_calls}")
        print(f"    Macros réussies: {self.successful_macros}")
        print(f"    Coût total: ${self.total_cost:.3f}")
        print(f"    Distance finale: {real_info.get('x_pos', 0) if 'real_info' in locals() else 0}")

        # Taux de réussite
        if self.deaths_count > 0:
            survival_rate = (step_count - self.deaths_count * 20) / step_count * 100  # Approximation
            print(f"    Taux de survie: {survival_rate:.1f}%")

        # Afficher les statistiques d'historique
        print(f"\n HISTORIQUE GLOBAL:")
        updated_stats = self.history_manager.get_run_stats()
        print(f"    Runs totaux: {updated_stats.get('total_runs', 0)}")
        if updated_stats.get('total_runs', 0) > 0:
            print(f"    Record distance: {updated_stats['best_distance']} pixels")
            print(f"    Record vitesse: {updated_stats['best_speed']:.2f} px/s")
            completion = updated_stats.get('completion_rates', {})
            print(f"    Victoires: {completion.get('victory', 0)} | Morts: {completion.get('death', 0)} | Interruptions: {completion.get('interrupted', 0)}")

            # Comparer avec le meilleur run
            best_run = self.history_manager.get_best_run()
            if best_run:
                current_distance = real_info.get('x_pos', 0) if 'real_info' in locals() else 0
                if current_distance > best_run.max_position_x:
                    print(f"    NOUVEAU RECORD! Ancien: {best_run.max_position_x} → Nouveau: {current_distance}")
                else:
                    print(f"    Performance: {current_distance}/{best_run.max_position_x} pixels du record")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mario Bros IA - Claude LLM")
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Nombre maximum de steps avant arrêt automatique (défaut: illimité)"
    )
    args = parser.parse_args()

    print("Mario Bros FLUIDE - Claude LLM avec Macro-Actions")
    print("Mario exécute les décisions de Claude de façon naturelle!")
    if args.max_steps:
        print(f"⏱Arrêt automatique après {args.max_steps} steps")

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print(" ANTHROPIC_API_KEY requise!")
        return

    while True:
        mario_fluid = None
        try:
            mario_fluid = MarioFluidLLM()
            mario_fluid.play_fluid_mario(max_steps=args.max_steps)

        except KeyboardInterrupt:
            print("\n Arrêt demandé (Ctrl+C)")
            break
        except Exception as e:
            print(f" Erreur: {e}")
            import traceback
            traceback.print_exc()

        # Récupérer la raison d'arrêt
        exit_reason = getattr(mario_fluid, '_exit_reason', 'unknown') if mario_fluid else 'error'

        # Arrêts volontaires → quitter
        if exit_reason in ('keyboard_interrupt', 'user_esc', 'window_closed', 'user_quit'):
            print("\nÀ bientôt !")
            break

        # Fin naturelle (game_over, victoire, max_steps) → retour au menu automatiquement
        if exit_reason in ('game_over', 'victory', 'max_steps'):
            print("\n" + "="*60)
            print("Retour au menu...")
            continue

        # Autres cas (erreur inconnue) → demander
        print("\n" + "="*60)
        try:
            again = input("Nouvelle partie ? (o/N) : ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            again = "n"
        if again != "o":
            print("À bientôt !")
            break

if __name__ == "__main__":
    main()