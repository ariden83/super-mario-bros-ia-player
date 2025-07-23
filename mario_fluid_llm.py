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
from PIL import Image
import io
import tempfile

class MarioFluidLLM:
    def __init__(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        
        # Actions de base
        self.actions = {
            0: 'NOOP', 1: 'RIGHT', 2: 'JUMP', 3: 'RUN', 
            4: 'RUN_JUMP', 5: 'JUMP_ONLY', 6: 'LEFT'
        }
        
        # Actions étendues que Claude peut commander (basées sur la recherche Super Mario Bros NES)
        self.macro_actions = {
            # Mouvements de base
            'walk_right': {'base_action': 1, 'duration': 8, 'description': 'Marcher à droite'},
            'run_forward': {'base_action': 3, 'duration': 10, 'description': 'Courir vers la droite (plus rapide)'},
            'step_back': {'base_action': 6, 'duration': 6, 'description': 'Reculer/éviter danger'},
            'wait': {'base_action': 0, 'duration': 4, 'description': 'Attendre/observer'},
            
            # Sauts tactiques
            'short_jump': {'base_action': 2, 'duration': 10, 'description': 'Petit saut pour petits obstacles'},
            'high_jump': {'base_action': 5, 'duration': 8, 'description': 'Saut vertical haut'},
            'long_jump': {'base_action': 4, 'duration': 12, 'description': 'Course + saut pour longues distances'},
            'precise_jump': {'base_action': 2, 'duration': 10, 'description': 'Saut précis sur ennemis/blocs'},
            
            # Actions spéciales Mario Bros
            'stomp_enemy': {'base_action': 2, 'duration': 8, 'description': 'Sauter sur Goomba/Koopa pour les tuer'},
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
            
            # Actions tactiques spécifiques
            'wait_for_enemy': {'base_action': 0, 'duration': 10, 'description': 'Attendre que l\'ennemi passe (timing)'},
            'retreat_and_jump': {'base_action': 6, 'duration': 12, 'description': 'Reculer puis sauter (éviter puis attaquer)'},
            'run_jump_over': {'base_action': 4, 'duration': 20, 'description': 'Course + saut pour passer par-dessus obstacle'},
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
        self.screenshot_frequency = 15  # Prendre un screenshot toutes les 15 steps (meilleure anticipation)
        self.use_visual_analysis = True  # Utiliser l'analyse d'image Claude
        self.screenshot_cost_limit = 1.00  # Limite de coût pour les screenshots ($1.00) - augmentée pour meilleure anticipation
        self.screenshot_costs = 0.0  # Coût cumulé des screenshots
        self.ultra_low_cost_mode = False  # Mode très économique désactivé pour améliorer la vision
        
        # Statistiques
        self.api_calls = 0
        self.total_cost = 0.0
        self.successful_macros = 0
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
        
        print("✅ Mario Fluide LLM initialisé!")
    
    def analyze_situation(self, obs, info, step_count):
        """Analyser la situation pour Claude"""
        
        mario_x = info.get('x_pos', 0)
        mario_y = info.get('y_pos', 0)
        score = info.get('score', 0)
        
        self.position_history.append(mario_x)
        
        # Analyser la progression
        progress_analysis = self.analyze_progression()
        
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
        
        return {
            'mario': {'x': mario_x, 'y': mario_y, 'score': score},
            'progress': progress_analysis,
            'screen': screen_analysis,
            'history': {
                'positions': list(self.position_history)[-8:],
                'recent_macros': list(self.macro_history)[-4:]
            },
            'step': step_count,
            'lives': info.get('life', 3)  # Ajouter les vies
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
            # Convertir de BGR (OpenCV) vers RGB
            rgb_image = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
            
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
            
            # OPTIMISATION 3: Appliquer des filtres pour améliorer la détection
            enhanced_image = self.apply_detection_filters(resized_image)
            
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
            
            return img_base64
            
        except Exception as e:
            print(f"❌ Erreur capture screenshot: {e}")
            return None
    
    def apply_detection_filters(self, image):
        """Appliquer des filtres pour améliorer la détection de Claude"""
        try:
            # Convertir en numpy pour OpenCV
            img_array = np.array(image)
            
            # FILTRE 1: Augmenter le contraste pour différencier les éléments
            # Utiliser CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # FILTRE 2: Légère netteté pour clarifier les contours
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel * 0.3)  # Facteur réduit pour éviter l'over-sharpening
            
            # FILTRE 3: Réduction du bruit tout en préservant les détails
            denoised = cv2.bilateralFilter(sharpened, 5, 50, 50)
            
            # Reconvertir en PIL
            filtered_image = Image.fromarray(denoised)
            
            return filtered_image
            
        except Exception as e:
            print(f"⚠️ Erreur filtres, image originale utilisée: {e}")
            return image
    
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
            
            prompt = f"""Tu es Claude, expert Mario Bros ! Analyse cette capture d'écran du jeu en temps réel.

CONTEXTE DÉTAILLÉ MARIO:
🔹 Position actuelle: X={mario_x}, Y={mario_y} (sur écran {screen_width}px de large)
🔹 Vitesse: {mario_speed:.1f} pixels/step ({'vers droite' if mario_speed > 0 else 'stationnaire' if mario_speed == 0 else 'vers gauche'})
🔹 Score: {mario['score']} | Step: {step_count}
🔹 Progression: {progress['status']} (tendance: {progress['trend']}px sur 30 steps)
🔹 Zone d'écran: Mario occupe ~{(mario_x/screen_width)*100:.0f}% de la largeur écran
🔹 Morts: {self.deaths_count} | Vies utilisées: {self.lives_used}

📚 HISTORIQUE D'APPRENTISSAGE - APPRENDS DE TES ERREURS:
{self.get_learning_context()}

VITESSES DE RÉFÉRENCE (pour calculs de timing):
- Goomba: ~0.5 pixels/step vers la gauche
- Mario marche: ~1-2 pixels/step vers droite  
- Mario court: ~3-4 pixels/step vers droite
- Collision dans ~{abs(mario_x-200)//2:.0f} steps si ennemi à droite et vitesses normales

🔍 ÉVALUATION DE SÉCURITÉ PRIORITAIRE:
Regarde attentivement cette image et identifie EN PRIORITÉ ABSOLUE:

1. ENNEMIS ET DANGERS MORTELS: 
   - QUELS ennemis vois-tu (Goombas bruns, Koopas verts)?
   - DISTANCE CRITIQUE: À quelle distance EXACTE de Mario (très proche <15px = DANGER IMMÉDIAT, proche 15-30px = ATTENTION, loin >30px = TEMPORAIREMENT SÛR)?
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
   - TUYAUX: Si tu vois un tuyau vert devant Mario, utilise 'pipe_down' pour descendre ou 'high_jump' pour passer par-dessus
   - BLOCAGE: Si Mario semble coincé (vitesse négative), utilise 'step_back' puis 'high_jump' ou 'pipe_down'
   - Espace libre devant Mario pour avancer?

ACTIONS MARIO DISPONIBLES:
{chr(10).join([f"'{key}': {action['description']}" for key, action in self.macro_actions.items()])}

STRATÉGIES CRITIQUES:

🔥 ORDRE DE PRIORITÉS ABSOLU (SURVIE D'ABORD):

1. SURVIE ABSOLUE (PRIORITÉ MAXIMALE):
   - Si ennemi très proche (<15px): 'stomp_enemy' IMMÉDIATEMENT ou fuir avec 'retreat_and_jump'
   - Si collision imminente (1-2 steps): ÉVITER à tout prix - 'step_back', 'small_hop_left', 'wait_for_enemy'
   - JAMAIS marcher vers un ennemi sans plan de saut sur sa tête
   - TOUJOURS prioriser la survie sur tout autre objectif

2. COLLECTE BLOCS QUESTION MARKS (PRIORITÉ HAUTE):
   - UNIQUEMENT si aucun danger immédiat
   - Si bloc ? visible ET zone sécurisée: 'approach_and_hit_block' ou 'hop_on_platform' puis 'hit_block'
   - ABANDONNER la collecte si un ennemi s'approche
   
3. DÉBLOCAGE/PROGRESSION:
   - Si coincé ET sûr: actions de déblocage
   - Si voie libre: progression normale

🚨 RÈGLES DE SURVIE STRICTES:
- INTERDIT de marcher directement vers un ennemi avec 'walk_right' ou 'run_forward'
- INTERDIT d'ignorer un ennemi qui s'approche de Mario
- Les Goombas BOUGENT de DROITE vers GAUCHE automatiquement à ~0.5px/step
- DISTANCE CRITIQUE: Si ennemi <20px de Mario = DANGER IMMÉDIAT
- ACTIONS SURVIE: 'stomp_enemy' (sauter DESSUS), 'retreat_and_jump', 'wait_for_enemy', 'step_back'
- Si ennemi loin (>50px): Mario peut collecter blocs EN SURVEILLANT constamment l'ennemi

🎯 BLOCS QUESTION MARKS (PRIORITÉ HAUTE MAIS SÉCURISÉE):
- COLLECTE SYSTÉMATIQUE: Frappe TOUS les blocs ? visibles quand c'est SÉCURISÉ
- VÉRIFICATION SÉCURITÉ: Avant d'agir, confirme qu'aucun ennemi n'est proche (<30px)
- ACTIONS RECOMMANDÉES:
  * Bloc ? visible + zone sûre: 'approach_and_hit_block' (30 frames)
  * Mario sous bloc ? + sûr: 'hit_block' (20 frames)  
  * Bloc sur plateforme: 'hop_on_platform' puis 'hit_block'
- ABANDON TACTIQUE: Si ennemi s'approche pendant collecte, INTERROMPRE et fuir
- Ces blocs donnent des pièces, power-ups (champignons, fleurs) et des vies extra

TUYAUX ET PLATEFORMES:
- TUYAU COURT (comme dans l'image): utilise 'jump_on_pipe' ou 'hop_on_platform' pour monter dessus
- GROS TUYAU: utilise 'pipe_down' pour entrer (zone bonus) ou 'run_jump_over' pour passer par-dessus
- BLOCAGE: Si Mario est coincé, utilise 'retreat_and_jump' ou 'wait_for_enemy' selon la situation
- PLATEFORME + ENNEMI: utilise 'wait_for_enemy' puis 'jump_on_pipe' quand l'ennemi passe

ACTIONS GRANULAIRES DISPONIBLES:
- 'jump_on_pipe': Monter sur tuyau court/plateforme
- 'small_hop_right/left': Petits sauts directionnels
- 'big_jump_right': Grand saut pour franchir obstacles
- 'wait_for_enemy': Timing pour laisser passer les ennemis
- 'retreat_and_jump': Reculer puis attaquer
- 'run_jump_over': Course + saut pour passer par-dessus

🎯 DÉCISION FINALE - ORDRE STRICT:
1. D'ABORD: Évalue TOUS les dangers (ennemis, distances, timing)
2. ENSUITE: Si zone SÉCURISÉE, identifie les blocs ? à collecter  
3. ENFIN: Choisis actions qui GARANTISSENT la survie ET maximisent la collecte

DONNE 2-3 ACTIONS TACTIQUES basées sur cette analyse SÉCURISÉE!

Réponds en JSON compact avec analyse de sécurité détaillée:
{{"actions":[{{"macro_action":"<nom>","reasoning":"<court + distance/timing + sécurité>"}}],"strategy":"<court>","urgency":<1-10>,"spatial_analysis":"<positions, distances, timing des ennemis/blocs>","immediate_danger":"<oui/non + détails + distance exacte>","safety_assessment":"<niveau de sécurité pour collecte>","next_target":"<prochain objectif sécurisé>"}}"""

            self.api_calls += 1
            print(f"📸 Envoi screenshot à Claude (appel #{self.api_calls})...")
            print("="*80)
            print("🔍 PROMPT ENVOYÉ À CLAUDE:")
            print(prompt)
            print("="*80)
            
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=400,  # Augmenté pour permettre des réponses JSON complètes avec plusieurs actions
                temperature=0.1,
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
            print("="*80)
            print("💭 RÉPONSE DE CLAUDE:")
            print(response_text)
            print("="*80)
            
            # Coût plus élevé pour les images (estimé selon la taille optimisée)
            image_cost = min(0.01, len(screenshot_b64) * 0.000001)  # Coût proportionnel à la taille
            text_cost = len(prompt) * 0.25 / 1000000 + len(response_text) * 1.25 / 1000000
            cost = text_cost + image_cost
            
            self.total_cost += cost
            self.screenshot_costs += image_cost
            
            # Ajuster la fréquence si on dépasse le budget
            if self.screenshot_costs > self.screenshot_cost_limit:
                self.screenshot_frequency = min(100, self.screenshot_frequency + 10)  # Réduire la fréquence
                print(f"💰 Budget screenshots dépassé, fréquence réduite à {self.screenshot_frequency} steps")
            
            print(f"💰 Coût screenshot: ${image_cost:.4f} (total screenshots: ${self.screenshot_costs:.3f})")
            
            return response_text
            
        except Exception as e:
            print(f"❌ Erreur analyse screenshot: {e}")
            return None
    
    def create_claude_prompt(self, situation):
        """Créer un prompt pour que Claude choisisse une macro-action"""
        
        mario = situation['mario']
        progress = situation['progress']
        screen = situation['screen']
        history = situation['history']
        
        # Construire la liste des macro-actions
        macro_list = []
        for key, macro in self.macro_actions.items():
            macro_list.append(f"'{key}': {macro['description']} ({macro['duration']} frames)")
        
        prompt = f"""🍄 Tu es Claude, EXPERT MARIO BROS NES ! Mario a besoin de 2-3 actions RAPIDES car le jeu est dangereux !

📍 SITUATION MARIO:
• Position: X={mario['x']}, Y={mario['y']} | Score: {mario['score']} | Step: {situation['step']}
• Progression: {progress['status']} (tendance: {progress['trend']}px)

🔍 ANALYSE VISUELLE MARIO BROS:
• Obstacles: {'OUI' if screen['immediate_obstacles'] else 'NON'} | Sol stable: {'OUI' if screen['ground_stable'] else 'NON'}
• Blocs ?: {'OUI - FRAPPE-LES!' if screen.get('question_blocks') else 'NON'}
• Ennemis (Goomba): {'OUI - DANGER!' if screen.get('enemies_nearby') else 'NON'}
• Power-ups: {'OUI - RÉCUPÈRE!' if screen.get('power_ups') else 'NON'}
• Environnement: {screen['environment_type']}

🗺️ CARTE SPATIALE (distances/directions):
{chr(10).join(screen.get('spatial_map', ['Aucun élément détecté'])[:5])}

🎮 ACTIONS MARIO BROS DISPONIBLES:
{chr(10).join(macro_list)}

🧠 STRATÉGIES MARIO BROS:
• Stomp enemies (Goomba/Koopa) en sautant dessus pour les tuer et gagner points
• Hit blocks ? par dessous pour obtenir coins/power-ups/champignons
• Collect power-ups pour devenir Super Mario ou Fire Mario
• Avoid Piranha Plants en attendant qu'elles rentrent ou en courant vite
• Use pipes pour accéder aux zones bonus souterraines
• Kick shells de Koopa pour tuer autres ennemis

🎯 DONNE 2-3 ACTIONS ADAPTÉES À LA SITUATION!

JSON compact seulement:
{{"actions":[{{"macro_action":"<nom>","reasoning":"<court>"}},{{"macro_action":"<nom>","reasoning":"<court>"}}],"strategy":"<court>","urgency":<1-10>}}

Exemples PRIORITÉ SURVIE:
Danger immédiat: {{"actions":[{{"macro_action":"stomp_enemy","reasoning":"Goomba <15px - ÉLIMINER IMMÉDIATEMENT"}},{{"macro_action":"step_back","reasoning":"Sécuriser position après élimination"}}],"strategy":"SURVIE ABSOLUE","urgency":10}}
Ennemi proche: {{"actions":[{{"macro_action":"wait_for_enemy","reasoning":"Goomba à 25px - laisser passer d'abord"}},{{"macro_action":"approach_and_hit_block","reasoning":"Zone sécurisée après passage - collecter bloc ?"}}],"strategy":"Sécurité puis collecte","urgency":8}}
Zone sûre + bloc: {{"actions":[{{"macro_action":"approach_and_hit_block","reasoning":"Aucun ennemi visible - bloc ? collecte SÉCURISÉE"}},{{"macro_action":"collect_powerup","reasoning":"Récupérer item en sécurité"}}],"strategy":"Collecte systématique sécurisée","urgency":7}}
Fuite nécessaire: {{"actions":[{{"macro_action":"retreat_and_jump","reasoning":"Ennemi trop proche - fuir vers plateforme"}},{{"macro_action":"hop_on_platform","reasoning":"Monter sur tuyau pour sécurité"}}],"strategy":"Fuite tactique","urgency":9}}"""

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
            
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=120,  # Plus court pour éviter la troncature
                temperature=0.1,  # Plus déterministe
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            
            # Coût estimé
            cost = len(prompt) * 0.25 / 1000000 + len(response_text) * 1.25 / 1000000
            self.total_cost += cost
            
            return response_text
            
        except Exception as e:
            print(f"❌ Erreur Claude: {e}")
            return None
    
    def call_claude_async(self, situation, obs=None, step_count=0):
        """Appeler Claude en arrière-plan avec système hybride optimisé"""
        
        def claude_worker():
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
                for action in actions:
                    if len(self.action_queue) < 8:  # Éviter l'overflow
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
            # Chercher un JSON complet avec regex plus robuste
            json_patterns = [
                r'\{[^{}]*"actions"[^{}]*\[[^\]]*\][^{}]*\}',  # JSON complet avec actions
                r'\{[^{}]*"actions"[^{}]*\}',  # JSON partiel avec actions
                r'\{.*?\}',  # Tout JSON
            ]
            
            data = None
            for pattern in json_patterns:
                json_match = re.search(pattern, response_text, re.DOTALL)
                if json_match:
                    try:
                        json_text = json_match.group()
                        # Nettoyer le JSON (fermer les crochets/accolades manquants)
                        json_text = self.fix_broken_json(json_text)
                        data = json.loads(json_text)
                        break
                    except json.JSONDecodeError:
                        continue
            
            if data:
                actions_list = []
                actions_data = data.get('actions', [])
                strategy = data.get('strategy', 'Stratégie Claude')
                urgency = int(data.get('urgency', 5))
                
                for action_data in actions_data:
                    if isinstance(action_data, dict):
                        macro_name = action_data.get('macro_action', 'walk_right')
                        reasoning = action_data.get('reasoning', 'Action Claude')
                        
                        # Valider la macro-action
                        if macro_name not in self.macro_actions:
                            print(f"⚠️ Macro inconnue '{macro_name}', utilisation de walk_right")
                            macro_name = 'walk_right'
                        
                        actions_list.append({
                            'macro_name': macro_name,
                            'reasoning': reasoning,
                            'strategy': strategy,
                            'urgency': urgency,
                            'confidence': 80
                        })
                
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
    
    def get_fallback_macro(self):
        """Macro-action par défaut en cas d'erreur - SÉCURISÉE"""
        return {
            'macro_name': 'wait',
            'reasoning': 'Attente sécurisée en attendant les instructions Claude',
            'strategy': 'Mode sécurisé - attente',
            'urgency': 1,
            'confidence': 90
        }
    
    def execute_macro_action(self, macro_decision):
        """Démarrer l'exécution d'une macro-action"""
        
        macro_name = macro_decision['macro_name']
        macro_config = self.macro_actions[macro_name]
        
        self.current_macro = {
            'name': macro_name,
            'base_action': macro_config['base_action'],
            'frames_left': macro_config['duration'],
            'decision': macro_decision
        }
        
        self.macro_history.append({
            'name': macro_name,
            'reasoning': macro_decision['reasoning'][:30]
        })
        
        # Enregistrer l'action pour apprentissage
        if hasattr(self, 'last_situation'):
            self.record_action(
                macro_name, 
                self.last_situation, 
                getattr(self, 'current_step', 0),
                macro_decision.get('reasoning', '')
            )
        
        print(f"🎮 Exécution: {macro_name} ({macro_config['duration']} frames) - {macro_decision['reasoning'][:50]}")
    
    def get_current_action(self):
        """Obtenir l'action à exécuter cette frame"""
        
        if self.current_macro and self.current_macro['frames_left'] > 0:
            # Continuer la macro en cours
            self.current_macro['frames_left'] -= 1
            return self.current_macro['base_action']
        else:
            # Macro terminée, essayer de prendre la suivante dans la queue
            if self.current_macro:
                self.successful_macros += 1
                self.current_macro = None
            
            # Vérifier s'il y a des actions en attente
            if self.action_queue:
                next_action = self.action_queue.popleft()
                self.execute_macro_action(next_action)
                return self.get_current_action()  # Récursion pour obtenir l'action
            
            return None  # Pas d'action, temps de demander à Claude
    
    def create_display(self, frame, situation, mario_decision, total_reward, step_count):
        """Créer l'affichage avec informations"""
        
        display_frame = cv2.resize(frame, (600, 480))
        canvas = np.zeros((700, 1000, 3), dtype=np.uint8)
        
        # Placer le jeu
        canvas[80:560, 50:650] = display_frame
        
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
            f"Step: {step_count}",
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
        
        # Macro actuelle et queue
        y_pos += 20
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
        
        # Contrôles
        cv2.putText(canvas, "ESC: Quitter | ESPACE: Pause | Mario joue en FLUIDE!", (50, 680), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
        
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
        action_record = {
            'timestamp': step_count,
            'action': action_name,
            'mario_position': situation.get('mario', {}).get('x', 0),
            'mario_y': situation.get('mario', {}).get('y', 0),
            'progress_status': situation.get('progress', {}).get('status', 'unknown'),
            'reasoning': reasoning,
            'lives_remaining': situation.get('lives', 3)
        }
        self.action_history.append(action_record)
        self.last_actions_before_death.append(action_record)
    
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
                
                for goomba_x in goomba_x_positions:
                    # Calculer position relative à Mario et distance
                    distance_from_mario = abs(goomba_x - (mario_x % width))
                    enemies_list.append({
                        'type': 'Goomba',
                        'x': goomba_x,
                        'y': height - 50,  # Goombas sont au sol
                        'distance_from_mario': distance_from_mario,
                        'threat_level': 'HIGH' if distance_from_mario < 30 else 'MEDIUM' if distance_from_mario < 60 else 'LOW'
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
                
                for block_x in question_x_positions:
                    distance_from_mario = abs(block_x - (mario_x % width))
                    question_blocks.append({
                        'type': 'QuestionBlock',
                        'x': block_x,
                        'y': air_level,
                        'distance_from_mario': distance_from_mario,
                        'collectible': True
                    })
            
            # Retourner un résumé des changements depuis la dernière update
            changes = self.detect_position_changes(step_count)
            
            return {
                'mario': self.tracked_elements['mario'],
                'enemies': enemies_list,
                'question_blocks': question_blocks,
                'changes': changes,
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
        
        prompt += f"""

📚 HISTORIQUE D'APPRENTISSAGE:
{learning_context}

🎯 DÉCISION RAPIDE REQUISE:
Basé sur ces positions exactes et ton contexte du niveau, choisis 2-3 actions IMMÉDIATES.

PRIORITÉS ABSOLUES:
1. SURVIE: Éviter les ennemis HIGH/MEDIUM threat
2. COLLECTE: Frapper les blocs ? si zone sécurisée  
3. PROGRESSION: Avancer si voie libre

Réponds en JSON compact:
{{"actions":[{{"macro_action":"<nom>","reasoning":"<court + distance exacte>"}}],"urgency":<1-10>,"threat_analysis":"<menaces immédiates avec distances>","next_target":"<prochain objectif>"}}"""
        
        return prompt
    
    def call_claude_for_positions_update(self, positions_data, step_count):
        """Appeler Claude avec une mise à jour positionnelle (texte seulement)"""
        try:
            prompt = self.create_positional_update_prompt(positions_data, step_count)
            
            self.api_calls += 1
            print(f"📍 Envoi mise à jour positionnelle à Claude (appel #{self.api_calls})...")
            print("="*50)
            print("🔍 PROMPT POSITIONS ENVOYÉ À CLAUDE:")
            print(prompt)
            print("="*50)
            
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,  # Plus petit que screenshot (seulement JSON)
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            response_text = response.content[0].text if response.content else ""
            
            print("✅ Claude analyse reçue (texte)", f"({len(response_text)} chars)")
            print("="*50)
            print("💭 RÉPONSE DE CLAUDE:")
            print(response_text)
            print("="*50)
            
            # Coût estimé pour mise à jour textuelle (beaucoup moins cher qu'une image)
            estimated_cost = 0.001  # $0.001 vs $0.01 pour screenshot
            self.total_cost += estimated_cost
            print(f"💰 Coût mise à jour: ${estimated_cost:.4f} (total: ${self.total_cost:.3f})")
            
            return response_text
            
        except Exception as e:
            print(f"❌ Erreur mise à jour positionnelle: {e}")
            return None
    
    def should_use_screenshot_vs_positions(self, step_count):
        """Décider entre screenshot complet ou mise à jour positionnelle"""
        # Premier appel : toujours screenshot pour établir le contexte
        if not self.level_context_established:
            return True, "Premier screenshot pour établir la carte du niveau"
        
        # Screenshot de recalibrage périodique
        if step_count - self.last_screenshot_step >= self.context_recalibration_frequency:
            return True, f"Recalibrage après {self.context_recalibration_frequency} steps"
        
        # Screenshot si Mario semble en difficulté (progression négative)
        if len(self.position_history) >= 3:
            recent_positions = list(self.position_history)[-3:]
            if recent_positions[-1] <= recent_positions[0]:  # Pas de progrès
                return True, "Mario semble bloqué - screenshot pour diagnostic"
        
        # Sinon, mise à jour positionnelle
        return False, "Conditions normales - mise à jour positionnelle suffisante"
    
    def play_fluid_mario(self, max_steps=2000):
        """Jouer avec Mario fluide et Claude intelligent"""
        
        print("🎮 MARIO FLUIDE avec CLAUDE LLM")
        print("Claude donne des macro-actions, Mario les exécute fluidement!")
        print("=" * 60)
        
        obs = self.env.reset()
        total_reward = 0
        step_count = 0
        paused = False
        last_mario_decision = None
        
        cv2.namedWindow('Mario Fluide - Claude LLM', cv2.WINDOW_AUTOSIZE)
        
        try:
            while step_count < max_steps:
                if not paused:
                    # Obtenir l'action courante
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
                        if not self.claude_thinking and len(self.action_queue) < 2:
                            print(f"Déclenchement Claude - thinking:{self.claude_thinking}, queue:{len(self.action_queue)}, step:{step_count}")
                            self.call_claude_async(situation, obs, step_count)
                        
                        # Action d'urgence si on n'a rien - ATTENTE SÉCURISÉE
                        if len(self.action_queue) == 0:
                            if not self.claude_thinking:
                                # Claude n'est pas en train de réfléchir, on peut lui demander une action d'urgence
                                emergency_action = self.get_fallback_macro()
                                self.execute_macro_action(emergency_action)
                                current_action = self.get_current_action()
                                print("🚨 Attente sécurisée - Claude réfléchit...")
                            else:
                                # Claude réfléchit, on attend en sécurité
                                current_action = 0  # NOOP - Mario reste immobile
                        else:
                            # Utiliser l'action en queue
                            current_action = self.get_current_action()
                    
                    # Exécuter l'action dans le jeu
                    if current_action is not None:
                        obs, reward, done, real_info = self.env.step(current_action)
                        total_reward += reward
                        step_count += 1
                    
                    # 🚀 DÉCLENCHEMENT HYBRIDE OPTIMISÉ: Plus fréquent avec mises à jour positionnelles
                    should_trigger_claude = (
                        len(self.action_queue) <= 1 and not self.claude_thinking and 
                        step_count - self.last_positions_update >= self.positions_update_frequency
                    )
                    
                    if should_trigger_claude:
                        trigger_type = "📍 Positions" if self.level_context_established else "📸 Initial"
                        print(f"🚀 Déclenchement hybride {trigger_type} - queue:{len(self.action_queue)}, step:{step_count}")
                        
                        situation = self.analyze_situation(obs, real_info if 'real_info' in locals() else {
                            'x_pos': 40 + step_count * 2, 'y_pos': 200, 'score': total_reward
                        }, step_count)
                        
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
                            time.sleep(3)  # Pause pour admirer la victoire
                            break
                        else:
                            # Mario est mort - mettre à jour le compteur interne
                            self.mario_lives_remaining = mario_lives_env
                            mario_x_death = situation.get('mario', {}).get('x', 0)
                            self.record_death(step_count, mario_x_death)
                            self.lives_used += 1
                            print(f"💀 Mario est mort! (Mort #{self.deaths_count}) Vies restantes: {self.mario_lives_remaining}")
                            
                            # Vérifier si c'est vraiment game over (3 morts)
                            if self.deaths_count >= 3:
                                print("💀 GAME OVER - Mario a utilisé ses 3 vies!")
                                break
                            else:
                                print("🔄 Redémarrage automatique...")
                                # Réinitialiser l'état pour la nouvelle vie
                                obs = self.env.reset()
                                self.current_macro = None
                                self.position_history.clear()
                                self.action_queue.clear()  # Vider la queue d'actions
                                time.sleep(1)  # Pause pour voir le redémarrage
                
                # Affichage fluide
                situation = self.analyze_situation(obs, real_info if 'real_info' in locals() else {
                    'x_pos': 40 + step_count * 2, 'y_pos': 200, 'score': total_reward
                }, step_count)
                
                display = self.create_display(obs, situation, last_mario_decision, total_reward, step_count)
                cv2.imshow('Mario Fluide - Claude LLM', display)
                
                # Contrôles (30 FPS pour être plus réactif avec Claude)
                key = cv2.waitKey(33) & 0xFF  # ~30 FPS pour Claude
                if key == 27:  # ESC
                    break
                elif key == 32:  # ESPACE
                    paused = not paused
                    print("⏸️ Pause" if paused else "▶️ Reprise")
                    
        except KeyboardInterrupt:
            print("\n⏹️ Arrêt demandé")
        
        finally:
            cv2.destroyAllWindows()
            self.env.close()
        
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

def main():
    print("🚀 Mario Bros FLUIDE - Claude LLM avec Macro-Actions")
    print("Mario exécute les décisions de Claude de façon naturelle!")
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY requise!")
        return
    
    try:
        mario_fluid = MarioFluidLLM()
        mario_fluid.play_fluid_mario()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()