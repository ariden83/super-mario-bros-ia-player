#!/usr/bin/env python3
"""
Système de logging complet pour Mario
Enregistre toutes les actions, prompts, réponses et événements
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import json
import numpy as np

def convert_numpy_types(obj):
    """Convertir les types NumPy en types Python standard pour JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class MarioLogger:
    """Logger spécialisé pour Mario avec différents niveaux de détail"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.session_id = f"mario_session_{int(time.time())}"
        
        # Créer le dossier de logs
        os.makedirs(log_dir, exist_ok=True)
        
        # Configuration des loggers
        self.setup_loggers()
        
        print(f"📝 Logging activé - Session: {self.session_id}")
    
    def setup_loggers(self):
        """Configurer les différents loggers"""
        
        # Format des logs
        log_format = '%(asctime)s | %(levelname)8s | %(name)15s | %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Logger principal (toutes les activités)
        self.main_logger = logging.getLogger('mario.main')
        self.main_logger.setLevel(logging.INFO)
        
        main_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.session_id}_main.log"),
            encoding='utf-8'
        )
        main_handler.setFormatter(logging.Formatter(log_format, date_format))
        self.main_logger.addHandler(main_handler)
        
        # Logger pour les actions de Mario
        self.action_logger = logging.getLogger('mario.actions')
        self.action_logger.setLevel(logging.INFO)
        
        action_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.session_id}_actions.log"),
            encoding='utf-8'
        )
        action_handler.setFormatter(logging.Formatter(log_format, date_format))
        self.action_logger.addHandler(action_handler)
        
        # Logger pour Claude (prompts + réponses)
        self.claude_logger = logging.getLogger('mario.claude')
        self.claude_logger.setLevel(logging.INFO)
        
        claude_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.session_id}_claude.log"),
            encoding='utf-8'
        )
        claude_handler.setFormatter(logging.Formatter(log_format, date_format))
        self.claude_logger.addHandler(claude_handler)
        
        # Logger pour les événements de jeu
        self.game_logger = logging.getLogger('mario.game')
        self.game_logger.setLevel(logging.INFO)
        
        game_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.session_id}_game.log"),
            encoding='utf-8'
        )
        game_handler.setFormatter(logging.Formatter(log_format, date_format))
        self.game_logger.addHandler(game_handler)
        
        # Logger pour le replay
        self.replay_logger = logging.getLogger('mario.replay')
        self.replay_logger.setLevel(logging.INFO)
        
        replay_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.session_id}_replay.log"),
            encoding='utf-8'
        )
        replay_handler.setFormatter(logging.Formatter(log_format, date_format))
        self.replay_logger.addHandler(replay_handler)
    
    def log_session_start(self, mode: str, run_id: str):
        """Logger le début d'une session"""
        self.main_logger.info(f"SESSION START - Mode: {mode} | Run ID: {run_id}")
        self.game_logger.info(f"Nouvelle session Mario - Mode: {mode}")
    
    def log_action(self, step_count: int, action_name: str, reasoning: str, 
                   position_x: int, position_y: int, score: int, source: str = "AI"):
        """Logger une action de Mario"""
        action_data = {
            'step': step_count,
            'action': action_name,
            'reasoning': reasoning,
            'position': {'x': position_x, 'y': position_y},
            'score': score,
            'source': source  # AI, REPLAY, EMERGENCY
        }
        
        self.action_logger.info(f"ACTION - Step {step_count:4d} | {source:7s} | {action_name:20s} | Pos({position_x:4d},{position_y:3d}) | Score:{score:6d} | {reasoning}")
        self.main_logger.info(f"Mario action: {action_name} at step {step_count}")
    
    def log_claude_prompt(self, prompt_type: str, prompt: str, step_count: int):
        """Logger un prompt envoyé à Claude"""
        prompt_short = prompt.replace('\n', ' ')[:100] + "..." if len(prompt) > 100 else prompt.replace('\n', ' ')
        
        self.claude_logger.info(f"PROMPT [{prompt_type}] - Step {step_count:4d} | Length: {len(prompt):4d} chars")
        self.claude_logger.info(f"PROMPT CONTENT: {prompt_short}")
        
        # Sauvegarder le prompt complet dans un fichier séparé
        prompt_file = os.path.join(self.log_dir, f"{self.session_id}_prompts_full.txt")
        with open(prompt_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"TYPE: {prompt_type} | STEP: {step_count}\n")
            f.write(f"{'='*80}\n")
            f.write(prompt)
            f.write(f"\n{'='*80}\n\n")
    
    def log_claude_response(self, response: str, step_count: int, cost: float = 0.0):
        """Logger une réponse de Claude"""
        response_short = response.replace('\n', ' ')[:100] + "..." if len(response) > 100 else response.replace('\n', ' ')
        
        self.claude_logger.info(f"RESPONSE - Step {step_count:4d} | Length: {len(response):4d} chars | Cost: ${cost:.4f}")
        self.claude_logger.info(f"RESPONSE CONTENT: {response_short}")
        
        # Sauvegarder la réponse complète
        response_file = os.path.join(self.log_dir, f"{self.session_id}_responses_full.txt")
        with open(response_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"STEP: {step_count} | COST: ${cost:.4f}\n")
            f.write(f"{'='*80}\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")
    
    def log_replay_event(self, event_type: str, details: Dict[str, Any]):
        """Logger les événements de replay"""
        self.replay_logger.info(f"REPLAY {event_type.upper()} - {details}")
        self.main_logger.info(f"Replay event: {event_type}")
    
    def log_game_event(self, event_type: str, step_count: int, details: Dict[str, Any]):
        """Logger les événements de jeu (mort, victoire, etc.)"""
        self.game_logger.info(f"GAME {event_type.upper()} - Step {step_count:4d} | {details}")
        self.main_logger.info(f"Game event: {event_type} at step {step_count}")
    
    def log_mario_state(self, step_count: int, mario_data: Dict[str, Any], 
                       progress_data: Dict[str, Any]):
        """Logger l'état détaillé de Mario"""
        state_info = f"Mario State - Step {step_count:4d} | Pos({mario_data.get('x', 0):4d},{mario_data.get('y', 0):3d}) | Score:{mario_data.get('score', 0):6d} | Progress:{progress_data.get('status', 'unknown')}"
        self.main_logger.debug(state_info)
    
    def log_screenshot_analysis(self, step_count: int, cost: float, analysis_short: str):
        """Logger les analyses de screenshots"""
        self.claude_logger.info(f"SCREENSHOT - Step {step_count:4d} | Cost: ${cost:.4f} | Analysis: {analysis_short}")
    
    def log_error(self, error_type: str, error_msg: str, step_count: int = 0):
        """Logger les erreurs"""
        self.main_logger.error(f"ERROR [{error_type}] - Step {step_count:4d} | {error_msg}")
        self.game_logger.error(f"Error: {error_type} - {error_msg}")
    
    def log_session_end(self, final_stats: Dict[str, Any]):
        """Logger la fin de session avec statistiques"""
        self.main_logger.info(f"SESSION END - Final stats: {final_stats}")
        self.game_logger.info(f"Session terminée - Statistiques finales")
        
        # Créer un résumé de session
        summary_file = os.path.join(self.log_dir, f"{self.session_id}_summary.json")
        summary_data = convert_numpy_types({
            'session_id': self.session_id,
            'end_time': datetime.now().isoformat(),
            'final_stats': final_stats
        })
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    def log_menu_choice(self, choice: str, selected_run: Optional[str] = None):
        """Logger les choix du menu"""
        menu_info = f"Menu choice: {choice}"
        if selected_run:
            menu_info += f" | Selected run: {selected_run}"
        
        self.main_logger.info(menu_info)
        self.game_logger.info(f"Menu: {choice}")
    
    def log_ai_takeover(self, step_count: int, replay_progress: str):
        """Logger la transition replay -> IA"""
        self.replay_logger.info(f"AI TAKEOVER - Step {step_count:4d} | Replay progress: {replay_progress}")
        self.main_logger.info(f"Transition replay -> IA at step {step_count}")
    
    def get_session_files(self):
        """Retourner la liste des fichiers de log de cette session"""
        files = [
            f"{self.session_id}_main.log",
            f"{self.session_id}_actions.log", 
            f"{self.session_id}_claude.log",
            f"{self.session_id}_game.log",
            f"{self.session_id}_replay.log",
            f"{self.session_id}_prompts_full.txt",
            f"{self.session_id}_responses_full.txt",
            f"{self.session_id}_summary.json"
        ]
        
        return [os.path.join(self.log_dir, f) for f in files if os.path.exists(os.path.join(self.log_dir, f))]
    
    def close(self):
        """Fermer proprement les loggers"""
        for logger in [self.main_logger, self.action_logger, self.claude_logger, 
                      self.game_logger, self.replay_logger]:
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)