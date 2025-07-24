#!/usr/bin/env python3
"""
Gestionnaire d'historique des actions de Mario
Sauvegarde et compare les performances pour conserver les meilleurs runs
"""

import json
import os
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import glob
import numpy as np

@dataclass
class ActionRecord:
    """Enregistrement d'une action de Mario"""
    timestamp: float  # Temps en secondes depuis le début
    step_count: int   # Numéro du step
    position_x: int   # Position X de Mario
    position_y: int   # Position Y de Mario
    action_name: str  # Nom de l'action exécutée
    reasoning: str    # Raison de l'action
    mario_speed: float # Vitesse de Mario
    score: int        # Score actuel

@dataclass 
class RunSummary:
    """Résumé d'un run de Mario"""
    run_id: str
    start_time: str
    duration: float  # Durée totale en secondes
    max_position_x: int  # Position maximale atteinte
    final_score: int
    total_steps: int
    deaths_count: int
    actions_count: int
    completion_status: str  # "victory", "death", "interrupted"
    average_speed: float  # Vitesse moyenne (position/temps)
    
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

class MarioHistoryManager:
    """Gestionnaire d'historique des performances de Mario"""
    
    def __init__(self, history_dir: str = "historic"):
        self.history_dir = history_dir
        self.current_run: List[ActionRecord] = []
        self.run_start_time = None
        self.run_id = None
        
        # Assurer que le dossier existe
        os.makedirs(history_dir, exist_ok=True)
    
    def start_new_run(self) -> str:
        """Démarrer un nouveau run et retourner l'ID"""
        self.run_start_time = time.time()
        self.run_id = f"mario_run_{int(self.run_start_time)}"
        self.current_run = []
        
        print(f"🎮 Nouveau run démarré: {self.run_id}")
        return self.run_id
    
    def record_action(self, step_count: int, position_x: int, position_y: int, 
                     action_name: str, reasoning: str = "", mario_speed: float = 0.0, 
                     score: int = 0):
        """Enregistrer une action de Mario"""
        if self.run_start_time is None:
            self.start_new_run()
        
        current_time = time.time()
        timestamp = current_time - self.run_start_time
        
        action = ActionRecord(
            timestamp=timestamp,
            step_count=step_count,
            position_x=position_x,
            position_y=position_y,
            action_name=action_name,
            reasoning=reasoning,
            mario_speed=mario_speed,
            score=score
        )
        
        self.current_run.append(action)
    
    def end_run(self, completion_status: str = "death", final_score: int = 0) -> RunSummary:
        """Terminer le run actuel et créer le résumé"""
        if not self.current_run or self.run_start_time is None:
            return None
        
        end_time = time.time()
        duration = end_time - self.run_start_time
        
        # Calculer les statistiques
        max_position_x = max(action.position_x for action in self.current_run)
        total_steps = max(action.step_count for action in self.current_run) if self.current_run else 0
        actions_count = len(self.current_run)
        
        # Calculer la vitesse moyenne (progression / temps)
        start_x = self.current_run[0].position_x if self.current_run else 0
        average_speed = (max_position_x - start_x) / duration if duration > 0 else 0
        
        # Estimer les morts (basé sur les actions répétitives à des positions similaires)
        deaths_count = self._estimate_deaths()
        
        summary = RunSummary(
            run_id=self.run_id,
            start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.run_start_time)),
            duration=duration,
            max_position_x=max_position_x,
            final_score=final_score,
            total_steps=total_steps,
            deaths_count=deaths_count,
            actions_count=actions_count,
            completion_status=completion_status,
            average_speed=average_speed
        )
        
        # Sauvegarder le run
        self._save_run(summary)
        
        # Nettoyer les anciens runs si nécessaire
        self._cleanup_inferior_runs(summary)
        
        print(f"🏁 Run terminé: {self.run_id} - Position max: {max_position_x}, Durée: {duration:.1f}s")
        
        return summary
    
    def _estimate_deaths(self) -> int:
        """Estimer le nombre de morts basé sur les changements brusques de position"""
        deaths = 0
        if len(self.current_run) < 2:
            return deaths
        
        prev_x = self.current_run[0].position_x
        for action in self.current_run[1:]:
            # Si Mario revient brutalement en arrière (probablement une mort)
            if action.position_x < prev_x - 100:  # Recul de plus de 100 pixels
                deaths += 1
            prev_x = action.position_x
        
        return deaths
    
    def _save_run(self, summary: RunSummary):
        """Sauvegarder le run sur disque"""
        # Sauvegarder les actions détaillées
        actions_file = os.path.join(self.history_dir, f"{self.run_id}_actions.json")
        actions_data = [convert_numpy_types(asdict(action)) for action in self.current_run]
        
        with open(actions_file, 'w', encoding='utf-8') as f:
            json.dump(actions_data, f, indent=2, ensure_ascii=False)
        
        # Sauvegarder le résumé
        summary_file = os.path.join(self.history_dir, f"{self.run_id}_summary.json")
        summary_data = convert_numpy_types(asdict(summary))
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Run sauvegardé: {actions_file}, {summary_file}")
    
    def _cleanup_inferior_runs(self, new_summary: RunSummary):
        """Supprimer les runs inférieurs selon les critères"""
        # Charger tous les résumés existants
        existing_summaries = self._load_all_summaries()
        
        to_delete = []
        
        for existing_summary in existing_summaries:
            if existing_summary.run_id == new_summary.run_id:
                continue  # Ne pas se comparer à soi-même
            
            should_delete = False
            
            # Critère 1 : Le nouveau run va plus loin
            if new_summary.max_position_x > existing_summary.max_position_x:
                should_delete = True
                print(f"🗑️ Suppression: {existing_summary.run_id} - distance inférieure ({existing_summary.max_position_x} < {new_summary.max_position_x})")
            
            # Critère 2 : Distance similaire mais plus rapide
            elif abs(new_summary.max_position_x - existing_summary.max_position_x) <= 50:  # Distance similaire
                if new_summary.average_speed > existing_summary.average_speed * 1.1:  # 10% plus rapide
                    should_delete = True
                    print(f"🗑️ Suppression: {existing_summary.run_id} - plus lent ({existing_summary.average_speed:.2f} < {new_summary.average_speed:.2f})")
            
            if should_delete:
                to_delete.append(existing_summary.run_id)
        
        # Supprimer les fichiers
        for run_id in to_delete:
            self._delete_run_files(run_id)
    
    def _load_all_summaries(self) -> List[RunSummary]:
        """Charger tous les résumés existants"""
        summaries = []
        summary_files = glob.glob(os.path.join(self.history_dir, "*_summary.json"))
        
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summary = RunSummary(**data)
                    summaries.append(summary)
            except Exception as e:
                print(f"⚠️ Erreur chargement {summary_file}: {e}")
        
        return summaries
    
    def _delete_run_files(self, run_id: str):
        """Supprimer les fichiers d'un run"""
        files_to_delete = [
            os.path.join(self.history_dir, f"{run_id}_actions.json"),
            os.path.join(self.history_dir, f"{run_id}_summary.json")
        ]
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Fichier supprimé: {file_path}")
    
    def get_best_run(self) -> Optional[RunSummary]:
        """Obtenir le meilleur run (distance maximale)"""
        summaries = self._load_all_summaries()
        if not summaries:
            return None
        
        return max(summaries, key=lambda s: s.max_position_x)
    
    def get_fastest_run(self) -> Optional[RunSummary]:
        """Obtenir le run le plus rapide (vitesse moyenne)"""
        summaries = self._load_all_summaries()
        if not summaries:
            return None
        
        return max(summaries, key=lambda s: s.average_speed)
    
    def get_run_stats(self) -> Dict:
        """Obtenir les statistiques générales"""
        summaries = self._load_all_summaries()
        if not summaries:
            return {"total_runs": 0}
        
        return {
            "total_runs": len(summaries),
            "best_distance": max(s.max_position_x for s in summaries),
            "best_speed": max(s.average_speed for s in summaries),
            "total_playtime": sum(s.duration for s in summaries),
            "completion_rates": {
                "victory": len([s for s in summaries if s.completion_status == "victory"]),
                "death": len([s for s in summaries if s.completion_status == "death"]),
                "interrupted": len([s for s in summaries if s.completion_status == "interrupted"])
            }
        }
    
    def load_run_actions(self, run_id: str) -> List[ActionRecord]:
        """Charger les actions détaillées d'un run"""
        actions_file = os.path.join(self.history_dir, f"{run_id}_actions.json")
        
        if not os.path.exists(actions_file):
            return []
        
        try:
            with open(actions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [ActionRecord(**action_data) for action_data in data]
        except Exception as e:
            print(f"⚠️ Erreur chargement actions {run_id}: {e}")
            return []
    
    def print_run_summary(self, summary: RunSummary):
        """Afficher un résumé de run de façon lisible"""
        print(f"\n📊 RÉSUMÉ DU RUN: {summary.run_id}")
        print(f"   🕐 Démarré: {summary.start_time}")
        print(f"   ⏱️ Durée: {summary.duration:.1f}s")
        print(f"   🏁 Distance max: {summary.max_position_x} pixels")
        print(f"   🏆 Score final: {summary.final_score}")
        print(f"   👣 Steps totaux: {summary.total_steps}")
        print(f"   💀 Morts: {summary.deaths_count}")
        print(f"   🎮 Actions: {summary.actions_count}")
        print(f"   🚀 Vitesse moy: {summary.average_speed:.2f} px/s")
        print(f"   🎯 Statut: {summary.completion_status}")
    
    def get_available_runs_for_replay(self) -> List[RunSummary]:
        """Obtenir les runs disponibles pour le replay, triés par performance"""
        summaries = self._load_all_summaries()
        # Trier par distance maximale (meilleurs en premier)
        return sorted(summaries, key=lambda s: s.max_position_x, reverse=True)
    
    def create_replay_run(self, original_run_id: str) -> str:
        """Créer un nouveau run basé sur un run existant (pour replay)"""
        self.run_start_time = time.time()
        self.run_id = f"replay_{original_run_id}_{int(self.run_start_time)}"
        self.current_run = []
        
        print(f"🔄 Nouveau run replay démarré: {self.run_id} (basé sur {original_run_id})")
        return self.run_id