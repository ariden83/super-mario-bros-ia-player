#!/usr/bin/env python3
"""
Test rapide du panneau LLM pour vérifier l'affichage
"""

import cv2
import numpy as np
import time

# Simulation de la classe MarioLLM avec le panneau
class TestMarioLLM:
    def __init__(self):
        # Interface - Historique des instructions LLM
        self.llm_responses_history = []
        self.current_claude_analysis = ""
        self.current_claude_actions = []
        self.max_llm_history = 5
        self.claude_thinking = False
        self.action_queue = []
        
        # Ajouter des données de test
        self.add_test_data()
    
    def add_test_data(self):
        """Ajouter des données de test pour l'affichage"""
        test_entry = {
            'step': 150,
            'timestamp': time.strftime("%H:%M:%S"),
            'analysis': "Mario est au sol, à environ 40px de la gauche. Un Goomba visible à 30px à droite...",
            'danger': "Oui, le Goomba à 30px représente un danger immédiat",
            'strategy': "Sécuriser zone, collecter power-up",
            'actions': ['wait', 'micro_step_right', 'hit_block'],
            'reasoning': "Se positionner précisément sous le bloc question mark"
        }
        self.llm_responses_history.append(test_entry)
    
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
    
    def draw_llm_panel(self, canvas, step_count):
        """Dessiner le panneau des instructions LLM en temps réel"""
        
        # Couleurs
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        YELLOW = (0, 255, 255)
        CYAN = (255, 255, 0)
        ORANGE = (0, 165, 255)
        RED = (0, 0, 255)
        
        # Zone du panneau LLM (en bas)
        panel_start_y = 700
        panel_height = 120
        
        # Fond du panneau
        cv2.rectangle(canvas, (20, panel_start_y), (980, panel_start_y + panel_height), (40, 40, 40), -1)
        cv2.rectangle(canvas, (20, panel_start_y), (980, panel_start_y + panel_height), WHITE, 2)
        
        # Titre du panneau
        cv2.putText(canvas, "🧠 INSTRUCTIONS CLAUDE LLM EN TEMPS RÉEL", 
                   (30, panel_start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
        
        y_pos = panel_start_y + 45
        
        # Colonne de gauche : Dernière analyse
        if self.llm_responses_history:
            latest = self.llm_responses_history[-1]
            
            # Timestamp et step
            time_text = f"⏰ {latest['timestamp']} | Step: {latest['step']}"
            cv2.putText(canvas, time_text, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
            
            # Actions actuelles
            actions_text = f"🎯 Actions: {', '.join(latest['actions'][:3])}"  # Max 3 actions
            cv2.putText(canvas, actions_text, (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
            
            # Stratégie
            strategy_text = f"📋 Stratégie: {latest['strategy']}"
            cv2.putText(canvas, strategy_text, (450, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)
            
            y_pos += 18
            
            # Analyse spatiale
            analysis_lines = self.wrap_text(f"🔍 Analyse: {latest['analysis']}", 70)
            for i, line in enumerate(analysis_lines[:1]):  # Une seule ligne
                cv2.putText(canvas, line, (30, y_pos + i * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)
            
            y_pos += 18
            
            # Évaluation de danger
            danger_color = RED if "oui" in latest['danger'].lower() else GREEN
            danger_lines = self.wrap_text(f"⚠️ Danger: {latest['danger']}", 50)
            for i, line in enumerate(danger_lines[:1]):
                cv2.putText(canvas, line, (30, y_pos + i * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, danger_color, 1)
            
            # Raisonnement (colonne droite)
            reasoning_lines = self.wrap_text(f"💭 {latest['reasoning']}", 45)
            for i, line in enumerate(reasoning_lines[:2]):  # Max 2 lignes
                cv2.putText(canvas, line, (500, y_pos - 18 + i * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ORANGE, 1)
        
        else:
            # Pas d'historique LLM encore
            cv2.putText(canvas, "⏳ En attente des premières instructions de Claude...", 
                       (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        
        # Séparateur vertical
        cv2.line(canvas, (480, panel_start_y + 35), (480, panel_start_y + panel_height - 10), WHITE, 1)
        
        # Indicateur d'état Claude
        thinking_status = "🧠 RÉFLÉCHIT..." if self.claude_thinking else "⚡ ACTIF"
        status_color = ORANGE if self.claude_thinking else GREEN
        cv2.putText(canvas, thinking_status, (850, panel_start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Queue d'actions
        queue_size = len(self.action_queue)
        cv2.putText(canvas, f"📋 Queue: {queue_size} actions", (750, panel_start_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)

def test_panel():
    """Tester l'affichage du panneau LLM"""
    mario_llm = TestMarioLLM()
    
    # Créer une image de test
    canvas = np.zeros((850, 1000, 3), dtype=np.uint8)
    
    # Zone de jeu simulée
    cv2.rectangle(canvas, (50, 80), (650, 560), (100, 50, 0), -1)
    cv2.putText(canvas, "ZONE DE JEU MARIO", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Dessiner le panneau LLM
    mario_llm.draw_llm_panel(canvas, 150)
    
    # Contrôles
    cv2.putText(canvas, "ESC: Quitter | Test du panneau LLM", (50, 820), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Afficher
    cv2.imshow('Test Panneau LLM', canvas)
    print("🎯 Test du panneau LLM - Appuyez sur une touche pour fermer")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_panel()