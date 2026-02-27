#!/usr/bin/env python3
"""
Utilitaire pour convertir les distances dans les réponses de Claude
et assurer la cohérence entre screenshot et pixels de jeu.
"""

import re
import json
from typing import Dict, Any, Optional

class DistanceConverter:
    """Convertit automatiquement les distances des réponses Claude"""
    
    def __init__(self, scale_factor: float = 1.0):
        self.scale_factor = scale_factor
        
    def update_scale_factor(self, scale_factor: float):
        """Met à jour le facteur d'échelle"""
        self.scale_factor = scale_factor
        
    def convert_distance_in_text(self, text: str) -> str:
        """Convertit les distances trouvées dans un texte"""
        # Pattern pour trouver les distances en pixels
        # Ex: "15px", "30 pixels", "distance: 45px"
        patterns = [
            r'(\d+(?:\.\d+)?)\s*px(?:els?)?',
            r'(\d+(?:\.\d+)?)\s*pixels?',
            r'distance[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*units?'
        ]
        
        converted_text = text
        for pattern in patterns:
            def replace_distance(match):
                distance = float(match.group(1))
                converted = distance * self.scale_factor
                return f"{converted:.1f}px_jeu"
            
            converted_text = re.sub(pattern, replace_distance, converted_text, flags=re.IGNORECASE)
            
        return converted_text
    
    def convert_json_distances(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convertit les distances dans un objet JSON"""
        if not isinstance(json_data, dict):
            return json_data
            
        converted = json_data.copy()
        
        # Champs qui contiennent des distances à convertir
        distance_fields = [
            'spatial_analysis',
            'immediate_danger', 
            'next_target',
            'threat_analysis',
            'reasoning'
        ]
        
        for field in distance_fields:
            if field in converted and isinstance(converted[field], str):
                converted[field] = self.convert_distance_in_text(converted[field])
                
        # Traitement spécial pour les actions avec reasoning
        if 'actions' in converted and isinstance(converted['actions'], list):
            for action in converted['actions']:
                if isinstance(action, dict) and 'reasoning' in action:
                    action['reasoning'] = self.convert_distance_in_text(action['reasoning'])
                    
        return converted
    
    def process_claude_response(self, response_text: str) -> str:
        """Traite une réponse complète de Claude pour convertir les distances"""
        if not response_text:
            return response_text
            
        # Essayer d'extraire et convertir le JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                json_text = json_match.group()
                json_data = json.loads(json_text)
                converted_json = self.convert_json_distances(json_data)
                
                # Remplacer le JSON dans le texte original
                converted_json_text = json.dumps(converted_json, ensure_ascii=False, indent=2)
                response_text = response_text.replace(json_text, converted_json_text)
                
            except json.JSONDecodeError:
                # Si échec JSON, convertir juste le texte
                response_text = self.convert_distance_in_text(response_text)
        else:
            # Pas de JSON, convertir tout le texte
            response_text = self.convert_distance_in_text(response_text)
            
        return response_text
    
    def add_scale_info_to_response(self, response_text: str) -> str:
        """Ajoute des informations d'échelle à la réponse"""
        scale_info = f"\n\n📐 ÉCHELLE APPLIQUÉE: Distances converties × {self.scale_factor:.2f} pour correspondre aux pixels du jeu"
        return response_text + scale_info

def test_converter():
    """Test du convertisseur de distances"""
    converter = DistanceConverter(scale_factor=1.5)
    
    # Test JSON
    test_json = {
        "actions": [{"macro_action": "run_forward", "reasoning": "Ennemi à 20px, sécurisé"}],
        "spatial_analysis": "Goomba à 15px vers la droite, bloc à 45 pixels",
        "immediate_danger": "Non, distance de 30px"
    }
    
    converted = converter.convert_json_distances(test_json)
    print("JSON converti:", json.dumps(converted, indent=2, ensure_ascii=False))
    
    # Test texte
    test_text = "Mario voit un ennemi à 20px. Le prochain bloc est à 40 pixels."
    converted_text = converter.convert_distance_in_text(test_text)
    print("Texte converti:", converted_text)

if __name__ == "__main__":
    test_converter()