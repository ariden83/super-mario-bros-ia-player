#!/usr/bin/env python3
"""
Base de données des éléments de chaque niveau Super Mario Bros NES
Contient tous les ennemis, blocs, power-ups et obstacles par niveau
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Enemy:
    """Données d'un ennemi"""
    name: str
    behavior: str
    speed: float
    threat_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    defeat_methods: List[str]
    points: int
    special_notes: str = ""

@dataclass
class Block:
    """Données d'un bloc interactif"""
    name: str
    contents: str
    behavior: str
    breakable: bool
    special_notes: str = ""

@dataclass
class PowerUp:
    """Données d'un power-up"""
    name: str
    effect: str
    rarity: str  # COMMON, RARE, VERY_RARE
    points: int
    special_notes: str = ""

@dataclass
class Obstacle:
    """Données d'un obstacle"""
    name: str
    behavior: str
    avoidance_strategy: str
    threat_level: str
    special_notes: str = ""

@dataclass
class LevelData:
    """Données complètes d'un niveau"""
    world: int
    level: int
    level_type: str  # OVERWORLD, UNDERGROUND, UNDERWATER, CASTLE
    time_limit: int
    enemies: List[Enemy]
    blocks: List[Block]
    power_ups: List[PowerUp]
    obstacles: List[Obstacle]
    special_features: List[str]
    background_music: str
    completion_strategy: str

class MarioLevelDatabase:
    """Base de données complète des niveaux Super Mario Bros"""
    
    def __init__(self):
        self.levels = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialiser la base de données avec tous les niveaux"""
        
        # === WORLD 1 ===
        
        # World 1-1 (Premier niveau)
        self.levels["1-1"] = LevelData(
            world=1, level=1,
            level_type="OVERWORLD",
            time_limit=400,
            enemies=[
                Enemy("Goomba", "Marche vers Mario à ~0.5px/step", 0.5, "LOW", 
                      ["stomp_enemy", "fireball"], 100, "Premier ennemi rencontré"),
                Enemy("Koopa Troopa", "Marche vers Mario, devient carapace", 0.7, "MEDIUM",
                      ["stomp_enemy", "fireball", "kick_shell"], 100, "Carapace réutilisable")
            ],
            blocks=[
                Block("Question Block", "Coin/Power-up", "Frappe par dessous", False, "Blocs bleus"),
                Block("Brick Block", "Coin/Power-up/Vide", "Cassable par Super Mario", True, "Blocs bruns"),
                Block("Hidden Block", "1-Up/Coin", "Invisible jusqu'à frappe", False, "Position précise requise")
            ],
            power_ups=[
                PowerUp("Magic Mushroom", "Devient Super Mario", "COMMON", 1000, "Première transformation"),
                PowerUp("Fire Flower", "Devient Fire Mario", "COMMON", 1000, "Si déjà Super Mario"),
                PowerUp("Starman", "Invincibilité temporaire", "RARE", 1000, "Dans bloc brick"),
                PowerUp("1-Up Mushroom", "Vie supplémentaire", "VERY_RARE", 0, "Bloc invisible")
            ],
            obstacles=[
                Obstacle("Pit", "Trou mortel", "Saut précis", "CRITICAL", "Mort instantanée"),
                Obstacle("Pipe", "Bloque passage", "Saut par-dessus", "LOW", "Certains accessibles")
            ],
            special_features=[
                "Premier niveau tutoriel",
                "Warp zone cachée",
                "Bloc 10 coins",
                "Formation pyramidale de blocs"
            ],
            background_music="Ground Theme",
            completion_strategy="Apprendre les bases, collecter power-ups, éviter ennemis"
        )
        
        # World 1-2 (Premier niveau souterrain) 
        self.levels["1-2"] = LevelData(
            world=1, level=2,
            level_type="UNDERGROUND", 
            time_limit=400,
            enemies=[
                Enemy("Goomba", "Marche vers Mario", 0.5, "LOW", ["stomp_enemy", "fireball"], 100, "Couleur teal souterrain"),
                Enemy("Koopa Troopa", "Marche vers Mario", 0.7, "MEDIUM", ["stomp_enemy", "fireball"], 100, "Plus dangereux en souterrain")
            ],
            blocks=[
                Block("Question Block", "Power-up/Coin", "5 blocs au début", False, "Premier contient power-up"),
                Block("Brick Block", "10 coins/Starman", "Tour de blocs", True, "Bloc spécial 10 coins"),
                Block("Lift Platform", "Transport vertical", "Monte/descend automatiquement", False, "Timing crucial")
            ],
            power_ups=[
                PowerUp("Magic Mushroom", "Super Mario", "COMMON", 1000, "Dans premier ? bloc"),
                PowerUp("Fire Flower", "Fire Mario", "COMMON", 1000, "Si déjà grand"),
                PowerUp("Starman", "Invincibilité", "RARE", 1000, "Dans brick block spécifique")
            ],
            obstacles=[
                Obstacle("Moving Lifts", "Plateformes mobiles", "Timing des sauts", "MEDIUM", "Une descend, une monte")
            ],
            special_features=[
                "Premier niveau souterrain",
                "Warp Zone (pipes vers mondes 2, 3, 4)",
                "Passage secret vers Minus World",
                "Lifts mobiles introduction"
            ],
            background_music="Underground Theme",
            completion_strategy="Utiliser lifts pour accéder warp zone cachée"
        )
        
        # World 1-3 (Niveau arbres)
        self.levels["1-3"] = LevelData(
            world=1, level=3,
            level_type="OVERWORLD",
            time_limit=300,
            enemies=[
                Enemy("Goomba", "Marche sur plateformes", 0.5, "LOW", ["stomp_enemy", "fireball"], 100, "Sur arbres"),
                Enemy("Koopa Troopa", "Marche sur plateformes", 0.7, "MEDIUM", ["stomp_enemy", "fireball"], 100, "Chute mortelle possible")
            ],
            blocks=[
                Block("Tree Platform", "Plateforme", "Support solide", False, "Différentes hauteurs"),
                Block("Question Block", "Coin/Power-up", "Sur arbres", False, "Positions élevées")
            ],
            power_ups=[
                PowerUp("Magic Mushroom", "Super Mario", "COMMON", 1000, "Dans blocs ?"),
                PowerUp("Fire Flower", "Fire Mario", "COMMON", 1000, "Dans blocs ?")
            ],
            obstacles=[
                Obstacle("Tree Gaps", "Espaces entre arbres", "Saut précis", "HIGH", "Chute mortelle"),
                Obstacle("Height Variations", "Plateformes différentes", "Navigation verticale", "MEDIUM", "Planification requise")
            ],
            special_features=[
                "Niveau plateforme arbre",
                "Pas de sol continu",
                "Navigation verticale importante",
                "Flagpole final"
            ],
            background_music="Ground Theme",
            completion_strategy="Navigation prudente entre arbres, éviter chutes"
        )
        
        # World 1-4 (Premier château)
        self.levels["1-4"] = LevelData(
            world=1, level=4,
            level_type="CASTLE",
            time_limit=300,
            enemies=[
                Enemy("Fake Bowser", "Respire feu, saute", 1.0, "CRITICAL", ["axe", "fireball"], 5000, "En réalité un Goomba déguisé"),
                Enemy("Goomba", "Marche vers Mario", 0.5, "LOW", ["stomp_enemy", "fireball"], 100, "Couleur grise château")
            ],
            blocks=[
                Block("Brick Block", "Vide généralement", "Cassable", True, "Décoration château"),
                Block("Axe", "Termine niveau", "Coupe pont", False, "Objectif principal")
            ],
            power_ups=[
                PowerUp("Magic Mushroom", "Super Mario", "RARE", 1000, "Très peu présents"),
                PowerUp("Fire Flower", "Fire Mario", "RARE", 1000, "Efficace contre Bowser")
            ],
            obstacles=[
                Obstacle("Fire Bar", "Barres de feu rotatives", "Timing précis", "HIGH", "6 boules de feu"),
                Obstacle("Lava", "Lave en dessous", "Ne pas tomber", "CRITICAL", "Mort instantanée"),
                Obstacle("Bridge", "Pont de Bowser", "Atteindre axe", "MEDIUM", "Objectif final")
            ],
            special_features=[
                "Premier château",
                "Premier boss (faux Bowser)",
                "Fire Bars introduction",
                "Axe coupe pont",
                "Princess saved (faux)"
            ],
            background_music="Castle Theme",
            completion_strategy="Éviter Fire Bars, courir sous Bowser quand il saute, atteindre axe"
        )
        
        # === WORLD 2 ===
        
        # World 2-1
        self.levels["2-1"] = LevelData(
            world=2, level=1,
            level_type="OVERWORLD",
            time_limit=400,
            enemies=[
                Enemy("Goomba", "Marche vers Mario", 0.5, "LOW", ["stomp_enemy", "fireball"], 100, "Plus nombreux"),
                Enemy("Koopa Troopa", "Marche vers Mario", 0.7, "MEDIUM", ["stomp_enemy", "fireball"], 100, "Vert et rouge"),
                Enemy("Piranha Plant", "Sort de tuyau", 0.0, "HIGH", ["fireball", "avoid"], 200, "Ne sort pas si Mario près")
            ],
            blocks=[
                Block("Question Block", "Coin/Power-up", "Dispersés", False, "Plus nombreux"),
                Block("Brick Block", "Divers contenus", "Cassable", True, "Formations complexes")
            ],
            power_ups=[
                PowerUp("Magic Mushroom", "Super Mario", "COMMON", 1000, "Standard"),
                PowerUp("Fire Flower", "Fire Mario", "COMMON", 1000, "Efficace contre Piranha"),
                PowerUp("Starman", "Invincibilité", "RARE", 1000, "Placement stratégique")
            ],
            obstacles=[
                Obstacle("Pipe avec Piranha", "Tuyau avec plante", "Timing ou feu", "HIGH", "Attendre qu'elle rentre"),
                Obstacle("Pit", "Trous", "Saut précis", "CRITICAL", "Plus larges")
            ],
            special_features=[
                "Introduction Piranha Plant",
                "Tuyaux plus complexes",
                "Difficulté accrue"
            ],
            background_music="Ground Theme",
            completion_strategy="Patience avec Piranha Plants, utiliser Fire Flower"
        )
        
        # World 2-2 (Aquatique)
        self.levels["2-2"] = LevelData(
            world=2, level=2,
            level_type="UNDERWATER",
            time_limit=400,
            enemies=[
                Enemy("Bloober", "Nage vers Mario", 0.3, "MEDIUM", ["fireball"], 200, "Mouvement ondulant"),
                Enemy("Cheep-Cheep", "Nage horizontalement", 1.0, "LOW", ["fireball"], 100, "Poisson basique")
            ],
            blocks=[
                Block("Coral", "Décoration", "Indestructible", False, "Obstacles naturels"),
                Block("Question Block", "Power-up", "Rares", False, "Power-ups limités")
            ],
            power_ups=[
                PowerUp("Fire Flower", "Fire Mario", "RARE", 1000, "Fonctionne sous l'eau"),
                PowerUp("Magic Mushroom", "Super Mario", "RARE", 1000, "Très rare sous l'eau")
            ],
            obstacles=[
                Obstacle("Water Current", "Courant d'eau", "Contrôle difficile", "MEDIUM", "Inertie modifiée"),
                Obstacle("Coral Formations", "Coraux", "Navigation complexe", "MEDIUM", "Chemins étroits")
            ],
            special_features=[
                "Premier niveau aquatique",
                "Physique modifiée (natation)",
                "Ennemis spécifiques eau",
                "Fire Flower fonctionne sous l'eau"
            ],
            background_music="Underwater Theme", 
            completion_strategy="Maîtriser physique natation, utiliser Fire Flower"
        )
        
        # === WORLD 3 et plus (données essentielles) ===
        
        # World 3-1 (Introduction Hammer Bro)
        self.levels["3-1"] = LevelData(
            world=3, level=1,
            level_type="OVERWORLD", 
            time_limit=400,
            enemies=[
                Enemy("Hammer Bro", "Lance marteaux, saute", 0.8, "HIGH", ["stomp_enemy", "fireball"], 1000, "Premier apparition"),
                Enemy("Goomba", "Standard", 0.5, "LOW", ["stomp_enemy", "fireball"], 100, "Support"),
                Enemy("Koopa Troopa", "Standard", 0.7, "MEDIUM", ["stomp_enemy", "fireball"], 100, "Support")
            ],
            blocks=[
                Block("Brick Block", "Formations défensives", "Utilisées par Hammer Bro", True, "Protection ennemis"),
                Block("Question Block", "Power-ups", "Stratégiques", False, "Aide contre Hammer Bro")
            ],
            power_ups=[
                PowerUp("Fire Flower", "Fire Mario", "COMMON", 1000, "Essentiel vs Hammer Bro"),
                PowerUp("Starman", "Invincibilité", "RARE", 1000, "Solution facile")
            ],
            obstacles=[
                Obstacle("Hammer Projectiles", "Marteaux volants", "Esquive timing", "HIGH", "Trajectoire prévisible")
            ],
            special_features=[
                "Introduction Hammer Bro",
                "Combat tactique requis",
                "Apprentissage projectiles"
            ],
            background_music="Ground Theme",
            completion_strategy="Fire Flower ou approche prudente par dessous"
        )
        
        # World 4-2 (Introduction Buzzy Beetle)
        self.levels["4-2"] = LevelData(
            world=4, level=2,
            level_type="UNDERGROUND",
            time_limit=400,
            enemies=[
                Enemy("Buzzy Beetle", "Résistant au feu", 0.5, "MEDIUM", ["stomp_enemy"], 100, "Premier apparition, résiste Fire Flower"),
                Enemy("Goomba", "Souterrain", 0.5, "LOW", ["stomp_enemy", "fireball"], 100, "Couleur teal")
            ],
            blocks=[
                Block("Question Block", "Power-ups", "Standard", False, "Moins utiles ici"),
                Block("Pipe", "Transport/obstacle", "Certains accessibles", False, "1-up caché dessus")
            ],
            power_ups=[
                PowerUp("Magic Mushroom", "Super Mario", "COMMON", 1000, "Plus utile que Fire Flower"),
                PowerUp("1-Up Mushroom", "Vie extra", "VERY_RARE", 0, "Au-dessus tuyau")
            ],
            obstacles=[
                Obstacle("Buzzy Beetle Shell", "Carapace robuste", "Stomp seulement", "MEDIUM", "Fire Flower inefficace")
            ],
            special_features=[
                "Introduction Buzzy Beetle", 
                "Fire Flower partiellement inefficace",
                "1-up secret",
                "Nouvelles tactiques requises"
            ],
            background_music="Underground Theme",
            completion_strategy="Stomp au lieu de Fire Flower, chercher 1-up"
        )
        
        # Ajouter les niveaux critiques suivants...
        
        # World 8-4 (Boss final)
        self.levels["8-4"] = LevelData(
            world=8, level=4,
            level_type="CASTLE",
            time_limit=300,
            enemies=[
                Enemy("Real Bowser", "Feu + marteaux", 1.2, "CRITICAL", ["axe", "fireball"], 5000, "Vrai boss final"),
                Enemy("Hammer Bro", "Garde avant boss", 0.8, "HIGH", ["stomp_enemy", "fireball"], 1000, "Garde final")
            ],
            blocks=[
                Block("Brick Block", "Labyrinthe", "Navigation complexe", True, "Château complexe"),
                Block("Axe", "Victoire finale", "Coupe pont final", False, "Vraie fin")
            ],
            power_ups=[
                PowerUp("Fire Flower", "Fire Mario", "RARE", 1000, "Essentiel pour boss"),
                PowerUp("Magic Mushroom", "Super Mario", "RARE", 1000, "Protection vitale")
            ],
            obstacles=[
                Obstacle("Fire Bar", "12 boules max", "Navigation expert", "CRITICAL", "Plus long du jeu"),
                Obstacle("Lava", "Lave partout", "Précision absolue", "CRITICAL", "Mort certaine"),
                Obstacle("Maze", "Labyrinthe", "Mémorisation chemin", "HIGH", "Plusieurs faux chemins")
            ],
            special_features=[
                "Boss final (vrai Bowser)",
                "Château le plus complexe", 
                "Fire Bars géants",
                "Zone aquatique dans château",
                "Vraie Princess à sauver"
            ],
            background_music="Castle Theme",
            completion_strategy="Mémoriser labyrinthe, Fire Flower obligatoire, timing parfait"
        )
    
    def get_level_data(self, world: int, level: int) -> LevelData:
        """Obtenir les données d'un niveau spécifique"""
        level_key = f"{world}-{level}"
        return self.levels.get(level_key)
    
    def get_enemies_for_level(self, world: int, level: int) -> List[Enemy]:
        """Obtenir les ennemis d'un niveau"""
        level_data = self.get_level_data(world, level)
        return level_data.enemies if level_data else []
    
    def get_blocks_for_level(self, world: int, level: int) -> List[Block]:
        """Obtenir les blocs d'un niveau"""
        level_data = self.get_level_data(world, level)
        return level_data.blocks if level_data else []
    
    def get_level_strategy(self, world: int, level: int) -> str:
        """Obtenir la stratégie recommandée pour un niveau"""
        level_data = self.get_level_data(world, level)
        return level_data.completion_strategy if level_data else "Stratégie générique"
    
    def get_level_type(self, world: int, level: int) -> str:
        """Obtenir le type de niveau"""
        level_data = self.get_level_data(world, level)
        return level_data.level_type if level_data else "OVERWORLD"
    
    def get_threat_analysis(self, world: int, level: int) -> Dict:
        """Analyser les menaces d'un niveau"""
        enemies = self.get_enemies_for_level(world, level)
        
        threat_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        high_value_targets = []
        
        for enemy in enemies:
            threat_counts[enemy.threat_level] += 1
            if enemy.points >= 1000:
                high_value_targets.append(enemy.name)
        
        return {
            "threat_distribution": threat_counts,
            "high_value_targets": high_value_targets,
            "max_threat_level": max(threat_counts.keys(), key=lambda x: threat_counts[x]) if any(threat_counts.values()) else "LOW"
        }
    
    def get_recommended_powerups(self, world: int, level: int) -> List[str]:
        """Recommander les power-ups optimaux pour un niveau"""
        level_data = self.get_level_data(world, level)
        if not level_data:
            return ["Magic Mushroom", "Fire Flower"]
        
        # Analyser les ennemis pour recommander
        recommendations = []
        
        # Fire Flower recommandé si beaucoup d'ennemis
        enemy_count = len(level_data.enemies)
        if enemy_count > 2:
            recommendations.append("Fire Flower")
        
        # Starman pour niveaux dangereux
        threat_analysis = self.get_threat_analysis(world, level)
        if threat_analysis["threat_distribution"]["HIGH"] > 0 or threat_analysis["threat_distribution"]["CRITICAL"] > 0:
            recommendations.append("Starman")
        
        # Mushroom toujours recommandé
        recommendations.append("Magic Mushroom")
        
        return recommendations
    
    def detect_current_level(self, mario_x: int, step_count: int, level_features: Dict) -> Tuple[int, int]:
        """Détecter le niveau actuel basé sur les données de jeu"""
        # Heuristiques pour détecter le niveau
        
        # Pour l'instant, assume World 1-1 par défaut
        # Cette méthode serait étendue avec plus de logique de détection
        return (1, 1)
    
    def available_levels(self) -> List[str]:
        """Retourner la liste des niveaux disponibles"""
        return list(self.levels.keys())