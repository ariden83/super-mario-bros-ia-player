#!/usr/bin/env python3
"""
Script de test pour le système de détection de niveau et contexte spécifique
"""

from mario_level_database import MarioLevelDatabase

def test_level_database():
    """Tester la base de données des niveaux"""
    print("🧪 TEST - BASE DE DONNÉES DES NIVEAUX")
    print("="*60)
    
    db = MarioLevelDatabase()
    
    # Tester les niveaux disponibles
    levels = db.available_levels()
    print(f"📋 Niveaux disponibles: {levels}")
    
    # Tester World 1-1
    print(f"\n🗺️ TEST WORLD 1-1:")
    level_data = db.get_level_data(1, 1)
    if level_data:
        print(f"   Type: {level_data.level_type}")
        print(f"   Temps: {level_data.time_limit}s")
        print(f"   Ennemis: {len(level_data.enemies)}")
        print(f"   Blocs: {len(level_data.blocks)}")
        print(f"   Power-ups: {len(level_data.power_ups)}")
        print(f"   Stratégie: {level_data.completion_strategy}")
    
    # Tester analyse des menaces
    print(f"\n⚠️ ANALYSE DES MENACES 1-1:")
    threats = db.get_threat_analysis(1, 1)
    print(f"   Distribution: {threats['threat_distribution']}")
    print(f"   Niveau max: {threats['max_threat_level']}")
    print(f"   Cibles prioritaires: {threats['high_value_targets']}")
    
    # Tester recommandations power-ups
    print(f"\n💊 POWER-UPS RECOMMANDÉS 1-1:")
    recommended = db.get_recommended_powerups(1, 1)
    print(f"   {recommended}")
    
    # Tester World 1-4 (château)
    print(f"\n🏰 TEST WORLD 1-4 (Château):")
    castle_data = db.get_level_data(1, 4)
    if castle_data:
        print(f"   Type: {castle_data.level_type}")
        print(f"   Ennemis: {[e.name for e in castle_data.enemies]}")
        print(f"   Obstacles: {[o.name for o in castle_data.obstacles]}")

def test_level_context_generation():
    """Tester la génération du contexte de niveau"""
    print(f"\n🧠 TEST - GÉNÉRATION DU CONTEXTE")
    print("="*60)
    
    # Simuler une instance MarioFluidLLM minimale
    class TestMario:
        def __init__(self):
            self.level_db = MarioLevelDatabase()
            self.current_world = 1
            self.current_level = 1
        
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
            
            # Stratégie recommandée
            context_parts.append(f"\n🎯 STRATÉGIE RECOMMANDÉE:")
            context_parts.append(f"   📋 {level_data.completion_strategy}")
            
            return "\n".join(context_parts)
    
    test_mario = TestMario()
    
    # Tester World 1-1
    print("📋 CONTEXTE WORLD 1-1:")
    context_1_1 = test_mario.get_level_specific_context()
    print(context_1_1)
    
    # Tester World 1-4
    print(f"\n📋 CONTEXTE WORLD 1-4:")
    test_mario.current_world = 1
    test_mario.current_level = 4
    context_1_4 = test_mario.get_level_specific_context()
    print(context_1_4)

def test_enemy_specific_info():
    """Tester les informations spécifiques aux ennemis"""
    print(f"\n👾 TEST - INFORMATIONS ENNEMIS SPÉCIFIQUES")
    print("="*60)
    
    db = MarioLevelDatabase()
    
    # Tester différents niveaux pour voir les ennemis
    test_levels = [(1, 1), (1, 2), (1, 4), (3, 1), (4, 2)]
    
    for world, level in test_levels:
        enemies = db.get_enemies_for_level(world, level)
        print(f"\n🗺️ World {world}-{level}:")
        
        if enemies:
            for enemy in enemies:
                print(f"   {enemy.name}: {enemy.behavior}")
                print(f"   └─ Vitesse: {enemy.speed}px/step | Menace: {enemy.threat_level}")
                print(f"   └─ Défaite: {', '.join(enemy.defeat_methods)}")
                if enemy.special_notes:
                    print(f"   └─ Note: {enemy.special_notes}")
        else:
            print("   Aucun ennemi trouvé")

def main():
    """Lancer tous les tests"""
    print("🧪 SYSTÈME DE TEST - DÉTECTION NIVEAU MARIO")
    print("="*80)
    
    test_level_database()
    test_level_context_generation()
    test_enemy_specific_info()
    
    print(f"\n✅ TESTS TERMINÉS")
    print("="*80)
    
    print(f"\n💡 UTILISATION:")
    print("   - Les prompts incluront maintenant des informations spécifiques au niveau")
    print("   - Le LLM recevra des détails sur les ennemis exacts du niveau actuel")
    print("   - Stratégies adaptées selon le type de niveau (overworld/underground/castle)")
    print("   - Recommandations de power-ups optimisées par niveau")

if __name__ == "__main__":
    main()