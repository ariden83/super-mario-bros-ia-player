#!/usr/bin/env python3
"""
Script utilitaire pour visualiser les historiques de Mario
"""

from mario_history_manager import MarioHistoryManager
import sys

def main():
    print("📊 VISUALISEUR D'HISTORIQUE MARIO")
    print("=" * 50)
    
    manager = MarioHistoryManager()
    
    # Statistiques générales
    stats = manager.get_run_stats()
    
    if stats.get("total_runs", 0) == 0:
        print("📝 Aucun historique trouvé!")
        print("🎮 Lancez Mario pour créer votre premier run!")
        return
    
    print(f"🏃 Runs totaux: {stats['total_runs']}")
    print(f"🏆 Meilleure distance: {stats['best_distance']} pixels")
    print(f"🚀 Meilleure vitesse: {stats['best_speed']:.2f} px/s")
    print(f"⏱️  Temps de jeu total: {stats['total_playtime']:.1f}s")
    
    completion = stats.get('completion_rates', {})
    print(f"🎯 Résultats: Victoires: {completion.get('victory', 0)} | Morts: {completion.get('death', 0)} | Interruptions: {completion.get('interrupted', 0)}")
    
    # Meilleurs runs
    print(f"\n🏆 TOP RUNS:")
    available_runs = manager.get_available_runs_for_replay()
    
    for i, run in enumerate(available_runs[:10], 1):  # Top 10
        status_emoji = "🏆" if run.completion_status == "victory" else "💀" if run.completion_status == "death" else "⏸️"
        print(f"{i:2d}. {status_emoji} {run.run_id}")
        print(f"     📍 {run.max_position_x}px | ⏱️ {run.duration:.1f}s | 🎮 {run.actions_count} actions | 🚀 {run.average_speed:.2f}px/s")
    
    # Options interactives
    if len(sys.argv) > 1 and sys.argv[1] == "--details":
        print(f"\n🔍 DÉTAILS D'UN RUN:")
        
        try:
            choice = input("Entrez le numéro du run à détailler (1-10): ").strip()
            run_index = int(choice) - 1
            
            if 0 <= run_index < len(available_runs):
                selected_run = available_runs[run_index]
                manager.print_run_summary(selected_run)
                
                # Charger et afficher quelques actions
                actions = manager.load_run_actions(selected_run.run_id)
                if actions:
                    print(f"\n🎮 APERÇU DES ACTIONS ({len(actions)} total):")
                    for i, action in enumerate(actions[:10], 1):  # 10 premières actions
                        print(f"{i:2d}. Step {action.step_count:4d} | Pos: {action.position_x:4d} | Action: {action.action_name}")
                    
                    if len(actions) > 10:
                        print(f"    ... et {len(actions) - 10} autres actions")
            else:
                print("❌ Numéro invalide!")
                
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Au revoir!")

if __name__ == "__main__":
    main()