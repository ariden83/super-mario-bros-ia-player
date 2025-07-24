#!/usr/bin/env python3
"""
Script d'analyse des logs Mario
Permet de visualiser et analyser les sessions de jeu
"""

import os
import json
import glob
from datetime import datetime
import argparse

def list_sessions():
    """Lister toutes les sessions disponibles"""
    log_files = glob.glob("logs/*_main.log")
    sessions = []
    
    for log_file in log_files:
        session_id = os.path.basename(log_file).replace('_main.log', '')
        summary_file = f"logs/{session_id}_summary.json"
        
        session_info = {
            'id': session_id,
            'timestamp': session_id.split('_')[-1] if '_' in session_id else 'unknown',
            'has_summary': os.path.exists(summary_file)
        }
        
        if session_info['has_summary']:
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    session_info['stats'] = summary.get('final_stats', {})
            except:
                session_info['stats'] = {}
        
        sessions.append(session_info)
    
    # Trier par timestamp (plus récent d'abord)
    sessions.sort(key=lambda x: x['timestamp'], reverse=True)
    return sessions

def analyze_session(session_id):
    """Analyser une session spécifique"""
    print(f"\n📊 ANALYSE DE LA SESSION: {session_id}")
    print("=" * 60)
    
    # Fichiers de la session
    files = {
        'main': f"logs/{session_id}_main.log",
        'actions': f"logs/{session_id}_actions.log",
        'claude': f"logs/{session_id}_claude.log",
        'game': f"logs/{session_id}_game.log",
        'replay': f"logs/{session_id}_replay.log",
        'prompts': f"logs/{session_id}_prompts_full.txt",
        'responses': f"logs/{session_id}_responses_full.txt",
        'summary': f"logs/{session_id}_summary.json"
    }
    
    # Vérifier quels fichiers existent
    existing_files = {k: v for k, v in files.items() if os.path.exists(v)}
    
    print(f"📁 Fichiers disponibles: {len(existing_files)}/{len(files)}")
    for name, path in existing_files.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"   📄 {name:<10} : {os.path.basename(path)} ({size_kb:.1f} KB)")
    
    # Analyser le résumé
    if 'summary' in existing_files:
        print(f"\n📈 STATISTIQUES FINALES:")
        try:
            with open(files['summary'], 'r', encoding='utf-8') as f:
                summary = json.load(f)
                stats = summary.get('final_stats', {})
                
                print(f"   🎮 Steps total: {stats.get('steps_total', 0)}")
                print(f"   🏆 Score final: {stats.get('final_score', 0)}")
                print(f"   💀 Morts: {stats.get('deaths', 0)}")
                print(f"   🧠 Appels Claude: {stats.get('api_calls', 0)}")
                print(f"   💰 Coût total: ${stats.get('total_cost', 0):.3f}")
                print(f"   🚀 Position finale: {stats.get('final_position', 0)} pixels")
                
        except Exception as e:
            print(f"   ❌ Erreur lecture résumé: {e}")
    
    # Analyser les actions
    if 'actions' in existing_files:
        print(f"\n🎮 ANALYSE DES ACTIONS:")
        try:
            actions_by_source = {}
            action_types = {}
            
            with open(files['actions'], 'r', encoding='utf-8') as f:
                for line in f:
                    if 'ACTION -' in line:
                        parts = line.split('|')
                        if len(parts) >= 4:
                            source = parts[2].strip()
                            action = parts[3].split()[0] if parts[3].strip() else 'unknown'
                            
                            actions_by_source[source] = actions_by_source.get(source, 0) + 1
                            action_types[action] = action_types.get(action, 0) + 1
            
            print("   Par source:")
            for source, count in sorted(actions_by_source.items()):
                print(f"     {source:<8} : {count:4d} actions")
            
            print("   Top actions:")
            for action, count in sorted(action_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"     {action:<20} : {count:3d} fois")
                
        except Exception as e:
            print(f"   ❌ Erreur analyse actions: {e}")
    
    # Analyser Claude
    if 'claude' in existing_files:
        print(f"\n🧠 ANALYSE CLAUDE:")
        try:
            prompt_types = {}
            total_cost = 0
            response_count = 0
            
            with open(files['claude'], 'r', encoding='utf-8') as f:
                for line in f:
                    if 'PROMPT [' in line:
                        # Extraire le type de prompt
                        start = line.find('[') + 1
                        end = line.find(']')
                        if start > 0 and end > start:
                            prompt_type = line[start:end]
                            prompt_types[prompt_type] = prompt_types.get(prompt_type, 0) + 1
                    
                    elif 'RESPONSE -' in line and 'Cost:' in line:
                        response_count += 1
                        # Extraire le coût
                        cost_start = line.find('Cost: $') + 7
                        cost_end = line.find(' ', cost_start)
                        if cost_start > 6:
                            try:
                                cost = float(line[cost_start:cost_end] if cost_end > 0 else line[cost_start:])
                                total_cost += cost
                            except:
                                pass
            
            print("   Types de prompts:")
            for ptype, count in sorted(prompt_types.items()):
                print(f"     {ptype:<12} : {count:3d} prompts")
            
            print(f"   💰 Coût estimé: ${total_cost:.4f}")
            print(f"   📝 Réponses: {response_count}")
            
        except Exception as e:
            print(f"   ❌ Erreur analyse Claude: {e}")
    
    # Analyser les événements de jeu
    if 'game' in existing_files:
        print(f"\n🎯 ÉVÉNEMENTS DE JEU:")
        try:
            events = {}
            
            with open(files['game'], 'r', encoding='utf-8') as f:
                for line in f:
                    if 'GAME ' in line:
                        # Extraire le type d'événement
                        game_start = line.find('GAME ') + 5
                        game_end = line.find(' -', game_start)
                        if game_start > 4 and game_end > game_start:
                            event = line[game_start:game_end]
                            events[event] = events.get(event, 0) + 1
            
            for event, count in sorted(events.items()):
                print(f"   {event:<12} : {count} fois")
                
        except Exception as e:
            print(f"   ❌ Erreur analyse événements: {e}")

def show_recent_actions(session_id, limit=20):
    """Afficher les dernières actions d'une session"""
    actions_file = f"logs/{session_id}_actions.log"
    
    if not os.path.exists(actions_file):
        print(f"❌ Fichier d'actions non trouvé: {actions_file}")
        return
    
    print(f"\n🎮 DERNIÈRES {limit} ACTIONS - SESSION: {session_id}")
    print("=" * 80)
    
    try:
        with open(actions_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        action_lines = [line for line in lines if 'ACTION -' in line]
        recent_actions = action_lines[-limit:] if len(action_lines) > limit else action_lines
        
        for line in recent_actions:
            # Nettoyer et formater la ligne
            clean_line = line.strip()
            if '|' in clean_line:
                parts = clean_line.split('|')
                if len(parts) >= 6:
                    timestamp = parts[0].strip()
                    step = parts[3].strip() if 'Step' in parts[3] else ''
                    source = parts[2].strip()
                    action_details = '|'.join(parts[4:]).strip()
                    
                    print(f"{timestamp} | {step:<12} | {source:<7} | {action_details}")
        
        print(f"\nTotal actions dans le fichier: {len(action_lines)}")
        
    except Exception as e:
        print(f"❌ Erreur lecture actions: {e}")

def show_claude_conversation(session_id, limit=5):
    """Afficher la conversation avec Claude (prompts + réponses)"""
    prompts_file = f"logs/{session_id}_prompts_full.txt"
    responses_file = f"logs/{session_id}_responses_full.txt"
    
    print(f"\n🧠 CONVERSATION CLAUDE - SESSION: {session_id}")
    print("=" * 80)
    
    # Lire les prompts
    prompts = []
    if os.path.exists(prompts_file):
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                content = f.read()
                prompt_blocks = content.split('=' * 80)
                
                for block in prompt_blocks:
                    if 'TYPE:' in block and 'STEP:' in block:
                        lines = block.strip().split('\n')
                        if len(lines) >= 3:
                            meta_line = [l for l in lines if 'TYPE:' in l and 'STEP:' in l]
                            if meta_line:
                                prompts.append({
                                    'meta': meta_line[0].strip(),
                                    'content': '\n'.join(lines[3:])[:500] + '...' if len('\n'.join(lines[3:])) > 500 else '\n'.join(lines[3:])
                                })
        except Exception as e:
            print(f"❌ Erreur lecture prompts: {e}")
    
    # Lire les réponses
    responses = []
    if os.path.exists(responses_file):
        try:
            with open(responses_file, 'r', encoding='utf-8') as f:
                content = f.read()
                response_blocks = content.split('=' * 80)
                
                for block in response_blocks:
                    if 'STEP:' in block and 'COST:' in block:
                        lines = block.strip().split('\n')
                        if len(lines) >= 3:
                            meta_line = [l for l in lines if 'STEP:' in l and 'COST:' in l]
                            if meta_line:
                                responses.append({
                                    'meta': meta_line[0].strip(),
                                    'content': '\n'.join(lines[3:])[:300] + '...' if len('\n'.join(lines[3:])) > 300 else '\n'.join(lines[3:])
                                })
        except Exception as e:
            print(f"❌ Erreur lecture réponses: {e}")
    
    # Afficher les échanges les plus récents
    recent_prompts = prompts[-limit:] if len(prompts) > limit else prompts
    recent_responses = responses[-limit:] if len(responses) > limit else responses
    
    max_exchanges = max(len(recent_prompts), len(recent_responses))
    
    for i in range(max_exchanges):
        if i < len(recent_prompts):
            print(f"\n🔍 PROMPT {i+1}:")
            print(f"   Meta: {recent_prompts[i]['meta']}")
            print(f"   Content: {recent_prompts[i]['content'][:200]}...")
        
        if i < len(recent_responses):
            print(f"\n💭 RÉPONSE {i+1}:")
            print(f"   Meta: {recent_responses[i]['meta']}")
            print(f"   Content: {recent_responses[i]['content']}")
        
        if i < max_exchanges - 1:
            print("\n" + "-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Analyser les logs Mario")
    parser.add_argument('--list', action='store_true', help='Lister toutes les sessions')
    parser.add_argument('--analyze', help='Analyser une session spécifique')
    parser.add_argument('--actions', help='Afficher les dernières actions d\'une session')
    parser.add_argument('--claude', help='Afficher la conversation Claude d\'une session')
    parser.add_argument('--limit', type=int, default=20, help='Limite d\'éléments à afficher')
    
    args = parser.parse_args()
    
    if args.list:
        sessions = list_sessions()
        print(f"\n📝 SESSIONS DISPONIBLES ({len(sessions)}):")
        print("=" * 80)
        
        for i, session in enumerate(sessions, 1):
            timestamp = session['timestamp']
            if timestamp != 'unknown':
                try:
                    dt = datetime.fromtimestamp(int(timestamp))
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_time = timestamp
            else:
                formatted_time = 'Date inconnue'
            
            print(f"{i:2d}. {session['id']}")
            print(f"    📅 {formatted_time}")
            
            if 'stats' in session and session['stats']:
                stats = session['stats']
                print(f"    🎮 Steps: {stats.get('steps_total', 0)} | Score: {stats.get('final_score', 0)} | Morts: {stats.get('deaths', 0)}")
                print(f"    💰 Coût: ${stats.get('total_cost', 0):.3f} | Position: {stats.get('final_position', 0)}px")
            
            print()
    
    elif args.analyze:
        analyze_session(args.analyze)
    
    elif args.actions:
        show_recent_actions(args.actions, args.limit)
    
    elif args.claude:
        show_claude_conversation(args.claude, args.limit)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()