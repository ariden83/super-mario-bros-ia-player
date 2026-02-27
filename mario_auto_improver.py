#!/usr/bin/env python3
"""
Mario Auto-Improver : analyse les logs d'une session et génère des améliorations
automatiques (paramètres + prompts) via Claude API.
Les améliorations sont persistées dans mario_config_override.json et chargées
au démarrage de la prochaine session.
"""

import os
import json
import glob
import anthropic
from datetime import datetime
from typing import Optional

CONFIG_FILE = "mario_config_override.json"
LOGS_DIR = "logs"

# Paramètres tunable et leurs contraintes (min, max, défaut)
TUNABLE_PARAMS = {
    "stuck_check_frequency":        (20,  200,  60),
    "positions_update_frequency":   (3,    30,   5),
    "known_solution_cooldown_px":   (10,  200,  40),   # seuil dans inject_known_solution
    "stuck_mode_max_tokens":        (150, 600,  200),
    "reflex_cooldown_frames":       (30,  200,  80),
    "hole_reflex_cooldown_frames":  (5,    60,  15),
}

DEFAULT_CONFIG = {
    "version": 0,
    "parameters": {k: v[2] for k, v in TUNABLE_PARAMS.items()},
    "prompt_additions": {
        "stuck_mode": [],
        "main_context": [],
    },
    "improvement_history": [],
}


class MarioAutoImprover:
    def __init__(self, api_key: str, logs_dir: str = LOGS_DIR):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.logs_dir = logs_dir

    # ------------------------------------------------------------------
    # Config I/O
    # ------------------------------------------------------------------

    def load_config(self) -> dict:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                # Compléter les clés manquantes avec les défauts
                cfg.setdefault("parameters", {})
                for k, (_, _, default) in TUNABLE_PARAMS.items():
                    cfg["parameters"].setdefault(k, default)
                cfg.setdefault("prompt_additions", {"stuck_mode": [], "main_context": []})
                cfg["prompt_additions"].setdefault("stuck_mode", [])
                cfg["prompt_additions"].setdefault("main_context", [])
                cfg.setdefault("improvement_history", [])
                return cfg
            except Exception as e:
                print(f"⚠️  Config override illisible ({e}), réinitialisation.")
        return dict(DEFAULT_CONFIG)

    def save_config(self, cfg: dict):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Lecture des logs
    # ------------------------------------------------------------------

    def get_latest_session_id(self) -> Optional[str]:
        summaries = glob.glob(os.path.join(self.logs_dir, "*_summary.json"))
        if not summaries:
            return None
        summaries.sort(key=os.path.getmtime, reverse=True)
        basename = os.path.basename(summaries[0])
        return basename.replace("_summary.json", "")

    def load_session_data(self, session_id: str) -> dict:
        """Charge un résumé compact de la session (pas trop gros pour l'API)."""
        data = {"session_id": session_id}

        # Summary JSON
        summary_path = os.path.join(self.logs_dir, f"{session_id}_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                data["summary"] = json.load(f)

        # Actions log — on garde les 80 dernières lignes (là où le blocage se passe)
        actions_path = os.path.join(self.logs_dir, f"{session_id}_actions.log")
        if os.path.exists(actions_path):
            with open(actions_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if "ACTION -" in l]
            data["actions_last80"] = lines[-80:]
            data["actions_total"] = len(lines)
            # Compter les macros utilisées
            counts: dict = {}
            for l in lines:
                parts = l.split("|")
                if len(parts) >= 4:
                    macro = parts[3].strip().split()[0]
                    counts[macro] = counts.get(macro, 0) + 1
            data["macro_counts"] = dict(sorted(counts.items(), key=lambda x: -x[1]))

        # Réponses Claude (petit fichier en général)
        responses_path = os.path.join(self.logs_dir, f"{session_id}_responses_full.txt")
        if os.path.exists(responses_path):
            with open(responses_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Garder les 3000 derniers caractères
            data["claude_responses_tail"] = content[-3000:] if len(content) > 3000 else content

        return data

    # ------------------------------------------------------------------
    # Appel Claude pour analyse
    # ------------------------------------------------------------------

    def _build_analysis_prompt(self, session_data: dict, current_cfg: dict) -> str:
        summary = session_data.get("summary", {})
        stats = summary.get("final_stats", {})
        params = current_cfg.get("parameters", {})
        prev_additions = current_cfg.get("prompt_additions", {})

        actions_tail = "\n".join(session_data.get("actions_last80", []))
        responses_tail = session_data.get("claude_responses_tail", "(aucune réponse)")
        macro_counts = json.dumps(session_data.get("macro_counts", {}), ensure_ascii=False)

        prompt_additions_str = json.dumps(prev_additions, ensure_ascii=False, indent=2)
        params_str = json.dumps(params, ensure_ascii=False, indent=2)
        history = current_cfg.get("improvement_history", [])
        history_str = json.dumps(history[-5:], ensure_ascii=False, indent=2)  # 5 dernières

        tunable_doc = "\n".join(
            f"  - {k}: actuel={params.get(k, v[2])}, min={v[0]}, max={v[1]}, défaut={v[2]}"
            for k, v in TUNABLE_PARAMS.items()
        )

        return f"""Tu es un expert en optimisation d'IA pour jeux vidéo.
Tu dois analyser une session de jeu Mario Bros et proposer des améliorations concrètes.

## STATISTIQUES DE LA SESSION
- Steps total: {stats.get('steps_total', '?')}
- Score final: {stats.get('final_score', '?')}
- Morts: {stats.get('deaths', '?')}
- Appels Claude: {stats.get('api_calls', '?')}
- Position finale: {stats.get('final_position', '?')}px
- Coût: ${stats.get('total_cost', 0):.4f}

## DISTRIBUTION DES MACROS UTILISÉES
{macro_counts}

## DERNIÈRES 80 ACTIONS (là où le blocage survient)
{actions_tail}

## RÉPONSES CLAUDE (fin de session)
{responses_tail}

## PARAMÈTRES ACTUELS
{params_str}

## AJOUTS AUX PROMPTS ACTUELS
{prompt_additions_str}

## HISTORIQUE DES 5 DERNIÈRES AMÉLIORATIONS
{history_str}

## PARAMÈTRES MODIFIABLES (avec contraintes)
{tunable_doc}

## INSTRUCTIONS
Analyse les patterns de blocage dans les logs.
En particulier:
- Si Mario répète la même macro > 20 fois → blocage détecté
- Si step counter est figé avec position fixe → inject_known_solution en boucle
- Si pipe_jump n'est jamais exécuté malgré le blocage → prompt insuffisant
- Si Claude n'est appelé que 5 fois pour 800 steps → queue toujours pleine

Retourne UNIQUEMENT un JSON valide (pas de texte avant/après) avec cette structure exacte:
{{
  "analysis": "Description courte du problème principal observé (2-3 phrases)",
  "root_cause": "Cause technique précise",
  "improvements": [
    {{
      "type": "parameter",
      "key": "<nom_paramètre_exact>",
      "new_value": <nombre>,
      "reason": "Pourquoi cette valeur"
    }},
    {{
      "type": "prompt_addition",
      "section": "stuck_mode",
      "text": "Texte à ajouter au prompt stuck mode",
      "reason": "Pourquoi ce texte"
    }}
  ],
  "priority": <1-10>,
  "expected_improvement": "Ce qui devrait s'améliorer au prochain run"
}}

Règles:
- Au maximum 3 improvements
- Les valeurs de paramètres doivent rester dans les bornes min/max
- Les textes de prompt doivent être courts (< 80 chars) et en français
- Ne propose que des changements que tu es confiant d'améliorer, pas de changements par défaut
- Si la session s'est bien passée (position > 1000px), propose des micro-optimisations seulement
"""

    def analyze_session(self, session_id: Optional[str] = None) -> Optional[dict]:
        """
        Analyse la session (ou la dernière si session_id=None).
        Retourne le rapport d'amélioration ou None en cas d'erreur.
        """
        if session_id is None:
            session_id = self.get_latest_session_id()
        if session_id is None:
            print("⚠️  Aucune session à analyser.")
            return None

        print(f"\n🔍 Analyse de la session {session_id}...")
        session_data = self.load_session_data(session_id)
        current_cfg = self.load_config()

        prompt = self._build_analysis_prompt(session_data, current_cfg)

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",  # haiku: rapide et bon marché pour l'analyse
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
            cost = (response.usage.input_tokens * 0.00000025 +
                    response.usage.output_tokens * 0.00000125)
            print(f"   💰 Coût analyse: ${cost:.5f}")
        except Exception as e:
            print(f"❌ Erreur appel Claude pour analyse: {e}")
            return None

        # Parser le JSON
        try:
            # Extraire le JSON si du texte l'entoure
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("Pas de JSON trouvé dans la réponse")
            report = json.loads(raw[start:end])
        except Exception as e:
            print(f"❌ Impossible de parser la réponse Claude: {e}")
            print(f"   Réponse brute: {raw[:300]}")
            return None

        return report

    # ------------------------------------------------------------------
    # Application des améliorations
    # ------------------------------------------------------------------

    def apply_improvements(self, report: dict, session_id: str) -> dict:
        """Applique les améliorations au config et le sauvegarde."""
        cfg = self.load_config()
        improvements = report.get("improvements", [])
        applied = []

        for imp in improvements:
            imp_type = imp.get("type")
            if imp_type == "parameter":
                key = imp.get("key")
                new_val = imp.get("new_value")
                if key in TUNABLE_PARAMS and new_val is not None:
                    mn, mx, _ = TUNABLE_PARAMS[key]
                    # Clamp dans les bornes
                    clamped = max(mn, min(mx, int(new_val)))
                    old_val = cfg["parameters"].get(key)
                    cfg["parameters"][key] = clamped
                    applied.append(f"  📐 {key}: {old_val} → {clamped} ({imp.get('reason', '')})")
                else:
                    applied.append(f"  ⚠️  Paramètre inconnu ignoré: {key}")

            elif imp_type == "prompt_addition":
                section = imp.get("section", "stuck_mode")
                text = imp.get("text", "").strip()
                if section in cfg["prompt_additions"] and text:
                    # Eviter les doublons
                    if text not in cfg["prompt_additions"][section]:
                        # Garder au maximum 5 additions par section
                        if len(cfg["prompt_additions"][section]) >= 5:
                            cfg["prompt_additions"][section].pop(0)
                        cfg["prompt_additions"][section].append(text)
                    applied.append(f"  📝 Prompt [{section}]: \"{text[:60]}\"")

        # Enregistrer dans l'historique
        entry = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "analysis": report.get("analysis", ""),
            "root_cause": report.get("root_cause", ""),
            "applied": applied,
            "priority": report.get("priority", 0),
        }
        cfg["improvement_history"].append(entry)
        # Garder les 20 dernières entrées
        cfg["improvement_history"] = cfg["improvement_history"][-20:]
        cfg["version"] = cfg.get("version", 0) + 1

        self.save_config(cfg)
        return cfg, applied

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def run(self, session_id: Optional[str] = None) -> bool:
        """
        Lance le cycle complet : analyse → affichage → application.
        Retourne True si des améliorations ont été appliquées.
        """
        report = self.analyze_session(session_id)
        if report is None:
            return False

        if not session_id:
            session_id = self.get_latest_session_id() or "unknown"

        print(f"\n{'='*60}")
        print("🧠 RAPPORT D'AMÉLIORATION AUTOMATIQUE")
        print(f"{'='*60}")
        print(f"📋 Analyse : {report.get('analysis', '(vide)')}")
        print(f"🔎 Cause   : {report.get('root_cause', '(vide)')}")
        print(f"⭐ Priorité: {report.get('priority', '?')}/10")

        improvements = report.get("improvements", [])
        if not improvements:
            print("ℹ️  Aucune amélioration suggérée pour cette session.")
            return False

        print(f"\n🔧 Améliorations proposées ({len(improvements)}) :")
        for imp in improvements:
            if imp.get("type") == "parameter":
                print(f"  📐 {imp['key']} → {imp['new_value']}  ({imp.get('reason', '')})")
            elif imp.get("type") == "prompt_addition":
                print(f"  📝 [{imp.get('section')}] \"{imp.get('text', '')[:70]}\"  ({imp.get('reason', '')})")

        print(f"\n✨ Amélioration attendue : {report.get('expected_improvement', '?')}")

        _, applied = self.apply_improvements(report, session_id)
        print(f"\n✅ Appliqué ({len(applied)}) :")
        for a in applied:
            print(a)
        print(f"{'='*60}")
        return len(applied) > 0
