#!/usr/bin/env python3
"""
Mario Auto-Improver v2 : analyse les logs d'une session et modifie directement
mario_fluid_llm.py via des patches search/replace générés par Claude Sonnet.
Un backup est créé avant chaque modification.
"""

import os
import re
import json
import glob
import shutil
import anthropic
from datetime import datetime
from typing import Optional

CONFIG_FILE = "mario_config_override.json"
GAME_FILE = "mario_fluid_llm.py"
LOGS_DIR = "logs"
BACKUP_DIR = "backups"

# Sections de code exposées à l'analyse (extraites par marqueur de fonction)
CODE_SECTIONS = [
    "call_claude_stuck_mode",     # prompt de déblocage
    "inject_known_solution",      # logique mémoire
    "detect_stuck",               # détection blocage
]

# Constantes numériques modifiables (avec borne min/max)
TUNABLE_PARAMS = {
    "stuck_check_frequency":      (20,  200,  60),
    "positions_update_frequency": (3,    30,   5),
    "known_solution_cooldown_px": (10,  200,  40),
    "stuck_mode_max_tokens":      (150, 600,  200),
    "reflex_cooldown_frames":     (30,  200,  80),
    "hole_reflex_cooldown_frames":(5,    60,  15),
}

# Pour rétrocompatibilité avec mario_fluid_llm.py qui importe ces noms
DEFAULT_CONFIG = {"version": 0, "parameters": {}, "prompt_additions": {}, "improvement_history": []}


# ---------------------------------------------------------------------------
# Utilitaires fichier
# ---------------------------------------------------------------------------

def _backup(filepath: str) -> str:
    """Crée un backup horodaté du fichier et retourne le chemin du backup."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(BACKUP_DIR, f"{os.path.basename(filepath)}.{ts}.bak")
    shutil.copy2(filepath, dest)
    return dest


def _extract_function(source: str, func_name: str, max_lines: int = 60) -> str:
    """Extrait le corps d'une fonction depuis le source Python (simplifié)."""
    pattern = rf"def {func_name}\("
    m = re.search(pattern, source)
    if not m:
        return ""
    start = source.rfind("\n", 0, m.start()) + 1
    lines = source[start:].splitlines()
    result = []
    in_func = False
    indent = None
    for line in lines:
        if not in_func:
            result.append(line)
            in_func = True
            continue
        stripped = line.lstrip()
        if not stripped:
            result.append(line)
            continue
        cur_indent = len(line) - len(stripped)
        if indent is None:
            indent = cur_indent
        if cur_indent < indent and stripped:
            break
        result.append(line)
        if len(result) >= max_lines:
            result.append("    # ... (tronqué)")
            break
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class MarioAutoImprover:
    def __init__(self, api_key: str, logs_dir: str = LOGS_DIR):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.logs_dir = logs_dir

    # -----------------------------------------------------------------------
    # Compatibilité : load/save config (pour mario_fluid_llm.py)
    # -----------------------------------------------------------------------

    def load_config(self) -> dict:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return dict(DEFAULT_CONFIG)

    def save_config(self, cfg: dict):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # Lecture des logs
    # -----------------------------------------------------------------------

    def get_latest_session_id(self) -> Optional[str]:
        summaries = glob.glob(os.path.join(self.logs_dir, "*_summary.json"))
        if not summaries:
            return None
        summaries.sort(key=os.path.getmtime, reverse=True)
        return os.path.basename(summaries[0]).replace("_summary.json", "")

    def load_session_data(self, session_id: str) -> dict:
        data = {"session_id": session_id}

        path = os.path.join(self.logs_dir, f"{session_id}_summary.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data["summary"] = json.load(f)

        path = os.path.join(self.logs_dir, f"{session_id}_actions.log")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if "ACTION -" in l]
            data["actions_last100"] = lines[-100:]
            counts: dict = {}
            for l in lines:
                # Format: timestamp | INFO | mario.actions | ACTION - Step N | source | macro_name | Pos | Score | reason
                parts = l.split("|")
                if len(parts) >= 6:
                    macro = parts[5].strip().split()[0]
                    counts[macro] = counts.get(macro, 0) + 1
            data["macro_counts"] = dict(sorted(counts.items(), key=lambda x: -x[1])[:15])

        path = os.path.join(self.logs_dir, f"{session_id}_responses_full.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            data["claude_responses"] = content[-4000:] if len(content) > 4000 else content

        return data

    def _read_code_sections(self) -> str:
        """Lit les sections de code pertinentes depuis mario_fluid_llm.py."""
        if not os.path.exists(GAME_FILE):
            return "(fichier introuvable)"
        with open(GAME_FILE, "r", encoding="utf-8") as f:
            source = f.read()
        parts = []
        for fn in CODE_SECTIONS:
            snippet = _extract_function(source, fn)
            if snippet:
                parts.append(f"### def {fn}(...)\n```python\n{snippet}\n```")
        # Extraire aussi les constantes clés de __init__
        init_lines = []
        for line in source.splitlines():
            if any(k in line for k in ("stuck_check_frequency", "positions_update_frequency",
                                        "max_tokens", "reflex_cooldown", "cooldown_px")):
                init_lines.append(line.strip())
        if init_lines:
            parts.append("### Constantes clés\n```python\n" + "\n".join(init_lines[:20]) + "\n```")
        return "\n\n".join(parts)

    # -----------------------------------------------------------------------
    # Prompt d'analyse
    # -----------------------------------------------------------------------

    def _build_prompt(self, session_data: dict, code_sections: str) -> str:
        stats = session_data.get("summary", {}).get("final_stats", {})
        actions_tail = "\n".join(session_data.get("actions_last100", []))
        macro_counts = json.dumps(session_data.get("macro_counts", {}), ensure_ascii=False)
        responses = session_data.get("claude_responses", "(aucune)")

        return f"""Tu es un expert en développement de jeux vidéo et en IA.
Tu analyses une session de Mario Bros jouée par une IA (Claude LLM) et tu dois
proposer des améliorations CONCRÈTES au code Python du jeu.

## STATISTIQUES DE LA SESSION
- Steps: {stats.get('steps_total','?')} | Score: {stats.get('final_score','?')}
- Morts: {stats.get('deaths','?')} | API calls: {stats.get('api_calls','?')}
- Position finale: {stats.get('final_position','?')}px | Coût: ${stats.get('total_cost',0):.4f}

## DISTRIBUTION DES MACROS (fréquence d'usage)
{macro_counts}

## DERNIÈRES 100 ACTIONS (chronologiques, là où les problèmes surviennent)
{actions_tail}

## RÉPONSES CLAUDE PENDANT LA SESSION (fin de partie)
{responses}

## CODE SOURCE ACTUEL (sections pertinentes de mario_fluid_llm.py)
{code_sections}

## PATTERNS DE PROBLÈMES À CHERCHER
- Si `run_forward` apparaît > 20 fois de suite avec position fixe → inject_known_solution en boucle
- Si `pipe_jump` est suivi immédiatement de `run_forward` → pipe_jump écrasé
- Si `api_calls` < 10 pour > 500 steps → Claude sous-utilisé, queue toujours pleine
- Si position finale < 500px → Mario bloqué avant premier tuyau
- Si même action répétée en boucle → stuck mode insuffisant

## INSTRUCTIONS
Propose des améliorations DIRECTES au code.
Pour chaque amélioration, fournis un patch search/replace EXACT.

Retourne UNIQUEMENT un JSON valide avec cette structure :
{{
  "analyse": "2-3 phrases décrivant le problème principal observé",
  "cause_racine": "cause technique précise",
  "patches": [
    {{
      "description": "Ce que le patch change",
      "fichier": "mario_fluid_llm.py",
      "search": "texte EXACT à rechercher dans le fichier (doit être unique, copie exacte)",
      "replace": "texte de remplacement",
      "risque": "faible|moyen|élevé"
    }}
  ],
  "amelioration_attendue": "ce qui devrait s'améliorer au prochain run"
}}

Règles importantes :
- Maximum 3 patches par session
- Le champ "search" doit être un extrait EXACT du code fourni (copie/colle)
- Préfère modifier les prompts, les constantes numériques, les durées de macros
- NE PAS modifier le moteur de jeu, les boucles principales, les imports
- Si la session est bonne (position > 1000px), ne propose rien ou micro-optimisation
- Chaque patch doit vraiment résoudre le problème identifié dans les logs
"""

    # -----------------------------------------------------------------------
    # Application des patches
    # -----------------------------------------------------------------------

    def _apply_patches(self, patches: list) -> list:
        """Applique les patches search/replace sur mario_fluid_llm.py."""
        if not patches:
            return []

        if not os.path.exists(GAME_FILE):
            print(f"❌ Fichier {GAME_FILE} introuvable")
            return []

        with open(GAME_FILE, "r", encoding="utf-8") as f:
            source = f.read()

        applied = []
        modified = source

        for patch in patches:
            search = patch.get("search", "")
            replace = patch.get("replace", "")
            desc = patch.get("description", "?")
            risk = patch.get("risque", "?")

            if not search or not replace:
                print(f"  ⚠️  Patch incomplet ignoré: {desc}")
                continue

            if search == replace:
                print(f"  ⚠️  Patch no-op ignoré: {desc}")
                continue

            if search not in modified:
                print(f"  ⚠️  Texte non trouvé, patch ignoré: {desc}")
                print(f"       Cherché: {repr(search[:80])}")
                continue

            count = modified.count(search)
            if count > 1:
                print(f"  ⚠️  Texte ambigu ({count} occurrences), patch ignoré: {desc}")
                continue

            modified = modified.replace(search, replace, 1)
            applied.append({
                "description": desc,
                "risque": risk,
                "search_preview": search[:100].replace("\n", "↵"),
                "replace_preview": replace[:100].replace("\n", "↵"),
            })
            print(f"  ✅ Patch appliqué [{risk}]: {desc}")

        if applied:
            backup_path = _backup(GAME_FILE)
            print(f"  💾 Backup: {backup_path}")
            with open(GAME_FILE, "w", encoding="utf-8") as f:
                f.write(modified)
            print(f"  📝 {GAME_FILE} mis à jour ({len(applied)} patch(es))")

        return applied

    # -----------------------------------------------------------------------
    # Historique
    # -----------------------------------------------------------------------

    def _save_history(self, session_id: str, report: dict, applied: list):
        cfg = self.load_config()
        cfg.setdefault("improvement_history", [])
        cfg["improvement_history"].append({
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "analyse": report.get("analyse", ""),
            "cause_racine": report.get("cause_racine", ""),
            "patches_applied": len(applied),
            "patches": [p["description"] for p in applied],
        })
        cfg["improvement_history"] = cfg["improvement_history"][-20:]
        cfg["version"] = cfg.get("version", 0) + 1
        self.save_config(cfg)

    # -----------------------------------------------------------------------
    # Point d'entrée principal
    # -----------------------------------------------------------------------

    def run(self, session_id: Optional[str] = None) -> bool:
        if session_id is None:
            session_id = self.get_latest_session_id()
        if session_id is None:
            print("⚠️  Aucune session à analyser.")
            return False

        print(f"\n{'='*60}")
        print(f"🔬 AUTO-AMÉLIORATION — session {session_id[-8:]}")
        print(f"{'='*60}")

        # 1. Charger les données
        print("📂 Lecture des logs...")
        session_data = self.load_session_data(session_id)
        print("📖 Lecture du code source...")
        code_sections = self._read_code_sections()

        # 2. Appeler Claude Sonnet
        prompt = self._build_prompt(session_data, code_sections)
        print("🧠 Appel Claude Sonnet pour analyse...")
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
            cost = (response.usage.input_tokens * 0.000003 +
                    response.usage.output_tokens * 0.000015)
            print(f"   💰 Coût analyse: ${cost:.4f}")
        except Exception as e:
            print(f"❌ Erreur API Claude: {e}")
            return False

        # 3. Parser le JSON
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("Pas de JSON dans la réponse")
            report = json.loads(raw[start:end])
        except Exception as e:
            print(f"❌ Réponse Claude non parseable: {e}")
            print(f"   Réponse brute:\n{raw[:500]}")
            return False

        # 4. Afficher l'analyse
        print(f"\n📋 Analyse : {report.get('analyse', '(vide)')}")
        print(f"🔎 Cause   : {report.get('cause_racine', '(vide)')}")

        patches = report.get("patches", [])
        if not patches:
            print("ℹ️  Aucun patch suggéré — comportement correct pour cette session.")
            self._save_history(session_id, report, [])
            return False

        print(f"\n🔧 {len(patches)} patch(es) proposé(s) :")
        for i, p in enumerate(patches, 1):
            print(f"  {i}. [{p.get('risque','?')}] {p.get('description','?')}")

        # 5. Appliquer les patches
        print(f"\n⚙️  Application des patches sur {GAME_FILE}...")
        applied = self._apply_patches(patches)

        # 6. Sauvegarder l'historique
        self._save_history(session_id, report, applied)

        print(f"\n✨ Amélioration attendue : {report.get('amelioration_attendue','?')}")
        print(f"{'='*60}")
        return len(applied) > 0
