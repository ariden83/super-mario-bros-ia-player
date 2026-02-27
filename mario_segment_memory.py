"""
Système de mémoire par segments pour Mario.

Le niveau est découpé en zones de SEGMENT_SIZE pixels.
Pour chaque zone, on mémorise entre sessions :
  - Nombre de passages
  - Meilleur temps (en steps)
  - Interactions : ennemis écrasés, blocs frappés, items collectés, items loupés
  - Morts : cause + pixel + dernière action → à ne pas reproduire
  - Progression maximale atteinte globalement

Les données persistent dans memory/segment_memory.json.
"""
import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional


SEGMENT_SIZE = 100  # pixels par zone
MEMORY_PATH = os.path.join(os.path.dirname(__file__), "memory", "segment_memory.json")

# Macros qui frappent des blocs ? directement (sans nécessiter détection coin/score)
BLOCK_HIT_MACROS = {'approach_and_hit_block', 'hit_block'}

# Nombre de segments avant la mort exclus du replay (zone danger → IA reprend la main)
DANGER_SEG_COUNT = 2


@dataclass
class DeathRecord:
    x: int                  # Position exacte de la mort
    cause: str              # "enemy_hit", "fell_in_hole", "time_out", "unknown"
    last_action: str        # Dernière macro exécutée
    approach_actions: List[str]  # 3 actions avant la mort
    count: int = 1          # Nombre de fois cette mort s'est produite


@dataclass
class InteractionRecord:
    x: int
    interaction_type: str   # "enemy_stomp", "block_hit", "item_collected", "item_missed"
    detail: str             # "Goomba", "question_block", "mushroom", "coin", etc.
    points: int = 0


@dataclass
class SuccessRecord:
    x: int                      # Position de l'obstacle franchi
    winning_sequence: List[str] # Ex: ["walk_right", "max_jump"]
    steps_to_clear: int         # Steps entre blocage et franchissement
    count: int = 1              # Confirmée N fois (fiabilité)


@dataclass
class StageRecord:
    """Meilleur run complet sauvegardé — du début jusqu'à la frontière sûre.

    La frontière sûre = max_x du run - DANGER_SEG_COUNT segments.
    Au-delà (danger_keys), l'IA reprend la main : le timing des ennemis
    n'était pas validé dans cette zone.
    """
    safe_max_x: int = 0                           # x max rejoué en sécurité
    total_blocks: int = 0                         # Blocs ? frappés (hors danger)
    total_steps: int = 0                          # Steps jusqu'à safe_max_x
    danger_keys: List[str] = field(default_factory=list)     # Segments IA (exclus du replay)
    sequences: Dict[str, List[tuple]] = field(default_factory=dict)  # {seg_key: [(macro,count)]}
    run_id: Optional[str] = None


@dataclass
class SegmentData:
    segment_key: str            # ex: "0-100"
    start_x: int
    end_x: int
    passages_count: int = 0
    approach_death_count: int = 0  # Fois où ce segment précédait une mort
    deaths: List[DeathRecord] = field(default_factory=list)
    interactions: List[InteractionRecord] = field(default_factory=list)
    successes: List[SuccessRecord] = field(default_factory=list)

    def add_death(self, x: int, cause: str, last_action: str, approach: List[str]):
        """Ajoute ou incrémente une mort à cette position."""
        for d in self.deaths:
            if abs(d.x - x) < 20 and d.cause == cause and d.last_action == last_action:
                d.count += 1
                return
        self.deaths.append(DeathRecord(
            x=x, cause=cause, last_action=last_action,
            approach_actions=approach[:3], count=1
        ))

    def add_interaction(self, x: int, itype: str, detail: str, points: int = 0):
        self.interactions.append(InteractionRecord(
            x=x, interaction_type=itype, detail=detail, points=points
        ))

    def add_success(self, x: int, sequence: List[str], steps: int):
        """Ajoute ou renforce une séquence gagnante pour franchir un obstacle."""
        for s in self.successes:
            if abs(s.x - x) < 20 and s.winning_sequence == sequence:
                s.count += 1
                s.steps_to_clear = min(s.steps_to_clear, steps)
                return
        self.successes.append(SuccessRecord(
            x=x, winning_sequence=sequence, steps_to_clear=steps, count=1
        ))

    def success_summary(self) -> List[str]:
        """Résumé des séquences gagnantes pour le prompt Claude."""
        lines = []
        for s in sorted(self.successes, key=lambda s: -s.count):
            seq = " → ".join(s.winning_sequence)
            lines.append(
                f"  🏆 SOLUTION x={s.x} ({s.count}x confirmée): {seq} "
                f"({s.steps_to_clear} steps) → UTILISER DIRECTEMENT"
            )
        return lines

    def death_summary(self) -> List[str]:
        """Résumé des morts pour le prompt Claude."""
        lines = []
        for d in sorted(self.deaths, key=lambda d: -d.count):
            lines.append(
                f"  ⚠️ Mort x={d.x} ({d.count}x): cause={d.cause}, "
                f"dernière action={d.last_action} → NE PAS répéter"
            )
        return lines

    def interaction_summary(self) -> List[str]:
        """Résumé des interactions pour le prompt Claude."""
        stomps = [i for i in self.interactions if i.interaction_type == "enemy_stomp"]
        blocks = [i for i in self.interactions if i.interaction_type == "block_hit"]
        items = [i for i in self.interactions if i.interaction_type == "item_collected"]
        missed = [i for i in self.interactions if i.interaction_type == "item_missed"]
        lines = []
        if stomps:
            lines.append(f"  ✅ Ennemis écrasés: {len(stomps)} (positions: {[s.x for s in stomps[:3]]})")
        if blocks:
            lines.append(f"  ✅ Blocs frappés: {len(blocks)}")
        if items:
            lines.append(f"  ✅ Items collectés: {[i.detail for i in items]}")
        if missed:
            lines.append(f"  ❌ Items loupés: {[i.detail for i in missed[:3]]} → à récupérer")
        return lines


class MarioSegmentMemory:
    """Mémoire persistante par segments de niveau."""

    def __init__(self, path: str = MEMORY_PATH):
        self.path = path
        self.segments: Dict[str, SegmentData] = {}
        self.furthest_x: int = 0
        self.total_runs: int = 0
        self.stage: StageRecord = StageRecord()   # Meilleur run complet
        self._load()

        # Buffer de la run en cours (vidé à chaque nouvelle run)
        self._run_events: List[dict] = []
        self._run_id: Optional[str] = None
        self._run_start_x: int = 0
        self._run_steps_per_segment: Dict[str, int] = {}
        self._run_macros_per_segment: Dict[str, List[tuple]] = {}  # {seg_key: [(macro, count)...]}
        self._run_blocks_per_segment: Dict[str, int] = {}  # Blocs ? frappés par segment
        self._segment_entry_step: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # API publique — appelée depuis mario_fluid_llm.py                    #
    # ------------------------------------------------------------------ #

    def start_run(self, run_id: str):
        """Démarre une nouvelle run."""
        self._run_id = run_id
        self._run_events = []
        self._run_steps_per_segment = {}
        self._run_macros_per_segment = {}
        self._run_blocks_per_segment = {}
        self._segment_entry_step = {}

    def record_position(self, x: int, step: int):
        """Appeler à chaque step pour tracker l'entrée dans un nouveau segment."""
        key = self._key(x)
        if key not in self._segment_entry_step:
            self._segment_entry_step[key] = step
            seg = self._get_or_create(key, x)
            seg.passages_count += 1

    def record_enemy_stomp(self, x: int, enemy_type: str, points: int):
        seg = self._get_or_create(self._key(x), x)
        seg.add_interaction(x, "enemy_stomp", enemy_type, points)

    def record_block_hit(self, x: int, block_type: str, points: int = 0):
        seg = self._get_or_create(self._key(x), x)
        seg.add_interaction(x, "block_hit", block_type, points)

    def record_item_collected(self, x: int, item: str, points: int):
        seg = self._get_or_create(self._key(x), x)
        seg.add_interaction(x, "item_collected", item, points)

    def record_item_missed(self, x: int, item: str):
        seg = self._get_or_create(self._key(x), x)
        seg.add_interaction(x, "item_missed", item)

    def record_block_hit_in_run(self, x: int):
        """Enregistre un bloc ? frappé dans ce segment pour le run courant."""
        key = self._key(x)
        self._run_blocks_per_segment[key] = self._run_blocks_per_segment.get(key, 0) + 1

    def reset_run_recording(self, x: int):
        """Réinitialise l'enregistrement du segment courant (appelé à chaque entrée de segment).
        Garantit que la séquence stockée correspond au DERNIER passage propre gauche→droite."""
        key = self._key(x)
        self._run_macros_per_segment[key] = []
        self._run_blocks_per_segment[key] = 0

    def record_macro_in_segment(self, x: int, macro_name: str):
        """Enregistre une macro avec son nombre d'exécutions consécutives (macro, count).
        Les macros de frappe de blocs (approach_and_hit_block, hit_block) incrémentent
        automatiquement le compteur de blocs du segment, sans attendre détection coin/score."""
        key = self._key(x)
        if key not in self._run_macros_per_segment:
            self._run_macros_per_segment[key] = []
        seq = self._run_macros_per_segment[key]
        if seq and seq[-1][0] == macro_name:
            seq[-1] = (macro_name, seq[-1][1] + 1)
        else:
            seq.append((macro_name, 1))
        # Détection directe : les macros de frappe comptent comme bloc frappé
        if macro_name in BLOCK_HIT_MACROS:
            self._run_blocks_per_segment[key] = self._run_blocks_per_segment.get(key, 0) + 1

    def get_stage_sequence(self, seg_key: str) -> Optional[List[tuple]]:
        """Retourne la séquence du meilleur run pour ce segment.
        Retourne None si le segment est dans la zone danger (IA doit prendre la main)."""
        if seg_key in self.stage.danger_keys:
            return None
        seq = self.stage.sequences.get(seg_key)
        return list(seq) if seq else None

    def get_stage_danger_frontier(self) -> int:
        """Retourne le x où l'IA doit prendre la main (début de la zone danger)."""
        return self.stage.safe_max_x

    def clear_memory(self):
        """Efface toute la mémoire persistante (segments + stage) et repart de zéro."""
        self.segments = {}
        self.furthest_x = 0
        self.total_runs = 0
        self.stage = StageRecord()
        self._save()

    def has_deaths_in_segment(self, x: int) -> bool:
        """True si ce segment contient des morts enregistrées."""
        seg = self.segments.get(self._key(x))
        return bool(seg and seg.deaths)

    def record_death(self, x: int, cause: str, last_action: str, approach: List[str]):
        """Enregistre une mort avec sa cause et les actions qui y ont mené."""
        seg = self._get_or_create(self._key(x), x)
        seg.add_death(x, cause, last_action, approach)
        self._save()

    def record_death_approach(self, x_death: int, n_approach: int = 2):
        """Marque les N segments précédant la mort comme faisant partie d'une approche fatale.
        Sert uniquement de contexte pour Claude (approach_death_count).
        La gestion du replay se fait au niveau du stage dans finalize_stage."""
        death_seg_idx = int(x_death) // SEGMENT_SIZE
        for i in range(1, n_approach + 1):
            seg_start = (death_seg_idx - i) * SEGMENT_SIZE
            if seg_start < 0:
                continue
            key = f"{seg_start}-{seg_start + SEGMENT_SIZE}"
            seg = self._get_or_create(key, seg_start)
            seg.approach_death_count += 1
        self._save()

    def record_success(self, x: int, winning_sequence: List[str], steps_to_clear: int):
        """Enregistre la séquence qui a permis de franchir un obstacle."""
        seg = self._get_or_create(self._key(x), x)
        seg.add_success(x, winning_sequence, steps_to_clear)
        self._save()

    def finalize_stage(self, max_x: int, total_steps: int, died: bool = True):
        """Finalise le run et sauvegarde si c'est un meilleur run complet.

        Un run est meilleur si (en comparant le safe_max_x, i.e. max_x - DANGER zones) :
          1. safe_max_x > meilleur safe_max_x précédent (aller plus loin)
          2. Même safe_max_x + plus de blocs
          3. Même safe_max_x + même blocs + moins de steps jusqu'à safe_max_x

        Quand Mario meurt, les DANGER_SEG_COUNT derniers segments sont exclus du replay
        (danger_keys) : l'IA reprend la main dans ces zones à timing imprévisible.
        """
        if max_x > self.furthest_x:
            self.furthest_x = max_x
        self.total_runs += 1

        if died and max_x > 0:
            death_seg_idx = int(max_x) // SEGMENT_SIZE
            safe_seg_idx = max(0, death_seg_idx - DANGER_SEG_COUNT)
            safe_max_x = safe_seg_idx * SEGMENT_SIZE
            danger_keys = [
                f"{(death_seg_idx - i) * SEGMENT_SIZE}-{(death_seg_idx - i + 1) * SEGMENT_SIZE}"
                for i in range(DANGER_SEG_COUNT, 0, -1)
                if (death_seg_idx - i) >= 0
            ]
        else:
            safe_max_x = max_x
            danger_keys = []

        # Blocs frappés hors zone danger
        total_blocks = sum(
            v for k, v in self._run_blocks_per_segment.items()
            if k not in danger_keys
        )

        # Steps jusqu'à l'entrée dans la zone danger (ou fin si pas de mort)
        if danger_keys:
            danger_entry_step = self._segment_entry_step.get(danger_keys[0], total_steps)
        else:
            danger_entry_step = total_steps

        current = self.stage
        is_better = (
            current.safe_max_x == 0 or  # Aucun stage sauvegardé encore
            safe_max_x > current.safe_max_x or
            (safe_max_x == current.safe_max_x and total_blocks > current.total_blocks) or
            (safe_max_x == current.safe_max_x and total_blocks == current.total_blocks and
             danger_entry_step < current.total_steps)
        )

        if is_better:
            # Construire les nouvelles séquences depuis ce run
            new_sequences = {}
            for key, seq in self._run_macros_per_segment.items():
                if key in danger_keys:
                    continue
                total_macros = sum(c for _, c in seq)
                if seq and total_macros <= 8:
                    new_sequences[key] = list(seq)

            # Si même distance, fusionner avec les anciennes séquences :
            # conserver les segments de l'ancien run non couverts par le nouveau
            # (évite de perdre des séquences clés comme max_jump x8 sur un tuyau)
            if safe_max_x == current.safe_max_x:
                for key, seq in current.sequences.items():
                    if key not in new_sequences and key not in danger_keys:
                        new_sequences[key] = seq

            self.stage = StageRecord(
                safe_max_x=safe_max_x,
                total_blocks=total_blocks,
                total_steps=danger_entry_step,
                danger_keys=danger_keys,
                sequences=new_sequences,
                run_id=self._run_id,
            )
            print(f"💾 Stage sauvegardé: safe_max_x={safe_max_x}, "
                  f"{total_blocks} blocs, {len(new_sequences)} segments, danger: {danger_keys}")
        else:
            print(f"⏭️ Stage non écrasé (safe_max_x={safe_max_x} vs {current.safe_max_x})")

        self._save()

    def get_context_for_position(self, x: int) -> str:
        """
        Génère le contexte mémoire pour Claude à la position actuelle.
        Inclut : le segment courant + les 3 segments suivants (anticipation élargie).
        """
        lines = [f"📚 MÉMOIRE DES RUNS PRÉCÉDENTES ({self.total_runs} runs, record: x={self.furthest_x}px):"]

        seg_idx = int(x) // SEGMENT_SIZE
        segments_to_show = [
            (self._key(x), "Zone actuelle"),
            (f"{(seg_idx + 1) * SEGMENT_SIZE}-{(seg_idx + 2) * SEGMENT_SIZE}", "Zone +100px"),
            (f"{(seg_idx + 2) * SEGMENT_SIZE}-{(seg_idx + 3) * SEGMENT_SIZE}", "Zone +200px"),
            (f"{(seg_idx + 3) * SEGMENT_SIZE}-{(seg_idx + 4) * SEGMENT_SIZE}", "Zone +300px"),
        ]

        any_data = False
        for key, label in segments_to_show:
            seg = self.segments.get(key)
            if not seg:
                continue
            any_data = True

            approach_warn = (f", ⚠️ approche mortelle x{seg.approach_death_count}"
                             if seg.approach_death_count >= 1 else "")
            lines.append(f"\n  [{label} {key}] ({seg.passages_count} passages"
                         + approach_warn + ")")

            success_lines = seg.success_summary()
            if success_lines:
                lines.append("  SOLUTIONS CONFIRMÉES (exécuter directement):")
                lines.extend(success_lines)

            death_lines = seg.death_summary()
            if death_lines:
                lines.append("  MORTS PRÉCÉDENTES (à éviter absolument):")
                lines.extend(death_lines)

            inter_lines = seg.interaction_summary()
            if inter_lines:
                lines.append("  INTERACTIONS CONNUES:")
                lines.extend(inter_lines)

        if not any_data:
            lines.append("  (aucune donnée pour cette zone — première exploration)")

        return "\n".join(lines)

    def is_deadly_approach(self, x: int, last_action: str) -> Optional[str]:
        """
        Retourne un avertissement si cette action a déjà causé une mort ici.
        Utilisé comme garde-fou avant d'exécuter une action.
        """
        seg = self.segments.get(self._key(x))
        if not seg:
            return None
        for d in seg.deaths:
            if abs(d.x - x) < 30 and d.last_action == last_action and d.count >= 2:
                return (f"⚠️ DANGER MÉMORISÉ: '{last_action}' a causé {d.count} morts "
                        f"à x={d.x} dans cette zone!")
        return None

    # ------------------------------------------------------------------ #
    # Interne                                                              #
    # ------------------------------------------------------------------ #

    def _key(self, x: int) -> str:
        start = (int(x) // SEGMENT_SIZE) * SEGMENT_SIZE
        return f"{start}-{start + SEGMENT_SIZE}"

    def _get_or_create(self, key: str, x: int) -> SegmentData:
        if key not in self.segments:
            start = (int(x) // SEGMENT_SIZE) * SEGMENT_SIZE
            self.segments[key] = SegmentData(
                segment_key=key, start_x=start, end_x=start + SEGMENT_SIZE
            )
        return self.segments[key]

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        stage_dict = {
            "safe_max_x": self.stage.safe_max_x,
            "total_blocks": self.stage.total_blocks,
            "total_steps": self.stage.total_steps,
            "danger_keys": self.stage.danger_keys,
            "sequences": {k: [list(t) for t in v] for k, v in self.stage.sequences.items()},
            "run_id": self.stage.run_id,
        }
        data = {
            "furthest_x": self.furthest_x,
            "total_runs": self.total_runs,
            "stage": stage_dict,
            "segments": {
                k: {
                    **{f: v for f, v in asdict(seg).items()
                       if f not in ("deaths", "interactions", "successes")},
                    "deaths": [asdict(d) for d in seg.deaths],
                    "interactions": [asdict(i) for i in seg.interactions],
                    "successes": [asdict(s) for s in seg.successes],
                }
                for k, seg in self.segments.items()
            }
        }

        def _default(obj):
            # gym/numpy renvoient des int64/float64 non sérialisables
            if hasattr(obj, 'item'):
                return obj.item()
            raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_default)

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.furthest_x = data.get("furthest_x", 0)
            self.total_runs = data.get("total_runs", 0)

            # Charger le StageRecord
            stage_data = data.get("stage", {})
            if stage_data:
                sequences = {
                    k: [tuple(t) for t in v]
                    for k, v in stage_data.get("sequences", {}).items()
                }
                self.stage = StageRecord(
                    safe_max_x=stage_data.get("safe_max_x", 0),
                    total_blocks=stage_data.get("total_blocks", 0),
                    total_steps=stage_data.get("total_steps", 0),
                    danger_keys=stage_data.get("danger_keys", []),
                    sequences=sequences,
                    run_id=stage_data.get("run_id"),
                )

            for key, seg_data in data.get("segments", {}).items():
                deaths = [DeathRecord(**d) for d in seg_data.pop("deaths", [])]
                interactions = [InteractionRecord(**i) for i in seg_data.pop("interactions", [])]
                successes = [SuccessRecord(**s) for s in seg_data.pop("successes", [])]
                # Supprimer les anciens champs per-segment qui n'existent plus
                seg_data.pop("best_sequence", None)
                seg_data.pop("best_steps", None)
                seg_data.pop("best_blocks_hit", None)
                seg_data.pop("best_run_id", None)
                seg = SegmentData(**seg_data)
                seg.deaths = deaths
                seg.interactions = interactions
                seg.successes = successes
                self.segments[key] = seg
        except Exception as e:
            print(f"⚠️ Erreur chargement mémoire segments: {e}")
