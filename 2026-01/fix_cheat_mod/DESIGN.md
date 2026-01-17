# Selectable Advances Cheat Mod - Design Document

## Overview

This mod for Europa Universalis V (EU5) allows players to unlock advances (technology/research options) from other nations, cultures, and religions through a cheat menu accessible by right-clicking their ruler.

---

## Part A: Original Design

### Architecture

The mod consists of four interconnected components:

#### 1. Entry Point: Character Interaction
**File:** `output_mod_folder/in_game/common/character_interactions/euro_tech_interaction.txt`

- Adds a menu option when right-clicking the ruler
- Only available to human players
- Triggers the main event `euro_advances_event.1`

#### 2. Event System (UI/Logic)
**File:** `output_mod_folder/in_game/events/euro_advances_event.txt` (7,403 lines, 77 events)

Hierarchical menu structure in Korean:
```
Main Menu (event.1)
├── 현존 국가 (Existing Countries) → event.10 → event.1000-1021 (by region)
├── 형성 국가 (Formable Countries) → event.11 → event.1100-1121 (by region)
├── 문화/언어 (Culture/Language) → event.20 → event.2000-2020 (by region)
├── 시대 진보 (Age Advances) → event.30 → event.3000-3005 (by age)
└── 기타 진보 (Other Advances) → event.40 (religions, regions, etc.)
```

Each country/culture option toggles a variable (e.g., `set_variable = euro_tech_ENG`).

#### 3. Scripted Triggers
**File:** `output_mod_folder/in_game/common/scripted_triggers/euro_advances_triggers.txt` (747 lines)

Defines helper triggers like `is_euro_advances_western_europe_on` that check if any country in a region has been selected. Used for UI state (showing selected/unselected status).

#### 4. Modified Advance Files
**Directory:** `output_mod_folder/in_game/common/advances/` (162 files)

Each advance's `potential` block is transformed to include cheat variables:

**Original (source):**
```pdx
english_tradition = {
    age = age_1_traditions
    potential = {
        has_or_had_tag = ENG
    }
    ...
}
```

**Transformed (output):**
```pdx
english_tradition = {
    age = age_1_traditions
    potential = {
        OR = {
            has_variable = euro_tech_ENG    # Cheat bypass
            AND = {
                has_or_had_tag = ENG        # Original condition preserved
            }
        }
    }
    ...
}
```

### Data Flow

```
Player clicks ruler → Character Interaction → Event Menu
                                                  ↓
Player selects England → set_variable = euro_tech_ENG
                                                  ↓
Advance files check: OR { has_variable = euro_tech_ENG, has_or_had_tag = ENG }
                                                  ↓
English advances become visible and researchable
```

### Current Statistics

| Component | Count |
|-----------|-------|
| Source advance files | 189 |
| Output advance files | 162 |
| Events | 77 |
| Event file lines | 7,403 |
| Scripted triggers | ~100 |
| Supported game version | 1.0.8 |

---

## Part B: Identified Issues

### 1. Missing Countries (6 country files not transformed)
- `country_ITA.txt` (Italy)
- `country_KRS.txt` (Kurdistan)
- `country_MCH.txt` (Manchuria)
- `country_MLC.txt` (Malacca)
- `country_MSA.txt` (Mysore?)
- `country_SKO.txt` (South Korea/Joseon-related?)

These countries exist in source data but have no transformed output, making their advances inaccessible via the cheat menu.

### 2. Hand-Coded Event File
The 7,403-line event file appears to be entirely hand-written:
- Country lists hard-coded in each regional event
- Menu structure manually maintained
- Korean text inline (no localization file)
- Prone to typos and omissions

### 3. Hand-Coded Triggers
The scripted triggers file has manually curated country lists that must match:
- The event file options
- The actual advance file transformations
- The source game data

### 4. Version Drift
- Mod targets version 1.0.8
- Git history shows "Full EU5 1.0.10 import"
- New countries/advances in 1.0.10 may not be covered

### 5. Inconsistent Transformation
Comparing source to output reveals the transformation was likely done semi-manually:
- Some files transformed differently than others
- The `4_choices_*.txt` files renamed to `zz_choices_*.txt` but not all advances included
- Culture checks (e.g., `culture = culture:welsh`) extract country tags inconsistently

### 6. No Generation Pipeline
Without automation, every game update requires:
- Manual identification of new/changed advance files
- Manual transformation of potential blocks
- Manual update of events for new countries
- Manual update of triggers
- Manual update of localization

---

## Part C: Recommended Changes

### Primary Recommendation: Create an Automated Generation Pipeline

Build a Python script (`generate_mod.py`) that:

#### Phase 1: Parse Source Data
```
common/advances/*.txt → Extract:
  - All advance definitions
  - potential blocks (has_or_had_tag, culture, religion checks)
  - Country tag associations
  - Culture associations
  - Religion associations
```

#### Phase 2: Build Country/Culture/Religion Registry
```
For each advance file:
  - Extract all referenced country tags (ENG, FRA, etc.)
  - Extract all referenced cultures (culture:welsh, etc.)
  - Extract all referenced religions
  - Map advance files → entities they unlock for
```

#### Phase 3: Generate Output Files

**A. Transform Advance Files**
- Parse each source file
- Wrap `potential` blocks with `OR { has_variable = euro_tech_XXX, AND { original } }`
- Handle multi-tag advances (e.g., ENG/WLS/GBR shares)
- Output to `output_mod_folder/in_game/common/advances/`

**B. Generate Events File**
- Build hierarchical menus from discovered data
- Generate event options programmatically
- Include proper toggle logic (set/remove variable)
- Output to `output_mod_folder/in_game/events/`

**C. Generate Scripted Triggers**
- Create `is_euro_advances_*_on` triggers from country lists
- Auto-generate all regional groupings
- Output to `output_mod_folder/in_game/common/scripted_triggers/`

**D. Generate Localization**
- Create English localization file
- Optionally keep Korean as alternative
- Output to `output_mod_folder/in_game/localization/`

### Data Model

```python
@dataclass
class AdvanceEntity:
    type: str  # "country", "culture", "religion", "region", "government"
    id: str    # "ENG", "welsh", "catholic", etc.
    display_name: str
    region: str  # "western_europe", "east_asia", etc.
    files: List[str]  # Which advance files reference this

@dataclass
class RegionGroup:
    id: str
    display_name: str
    entities: List[AdvanceEntity]
```

### File Organization After Changes

```
fix_cheat_mod/
├── common/                          # Source game data (unchanged)
├── output_mod_folder/               # Generated mod output
├── generate_mod.py                  # Generation script
├── config.yaml                      # Configuration (regions, display names)
├── DESIGN.md                        # This document
└── templates/                       # Jinja2 templates for events/triggers
    ├── event_menu.txt.j2
    ├── event_country.txt.j2
    └── triggers.txt.j2
```

### Benefits

1. **Accuracy**: All countries/cultures from source data automatically included
2. **Maintainability**: Game updates only require re-running the generator
3. **Consistency**: Events, triggers, and advance files always in sync
4. **Localization**: Proper localization support instead of inline Korean
5. **Debuggability**: Clear mapping from source → output
6. **Extensibility**: Easy to add new categories (governments, playstyles, etc.)

### Implementation Approach

1. Write a PDX script parser (the game's text format)
2. Parse all `common/advances/*.txt` files
3. Extract entity references from `potential` blocks
4. Group entities by region (using a configuration file)
5. Generate all output files using templates
6. Update metadata.json with current game version

### Configuration File Example (`config.yaml`)

```yaml
game_version: "1.0.10"
mod_version: "2.0"

regions:
  western_europe:
    display_name: "Western Europe"
    countries: [ENG, SCO, WLS, FRA, BUR, ...]  # Auto-populated, can override

  # ... other regions

excluded_files:
  - "0_age_*.txt"
  - "1_building_unlocks.txt"
  - "_advances_template.txt"
  # Files that shouldn't be transformed
```

---

## Summary

The current mod is functional but fragile due to hand-coded components that can drift out of sync with the game data. The recommended solution is to build an automated generation pipeline that:

1. Parses the source game data
2. Extracts all relevant entities (countries, cultures, religions)
3. Generates all mod components programmatically
4. Ensures consistency and completeness
5. Makes future updates trivial

This approach transforms maintenance from "manually edit 8,000+ lines across multiple files" to "run a script and review the output."
