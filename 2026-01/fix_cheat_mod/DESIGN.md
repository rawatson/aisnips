# Selectable Advances Cheat Mod - Design Document

## Overview

This mod for Europa Universalis V (EU5) allows players to unlock advances (technology/research options) from other nations, cultures, and religions through a cheat menu accessible by right-clicking their ruler.

The mod is fully auto-generated from game data using `generate_mod.py`, ensuring consistency and easy updates when the game changes.

---

## Architecture

### File Structure

```
fix_cheat_mod/
├── common/                          # Source game data (copy from game files)
│   ├── advances/                    # Advance definition files
│   └── ...
├── localization/                    # Source game localization
│   └── english/
│       └── country_names_l_english.yml
├── output_mod_folder/               # Generated mod output
│   ├── .metadata/
│   │   └── metadata.json
│   ├── in_game/
│   │   ├── common/
│   │   │   ├── advances/            # Transformed advance files
│   │   │   ├── character_interactions/
│   │   │   └── scripted_triggers/
│   │   └── events/
│   └── main_menu/
│       └── localization/
│           └── english/
│               └── adv_cheat_l_english.yml
├── generate_mod.py                  # Generation script
├── config.yaml                      # Configuration
├── DESIGN.md                        # This document
└── CLAUDE.md                        # AI assistant notes
```

### Components

#### 1. Generator Script (`generate_mod.py`)

The main Python script that:
- Parses PDX script files using a custom parser
- Extracts entity references (countries, cultures, religions, etc.) from `potential` blocks
- Deduplicates countries with identical advance sets
- Generates transformed advance files with cheat variable checks
- Creates the event menu system
- Generates scripted triggers for UI state
- Produces localization with game-native variable lookups
- Exports to the game mod folder

#### 2. Configuration (`config.yaml`)

Contains:
- **export_path**: Where to copy the generated mod (e.g., game's mod folder)
- **mod_name, mod_id, mod_version, game_version**: Metadata
- **excluded_files**: Glob patterns for files to skip (generic game mechanics)
- **formable_nations**: Country tags that get "(formable)" suffix in menu
- **country_regions**: Manual mapping of country tags to regions
- **culture_regions**: Manual mapping of cultures/culture groups to regions

#### 3. Entry Point: Character Interaction

**File:** `in_game/common/character_interactions/adv_cheat_interaction.txt`

- Adds a menu option when right-clicking the ruler
- Only available to human players
- Triggers the main event `adv_cheat_event.1`

#### 4. Event System (UI/Logic)

**File:** `in_game/events/adv_cheat_event.txt`

Hierarchical menu structure:
```
Main Menu (event 1)
├── Countries → Region Selector (event 10) → Region Menus (events 100+)
├── Cultures → Region Selector (event 20) → Region Menus (events 200+)
├── Religions → Direct Menu (event 30)
└── Governments → Direct Menu (event 40)
```

Each entity option toggles a variable (e.g., `set_variable = adv_cheat_ENG`).

Menu features:
- Selected items highlighted in green (`#g text#!` formatting)
- Custom tooltips showing advances grouped by age
- Regions ordered by continent (Europe → Asia → Africa → Americas → Oceania)
- Merged entries for countries with identical advance sets

#### 5. Scripted Triggers

**File:** `in_game/common/scripted_triggers/adv_cheat_triggers.txt`

Defines helper triggers like `is_adv_cheat_western_europe_on` that check if any entity in a region has been selected. Used for UI highlighting in region selectors.

#### 6. Transformed Advance Files

**Directory:** `in_game/common/advances/`

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
            has_variable = adv_cheat_ENG    # Cheat bypass
            AND = {
                has_or_had_tag = ENG        # Original condition preserved
            }
        }
    }
    ...
}
```

#### 7. Localization

**File:** `main_menu/localization/english/adv_cheat_l_english.yml`

Features:
- Uses `$TAG$` lookups for country names (pulls from game localization)
- Green text formatting: `#g text#!`
- Custom tooltips with advance details:
  ```
  4 advances
  $BULLET_WITH_TAB$[ShowShortAgeName('age_1_traditions')]: [ShowAdvanceName('advance_1')], [ShowAdvanceName('advance_2')]
  $BULLET_WITH_TAB$[ShowShortAgeName('age_2_renaissance')]: [ShowAdvanceName('advance_3')], [ShowAdvanceName('advance_4')]
  ```

---

## Entity Types Supported

| Type | Variable Prefix | Example |
|------|----------------|---------|
| Country | `adv_cheat_` | `adv_cheat_ENG` |
| Culture | `adv_cheat_culture_` | `adv_cheat_culture_welsh` |
| Culture Group | `adv_cheat_culgroup_` | `adv_cheat_culgroup_chinese_group` |
| Religion | `adv_cheat_religion_` | `adv_cheat_religion_catholic` |
| Government | `adv_cheat_gov_` | `adv_cheat_gov_monarchy` |
| Region | `adv_cheat_region_` | (detected but not shown in menu) |
| Area | `adv_cheat_area_` | (detected but not shown in menu) |

---

## Special Features

### Country Deduplication

Countries with identical advance sets are merged into single menu entries displaying all names:
- "Bavaria, Lower Bavaria, Upper Bavaria, ..." (7 Bavarian tags)
- "Cusco, Inca Empire"
- "Meissen, Saxony"

This reduces menu clutter while maintaining full functionality.

### Formable Nations

Countries in the `formable_nations` config list get a "(formable)" suffix:
- "Great Britain (formable)"
- "Roman Empire (formable)"

### Region Ordering

Regions are grouped by continent for logical menu navigation:
1. Europe: Western, Central, Southern, Northern, Eastern
2. Asia: Western, Central, South, Southeast, East, Japan
3. Africa: North, West, Central, East, Southern
4. Americas: North, Central, South
5. Other: Oceania, Special, Other

### Culture Group Support

Both individual cultures and culture groups are supported:
- Individual cultures: "Welsh", "Castilian"
- Culture groups: "Chinese (Group)", "Japanese (Group)"

The "(Group)" suffix distinguishes groups from individual cultures with similar names.

---

## Data Flow

```
Source Game Files (common/advances/*.txt)
            ↓
    PDX Parser extracts:
    - Advance definitions
    - Potential blocks
    - Entity references (tags, cultures, religions)
            ↓
    Entity Extractor builds:
    - Entity registry
    - Advance-to-entity mapping
            ↓
    Country Deduplicator finds:
    - Countries with identical advance sets
    - Creates merged entities
            ↓
    Mod Generator outputs:
    - Transformed advance files
    - Event menu system
    - Scripted triggers
    - Localization
            ↓
    Export copies to game mod folder
```

---

## Usage

### Generating the Mod

```bash
python generate_mod.py
```

Output:
- Processes all advance files
- Reports entity counts and advance breakdown by age
- Lists any merged/deduplicated countries
- Generates all mod files
- Exports to configured game mod folder

### Configuration

Edit `config.yaml` to:
- Set the export path for your game installation
- Update game version for compatibility
- Add new formable nations
- Override region assignments for countries/cultures

### Updating for New Game Versions

1. Copy new game files to `common/` and `localization/`
2. Review and update `excluded_files` if needed
3. Add any new formable nations to the config
4. Run `generate_mod.py`
5. Test in-game

---

## Statistics

| Component | Current Count |
|-----------|---------------|
| Source advance files | ~162 (processed) |
| Unique entities | ~175 |
| Country regions | 22 |
| Culture regions | ~15 |
| Supported game version | 1.0.10 |

---

## Benefits of Automated Generation

1. **Accuracy**: All entities from source data automatically included
2. **Maintainability**: Game updates only require re-running the generator
3. **Consistency**: Events, triggers, and advance files always in sync
4. **Localization**: Uses game's native variable lookups for proper names
5. **Debuggability**: Clear mapping from source to output
6. **Deduplication**: Automatically detects and merges identical advance sets
