#!/usr/bin/env python3
"""
Selectable Advances Cheat Mod Generator

Automatically generates mod files from EU5 game data:
- Transformed advance files with cheat variable checks
- Event menus for selecting countries/cultures/religions
- Scripted triggers for UI state tracking
- Localization files
"""

import re
import os
import json
import yaml
import fnmatch
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict


# =============================================================================
# PDX Script Parser
# =============================================================================

class PDXParser:
    """Parser for Paradox script files (.txt)"""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)

    def parse(self) -> Dict[str, Any]:
        """Parse the entire file into a dictionary"""
        result = {}
        while self.pos < self.length:
            self.skip_whitespace_and_comments()
            if self.pos >= self.length:
                break

            key = self.parse_key()
            if key is None:
                break

            self.skip_whitespace_and_comments()
            if self.pos >= self.length:
                break

            # Expect '='
            if self.text[self.pos] == '=':
                self.pos += 1
                self.skip_whitespace_and_comments()
                value = self.parse_value()

                # Handle duplicate keys by converting to list
                if key in result:
                    if not isinstance(result[key], list):
                        result[key] = [result[key]]
                    result[key].append(value)
                else:
                    result[key] = value
            else:
                # Bare key without value (shouldn't happen often)
                result[key] = True

        return result

    def skip_whitespace_and_comments(self):
        """Skip whitespace and # comments"""
        while self.pos < self.length:
            c = self.text[self.pos]
            if c in ' \t\n\r':
                self.pos += 1
            elif c == '#':
                # Skip to end of line
                while self.pos < self.length and self.text[self.pos] != '\n':
                    self.pos += 1
            elif c == '\ufeff':  # BOM
                self.pos += 1
            else:
                break

    def parse_key(self) -> Optional[str]:
        """Parse an identifier/key"""
        self.skip_whitespace_and_comments()
        if self.pos >= self.length:
            return None

        start = self.pos
        while self.pos < self.length:
            c = self.text[self.pos]
            if c.isalnum() or c in '_-.:':
                self.pos += 1
            else:
                break

        if self.pos == start:
            return None

        return self.text[start:self.pos]

    def parse_value(self) -> Any:
        """Parse a value (string, number, block, or identifier)"""
        self.skip_whitespace_and_comments()
        if self.pos >= self.length:
            return None

        c = self.text[self.pos]

        if c == '{':
            return self.parse_block()
        elif c == '"':
            return self.parse_quoted_string()
        elif c == '-' or c.isdigit():
            return self.parse_number_or_identifier()
        else:
            return self.parse_identifier()

    def parse_block(self) -> Dict[str, Any]:
        """Parse a { } block"""
        assert self.text[self.pos] == '{'
        self.pos += 1

        result = {}
        while self.pos < self.length:
            self.skip_whitespace_and_comments()
            if self.pos >= self.length:
                break

            if self.text[self.pos] == '}':
                self.pos += 1
                break

            key = self.parse_key()
            if key is None:
                # Might be a bare value in a list-like block
                break

            self.skip_whitespace_and_comments()

            if self.pos < self.length and self.text[self.pos] == '=':
                self.pos += 1
                self.skip_whitespace_and_comments()
                value = self.parse_value()
            else:
                # Bare key (like in lists: { tag1 tag2 tag3 })
                value = True

            # Handle duplicate keys
            if key in result:
                if not isinstance(result[key], list):
                    result[key] = [result[key]]
                result[key].append(value)
            else:
                result[key] = value

        return result

    def parse_quoted_string(self) -> str:
        """Parse a "quoted string" """
        assert self.text[self.pos] == '"'
        self.pos += 1

        start = self.pos
        while self.pos < self.length and self.text[self.pos] != '"':
            self.pos += 1

        result = self.text[start:self.pos]
        if self.pos < self.length:
            self.pos += 1  # Skip closing quote

        return result

    def parse_number_or_identifier(self) -> Any:
        """Parse a number or identifier starting with - or digit"""
        start = self.pos
        if self.text[self.pos] == '-':
            self.pos += 1

        has_dot = False
        while self.pos < self.length:
            c = self.text[self.pos]
            if c.isdigit():
                self.pos += 1
            elif c == '.' and not has_dot:
                has_dot = True
                self.pos += 1
            elif c.isalpha() or c == '_':
                # It's an identifier, not a number
                return self.parse_identifier_from(start)
            else:
                break

        num_str = self.text[start:self.pos]
        try:
            if has_dot:
                return float(num_str)
            else:
                return int(num_str)
        except ValueError:
            return num_str

    def parse_identifier_from(self, start: int) -> str:
        """Continue parsing an identifier from a given start position"""
        while self.pos < self.length:
            c = self.text[self.pos]
            if c.isalnum() or c in '_-.:':
                self.pos += 1
            else:
                break
        return self.text[start:self.pos]

    def parse_identifier(self) -> str:
        """Parse an identifier"""
        start = self.pos
        while self.pos < self.length:
            c = self.text[self.pos]
            if c.isalnum() or c in '_-.:':
                self.pos += 1
            else:
                break
        return self.text[start:self.pos]


def parse_pdx_file(filepath: Path) -> Dict[str, Any]:
    """Parse a PDX script file"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        text = f.read()
    parser = PDXParser(text)
    return parser.parse()


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class AdvanceEntity:
    """An entity (country, culture, religion) that can unlock advances"""
    entity_type: str  # "country", "culture", "religion", "government", "region"
    entity_id: str    # "ENG", "welsh", "catholic", etc.
    variable_name: str  # The variable to set (e.g., "adv_cheat_ENG")
    display_name: str = ""
    region: str = ""
    source_files: Set[str] = field(default_factory=set)

    def __hash__(self):
        return hash((self.entity_type, self.entity_id))

    def __eq__(self, other):
        return self.entity_type == other.entity_type and self.entity_id == other.entity_id


@dataclass
class Advance:
    """An advance definition from the game data"""
    advance_id: str
    source_file: str
    age: str = ""
    potential: Dict = field(default_factory=dict)
    raw_content: str = ""
    entities: Set[str] = field(default_factory=set)  # Entity IDs this advance is for


# =============================================================================
# Entity Extraction
# =============================================================================

class EntityExtractor:
    """Extracts entity references from advance potential blocks"""

    def __init__(self):
        self.entities: Dict[str, AdvanceEntity] = {}
        self.advance_entities: Dict[str, Set[str]] = defaultdict(set)  # advance_id -> entity_ids

    def extract_from_potential(self, potential: Dict, advance_id: str, source_file: str) -> Set[str]:
        """Extract all entity references from a potential block"""
        entities = set()
        self._extract_recursive(potential, entities, source_file)
        return entities

    def _extract_recursive(self, obj: Any, entities: Set[str], source_file: str):
        """Recursively extract entities from nested structures"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Country tags
                if key == 'has_or_had_tag':
                    self._add_country(value, entities, source_file)
                elif key == 'tag':
                    self._add_country(value, entities, source_file)

                # Culture checks
                elif key == 'culture':
                    self._add_culture(value, entities, source_file)
                elif key == 'has_culture_group':
                    self._add_culture_group(value, entities, source_file)
                elif key == 'culture_group':
                    self._add_culture_group(value, entities, source_file)

                # Religion checks
                elif key == 'religion':
                    self._add_religion(value, entities, source_file)
                elif key == 'has_religion':
                    self._add_religion(value, entities, source_file)

                # Government checks
                elif key == 'government':
                    self._add_government(value, entities, source_file)
                elif key == 'has_government':
                    self._add_government(value, entities, source_file)

                # Region/area checks
                elif key == 'capital_region':
                    self._add_region(value, entities, source_file)
                elif key == 'capital_area':
                    self._add_area(value, entities, source_file)

                # Recurse into nested structures
                elif isinstance(value, (dict, list)):
                    self._extract_recursive(value, entities, source_file)

        elif isinstance(obj, list):
            for item in obj:
                self._extract_recursive(item, entities, source_file)

    def _add_country(self, value: Any, entities: Set[str], source_file: str):
        """Add a country entity"""
        if isinstance(value, str):
            tag = value.upper()
            entity_id = f"country:{tag}"
            entities.add(entity_id)
            if entity_id not in self.entities:
                self.entities[entity_id] = AdvanceEntity(
                    entity_type="country",
                    entity_id=tag,
                    variable_name=f"adv_cheat_{tag}",
                    display_name=tag,
                    source_files={source_file}
                )
            else:
                self.entities[entity_id].source_files.add(source_file)
        elif isinstance(value, list):
            for v in value:
                self._add_country(v, entities, source_file)

    def _add_culture(self, value: Any, entities: Set[str], source_file: str):
        """Add a culture entity"""
        if isinstance(value, str):
            # Handle "culture:welsh" format
            if ':' in value:
                culture = value.split(':')[1]
            else:
                culture = value
            entity_id = f"culture:{culture}"
            entities.add(entity_id)
            if entity_id not in self.entities:
                self.entities[entity_id] = AdvanceEntity(
                    entity_type="culture",
                    entity_id=culture,
                    variable_name=f"adv_cheat_culture_{culture}",
                    display_name=culture.replace('_', ' ').title(),
                    source_files={source_file}
                )
            else:
                self.entities[entity_id].source_files.add(source_file)
        elif isinstance(value, list):
            for v in value:
                self._add_culture(v, entities, source_file)

    def _add_culture_group(self, value: Any, entities: Set[str], source_file: str):
        """Add a culture group entity"""
        if isinstance(value, str):
            if ':' in value:
                group = value.split(':')[1]
            else:
                group = value
            entity_id = f"culture_group:{group}"
            entities.add(entity_id)
            if entity_id not in self.entities:
                self.entities[entity_id] = AdvanceEntity(
                    entity_type="culture_group",
                    entity_id=group,
                    variable_name=f"adv_cheat_culgroup_{group}",
                    display_name=group.replace('_', ' ').title(),
                    source_files={source_file}
                )
            else:
                self.entities[entity_id].source_files.add(source_file)
        elif isinstance(value, list):
            for v in value:
                self._add_culture_group(v, entities, source_file)

    def _add_religion(self, value: Any, entities: Set[str], source_file: str):
        """Add a religion entity"""
        if isinstance(value, str):
            if ':' in value:
                religion = value.split(':')[1]
            else:
                religion = value
            entity_id = f"religion:{religion}"
            entities.add(entity_id)
            if entity_id not in self.entities:
                self.entities[entity_id] = AdvanceEntity(
                    entity_type="religion",
                    entity_id=religion,
                    variable_name=f"adv_cheat_religion_{religion}",
                    display_name=religion.replace('_', ' ').title(),
                    source_files={source_file}
                )
            else:
                self.entities[entity_id].source_files.add(source_file)
        elif isinstance(value, list):
            for v in value:
                self._add_religion(v, entities, source_file)

    def _add_government(self, value: Any, entities: Set[str], source_file: str):
        """Add a government entity"""
        if isinstance(value, str):
            entity_id = f"government:{value}"
            entities.add(entity_id)
            if entity_id not in self.entities:
                self.entities[entity_id] = AdvanceEntity(
                    entity_type="government",
                    entity_id=value,
                    variable_name=f"adv_cheat_gov_{value}",
                    display_name=value.replace('_', ' ').title(),
                    source_files={source_file}
                )
            else:
                self.entities[entity_id].source_files.add(source_file)

    def _add_region(self, value: Any, entities: Set[str], source_file: str):
        """Add a region entity"""
        if isinstance(value, str):
            if ':' in value:
                region = value.split(':')[1]
            else:
                region = value
            entity_id = f"region:{region}"
            entities.add(entity_id)
            if entity_id not in self.entities:
                self.entities[entity_id] = AdvanceEntity(
                    entity_type="region",
                    entity_id=region,
                    variable_name=f"adv_cheat_region_{region}",
                    display_name=region.replace('_', ' ').title(),
                    source_files={source_file}
                )
            else:
                self.entities[entity_id].source_files.add(source_file)

    def _add_area(self, value: Any, entities: Set[str], source_file: str):
        """Add an area entity"""
        if isinstance(value, str):
            if ':' in value:
                area = value.split(':')[1]
            else:
                area = value
            entity_id = f"area:{area}"
            entities.add(entity_id)
            if entity_id not in self.entities:
                self.entities[entity_id] = AdvanceEntity(
                    entity_type="area",
                    entity_id=area,
                    variable_name=f"adv_cheat_area_{area}",
                    display_name=area.replace('_', ' ').title(),
                    source_files={source_file}
                )
            else:
                self.entities[entity_id].source_files.add(source_file)


# =============================================================================
# Advance File Processor
# =============================================================================

class AdvanceProcessor:
    """Processes advance files and extracts data"""

    def __init__(self, common_path: Path, config: Dict):
        self.common_path = common_path
        self.config = config
        self.advances_path = common_path / "advances"
        self.extractor = EntityExtractor()
        self.advances: Dict[str, Advance] = {}
        self.files_to_transform: Dict[str, Dict] = {}  # filename -> parsed content

    def process_all(self):
        """Process all advance files"""
        excluded = set(self.config.get('excluded_files', []))

        for filepath in sorted(self.advances_path.glob("*.txt")):
            filename = filepath.name

            # Check exclusions
            if self._is_excluded(filename, excluded):
                print(f"  Skipping (excluded): {filename}")
                continue

            print(f"  Processing: {filename}")
            self._process_file(filepath)

    def _is_excluded(self, filename: str, excluded: Set[str]) -> bool:
        """Check if a file should be excluded using glob patterns"""
        for pattern in excluded:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def _process_file(self, filepath: Path):
        """Process a single advance file"""
        try:
            parsed = parse_pdx_file(filepath)
            filename = filepath.name

            # Store for transformation
            self.files_to_transform[filename] = parsed

            # Extract entities from each advance
            for advance_id, advance_data in parsed.items():
                if not isinstance(advance_data, dict):
                    continue

                potential = advance_data.get('potential', {})
                if potential:
                    entities = self.extractor.extract_from_potential(
                        potential, advance_id, filename
                    )

                    advance = Advance(
                        advance_id=advance_id,
                        source_file=filename,
                        age=advance_data.get('age', ''),
                        potential=potential,
                        entities=entities
                    )
                    self.advances[advance_id] = advance

        except Exception as e:
            print(f"    Error processing {filepath}: {e}")


# =============================================================================
# Output Generator
# =============================================================================

class ModGenerator:
    """Generates mod output files"""

    VARIABLE_PREFIX = "adv_cheat_"
    NAMESPACE = "adv_cheat_event"

    def __init__(self, processor: AdvanceProcessor, output_path: Path, config: Dict):
        self.processor = processor
        self.output_path = output_path
        self.config = config
        self.entities = processor.extractor.entities

        # Organize entities by type and region
        self.countries: Dict[str, List[AdvanceEntity]] = defaultdict(list)
        self.cultures: Dict[str, List[AdvanceEntity]] = defaultdict(list)
        self.religions: List[AdvanceEntity] = []
        self.governments: List[AdvanceEntity] = []
        self.regions: List[AdvanceEntity] = []
        self.areas: List[AdvanceEntity] = []

        self._organize_entities()

    def _organize_entities(self):
        """Organize entities into categories"""
        region_mapping = self.config.get('country_regions', {})
        culture_region_mapping = self.config.get('culture_regions', {})
        country_names = self.config.get('country_names', {})

        for entity_id, entity in self.entities.items():
            if entity.entity_type == "country":
                # Determine region from config or source file
                region = region_mapping.get(entity.entity_id, self._guess_region_from_file(entity))
                entity.region = region
                # Set display name from config
                entity.display_name = country_names.get(entity.entity_id, entity.entity_id)
                self.countries[region].append(entity)

            elif entity.entity_type in ("culture", "culture_group"):
                region = culture_region_mapping.get(entity.entity_id, "other")
                entity.region = region
                # Format culture names nicely
                entity.display_name = entity.entity_id.replace('_', ' ').title()
                self.cultures[region].append(entity)

            elif entity.entity_type == "religion":
                entity.display_name = entity.entity_id.replace('_', ' ').title()
                self.religions.append(entity)

            elif entity.entity_type == "government":
                entity.display_name = entity.entity_id.replace('_', ' ').title()
                self.governments.append(entity)

            elif entity.entity_type == "region":
                entity.display_name = entity.entity_id.replace('_', ' ').title()
                self.regions.append(entity)

            elif entity.entity_type == "area":
                entity.display_name = entity.entity_id.replace('_', ' ').title()
                self.areas.append(entity)

        # Sort entities within each category
        for region in self.countries:
            self.countries[region].sort(key=lambda e: e.entity_id)
        for region in self.cultures:
            self.cultures[region].sort(key=lambda e: e.entity_id)
        self.religions.sort(key=lambda e: e.entity_id)
        self.governments.sort(key=lambda e: e.entity_id)

    def _guess_region_from_file(self, entity: AdvanceEntity) -> str:
        """Guess region based on source file naming"""
        for filename in entity.source_files:
            # Check if filename hints at region
            fname_lower = filename.lower()
            if 'japan' in fname_lower:
                return 'east_asia'
            if 'china' in fname_lower or 'chi' in fname_lower:
                return 'east_asia'
            if 'india' in fname_lower:
                return 'south_asia'
        return 'other'

    def generate_all(self):
        """Generate all mod files"""
        self._ensure_directories()
        self.generate_advance_files()
        self.generate_events()
        self.generate_triggers()
        self.generate_character_interaction()
        self.generate_localization()
        self.generate_metadata()

    def _ensure_directories(self):
        """Create output directories"""
        dirs = [
            self.output_path / "in_game" / "common" / "advances",
            self.output_path / "in_game" / "common" / "scripted_triggers",
            self.output_path / "in_game" / "common" / "character_interactions",
            self.output_path / "in_game" / "events",
            self.output_path / "in_game" / "localization" / "english",
            self.output_path / ".metadata",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def generate_advance_files(self):
        """Generate transformed advance files (only those with entity-specific advances)"""
        print("\nGenerating advance files...")
        output_dir = self.output_path / "in_game" / "common" / "advances"

        generated_count = 0
        skipped_count = 0

        for filename, parsed_data in self.processor.files_to_transform.items():
            # Check if this file has any entity-specific advances
            has_entities = self._file_has_entity_advances(parsed_data, filename)

            if not has_entities:
                skipped_count += 1
                continue

            output_file = output_dir / filename
            content = self._transform_advances(parsed_data, filename)

            with open(output_file, 'w', encoding='utf-8-sig') as f:
                f.write(content)
            print(f"  Generated: {filename}")
            generated_count += 1

        print(f"  ({skipped_count} files skipped - no entity-specific advances)")

    def _file_has_entity_advances(self, parsed_data: Dict, filename: str) -> bool:
        """Check if a file contains any advances with entity-specific potential blocks"""
        for advance_id, advance_data in parsed_data.items():
            if not isinstance(advance_data, dict):
                continue

            potential = advance_data.get('potential', {})
            if potential:
                entities = self.processor.extractor.extract_from_potential(
                    potential, advance_id, filename
                )
                if entities:
                    return True
        return False

    def _transform_advances(self, parsed_data: Dict, filename: str) -> str:
        """Transform advance definitions to include cheat variables"""
        lines = []

        for advance_id, advance_data in parsed_data.items():
            if not isinstance(advance_data, dict):
                continue

            potential = advance_data.get('potential', {})
            entities = set()

            if potential:
                # Extract entities for this advance
                entities = self.processor.extractor.extract_from_potential(
                    potential, advance_id, filename
                )

            lines.append(self._generate_advance_block(advance_id, advance_data, entities))

        return '\n'.join(lines)

    def _generate_advance_block(self, advance_id: str, data: Dict, entities: Set[str]) -> str:
        """Generate a single advance block with transformed potential"""
        lines = [f"{advance_id} = {{"]

        for key, value in data.items():
            if key == 'potential':
                # Generate transformed potential
                lines.append(self._generate_transformed_potential(value, entities))
            else:
                # Pass through other properties
                lines.append(f"\t{self._serialize_value(key, value)}")

        lines.append("}")
        return '\n'.join(lines)

    def _generate_transformed_potential(self, original: Dict, entities: Set[str]) -> str:
        """Generate potential block with cheat variable OR conditions"""
        if not entities:
            # No entities, just serialize original
            return f"\tpotential = {self._serialize_dict(original, 1)}"

        lines = ["\tpotential = {", "\t\tOR = {"]

        # Add cheat variable checks
        for entity_id in sorted(entities):
            if entity_id in self.entities:
                var_name = self.entities[entity_id].variable_name
                lines.append(f"\t\t\thas_variable = {var_name}")

        # Add original conditions wrapped in AND
        lines.append("\t\t\tAND = {")
        for key, value in original.items():
            lines.append(f"\t\t\t\t{self._serialize_value(key, value, 4)}")
        lines.append("\t\t\t}")

        lines.append("\t\t}")
        lines.append("\t}")

        return '\n'.join(lines)

    def _serialize_value(self, key: str, value: Any, indent: int = 2) -> str:
        """Serialize a key-value pair to PDX format"""
        tabs = '\t' * indent

        if isinstance(value, dict):
            return f"{key} = {self._serialize_dict(value, indent)}"
        elif isinstance(value, bool):
            return f"{key} = {'yes' if value else 'no'}"
        elif isinstance(value, (int, float)):
            return f"{key} = {value}"
        elif isinstance(value, str):
            if ' ' in value or any(c in value for c in '{}='):
                return f'{key} = "{value}"'
            return f"{key} = {value}"
        elif isinstance(value, list):
            # Handle list of values with same key
            return '\n'.join(self._serialize_value(key, v, indent) for v in value)
        else:
            return f"{key} = {value}"

    def _serialize_dict(self, d: Dict, indent: int = 0) -> str:
        """Serialize a dictionary to PDX block format"""
        tabs = '\t' * indent
        inner_tabs = '\t' * (indent + 1)

        if not d:
            return "{ }"

        lines = ["{"]
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{inner_tabs}{key} = {self._serialize_dict(value, indent + 1)}")
            elif isinstance(value, bool):
                lines.append(f"{inner_tabs}{key} = {'yes' if value else 'no'}")
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, dict):
                        lines.append(f"{inner_tabs}{key} = {self._serialize_dict(v, indent + 1)}")
                    else:
                        lines.append(f"{inner_tabs}{key} = {v}")
            else:
                lines.append(f"{inner_tabs}{key} = {value}")
        lines.append(f"{tabs}}}")

        return '\n'.join(lines)

    def generate_events(self):
        """Generate the event menu system"""
        print("\nGenerating events...")
        output_file = self.output_path / "in_game" / "events" / f"{self.NAMESPACE}.txt"

        lines = [f"namespace = {self.NAMESPACE}", ""]

        # Build region lists and event number mappings
        country_regions = sorted([r for r in self.countries.keys() if self.countries[r]])
        culture_regions = sorted([r for r in self.cultures.keys() if self.cultures[r]])

        # Event numbering scheme:
        # 1 = Main menu
        # 10 = Country region selector
        # 100-199 = Country region menus (100 + index)
        # 20 = Culture region selector
        # 200-299 = Culture region menus (200 + index)
        # 30 = Religion menu
        # 40 = Government menu

        # Main menu event (event 1)
        lines.extend(self._generate_main_menu(country_regions, culture_regions))

        # Country region selector (event 10)
        lines.extend(self._generate_region_selector("country", country_regions, 10, 100))

        # Individual country region menus (events 100+)
        for idx, region in enumerate(country_regions):
            lines.extend(self._generate_entity_menu("country", region, self.countries[region], 100 + idx, 10))

        # Culture region selector (event 20)
        lines.extend(self._generate_region_selector("culture", culture_regions, 20, 200))

        # Individual culture region menus (events 200+)
        for idx, region in enumerate(culture_regions):
            lines.extend(self._generate_entity_menu("culture", region, self.cultures[region], 200 + idx, 20))

        # Religion menu (event 30)
        if self.religions:
            lines.extend(self._generate_entity_menu("religion", "all", self.religions, 30, 1))

        # Government menu (event 40)
        if self.governments:
            lines.extend(self._generate_entity_menu("government", "all", self.governments, 40, 1))

        with open(output_file, 'w', encoding='utf-8-sig') as f:
            f.write('\n'.join(lines))
        print(f"  Generated: {self.NAMESPACE}.txt")

    def _generate_main_menu(self, country_regions: List[str], culture_regions: List[str]) -> List[str]:
        """Generate the main menu event"""
        lines = [
            f"# Main menu",
            f"{self.NAMESPACE}.1 = {{",
            "\ttype = country_event",
            f"\ttitle = {self.NAMESPACE}.1.title",
            "",
            '\timage = "gfx/interface/illustrations/event/backgrounds/special/comet_sighted.dds"',
            "",
            "\timmediate = {",
            "\t\tset_variable = adv_cheat_menu_open",
            "\t}",
            "",
            "\ttrigger = {",
            "\t\tis_human = yes",
            "\t}",
            "",
        ]

        # Country option (leads to region selector)
        if country_regions:
            lines.extend([
                "\toption = {",
                f"\t\tname = {self.NAMESPACE}.1.countries",
                f"\t\ttrigger_event_silently = {{ id = {self.NAMESPACE}.10 }}",
                "\t}",
                "",
            ])

        # Culture option (leads to region selector)
        if culture_regions:
            lines.extend([
                "\toption = {",
                f"\t\tname = {self.NAMESPACE}.1.cultures",
                f"\t\ttrigger_event_silently = {{ id = {self.NAMESPACE}.20 }}",
                "\t}",
                "",
            ])

        # Religion option
        if self.religions:
            lines.extend([
                "\toption = {",
                f"\t\tname = {self.NAMESPACE}.1.religions",
                f"\t\ttrigger_event_silently = {{ id = {self.NAMESPACE}.30 }}",
                "\t}",
                "",
            ])

        # Government option
        if self.governments:
            lines.extend([
                "\toption = {",
                f"\t\tname = {self.NAMESPACE}.1.governments",
                f"\t\ttrigger_event_silently = {{ id = {self.NAMESPACE}.40 }}",
                "\t}",
                "",
            ])

        # Exit option
        lines.extend([
            "\toption = {",
            f"\t\tname = {self.NAMESPACE}.exit",
            "\t\tremove_variable = adv_cheat_menu_open",
            "\t}",
            "}",
            "",
        ])

        return lines

    def _generate_region_selector(self, entity_type: str, regions: List[str], event_num: int, target_base: int) -> List[str]:
        """Generate a region selector menu with conditional highlighting"""
        lines = [
            f"# {entity_type.title()} region selector",
            f"{self.NAMESPACE}.{event_num} = {{",
            "\ttype = country_event",
            f"\ttitle = {self.NAMESPACE}.{event_num}.title",
            "",
            '\timage = "gfx/interface/illustrations/event/backgrounds/special/comet_sighted.dds"',
            "",
            "\ttrigger = {",
            "\t\tis_human = yes",
            "\t\thas_variable = adv_cheat_menu_open",
            "\t}",
            "",
        ]

        # Option for each region with conditional highlighting
        for idx, region in enumerate(regions):
            target_event = target_base + idx
            trigger_name = f"is_adv_cheat_{region}_on" if entity_type == "country" else f"is_adv_cheat_culture_{region}_on"
            lines.extend([
                "\toption = {",
                "\t\tname = {",
                f"\t\t\ttext = {self.NAMESPACE}.region.{region}",
                f"\t\t\ttrigger = {{ NOT = {{ {trigger_name} = yes }} }}",
                "\t\t}",
                "\t\tname = {",
                f"\t\t\ttext = {self.NAMESPACE}.region.{region}.selected",
                f"\t\t\ttrigger = {{ {trigger_name} = yes }}",
                "\t\t}",
                f"\t\ttrigger_event_silently = {{ id = {self.NAMESPACE}.{target_event} }}",
                "\t}",
                "",
            ])

        # Back option
        lines.extend([
            "\toption = {",
            f"\t\tname = {self.NAMESPACE}.back",
            f"\t\ttrigger_event_silently = {{ id = {self.NAMESPACE}.1 }}",
            "\t}",
            "}",
            "",
        ])

        return lines

    def _generate_entity_menu(self, entity_type: str, region: str, entities: List[AdvanceEntity], event_num: int, back_event: int) -> List[str]:
        """Generate a menu for selecting individual entities"""
        if not entities:
            return []

        region_display = region.replace('_', ' ').title() if region != "all" else entity_type.title()

        lines = [
            f"# {entity_type.title()} - {region_display}",
            f"{self.NAMESPACE}.{event_num} = {{",
            "\ttype = country_event",
            f"\ttitle = {self.NAMESPACE}.{event_num}.title",
            "",
            '\timage = "gfx/interface/illustrations/event/backgrounds/special/comet_sighted.dds"',
            "",
            "\ttrigger = {",
            "\t\tis_human = yes",
            "\t\thas_variable = adv_cheat_menu_open",
            "\t}",
            "",
        ]

        # Options for each entity
        for entity in entities:
            lines.extend(self._generate_entity_option(entity, event_num))

        # Back option
        lines.extend([
            "\toption = {",
            f"\t\tname = {self.NAMESPACE}.back",
            f"\t\ttrigger_event_silently = {{ id = {self.NAMESPACE}.{back_event} }}",
            "\t}",
            "}",
            "",
        ])

        return lines

    def _generate_entity_option(self, entity: AdvanceEntity, event_num: int) -> List[str]:
        """Generate an option for toggling an entity"""
        var_name = entity.variable_name

        return [
            "\toption = {",
            "\t\tname = {",
            f"\t\t\ttext = {self.NAMESPACE}.{entity.entity_type}.{entity.entity_id}",
            f"\t\t\ttrigger = {{ NOT = {{ has_variable = {var_name} }} }}",
            "\t\t}",
            "\t\tname = {",
            f"\t\t\ttext = {self.NAMESPACE}.{entity.entity_type}.{entity.entity_id}.selected",
            f"\t\t\ttrigger = {{ has_variable = {var_name} }}",
            "\t\t}",
            "\t\tif = {",
            f"\t\t\tlimit = {{ NOT = {{ has_variable = {var_name} }} }}",
            f"\t\t\tset_variable = {var_name}",
            "\t\t}",
            "\t\telse = {",
            f"\t\t\tremove_variable = {var_name}",
            "\t\t}",
            f"\t\ttrigger_event_silently = {{ id = {self.NAMESPACE}.{event_num} }}",
            "\t}",
            "",
        ]

    def generate_triggers(self):
        """Generate scripted triggers for UI state"""
        print("\nGenerating triggers...")
        output_file = self.output_path / "in_game" / "common" / "scripted_triggers" / "adv_cheat_triggers.txt"

        lines = ["# Auto-generated scripted triggers for advance cheat mod", ""]

        # Triggers for each region (countries)
        for region, entities in sorted(self.countries.items()):
            if not entities:
                continue
            trigger_name = f"is_adv_cheat_{region}_on"
            lines.append(f"{trigger_name} = {{")
            lines.append("\tOR = {")
            for entity in entities:
                lines.append(f"\t\thas_variable = {entity.variable_name}")
            lines.append("\t}")
            lines.append("}")
            lines.append("")

        # Triggers for cultures
        for region, entities in sorted(self.cultures.items()):
            if not entities:
                continue
            trigger_name = f"is_adv_cheat_culture_{region}_on"
            lines.append(f"{trigger_name} = {{")
            lines.append("\tOR = {")
            for entity in entities:
                lines.append(f"\t\thas_variable = {entity.variable_name}")
            lines.append("\t}")
            lines.append("}")
            lines.append("")

        # Master "any selected" triggers
        lines.append("is_adv_cheat_any_country_on = {")
        lines.append("\tOR = {")
        for region in sorted(self.countries.keys()):
            if self.countries[region]:
                lines.append(f"\t\tis_adv_cheat_{region}_on = yes")
        lines.append("\t}")
        lines.append("}")
        lines.append("")

        lines.append("is_adv_cheat_any_culture_on = {")
        lines.append("\tOR = {")
        for region in sorted(self.cultures.keys()):
            if self.cultures[region]:
                lines.append(f"\t\tis_adv_cheat_culture_{region}_on = yes")
        lines.append("\t}")
        lines.append("}")
        lines.append("")

        with open(output_file, 'w', encoding='utf-8-sig') as f:
            f.write('\n'.join(lines))
        print(f"  Generated: adv_cheat_triggers.txt")

    def generate_character_interaction(self):
        """Generate character interaction for menu access"""
        print("\nGenerating character interaction...")
        output_file = self.output_path / "in_game" / "common" / "character_interactions" / "adv_cheat_interaction.txt"

        content = f"""adv_cheat_menu = {{
\ton_own_nation = yes
\tis_consort_action = no
\tmessage = no
\t
\tai_tick = never
\t
\tpotential = {{
\t\tis_human = yes
\t}}

\tallow = {{
\t\tis_human = yes
\t}}
\t
\tselect_trigger = {{
\t\tlooking_for_a = character
\t\tsource = actor
\t\ttarget_flag = recipient
\t\tname = "choose_character"
\t\tcolumn = {{
\t\t\tdata = name
\t\t}}
\t\tvisible = {{
\t\t\tscope:actor = {{
\t\t\t\truler ?= root
\t\t\t}}
\t\t}}
\t}}

\teffect = {{
\t\tscope:actor = {{
\t\t\ttrigger_event_non_silently = {self.NAMESPACE}.1
\t\t\tset_variable = adv_cheat_menu_open
\t\t}}
\t}}
\t
\tai_will_do = {{
\t\tbase = -1000
\t}}
}}
"""

        with open(output_file, 'w', encoding='utf-8-sig') as f:
            f.write(content)
        print(f"  Generated: adv_cheat_interaction.txt")

    def generate_localization(self):
        """Generate localization file"""
        print("\nGenerating localization...")
        output_file = self.output_path / "in_game" / "localization" / "english" / "adv_cheat_l_english.yml"

        # Get sorted region lists
        country_regions = sorted([r for r in self.countries.keys() if self.countries[r]])
        culture_regions = sorted([r for r in self.cultures.keys() if self.cultures[r]])

        lines = ["l_english:", ""]

        # Main menu
        lines.append(f' {self.NAMESPACE}.1.title: "Select Advances"')
        lines.append(f' {self.NAMESPACE}.1.countries: "Countries"')
        lines.append(f' {self.NAMESPACE}.1.cultures: "Cultures"')
        lines.append(f' {self.NAMESPACE}.1.religions: "Religions"')
        lines.append(f' {self.NAMESPACE}.1.governments: "Governments"')
        lines.append(f' {self.NAMESPACE}.exit: "Exit"')
        lines.append(f' {self.NAMESPACE}.back: "Back"')
        lines.append("")

        # Region selector titles
        lines.append(f' {self.NAMESPACE}.10.title: "Select Country Region"')
        lines.append(f' {self.NAMESPACE}.20.title: "Select Culture Region"')
        lines.append(f' {self.NAMESPACE}.30.title: "Select Religion"')
        lines.append(f' {self.NAMESPACE}.40.title: "Select Government Type"')
        lines.append("")

        # Region names for navigation
        all_regions = set(country_regions) | set(culture_regions)
        for region in sorted(all_regions):
            display = region.replace('_', ' ').title()
            lines.append(f' {self.NAMESPACE}.region.{region}: "{display}"')
            lines.append(f' {self.NAMESPACE}.region.{region}.selected: "#G{display}#!"')
        lines.append("")

        # Country region menu titles
        for idx, region in enumerate(country_regions):
            display = region.replace('_', ' ').title()
            lines.append(f' {self.NAMESPACE}.{100 + idx}.title: "{display} Countries"')

        # Culture region menu titles
        for idx, region in enumerate(culture_regions):
            display = region.replace('_', ' ').title()
            lines.append(f' {self.NAMESPACE}.{200 + idx}.title: "{display} Cultures"')
        lines.append("")

        # Entity names
        for entity in self.entities.values():
            base_key = f"{self.NAMESPACE}.{entity.entity_type}.{entity.entity_id}"
            display = entity.display_name or entity.entity_id.replace('_', ' ').title()
            lines.append(f' {base_key}: "{display}"')
            lines.append(f' {base_key}.selected: "#G{display}#!"')

        with open(output_file, 'w', encoding='utf-8-sig') as f:
            f.write('\n'.join(lines))
        print(f"  Generated: adv_cheat_l_english.yml")

    def generate_metadata(self):
        """Generate mod metadata"""
        print("\nGenerating metadata...")
        output_file = self.output_path / ".metadata" / "metadata.json"

        metadata = {
            "name": self.config.get('mod_name', "Selectable Advances Cheat Mod"),
            "id": self.config.get('mod_id', "selectable_advances_cheat"),
            "version": self.config.get('mod_version', "2.0"),
            "supported_game_version": self.config.get('game_version', "1.0.10"),
            "short_description": "Cheat mode to unlock other nations' advances.\nRight-click your ruler to access the cheat menu.\nSelect countries, cultures, or religions to unlock their advances.",
            "tags": ["Gameplay"],
            "relationships": [],
            "game_custom_data": {}
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent='\t')
        print(f"  Generated: metadata.json")


# =============================================================================
# Main
# =============================================================================

def load_config(config_path: Path) -> Dict:
    """Load configuration file"""
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def export_mod(output_path: Path, export_path: str) -> bool:
    """Export the generated mod to the target location (e.g., game mod folder)"""
    if not export_path:
        print("\nNo export path configured, skipping export.")
        return False

    target = Path(export_path)

    print("\n" + "=" * 60)
    print("Exporting mod...")
    print("=" * 60)
    print(f"  Source: {output_path}")
    print(f"  Target: {target}")

    # Check if target parent directory exists
    if not target.parent.exists():
        print(f"  ERROR: Parent directory does not exist: {target.parent}")
        print("  Make sure the game mod folder exists.")
        return False

    try:
        # Remove existing target if it exists
        if target.exists():
            print(f"  Removing existing mod folder...")
            shutil.rmtree(target)

        # Copy the entire output folder to target
        print(f"  Copying mod files...")
        shutil.copytree(output_path, target)

        print(f"  Export complete!")
        return True

    except PermissionError as e:
        print(f"  ERROR: Permission denied - {e}")
        print("  Make sure the target folder is not in use.")
        return False
    except Exception as e:
        print(f"  ERROR: Failed to export - {e}")
        return False


def main():
    # Paths
    base_path = Path(__file__).parent
    common_path = base_path / "common"
    output_path = base_path / "output_mod_folder"
    config_path = base_path / "config.yaml"

    # Load config
    config = load_config(config_path)

    print("=" * 60)
    print("Selectable Advances Cheat Mod Generator")
    print("=" * 60)

    # Process advance files
    print("\nProcessing advance files...")
    processor = AdvanceProcessor(common_path, config)
    processor.process_all()

    print(f"\nFound {len(processor.extractor.entities)} unique entities")
    print(f"Found {len(processor.advances)} advances with potential blocks")

    # Generate output
    print("\n" + "=" * 60)
    print("Generating mod files...")
    print("=" * 60)

    generator = ModGenerator(processor, output_path, config)
    generator.generate_all()

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)

    # Summary
    print(f"\nSummary:")
    print(f"  Countries: {sum(len(v) for v in generator.countries.values())}")
    print(f"  Cultures: {sum(len(v) for v in generator.cultures.values())}")
    print(f"  Religions: {len(generator.religions)}")
    print(f"  Governments: {len(generator.governments)}")

    # Export to game mod folder
    export_path = config.get('export_path', '')
    if export_path:
        export_mod(output_path, export_path)


if __name__ == "__main__":
    main()
