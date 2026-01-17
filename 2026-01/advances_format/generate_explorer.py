#!/usr/bin/env python3
"""
Generate an interactive HTML explorer for advances data.
"""

import os
import re
import json
from collections import defaultdict
from pathlib import Path

ADVANCES_DIR = Path("advances")

# Age mapping for display
AGE_NAMES = {
    "age_1_traditions": "Age 1: Traditions (1337)",
    "age_2_renaissance": "Age 2: Renaissance (1337-1437)",
    "age_3_discovery": "Age 3: Discovery (1437-1537)",
    "age_4_reformation": "Age 4: Reformation (1537-1637)",
    "age_5_absolutism": "Age 5: Absolutism (1637-1737)",
    "age_6_revolutions": "Age 6: Revolutions (1737-1837)",
}

AGE_ORDER = list(AGE_NAMES.keys())

# Country name mapping
COUNTRY_NAMES = {
    'ENG': 'England', 'FRA': 'France', 'CAS': 'Castile', 'ARA': 'Aragon',
    'POR': 'Portugal', 'HAB': 'Austria (Habsburg)', 'POL': 'Poland',
    'LIT': 'Lithuania', 'TUR': 'Ottomans', 'MOS': 'Muscovy', 'VEN': 'Venice',
    'GEN': 'Genoa', 'DAN': 'Denmark', 'SWE': 'Sweden', 'NOR': 'Norway',
    'SCO': 'Scotland', 'BYZ': 'Byzantium', 'NAP': 'Naples', 'MIL': 'Milan',
    'FLO': 'Florence', 'PAP': 'Papal States', 'HUN': 'Hungary', 'BOH': 'Bohemia',
    'BUR': 'Burgundy', 'ETH': 'Ethiopia', 'MAM': 'Mamluks', 'TIM': 'Timurids',
    'DLH': 'Delhi', 'VIJ': 'Vijayanagar', 'CHI': 'Ming China', 'KOR': 'Korea',
    'AYU': 'Ayutthaya', 'MAJ': 'Majapahit', 'AZT': 'Aztec', 'INC': 'Inca',
    'NED': 'Netherlands', 'PRU': 'Prussia', 'RUS': 'Russia', 'GBR': 'Great Britain',
    'SPA': 'Spain', 'ITA': 'Italy', 'USA': 'United States', 'MEI': 'Meissen',
    'SAX': 'Saxony', 'BRA': 'Brandenburg', 'TEU': 'Teutonic Order', 'LIV': 'Livonian Order',
    'HSA': 'Hansa', 'NOV': 'Novgorod', 'KAZ': 'Kazan', 'CRI': 'Crimea',
    'GRA': 'Granada', 'MAL': 'Mali', 'SON': 'Songhai', 'KON': 'Kongo',
    'ZAN': 'Zanzibar', 'OMA': 'Oman', 'YEM': 'Yemen', 'IRA': 'Persia',
    'MUG': 'Mughals', 'BEN': 'Bengal', 'SIA': 'Siam', 'KHM': 'Khmer',
    'JAP': 'Japan', 'MCH': 'Manchu', 'TIB': 'Tibet', 'CHG': 'Chagatai',
    'GLH': 'Golden Horde', 'NAV': 'Navarra', 'TUN': 'Tunisia', 'MOR': 'Morocco',
    'BUL': 'Bulgaria', 'SER': 'Serbia', 'WAL': 'Wallachia', 'CRO': 'Croatia',
    'GEO': 'Georgia', 'TRE': 'Trebizond', 'CIR': 'Circassia', 'ILK': 'Ilkhanate',
    'QAR': 'Qara Qoyunlu', 'AQQ': 'Aq Qoyunlu', 'KNI': 'Knights', 'CTH': 'Carthage',
    'DAH': 'Dahomey', 'ORY': 'Oyo', 'MAI': 'Mandinka',
    'KBO': 'Kanem Bornu', 'AIR': 'Air', 'HIN': 'Hindustan', 'ROM': 'Roman Empire',
    'MGE': 'Mongol Empire', 'SKO': 'Shogunate', 'DAI': 'Dai Viet', 'SMO': 'Smolensk',
    'NRM': 'Normandy', 'PRO': 'Provence', 'TLE': 'Tlaxcala', 'SUN': 'Sunda',
    'MLO': 'Malwa', 'ORM': 'Ormond', 'SIC': 'Sicily', 'SCA': 'Scandinavia',
    'TWS': 'Taiwan', 'KAR': 'Karamanids', 'KOJ': 'Joseon', 'KRS': 'Kursk',
    'MLC': 'Malacca', 'MSA': 'Mossi', 'VER': 'Verona', 'LON': 'London',
    'LOR': 'Lorraine', 'CHA': 'Champa', 'SAV': 'Savoy', 'TNC': 'Tencteri',
    'UBV': 'Urbino', 'ZMW': 'Zimbabwe', 'BOS': 'Bosnia', 'PAL': 'Palatinate',
    'TNK': 'Tonkin', 'SUK': 'Sukhothai', 'SME': 'Sukhothai', 'BRI': 'Britain',
    'ORI': 'Orissa', 'PLC': 'Poland-Lithuania', 'HOL': 'Holland', 'CSU': 'Cusco',
    'TUS': 'Tuscany', 'GER': 'Germany', 'FIN': 'Finland', 'IRE': 'Ireland',
    'WLS': 'Wales',
    # Japanese Daimyos
    'KGK': 'Kaga', 'AKT': 'Akita', 'ASA': 'Asai', 'CBA': 'Chiba', 'CSK': 'Chosokabe',
    'DTE': 'Date', 'HTK': 'Hatakeyama', 'HSK': 'Hosokawa', 'KTB': 'Kitabatake',
    'MRI': 'Mori', 'OGS': 'Ogasawara', 'OTM': 'Otomo', 'OUC': 'Ouchi', 'SBA': 'Shiba',
    'SOO': 'Soo', 'TKD': 'Takeda', 'UES': 'Uesugi', 'IMG': 'Imagawa', 'ASK': 'Ashikaga',
    'ODA': 'Oda', 'SMZ': 'Shimazu', 'YMN': 'Yamana', 'TRP': 'Tsurugi', 'HYM': 'Hayama',
    'AKM': 'Akama', 'SUW': 'Suwa', 'KNO': 'Kono', 'NBU': 'Nanbu', 'STK': 'Satake',
    'SHN': 'Shoni', 'TKI': 'Toki', 'UTN': 'Utsunomiya', 'ISK': 'Isshiki', 'ITO': 'Ito',
    'KKC': 'Kikuchi', 'NUS': 'Nishiuwa',
    # German states
    'FRE': 'Freising', 'REG': 'Regensburg', 'LBV': 'Lübeck', 'BAV': 'Bavaria',
    'PSS': 'Passau', 'SLZ': 'Salzburg', 'GRF': 'Graubünden', 'STY': 'Styria',
    'PIS': 'Pisa', 'SIE': 'Siena', 'TIR': 'Tyrol',
    # Lowlands
    'BRB': 'Brabant', 'FLA': 'Flanders', 'GEL': 'Gelre', 'HAI': 'Hainaut',
    'LIE': 'Liège', 'LUX': 'Luxembourg', 'UTR': 'Utrecht',
    # Native Americans
    'CEL': 'Cherokee', 'MLB': 'Maliseet', 'AHO': 'Ahojeunne', 'AIA': 'Aiabone',
    'PLE': 'Powhatan', 'NPL': 'Neapel', 'ARM': 'Armenia', 'MLK': 'Maluku', 'SBL': 'Sibir',
}

# Known modifier display names
MODIFIER_NAMES = {
    'army_infantry_power': 'Infantry Combat Ability',
    'global_maritime_presence_modifier': 'Maritime Presence',
    'discipline': 'Discipline',
    'monthly_prestige': 'Monthly Prestige',
    'monthly_legitimacy': 'Monthly Legitimacy',
    'spy_network_construction': 'Spy Network Construction',
    'prestige_from_land_battle': 'Prestige from Land Battles',
    'land_morale_modifier': 'Land Morale',
    'global_manpower_modifier': 'Manpower Modifier',
    'mercenary_maintenance_cost': 'Mercenary Maintenance',
    'global_max_rgo_size_modifier': 'Max RGO Size',
    'city_upgrade_cost_modifier': 'City Upgrade Cost',
    'town_upgrade_cost_modifier': 'Town Upgrade Cost',
    'global_max_literacy': 'Max Literacy',
    'prestige_decay': 'Prestige Decay',
    'monthly_diplomats': 'Monthly Diplomats',
    'cultural_tradition_modifier': 'Cultural Tradition',
    'stability_investment': 'Stability Investment',
    'monthly_army_tradition': 'Monthly Army Tradition',
    'monthly_religious_influence': 'Monthly Religious Influence',
    'clergy_estate_target_satisfaction': 'Clergy Satisfaction',
    'court_spending_cost': 'Court Spending Cost',
    'global_medicaments_output_modifier': 'Medicaments Output',
    'cultural_influence_modifier': 'Cultural Influence',
    'global_clergy_food_consumption': 'Clergy Food Consumption',
    'peasants_estate_target_satisfaction': 'Peasants Satisfaction',
    'cardinal_price_cost_modifier': 'Cardinal Price',
    'global_monthly_prosperity': 'Monthly Prosperity',
    'monthly_rebel_growth': 'Rebel Growth',
    'global_clergy_max_literacy': 'Clergy Max Literacy',
    'create_colonial_charter_cost_modifier': 'Colonial Charter Cost',
    'research_speed_modifier': 'Research Speed',
    'global_pop_conversion_speed_modifier': 'Conversion Speed',
    'tolerance_own': 'Tolerance of True Faith',
    'diplomatic_reputation': 'Diplomatic Reputation',
    'num_possible_artists': 'Available Artists',
    'legislative_efficiency': 'Legislative Efficiency',
    'global_separatism': 'Separatism',
    'embrace_institution_cost_modifier': 'Institution Embrace Cost',
    'hire_artist_cost_modifier': 'Artist Hire Cost',
    'building_missionary_effort_modifier': 'Missionary Effort',
    'navy_heavy_ship_power': 'Heavy Ship Combat Ability',
    'diplomatic_capacity': 'Diplomatic Capacity',
    'global_production_efficiency': 'Production Efficiency',
    'navy_maintenance_cost': 'Navy Maintenance',
    'global_estate_target_satisfaction': 'Estate Satisfaction',
    'trade_efficiency': 'Trade Efficiency',
    'global_urban_build_buildings_cost': 'Urban Building Cost',
    'army_disembark_speed': 'Disembark Speed',
    'blockade_efficiency': 'Blockade Efficiency',
    'navy_initiative': 'Naval Initiative',
    'global_tools_output_modifier': 'Tools Output',
    'global_cloth_output_modifier': 'Cloth Output',
    'global_lumber_output_modifier': 'Lumber Output',
    'global_monthly_food_modifier': 'Monthly Food',
    'peasants_estate_satisfaction_recovery': 'Peasant Satisfaction Recovery',
    'nobles_estate_target_satisfaction': 'Nobles Satisfaction',
    'experience_decay': 'Experience Decay',
    'army_tradition_decay': 'Army Tradition Decay',
    'navy_tradition_decay': 'Navy Tradition Decay',
    'global_mercenaries_modifier': 'Mercenary Pool',
    'max_diplomats': 'Max Diplomats',
    'monthly_republican_tradition': 'Monthly Republican Tradition',
    'monthly_devotion': 'Monthly Devotion',
    'monthly_horde_unity': 'Monthly Horde Unity',
    'army_initiative': 'Army Initiative',
    'antagonism_received_modifier': 'Antagonism Received',
    'diplomatic_capacity_modifier': 'Diplomatic Capacity',
    'hire_mercenary_premium_cost_modifier': 'Mercenary Premium Cost',
    'hire_mercenary_leader_cost_modifier': 'Mercenary Leader Cost',
    'global_integration_speed_modifier': 'Integration Speed',
    'global_build_buildings_cost': 'Building Cost',
    'cultures_capacity': 'Culture Capacity',
    'army_maintenance_cost': 'Army Maintenance',
    'global_distance_from_capital_cost_modifier': 'Distance from Capital Cost',
    'global_monthly_literacy': 'Monthly Literacy',
    'global_max_food_storage_modifier': 'Max Food Storage',
    'regiment_reinforcement_speed': 'Regiment Reinforcement Speed',
    'monthly_navy_tradition': 'Monthly Navy Tradition',
    'global_monthly_dev': 'Monthly Development',
    'colonization_speed': 'Colonization Speed',
    'global_colonial_growth': 'Colonial Growth',
    'war_exhaustion_cost': 'War Exhaustion Cost',
    'core_creation_cost': 'Core Creation Cost',
    'ae_impact': 'Aggressive Expansion Impact',
    'improve_relation_modifier': 'Improve Relations',
    'province_war_score_modifier': 'Province War Score',
    'siege_ability': 'Siege Ability',
    'fort_maintenance_modifier': 'Fort Maintenance',
    'global_goods_produced_modifier': 'Goods Produced',
    'global_trade_goods_size': 'Trade Goods Size',
    'fire_damage_received': 'Fire Damage Received',
    'shock_damage_received': 'Shock Damage Received',
    'fire_damage': 'Fire Damage',
    'shock_damage': 'Shock Damage',
}


def parse_advance_block(block_content, advance_name):
    """Parse a single advance block and extract all details."""
    advance = {
        'name': advance_name,
        'display_name': advance_name.replace('_', ' ').title(),
        'age': None,
        'requires': [],
        'unlocks': [],
        'modifiers': [],
        'potential': {},
        'allow': None,
        'icon': None,
    }

    # Extract age
    age_match = re.search(r'age\s*=\s*(\w+)', block_content)
    if age_match:
        advance['age'] = age_match.group(1)

    # Extract icon
    icon_match = re.search(r'icon\s*=\s*(\w+)', block_content)
    if icon_match:
        advance['icon'] = icon_match.group(1)

    # Extract requires
    for req in re.findall(r'requires\s*=\s*(\w+)', block_content):
        advance['requires'].append(req)

    # Extract unlocks
    unlock_patterns = [
        (r'unlock_unit\s*=\s*(\w+)', 'Unit'),
        (r'unlock_ability\s*=\s*(\w+)', 'Ability'),
        (r'unlock_building\s*=\s*(\w+)', 'Building'),
        (r'unlock_law\s*=\s*(\w+)', 'Law'),
        (r'unlock_levy\s*=\s*(\w+)', 'Levy'),
        (r'unlock_government_reform\s*=\s*(\w+)', 'Government Reform'),
        (r'unlock_casus_belli\s*=\s*(\w+)', 'Casus Belli'),
        (r'unlock_subject_type\s*=\s*(\w+)', 'Subject Type'),
        (r'unlock_production_method\s*=\s*(\w+)', 'Production Method'),
        (r'unlock_heir_selection\s*=\s*(\w+)', 'Heir Selection'),
        (r'unlock_estate_privilege\s*=\s*(\w+)', 'Estate Privilege'),
        (r'unlock_interaction\s*=\s*(\w+)', 'Interaction'),
        (r'unlock_country_interaction\s*=\s*(\w+)', 'Country Interaction'),
        (r'unlock_relation_type\s*=\s*(\w+)', 'Relation Type'),
    ]

    for pattern, unlock_type in unlock_patterns:
        for match in re.findall(pattern, block_content):
            advance['unlocks'].append({
                'type': unlock_type,
                'value': match.replace('_', ' ').title()
            })

    # Extract modifiers (lines with = that aren't special keywords)
    special_keywords = {'age', 'icon', 'requires', 'potential', 'allow', 'allow_children',
                       'government', 'country_type', 'for', 'ai_weight', 'has_or_had_tag',
                       'religion', 'culture', 'has_culture_group', 'has_reform', 'capital',
                       'OR', 'AND', 'NOT', 'any_market_present_in_country', 'is_produced_in_market',
                       'is_traded_in_market', 'has_unlocked_advance_trigger', 'type', 'add',
                       'scale', 'potential_trigger', 'modifier_while_progressing'}
    unlock_keywords = {'unlock_unit', 'unlock_ability', 'unlock_building', 'unlock_law',
                      'unlock_levy', 'unlock_government_reform', 'unlock_casus_belli',
                      'unlock_subject_type', 'unlock_production_method', 'unlock_heir_selection',
                      'unlock_estate_privilege', 'unlock_interaction', 'unlock_country_interaction',
                      'unlock_relation_type'}

    # Parse modifiers by tracking brace depth
    # Only capture modifiers at depth 1 (inside the advance block, but not inside nested blocks)
    depth = 0
    for line in block_content.split('\n'):
        # Track depth BEFORE processing the line
        open_braces = line.count('{')
        close_braces = line.count('}')

        line_stripped = line.strip()
        if line_stripped.startswith('#') or not line_stripped:
            depth += open_braces - close_braces
            continue

        # Only process lines at depth 1 (top level of advance, after opening brace)
        # and when the line doesn't contain braces (not start of nested block)
        if depth == 1 and '=' in line_stripped and '{' not in line_stripped:
            line = line_stripped
            # Check for simple modifier pattern: key = value
            mod_match = re.match(r'^(\w+)\s*=\s*(.+)$', line)
            if mod_match:
                key, value = mod_match.groups()
                value = value.strip()

                if key.lower() not in {k.lower() for k in special_keywords} and key not in unlock_keywords:
                    # Try to parse as number
                    try:
                        num_value = float(value)
                        display_name = MODIFIER_NAMES.get(key, key.replace('_', ' ').title())
                        # Format the value nicely
                        if num_value == int(num_value):
                            if num_value > 0:
                                formatted = f"+{int(num_value)}"
                            else:
                                formatted = str(int(num_value))
                        else:
                            if num_value > 0:
                                if abs(num_value) < 1:
                                    formatted = f"+{num_value*100:.1f}%"
                                else:
                                    formatted = f"+{num_value}"
                            else:
                                if abs(num_value) < 1:
                                    formatted = f"{num_value*100:.1f}%"
                                else:
                                    formatted = str(num_value)
                        advance['modifiers'].append({
                            'name': display_name,
                            'value': formatted,
                            'raw_key': key
                        })
                    except ValueError:
                        # It's a string value (like small_production_efficiency_bonus)
                        display_name = MODIFIER_NAMES.get(key, key.replace('_', ' ').title())
                        advance['modifiers'].append({
                            'name': display_name,
                            'value': value.replace('_', ' ').title(),
                            'raw_key': key
                        })

        # Update depth after processing the line
        depth += open_braces - close_braces

    # Check for special boolean modifiers
    if 'allow_conquistadors = yes' in block_content:
        advance['unlocks'].append({'type': 'Special', 'value': 'Allow Conquistadors'})

    # Extract potential conditions
    potential_match = re.search(r'potential\s*=\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', block_content, re.DOTALL)
    if potential_match:
        potential_content = potential_match.group(1)
        advance['potential'] = extract_potential(potential_content)

    # Extract government requirement
    gov_match = re.search(r'^[\t ]*government\s*=\s*(\w+)', block_content, re.MULTILINE)
    if gov_match:
        advance['potential']['governments'] = [gov_match.group(1)]

    return advance


def extract_potential(potential_content):
    """Extract entity requirements from potential block."""
    result = {
        'countries': [],
        'religions': [],
        'cultures': [],
        'culture_groups': [],
        'governments': [],
        'regions': [],
        'other': []
    }

    # Country tags
    for tag in re.findall(r'has_or_had_tag\s*=\s*(\w+)', potential_content):
        result['countries'].append(tag)

    # Religions
    for rel in re.findall(r'religion\s*=\s*religion:(\w+)', potential_content):
        result['religions'].append(rel)
    for rel in re.findall(r'religion\.group\s*=\s*religion_group:(\w+)', potential_content):
        result['religions'].append(f"group:{rel}")

    # Cultures
    for cult in re.findall(r'culture\s*=\s*culture:(\w+)', potential_content):
        result['cultures'].append(cult)
    for cult_group in re.findall(r'has_culture_group\s*=\s*culture_group:(\w+)', potential_content):
        result['culture_groups'].append(cult_group)

    # Regions
    for reg in re.findall(r'capital\.region\s*\?*=\s*region:(\w+)', potential_content):
        result['regions'].append(reg)

    # Government reforms
    for reform in re.findall(r'has_reform\s*=\s*government_reform:(\w+)', potential_content):
        result['other'].append(f"Reform: {reform.replace('_', ' ').title()}")

    return result


def parse_advances_file(filepath):
    """Parse an advances file and return list of advances with full details."""
    advances = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # Find all advance blocks
    current_pos = 0

    while current_pos < len(content):
        # Find the next advance definition
        match = re.search(r'^(\w+)\s*=\s*\{', content[current_pos:], re.MULTILINE)
        if not match:
            break

        advance_name = match.group(1)
        block_start = current_pos + match.end() - 1

        # Find the matching closing brace
        brace_count = 1
        pos = block_start + 1
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1

        block_content = content[block_start:pos]
        current_pos = pos

        advance = parse_advance_block(block_content, advance_name)
        if advance['age']:  # Only include if it has an age
            advance['source_file'] = filepath.name
            advances.append(advance)

    return advances


def categorize_file(filename):
    """Return the category of a file based on its name."""
    if filename.startswith('country_'):
        return 'country'
    elif filename.startswith('culture_'):
        return 'culture'
    elif filename.startswith('religion_'):
        return 'religion'
    elif filename.startswith('government_'):
        return 'government'
    elif filename.startswith('region_'):
        return 'region'
    elif filename.startswith('area_'):
        return 'area'
    elif filename.startswith('estate_'):
        return 'estate'
    elif filename.startswith('0_age_'):
        return 'base'
    elif filename.startswith(('1_', '2_', '3_', '4_')):
        return 'unlock'
    else:
        return 'other'


def build_entity_index(all_advances):
    """Build an index of advances by entity."""
    index = {
        'countries': defaultdict(list),
        'religions': defaultdict(list),
        'cultures': defaultdict(list),
        'culture_groups': defaultdict(list),
        'governments': defaultdict(list),
        'regions': defaultdict(list),
        'base': [],
        'unlocks': [],
    }

    for adv in all_advances:
        pot = adv.get('potential', {})
        added = False

        for country in pot.get('countries', []):
            index['countries'][country].append(adv)
            added = True

        for religion in pot.get('religions', []):
            index['religions'][religion].append(adv)
            added = True

        for culture in pot.get('cultures', []):
            index['cultures'][culture].append(adv)
            added = True

        for culture_group in pot.get('culture_groups', []):
            index['culture_groups'][culture_group].append(adv)
            added = True

        for government in pot.get('governments', []):
            index['governments'][government].append(adv)
            added = True

        for region in pot.get('regions', []):
            index['regions'][region].append(adv)
            added = True

        if not added:
            category = categorize_file(adv.get('source_file', ''))
            if category == 'base':
                index['base'].append(adv)
            elif category == 'unlock':
                index['unlocks'].append(adv)

    return index


def get_entity_display_name(entity_type, entity_id):
    """Get display name for an entity."""
    if entity_type == 'countries':
        return COUNTRY_NAMES.get(entity_id, entity_id)
    else:
        return entity_id.replace('_', ' ').title()


def generate_html(index, all_advances):
    """Generate the interactive HTML explorer."""

    # Prepare data for JavaScript
    entities_data = {}

    for entity_type in ['countries', 'religions', 'cultures', 'culture_groups', 'governments', 'regions']:
        entities_data[entity_type] = {}
        for entity_id, advances in index[entity_type].items():
            display_name = get_entity_display_name(entity_type, entity_id)
            by_age = defaultdict(list)
            for adv in advances:
                by_age[adv['age']].append(adv)
            entities_data[entity_type][entity_id] = {
                'display_name': display_name,
                'id': entity_id,
                'total': len(advances),
                'by_age': dict(by_age)
            }

    # Add base and unlock advances
    entities_data['base'] = {
        'base_advances': {
            'display_name': 'Base Advances (All Nations)',
            'id': 'base_advances',
            'total': len(index['base']),
            'by_age': {}
        }
    }
    by_age = defaultdict(list)
    for adv in index['base']:
        by_age[adv['age']].append(adv)
    entities_data['base']['base_advances']['by_age'] = dict(by_age)

    entities_data['unlocks'] = {
        'unlock_advances': {
            'display_name': 'Unlock Advances (Buildings, Units, etc.)',
            'id': 'unlock_advances',
            'total': len(index['unlocks']),
            'by_age': {}
        }
    }
    by_age = defaultdict(list)
    for adv in index['unlocks']:
        by_age[adv['age']].append(adv)
    entities_data['unlocks']['unlock_advances']['by_age'] = dict(by_age)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advances Explorer</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }}

        .container {{
            display: flex;
            min-height: 100vh;
        }}

        /* Sidebar */
        .sidebar {{
            width: 320px;
            background: #16213e;
            border-right: 1px solid #0f3460;
            display: flex;
            flex-direction: column;
            height: 100vh;
            position: fixed;
            overflow: hidden;
        }}

        .sidebar-header {{
            padding: 20px;
            background: #0f3460;
            border-bottom: 1px solid #1a1a2e;
        }}

        .sidebar-header h1 {{
            font-size: 1.4rem;
            color: #e94560;
            margin-bottom: 10px;
        }}

        .search-box {{
            width: 100%;
            padding: 10px;
            border: 1px solid #0f3460;
            border-radius: 5px;
            background: #1a1a2e;
            color: #fff;
            font-size: 14px;
        }}

        .search-box:focus {{
            outline: none;
            border-color: #e94560;
        }}

        .category-tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            padding: 10px;
            background: #0f3460;
        }}

        .category-tab {{
            padding: 8px 12px;
            background: #16213e;
            border: 1px solid #1a1a2e;
            border-radius: 5px;
            color: #aaa;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}

        .category-tab:hover {{
            background: #1a1a2e;
            color: #fff;
        }}

        .category-tab.active {{
            background: #e94560;
            color: #fff;
            border-color: #e94560;
        }}

        .entity-list {{
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }}

        .entity-item {{
            padding: 12px 15px;
            margin-bottom: 5px;
            background: #1a1a2e;
            border-radius: 5px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s;
        }}

        .entity-item:hover {{
            background: #0f3460;
        }}

        .entity-item.active {{
            background: #e94560;
        }}

        .entity-name {{
            font-weight: 500;
        }}

        .entity-count {{
            background: #0f3460;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 12px;
            color: #aaa;
        }}

        .entity-item.active .entity-count {{
            background: rgba(255,255,255,0.2);
            color: #fff;
        }}

        /* Main Content */
        .main-content {{
            flex: 1;
            margin-left: 320px;
            padding: 30px;
            overflow-y: auto;
        }}

        .welcome-message {{
            text-align: center;
            padding: 100px 20px;
            color: #888;
        }}

        .welcome-message h2 {{
            font-size: 2rem;
            margin-bottom: 20px;
            color: #e94560;
        }}

        .entity-detail {{
            display: none;
        }}

        .entity-detail.active {{
            display: block;
        }}

        .entity-header {{
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #0f3460;
        }}

        .entity-header h2 {{
            font-size: 2rem;
            color: #e94560;
            margin-bottom: 10px;
        }}

        .entity-header .stats {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}

        .stat-box {{
            background: #16213e;
            padding: 10px 20px;
            border-radius: 5px;
        }}

        .stat-box .label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }}

        .stat-box .value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #e94560;
        }}

        .age-section {{
            margin-bottom: 30px;
        }}

        .age-header {{
            background: #0f3460;
            padding: 15px 20px;
            border-radius: 5px 5px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }}

        .age-header h3 {{
            color: #fff;
            font-size: 1.1rem;
        }}

        .age-header .count {{
            background: #e94560;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 14px;
        }}

        .age-content {{
            background: #16213e;
            border-radius: 0 0 5px 5px;
            padding: 15px;
        }}

        .advance-card {{
            background: #1a1a2e;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 3px solid #e94560;
        }}

        .advance-card:last-child {{
            margin-bottom: 0;
        }}

        .advance-name {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #fff;
            margin-bottom: 10px;
        }}

        .advance-id {{
            font-size: 0.8rem;
            color: #666;
            font-family: monospace;
        }}

        .advance-section {{
            margin-top: 10px;
        }}

        .advance-section-title {{
            font-size: 0.75rem;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}

        .modifier-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .modifier-tag {{
            background: #0f3460;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 13px;
        }}

        .modifier-tag .value {{
            color: #4ade80;
            font-weight: 600;
        }}

        .unlock-tag {{
            background: #7c3aed;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 13px;
        }}

        .requires-tag {{
            background: #1e40af;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 13px;
            cursor: pointer;
        }}

        .requires-tag:hover {{
            background: #2563eb;
        }}

        .availability-tag {{
            background: #065f46;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 13px;
        }}

        .no-advances {{
            text-align: center;
            padding: 40px;
            color: #666;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: #1a1a2e;
        }}

        ::-webkit-scrollbar-thumb {{
            background: #0f3460;
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: #e94560;
        }}

        /* Age colors */
        .age-1 {{ border-left-color: #8b5cf6; }}
        .age-2 {{ border-left-color: #3b82f6; }}
        .age-3 {{ border-left-color: #10b981; }}
        .age-4 {{ border-left-color: #f59e0b; }}
        .age-5 {{ border-left-color: #ef4444; }}
        .age-6 {{ border-left-color: #ec4899; }}

        .age-header.age-1 {{ background: linear-gradient(90deg, #8b5cf6 0%, #0f3460 100%); }}
        .age-header.age-2 {{ background: linear-gradient(90deg, #3b82f6 0%, #0f3460 100%); }}
        .age-header.age-3 {{ background: linear-gradient(90deg, #10b981 0%, #0f3460 100%); }}
        .age-header.age-4 {{ background: linear-gradient(90deg, #f59e0b 0%, #0f3460 100%); }}
        .age-header.age-5 {{ background: linear-gradient(90deg, #ef4444 0%, #0f3460 100%); }}
        .age-header.age-6 {{ background: linear-gradient(90deg, #ec4899 0%, #0f3460 100%); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                <h1>Advances Explorer</h1>
                <input type="text" class="search-box" id="searchBox" placeholder="Search entities...">
            </div>
            <div class="category-tabs">
                <div class="category-tab active" data-category="countries">Countries</div>
                <div class="category-tab" data-category="religions">Religions</div>
                <div class="category-tab" data-category="cultures">Cultures</div>
                <div class="category-tab" data-category="culture_groups">Culture Groups</div>
                <div class="category-tab" data-category="governments">Governments</div>
                <div class="category-tab" data-category="regions">Regions</div>
                <div class="category-tab" data-category="base">Base</div>
                <div class="category-tab" data-category="unlocks">Unlocks</div>
            </div>
            <div class="entity-list" id="entityList">
                <!-- Populated by JS -->
            </div>
        </div>

        <div class="main-content" id="mainContent">
            <div class="welcome-message">
                <h2>Welcome to the Advances Explorer</h2>
                <p>Select a category and entity from the sidebar to explore their unique advances.</p>
                <p style="margin-top: 20px;">Use the search box to quickly find countries, religions, or cultures.</p>
            </div>
            <div class="entity-detail" id="entityDetail">
                <!-- Populated by JS -->
            </div>
        </div>
    </div>

    <script>
        const AGE_NAMES = {json.dumps(AGE_NAMES)};
        const AGE_ORDER = {json.dumps(AGE_ORDER)};
        const ENTITIES_DATA = {json.dumps(entities_data, default=str)};

        let currentCategory = 'countries';
        let currentEntity = null;

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            renderEntityList();
            setupEventListeners();
        }});

        function setupEventListeners() {{
            // Category tabs
            document.querySelectorAll('.category-tab').forEach(tab => {{
                tab.addEventListener('click', () => {{
                    document.querySelectorAll('.category-tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    currentCategory = tab.dataset.category;
                    currentEntity = null;
                    renderEntityList();
                    document.getElementById('entityDetail').classList.remove('active');
                    document.querySelector('.welcome-message').style.display = 'block';
                }});
            }});

            // Search
            document.getElementById('searchBox').addEventListener('input', (e) => {{
                renderEntityList(e.target.value.toLowerCase());
            }});
        }}

        function renderEntityList(filter = '') {{
            const list = document.getElementById('entityList');
            const data = ENTITIES_DATA[currentCategory] || {{}};

            // Sort by total advances (descending)
            const sorted = Object.entries(data).sort((a, b) => b[1].total - a[1].total);

            let html = '';
            for (const [id, entity] of sorted) {{
                const displayName = entity.display_name;
                if (filter && !displayName.toLowerCase().includes(filter) && !id.toLowerCase().includes(filter)) {{
                    continue;
                }}
                const activeClass = currentEntity === id ? 'active' : '';
                html += `
                    <div class="entity-item ${{activeClass}}" data-entity="${{id}}">
                        <span class="entity-name">${{displayName}}</span>
                        <span class="entity-count">${{entity.total}}</span>
                    </div>
                `;
            }}

            if (!html) {{
                html = '<div class="no-advances">No entities found</div>';
            }}

            list.innerHTML = html;

            // Add click handlers
            list.querySelectorAll('.entity-item').forEach(item => {{
                item.addEventListener('click', () => {{
                    list.querySelectorAll('.entity-item').forEach(i => i.classList.remove('active'));
                    item.classList.add('active');
                    currentEntity = item.dataset.entity;
                    renderEntityDetail();
                }});
            }});
        }}

        function renderEntityDetail() {{
            const detail = document.getElementById('entityDetail');
            const entity = ENTITIES_DATA[currentCategory][currentEntity];

            if (!entity) {{
                detail.classList.remove('active');
                return;
            }}

            document.querySelector('.welcome-message').style.display = 'none';
            detail.classList.add('active');

            // Count by age
            const ageCounts = {{}};
            for (const age of AGE_ORDER) {{
                ageCounts[age] = (entity.by_age[age] || []).length;
            }}

            let html = `
                <div class="entity-header">
                    <h2>${{entity.display_name}}</h2>
                    <div class="stats">
                        <div class="stat-box">
                            <div class="label">Total Advances</div>
                            <div class="value">${{entity.total}}</div>
                        </div>
            `;

            for (let i = 0; i < AGE_ORDER.length; i++) {{
                const age = AGE_ORDER[i];
                const count = ageCounts[age];
                if (count > 0) {{
                    html += `
                        <div class="stat-box">
                            <div class="label">Age ${{i + 1}}</div>
                            <div class="value">${{count}}</div>
                        </div>
                    `;
                }}
            }}

            html += `
                    </div>
                </div>
            `;

            // Build context for filtering redundant info
            const filterContext = {{
                category: currentCategory,
                entityId: currentEntity
            }};

            // Render advances by age
            for (let i = 0; i < AGE_ORDER.length; i++) {{
                const age = AGE_ORDER[i];
                const advances = entity.by_age[age] || [];
                if (advances.length === 0) continue;

                html += `
                    <div class="age-section">
                        <div class="age-header age-${{i + 1}}">
                            <h3>${{AGE_NAMES[age]}}</h3>
                            <span class="count">${{advances.length}} advances</span>
                        </div>
                        <div class="age-content">
                `;

                for (const adv of advances) {{
                    html += renderAdvanceCard(adv, i + 1, filterContext);
                }}

                html += `
                        </div>
                    </div>
                `;
            }}

            detail.innerHTML = html;
        }}

        function renderAdvanceCard(adv, ageNum, filterContext) {{
            let html = `
                <div class="advance-card age-${{ageNum}}">
                    <div class="advance-name">
                        ${{adv.display_name}}
                        <span class="advance-id">${{adv.name}}</span>
                    </div>
            `;

            // Filter out modifiers that are actually potential conditions that leaked through
            // Use exact matches only to avoid filtering out things like "global_export_modifier"
            const potentialExactKeywords = new Set(['has_or_had_tag', 'religion', 'culture',
                'has_culture_group', 'has_reform', 'capital', 'or', 'and', 'not', 'any_market_present_in_country',
                'is_produced_in_market', 'is_traded_in_market']);
            const filteredModifiers = (adv.modifiers || []).filter(mod => {{
                const key = mod.raw_key ? mod.raw_key.toLowerCase() : '';
                return !potentialExactKeywords.has(key);
            }});

            // Modifiers
            if (filteredModifiers.length > 0) {{
                html += `
                    <div class="advance-section">
                        <div class="advance-section-title">Effects</div>
                        <div class="modifier-list">
                `;
                for (const mod of filteredModifiers) {{
                    // Almost all advance modifiers are beneficial, so show in green
                    // (negative cost = good, positive stats = good)
                    html += `<span class="modifier-tag">${{mod.name}}: <span class="value">${{mod.value}}</span></span>`;
                }}
                html += `</div></div>`;
            }}

            // Unlocks
            if (adv.unlocks && adv.unlocks.length > 0) {{
                html += `
                    <div class="advance-section">
                        <div class="advance-section-title">Unlocks</div>
                        <div class="modifier-list">
                `;
                for (const unlock of adv.unlocks) {{
                    html += `<span class="unlock-tag">${{unlock.type}}: ${{unlock.value}}</span>`;
                }}
                html += `</div></div>`;
            }}

            // Requirements (advance prerequisites)
            if (adv.requires && adv.requires.length > 0) {{
                html += `
                    <div class="advance-section">
                        <div class="advance-section-title">Requires Advance</div>
                        <div class="modifier-list">
                `;
                for (const req of adv.requires) {{
                    html += `<span class="requires-tag">${{req.replace(/_/g, ' ')}}</span>`;
                }}
                html += `</div></div>`;
            }}

            // Show availability conditions (potential) - but filter out the current entity
            const pot = adv.potential || {{}};
            let availabilityItems = [];

            // Countries - filter out current if viewing countries
            if (pot.countries && pot.countries.length > 0) {{
                const filtered = filterContext.category === 'countries'
                    ? pot.countries.filter(c => c !== filterContext.entityId)
                    : pot.countries;
                if (filtered.length > 0) {{
                    availabilityItems.push(...filtered.map(c => `Country: ${{c}}`));
                }}
            }}

            // Religions - filter out current if viewing religions
            if (pot.religions && pot.religions.length > 0) {{
                const filtered = filterContext.category === 'religions'
                    ? pot.religions.filter(r => r !== filterContext.entityId)
                    : pot.religions;
                if (filtered.length > 0) {{
                    availabilityItems.push(...filtered.map(r => `Religion: ${{r.replace(/_/g, ' ')}}`));
                }}
            }}

            // Cultures - filter out current if viewing cultures
            if (pot.cultures && pot.cultures.length > 0) {{
                const filtered = filterContext.category === 'cultures'
                    ? pot.cultures.filter(c => c !== filterContext.entityId)
                    : pot.cultures;
                if (filtered.length > 0) {{
                    availabilityItems.push(...filtered.map(c => `Culture: ${{c.replace(/_/g, ' ')}}`));
                }}
            }}

            // Culture groups - filter out current if viewing culture_groups
            if (pot.culture_groups && pot.culture_groups.length > 0) {{
                const filtered = filterContext.category === 'culture_groups'
                    ? pot.culture_groups.filter(c => c !== filterContext.entityId)
                    : pot.culture_groups;
                if (filtered.length > 0) {{
                    availabilityItems.push(...filtered.map(c => `Culture Group: ${{c.replace(/_/g, ' ')}}`));
                }}
            }}

            // Governments - filter out current if viewing governments
            if (pot.governments && pot.governments.length > 0) {{
                const filtered = filterContext.category === 'governments'
                    ? pot.governments.filter(g => g !== filterContext.entityId)
                    : pot.governments;
                if (filtered.length > 0) {{
                    availabilityItems.push(...filtered.map(g => `Government: ${{g.replace(/_/g, ' ')}}`));
                }}
            }}

            // Regions
            if (pot.regions && pot.regions.length > 0) {{
                const filtered = filterContext.category === 'regions'
                    ? pot.regions.filter(r => r !== filterContext.entityId)
                    : pot.regions;
                if (filtered.length > 0) {{
                    availabilityItems.push(...filtered.map(r => `Region: ${{r.replace(/_/g, ' ')}}`));
                }}
            }}

            // Other conditions
            if (pot.other && pot.other.length > 0) {{
                availabilityItems.push(...pot.other);
            }}

            if (availabilityItems.length > 0) {{
                html += `
                    <div class="advance-section">
                        <div class="advance-section-title">Also Available To</div>
                        <div class="modifier-list">
                `;
                for (const item of availabilityItems) {{
                    html += `<span class="availability-tag">${{item}}</span>`;
                }}
                html += `</div></div>`;
            }}

            html += `</div>`;
            return html;
        }}
    </script>
</body>
</html>
'''

    return html


def main():
    all_advances = []

    # Parse all files
    for filepath in sorted(ADVANCES_DIR.glob('*.txt')):
        filename = filepath.name
        if filename.startswith('_') or filename == 'readme.txt':
            continue

        advances = parse_advances_file(filepath)
        all_advances.extend(advances)

    print(f"Parsed {len(all_advances)} total advances")

    # Build entity index
    index = build_entity_index(all_advances)

    # Count entities
    print(f"Countries: {len(index['countries'])}")
    print(f"Religions: {len(index['religions'])}")
    print(f"Cultures: {len(index['cultures'])}")
    print(f"Culture Groups: {len(index['culture_groups'])}")
    print(f"Governments: {len(index['governments'])}")
    print(f"Regions: {len(index['regions'])}")
    print(f"Base advances: {len(index['base'])}")
    print(f"Unlock advances: {len(index['unlocks'])}")

    # Generate HTML
    html = generate_html(index, all_advances)

    with open('advances_explorer.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("\nGenerated: advances_explorer.html")


if __name__ == '__main__':
    main()
