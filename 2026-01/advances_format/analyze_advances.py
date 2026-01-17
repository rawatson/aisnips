#!/usr/bin/env python3
"""
Analyze advances data and produce a summary report by entity and age.
"""

import os
import re
from collections import defaultdict
from pathlib import Path

ADVANCES_DIR = Path("advances")

# Age mapping for display
AGE_NAMES = {
    "age_1_traditions": "Age 1: Traditions",
    "age_2_renaissance": "Age 2: Renaissance",
    "age_3_discovery": "Age 3: Discovery",
    "age_4_reformation": "Age 4: Reformation",
    "age_5_absolutism": "Age 5: Absolutism",
    "age_6_revolutions": "Age 6: Revolutions",
}

AGE_ORDER = list(AGE_NAMES.keys())


def parse_advances_file(filepath):
    """Parse an advances file and return list of advances with their ages and entities."""
    advances = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # Find all advance blocks - pattern: name = { ... }
    # Using a simple state machine to track braces
    current_pos = 0

    while current_pos < len(content):
        # Find the next advance definition (name = {)
        match = re.search(r'^(\w+)\s*=\s*\{', content[current_pos:], re.MULTILINE)
        if not match:
            break

        advance_name = match.group(1)
        block_start = current_pos + match.end() - 1  # Position of opening brace

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

        # Extract age
        age_match = re.search(r'age\s*=\s*(\w+)', block_content)
        if not age_match:
            continue

        age = age_match.group(1)

        # Extract entities from potential block and government attribute
        entities = extract_entities(block_content, filepath)

        advances.append({
            'name': advance_name,
            'age': age,
            'entities': entities
        })

    return advances


def extract_entities(block_content, filepath):
    """Extract entity information from advance block."""
    entities = {
        'countries': set(),
        'religions': set(),
        'cultures': set(),
        'culture_groups': set(),
        'governments': set(),
        'regions': set(),
        'other': set()
    }

    # Extract from potential block
    potential_match = re.search(r'potential\s*=\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', block_content, re.DOTALL)
    if potential_match:
        potential_content = potential_match.group(1)

        # Country tags
        for tag in re.findall(r'has_or_had_tag\s*=\s*(\w+)', potential_content):
            entities['countries'].add(tag)

        # Religions
        for rel in re.findall(r'religion\s*=\s*religion:(\w+)', potential_content):
            entities['religions'].add(rel)
        for rel in re.findall(r'religion\.group\s*=\s*religion_group:(\w+)', potential_content):
            entities['religions'].add(f"group:{rel}")

        # Cultures
        for cult in re.findall(r'culture\s*=\s*culture:(\w+)', potential_content):
            entities['cultures'].add(cult)
        for cult_group in re.findall(r'has_culture_group\s*=\s*culture_group:(\w+)', potential_content):
            entities['culture_groups'].add(cult_group)

        # Regions from capital
        for reg in re.findall(r'capital\.region\s*\?*=\s*region:(\w+)', potential_content):
            entities['regions'].add(reg)

    # Government from direct attribute (not in potential)
    gov_match = re.search(r'^[\t ]*government\s*=\s*(\w+)', block_content, re.MULTILINE)
    if gov_match:
        entities['governments'].add(gov_match.group(1))

    # Infer from filename if no specific entities found
    filename = os.path.basename(filepath)
    if not any(entities[k] for k in entities):
        if filename.startswith('country_'):
            tag = filename[8:].replace('.txt', '').upper()
            entities['countries'].add(tag)
        elif filename.startswith('culture_'):
            cult = filename[8:].replace('.txt', '')
            entities['cultures'].add(cult)
        elif filename.startswith('religion_'):
            rel = filename[9:].replace('.txt', '')
            entities['religions'].add(rel)
        elif filename.startswith('government_'):
            gov = filename[11:].replace('.txt', '')
            entities['governments'].add(gov)
        elif filename.startswith('region_'):
            reg = filename[7:].replace('.txt', '')
            entities['regions'].add(reg)
        elif filename.startswith('0_age_'):
            entities['other'].add('base_advances')
        elif filename.startswith('1_') or filename.startswith('2_') or filename.startswith('3_') or filename.startswith('4_'):
            entities['other'].add('unlocks')

    return entities


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
    elif filename.startswith('ctype_'):
        return 'country_type'
    elif filename.startswith('0_age_'):
        return 'base'
    elif filename.startswith(('1_', '2_', '3_', '4_')):
        return 'unlock'
    else:
        return 'other'


def main():
    # Data structures to collect results
    # For each category, we track {entity_name: {age: count}}
    countries = defaultdict(lambda: defaultdict(int))
    religions = defaultdict(lambda: defaultdict(int))
    cultures = defaultdict(lambda: defaultdict(int))
    culture_groups = defaultdict(lambda: defaultdict(int))
    governments = defaultdict(lambda: defaultdict(int))
    regions = defaultdict(lambda: defaultdict(int))
    base_advances = defaultdict(int)
    other_categories = defaultdict(lambda: defaultdict(int))

    # Country name mapping (for better display)
    country_names = {
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

    # Process all files
    for filepath in sorted(ADVANCES_DIR.glob('*.txt')):
        filename = filepath.name

        # Skip template and readme
        if filename.startswith('_') or filename == 'readme.txt':
            continue

        category = categorize_file(filename)
        advances = parse_advances_file(filepath)

        for adv in advances:
            age = adv['age']
            entities = adv['entities']

            # Count for each entity type
            for country in entities['countries']:
                countries[country][age] += 1

            for religion in entities['religions']:
                religions[religion][age] += 1

            for culture in entities['cultures']:
                cultures[culture][age] += 1

            for culture_group in entities['culture_groups']:
                culture_groups[culture_group][age] += 1

            for government in entities['governments']:
                governments[government][age] += 1

            for region in entities['regions']:
                regions[region][age] += 1

            # Track base advances (those with no specific entity)
            if not any(entities[k] for k in ['countries', 'religions', 'cultures', 'culture_groups', 'governments', 'regions']):
                if category == 'base':
                    base_advances[age] += 1
                else:
                    other_categories[category][age] += 1

    # Generate report
    report = generate_report(countries, religions, cultures, culture_groups,
                            governments, regions, base_advances, other_categories,
                            country_names)

    # Write report
    with open('final_report.md', 'w') as f:
        f.write(report)

    print("Report generated: final_report.md")


def get_total(age_counts):
    """Sum all age counts."""
    return sum(age_counts.values())


def sort_by_total(data_dict):
    """Sort dictionary items by total advances (descending)."""
    return sorted(data_dict.items(), key=lambda x: get_total(x[1]), reverse=True)


def format_table(data_dict, title, name_mapping=None):
    """Generate a markdown table for a category."""
    if not data_dict:
        return ""

    sorted_data = sort_by_total(data_dict)

    lines = [f"\n## {title}\n"]
    lines.append(f"| Entity | {' | '.join(AGE_NAMES.values())} | **Total** |")
    lines.append("|" + "|".join([":---"] + [":---:"] * (len(AGE_ORDER) + 1)) + "|")

    for entity, age_counts in sorted_data:
        display_name = entity
        if name_mapping and entity in name_mapping:
            display_name = f"{name_mapping[entity]} ({entity})"
        elif name_mapping and entity.upper() in name_mapping:
            display_name = f"{name_mapping[entity.upper()]} ({entity})"

        counts = [str(age_counts.get(age, 0)) for age in AGE_ORDER]
        total = get_total(age_counts)
        lines.append(f"| {display_name} | {' | '.join(counts)} | **{total}** |")

    return '\n'.join(lines)


def generate_report(countries, religions, cultures, culture_groups,
                    governments, regions, base_advances, other_categories,
                    country_names):
    """Generate the full markdown report."""

    lines = ["# Advances Analysis Report\n"]
    lines.append("This report summarizes the number of unique advances available to each entity ")
    lines.append("(country, religion, culture, etc.) broken down by age.\n")

    # Summary statistics
    total_countries = len(countries)
    total_religions = len(religions)
    total_cultures = len(cultures) + len(culture_groups)
    total_governments = len(governments)

    lines.append("## Summary Statistics\n")
    lines.append(f"- **Countries with unique advances:** {total_countries}")
    lines.append(f"- **Religions with unique advances:** {total_religions}")
    lines.append(f"- **Cultures/Culture Groups with unique advances:** {total_cultures}")
    lines.append(f"- **Governments with unique advances:** {total_governments}")
    lines.append(f"- **Regions with unique advances:** {len(regions)}")
    lines.append("")

    # Base advances section
    if base_advances:
        lines.append("\n## Base Advances (Available to All)\n")
        lines.append("These are the core advances available to all nations.\n")
        lines.append(f"| Age | Count |")
        lines.append("|:---|:---:|")
        for age in AGE_ORDER:
            count = base_advances.get(age, 0)
            lines.append(f"| {AGE_NAMES[age]} | {count} |")
        lines.append(f"| **Total** | **{sum(base_advances.values())}** |")

    # Countries table
    lines.append(format_table(countries, "Country-Specific Advances", country_names))

    # Religions table
    lines.append(format_table(religions, "Religion-Specific Advances"))

    # Cultures table
    lines.append(format_table(cultures, "Culture-Specific Advances"))

    # Culture groups table
    lines.append(format_table(culture_groups, "Culture Group-Specific Advances"))

    # Governments table
    lines.append(format_table(governments, "Government-Specific Advances"))

    # Regions table
    lines.append(format_table(regions, "Region-Specific Advances"))

    # Other categories
    for category, age_counts in other_categories.items():
        if age_counts:
            lines.append(f"\n## {category.replace('_', ' ').title()} Advances\n")
            lines.append(f"| Age | Count |")
            lines.append("|:---|:---:|")
            for age in AGE_ORDER:
                count = age_counts.get(age, 0)
                lines.append(f"| {AGE_NAMES[age]} | {count} |")
            lines.append(f"| **Total** | **{sum(age_counts.values())}** |")

    return '\n'.join(lines)


if __name__ == '__main__':
    main()
