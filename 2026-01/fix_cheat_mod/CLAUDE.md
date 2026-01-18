# Claude Code Notes

Before making changes to this project, read `DESIGN.md` for important context about how the mod generator works and the EU5 modding conventions used.

Key points:
- Localization uses `$TAG$` syntax to look up localized country names
- Green text formatting uses `#g ` (lowercase g with space) before content, ending with `#!`
- The `generate_mod.py` script generates all mod files from EU5 game data
