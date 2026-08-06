#!/usr/bin/env python3
"""Static analysis: find cross-IIFE wiring bugs in the frontend.

The frontend was split from one monolithic app.js into 67 IIFE modules.
Each module factory receives an `options` object and destructures names
from it. The creation calls in app_application_composition.js build these
options objects. If a name is destructured from `options` in a module
but NOT present in the creation object, it's undefined at runtime — a
silent wiring bug.

This checker finds them statically.

Usage: python3 scripts/check_wiring.py [--fix]
"""
import re
import sys
from pathlib import Path

STATIC_DIR = Path(__file__).parent.parent / "codoxear" / "static"


def extract_destructured_names(source: str) -> set[str]:
    """Find all names destructured from `options` at the module level."""
    names = set()
    # Match: const { a, b, c } = options;
    # Also: const { a, b: c } = options;
    for m in re.finditer(r'(?:const|let|var)\s*\{([^}]+)\}\s*=\s*options\s*;', source):
        for part in m.group(1).split(','):
            part = part.strip()
            if not part:
                continue
            # Handle "name: alias" and "name = default"
            name = re.split(r'[:=]', part)[0].strip()
            if re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', name):
                names.add(name)
    # Also match multi-line destructuring (the god-file split has these)
    for m in re.finditer(r'(?:const|let|var)\s*\{([^}]+)\}\s*=\s*(?:requireObject\s*\(\s*options|options)', source, re.DOTALL):
        for part in m.group(1).split(','):
            part = part.strip()
            if not part:
                continue
            name = re.split(r'[:=]', part)[0].strip()
            if re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', name):
                names.add(name)
    return names


def extract_creation_keys(source: str, factory_name: str) -> set[str] | None:
    """Find the keys passed to a specific factory's options object.

    Looks for patterns like:
      const X = module.createYController({
        ...deps,
        a, b, c,
        d: () => ...,
      });
    """
    # Find the factory call
    # Match createXxx( up to the matching closing )
    pattern = rf'create{re.escape(factory_name)}\s*\(\s*\{{'
    results = []
    for m in re.finditer(pattern, source):
        # Find the matching closing brace
        start = m.end() - 1  # position of the opening {
        depth = 0
        i = start
        while i < len(source):
            if source[i] == '{':
                depth += 1
            elif source[i] == '}':
                depth -= 1
                if depth == 0:
                    block = source[start + 1:i]
                    break
            i += 1
        else:
            continue

        keys = set()
        # Find shorthand keys: "name," or "name\n" or "name}" at the start of a line/token
        # Also find "name:" (key-value pairs)
        for km in re.finditer(r'(?:^|[\s,])([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?=[,:}\n])', block, re.MULTILINE):
            key = km.group(1)
            keys.add(key)

        # Check for ...spread patterns
        has_spread = '...' in block
        results.append((keys, has_spread))

    return results if results else None


def check_module(module_file: str, composition_source: str) -> list[tuple[str, str]]:
    """Check if all destructured names in a module are present in its creation call."""
    module_path = STATIC_DIR / module_file
    if not module_path.exists():
        return []

    source = module_path.read_text()
    destructured = extract_destructured_names(source)
    if not destructured:
        return []

    # Find which factory function this module defines
    # Pattern: function createXxxController or createXxxController =
    factory_match = re.search(r'function\s+(create[A-Z]\w*Controller)', source)
    if not factory_match:
        factory_match = re.search(r'(create[A-Z]\w*Controller)\s*=', source)
    if not factory_match:
        # Try without "Controller" suffix
        factory_match = re.search(r'function\s+(create[A-Z]\w*)', source)

    if not factory_match:
        return []

    factory_name = factory_match.group(1)
    # Extract the part after "create" for searching in composition
    search_name = factory_name  # full name like "createMessageHistoryController"

    # Find creation calls in composition
    creation_results = extract_creation_keys(composition_source, search_name)

    if creation_results is None:
        # Factory might be called indirectly (via another controller)
        # Check if it's created inside another controller module
        return []

    bugs = []
    for keys, has_spread in creation_results:
        # If there's a spread (...deps, ...options), we can't verify statically
        # but we can still flag names that are NOT in the explicit keys
        # and would only be available via spread
        missing = destructured - keys
        if missing and has_spread:
            # Could be covered by spread — flag as WARNING
            for name in sorted(missing):
                bugs.append((name, "via-spread"))
        elif missing:
            # Not in keys and no spread — definitely broken
            for name in sorted(missing):
                bugs.append((name, "MISSING"))

    return bugs


def main():
    fix_mode = "--fix" in sys.argv

    comp_path = STATIC_DIR / "app_application_composition.js"
    if not comp_path.exists():
        print("ERROR: app_application_composition.js not found")
        sys.exit(1)

    comp_source = comp_path.read_text()

    # Also check app_chat_interaction.js which creates sub-controllers
    chat_interaction_path = STATIC_DIR / "app_chat_interaction.js"
    chat_interaction_source = chat_interaction_path.read_text() if chat_interaction_path.exists() else ""

    all_bugs = []
    js_files = sorted(f.name for f in STATIC_DIR.glob("app_*.js"))

    for module_file in js_files:
        bugs = check_module(module_file, comp_source)
        if bugs:
            for name, kind in bugs:
                all_bugs.append((module_file, name, kind))

    if not all_bugs:
        print("✓ No wiring bugs found in direct creation calls.")
    else:
        print(f"Found {len(all_bugs)} potential wiring issues:\n")
        for module, name, kind in all_bugs:
            symbol = "❌" if kind == "MISSING" else "⚠️"
            print(f"  {symbol} {module}: '{name}' ({kind})")

    # Check for indirect creation (controllers created inside other controllers)
    print("\n--- Indirect controller creations (via chat_interaction spread) ---")
    indirect_creators = {
        "app_chat_interaction.js": chat_interaction_source,
    }

    for module_file in js_files:
        module_path = STATIC_DIR / module_file
        source = module_path.read_text()
        destructured = extract_destructured_names(source)
        if not destructured:
            continue

        factory_match = re.search(r'function\s+(create[A-Z]\w*Controller)', source)
        if not factory_match:
            continue

        factory_name = factory_match.group(1)

        # Check if this factory is called in chat_interaction with spread
        for creator_file, creator_source in indirect_creators.items():
            results = extract_creation_keys(creator_source, factory_name)
            if results is None:
                continue

            for keys, has_spread in results:
                missing = destructured - keys
                if missing:
                    for name in sorted(missing):
                        kind = "via-spread" if has_spread else "MISSING"
                        entry = (module_file, name, kind, creator_file)
                        if entry not in [(b[0], b[1], b[2]) for b in all_bugs]:
                            symbol = "❌" if kind == "MISSING" else "⚠️"
                            print(f"  {symbol} {module_file}: '{name}' ({kind}, created in {creator_file})")

    sys.exit(1 if any(b[2] == "MISSING" for b in all_bugs) else 0)


if __name__ == "__main__":
    main()
