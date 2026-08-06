#!/usr/bin/env python3
"""Static analysis: find cross-IIFE wiring bugs in the frontend.

The frontend was split from one monolithic app.js into 67 IIFE modules.
Each module factory receives an `options` object and destructures names
from it. The creation calls build these options objects. If a name is
destructured from `options` but NOT present in the creation object,
it's undefined at runtime — a silent wiring bug.

This checker finds them statically by:
1. Extracting destructured names from each module factory
2. Tracing the full creation chain (composition → chat_interaction → children)
3. Flagging names not present anywhere in the chain

Usage: python3 scripts/check_wiring.py [static_dir]
"""
import re
import sys
from pathlib import Path


def get_static_dir() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
    else:
        p = Path(__file__).parent.parent / "codoxear" / "static"
    if not p.exists():
        print(f"ERROR: static dir not found: {p}", file=sys.stderr)
        sys.exit(2)
    return p


def extract_destructured_names(source: str) -> set[str]:
    """Find all names destructured from `options` in a module."""
    names = set()
    # Match: const { a, b, c } = options;  (single or multi-line)
    for m in re.finditer(r'(?:const|let|var)\s*\{([^}]+)\}\s*=\s*options\s*;', source, re.DOTALL):
        for part in m.group(1).split(','):
            part = part.strip()
            if not part:
                continue
            name = re.split(r'[:=]', part)[0].strip()
            if re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', name):
                names.add(name)
    # Also match: const { a, b } = requireObject(options, "...")
    for m in re.finditer(r'(?:const|let|var)\s*\{([^}]+)\}\s*=\s*requireObject\s*\(\s*options', source, re.DOTALL):
        for part in m.group(1).split(','):
            part = part.strip()
            if not part:
                continue
            name = re.split(r'[:=]', part)[0].strip()
            if re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', name):
                names.add(name)
    return names


def find_creation_object(source: str, factory_pattern: str) -> list[tuple[str, bool]]:
    """Find all object literals passed to a factory call.

    Returns list of (object_body, has_spread) tuples.
    """
    results = []
    # Search for the factory call pattern followed by an opening brace
    # The pattern may be: module.createXxx( or just createXxx(
    for m in re.finditer(rf'{factory_pattern}\s*\(\s*\{{', source):
        brace_start = m.end() - 1
        depth = 0
        i = brace_start
        while i < len(source):
            if source[i] == '{':
                depth += 1
            elif source[i] == '}':
                depth -= 1
                if depth == 0:
                    body = source[brace_start + 1:i]
                    has_spread = '...' in body
                    results.append((body, has_spread))
                    break
            i += 1
    return results


def extract_keys_from_object_body(body: str) -> set[str]:
    """Extract property names from a JS object literal body."""
    keys = set()
    # Shorthand: name, or name at line start
    for km in re.finditer(r'(?:^|[\s,])([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?=[,:}\n\r])', body, re.MULTILINE):
        keys.add(km.group(1))
    return keys


# Names that are built-in or provided by the JS runtime, not the options chain
BUILTINS = frozenset({
    'window', 'document', 'navigator', 'HTMLElement', 'Element', 'EventSource', 'AbortController',
    'getComputedStyle', 'requestAnimationFrame', 'setTimeout', 'clearTimeout', 'Node',
    'performance', 'console', 'Math', 'JSON', 'Object', 'Array', 'String', 'Number',
    'Date', 'Error', 'Promise', 'Set', 'Map', 'Path', 'fetch', 'URL', 'Event',
    'MutationObserver', 'ResizeObserver', 'IntersectionObserver', 'alert', 'confirm',
    'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'encodeURIComponent',
    'decodeURIComponent', 'requestAnimationFrame', 'matchMedia',
    'CSS', 'CustomEvent', 'DOMParser', 'XMLSerializer', 'TextEncoder', 'TextDecoder',
})


def find_module_factory(file_path: Path) -> str | None:
    """Find the main factory function name exported by a module."""
    source = file_path.read_text()
    # Look for function createXxxController
    for m in re.finditer(r'function\s+(create[A-Z]\w*Controller)\s*\(', source):
        return m.group(1)
    # Or assignment
    for m in re.finditer(r'(create[A-Z]\w*Controller)\s*=\s*(?:async\s+)?function', source):
        return m.group(1)
    # Without Controller suffix
    for m in re.finditer(r'function\s+(create[A-Z]\w*)\s*\(', source):
        name = m.group(1)
        if name not in ('createElement',):
            return name
    return None


def main():
    static_dir = get_static_dir()

    # Load all source files
    sources: dict[str, str] = {}
    for f in sorted(static_dir.glob("app_*.js")):
        sources[f.name] = f.read_text()

    # Collect all factory names and their destructured requirements
    module_deps: dict[str, tuple[str, set[str]]] = {}  # file -> (factory_name, destructured_names)
    for fname, source in sources.items():
        names = extract_destructured_names(source)
        if not names:
            continue
        factory = find_module_factory(static_dir / fname)
        if not factory:
            continue
        module_deps[fname] = (factory, names - BUILTINS)

    # Build a map of all factory call patterns to their creation sites
    # Check ALL source files for ALL factory calls
    all_creation_keys: dict[str, list[tuple[set[str], bool, str]]] = {}  # factory -> [(keys, has_spread, creator_file)]

    for creator_file, creator_source in sources.items():
        for module_file, (factory, _) in module_deps.items():
            # Search for this factory being called in this creator file
            # Pattern: the factory name may be called via module.createXxx or directly
            patterns = [
                factory,  # direct call: createXxxController({
                rf'\w+\.{factory}',  # namespaced: module.createXxxController({
            ]
            for pat in patterns:
                objects = find_creation_object(creator_source, re.escape(pat) if '\\' in pat else pat)
                for body, has_spread in objects:
                    keys = extract_keys_from_object_body(body)
                    all_creation_keys.setdefault(factory, []).append((keys, has_spread, creator_file))

    # Now check each module's deps against its creation sites
    definite_bugs = []
    spread_warnings = []

    for module_file, (factory, deps) in sorted(module_deps.items()):
        creation_sites = all_creation_keys.get(factory, [])

        if not creation_sites:
            continue

        for keys, has_spread, creator_file in creation_sites:
            missing = deps - keys
            if not missing:
                continue

            for name in sorted(missing):
                # Check if the name appears ANYWHERE in ANY source file
                # as: const/let/var/function definition, shorthand property,
                # destructured binding, or renamed import
                found_anywhere = False
                for src in sources.values():
                    # Direct definition
                    if re.search(rf'(?:const|let|var|function)\s+{re.escape(name)}\b', src):
                        found_anywhere = True
                        break
                    # Renamed destructuring: { x: name }
                    if re.search(rf':\s*{re.escape(name)}\b', src):
                        found_anywhere = True
                        break
                    # Shorthand property in object literal (passed as option)
                    if re.search(rf'\b{re.escape(name)}\s*[,}}\n]', src):
                        found_anywhere = True
                        break

                if not found_anywhere and not has_spread:
                    definite_bugs.append((module_file, factory, name, creator_file))
                elif not found_anywhere and has_spread:
                    spread_warnings.append((module_file, factory, name, creator_file))

    if definite_bugs:
        print(f"Found {len(definite_bugs)} DEFINITE wiring bugs:\n")
        for module, factory, name, creator in definite_bugs:
            print(f"  ❌ {module}: '{name}' destructured from options but not defined anywhere")
            print(f"     Factory: {factory}, created in: {creator}")

    if spread_warnings:
        print(f"\n{len(spread_warnings)} names only available via spread (cannot verify statically):")
        for module, factory, name, creator in spread_warnings[:10]:
            print(f"  ⚠️  {module}: '{name}' (via spread in {creator})")
        if len(spread_warnings) > 10:
            print(f"  ... and {len(spread_warnings) - 10} more")

    if definite_bugs:
        sys.exit(1)
    else:
        print("\n✓ No definite wiring bugs found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
