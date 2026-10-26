#!/usr/bin/env python3
"""Export a backed-up attachment v2 envelope for an explicitly planned downgrade.

Never edits the input or live application directory. Output directory must not
already exist. Revisions and upload identity outcomes cannot survive old code.
"""
import argparse
import json
import os
from pathlib import Path


def legacy_projection(envelope):
    if not isinstance(envelope, dict) or envelope.get('version') != 2 or not isinstance(envelope.get('sessions'), dict):
        raise ValueError('expected attachment resource version 2')
    attachments, pending = {}, []
    for sid, resource in envelope['sessions'].items():
        if not isinstance(sid, str) or not sid or not isinstance(resource, dict):
            raise ValueError('invalid session resource')
        items = resource.get('attachments')
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError('invalid attachments list')
        if type(resource.get('pending')) is not bool:
            raise ValueError('invalid pending flag')
        attachments[sid] = items
        if resource['pending'] or items:
            pending.append(sid)
    return attachments, sorted(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('v2_backup', type=Path)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    lists, pending = legacy_projection(json.loads(args.v2_backup.read_text()))
    args.output_directory.mkdir(mode=0o700)
    for name, value in [('staged_attachments.json', lists), ('pending_attachments.json', pending)]:
        path = args.output_directory / name
        with path.open('x') as stream:
            os.chmod(path, 0o600)
            json.dump(value, stream, ensure_ascii=False)
            stream.write('\n')
    print('Exported legacy lists/pending flags; deduplication and revisions are NOT preserved.')


if __name__ == '__main__':
    main()
