#!/usr/bin/env python3
"""Emit a small C++ resource index and assembler .incbin directives."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--assembly', required=True)
    parser.add_argument('--index', required=True)
    args = parser.parse_args()
    resources = [line.split('|', 1) for line in Path(args.manifest).read_text().splitlines() if line]
    assembly = ['.section .rodata.upscaler,"a",%progbits']
    declarations, entries = [], []
    for index, (name, filename) in enumerate(resources):
        path = Path(filename).resolve()
        if not path.is_file():
            raise SystemExit(f'Missing resource: {path}')
        symbol = f'upscaler_resource_{index}'
        # json.dumps quotes and escapes filenames for the GNU assembler string.
        assembly += ['.balign 16', f'.global {symbol}_begin, {symbol}_end',
                     f'.hidden {symbol}_begin, {symbol}_end', f'{symbol}_begin:',
                     f'.incbin {json.dumps(str(path))}', f'{symbol}_end:']
        declarations.append(f'extern const unsigned char {symbol}_begin[], {symbol}_end[];')
        entries.append(f'    {{{json.dumps(name)}, {symbol}_begin, {symbol}_end}}')
    assembly.append('.section .note.GNU-stack,"",%progbits')
    Path(args.assembly).write_text('\n'.join(assembly) + '\n')
    source = '''#include "resources.h"
#include <cstdint>
#include <stdexcept>
#include <string>
extern "C" {
''' + '\n'.join(declarations) + '''
}
namespace upscaler {
Resource embeddedResource(std::string_view name) {
    struct Entry { std::string_view name; const unsigned char* begin; const unsigned char* end; };
    static const Entry entries[] = {
''' + ',\n'.join(entries) + '''
    };
    for (const auto& entry : entries)
        if (entry.name == name)
            return {entry.begin, size_t(reinterpret_cast<uintptr_t>(entry.end) - reinterpret_cast<uintptr_t>(entry.begin))};
    throw std::runtime_error("Missing embedded resource: " + std::string(name));
}
} // namespace upscaler
'''
    Path(args.index).write_text(source)


if __name__ == '__main__':
    main()
