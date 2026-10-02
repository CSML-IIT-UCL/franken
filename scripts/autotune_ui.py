#!/usr/bin/env python3
"""Generate an offline autotune command builder, or serve it with parser validation.

Run from a Franken Python environment; no Node or JavaScript dependencies needed.
"""

import argparse
import contextlib
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys

# Also support running this script directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from franken.autotune.cli import (  # noqa: E402
    GroupWithIndividualOptions,
    build_parser,
    parse_cli,
)
from franken.config import HPSearchConfig  # noqa: E402


def readable_label(flag):
    """Make a readable title while retaining the exact flag as a subtitle."""
    name = flag.rsplit(".", 1)[-1].removeprefix("--")
    if name == "num-rf":
        return "Number of Random Features"
    words = {
        "rf": "RF",
        "l2": "L2",
        "id": "ID",
        "rng": "RNG",
        "num": "Number",
        "jac": "Jacobian",
        "dtype": "Data Type",
        "fmaps": "Feature Maps",
        "act": "Activation",
        "norm": "Normalization",
    }
    return " ".join(words.get(word, word.capitalize()) for word in name.split("-"))


def build_schema():
    """Read options, help, defaults and conditional requirements from the CLI."""
    parser, groups = build_parser(return_groups=True)
    membership = {}
    required = set()
    sections = []
    # Show all steps expanded, with optional general settings last.
    for key in ("dataset", "backbone", "rfs", "solver"):
        group = groups[key]
        if isinstance(group, GroupWithIndividualOptions):
            sections.append(
                {
                    "id": group.group_name,
                    "title": group.group_name.upper() if key == "rfs" else key.title(),
                    "help": group.help_text,
                }
            )
            membership[group.group_name] = (group.group_name, None)
            for name, subgroup in group.me_groups.items():
                for arg in subgroup.arguments:
                    membership[arg.dest] = (group.group_name, name)
                    if arg.is_arg_required:
                        required.add(arg.dest)
        else:
            sections.append(
                {"id": group.name, "title": group.title, "help": group.desc}
            )
            for arg in group.arguments:
                membership[arg.dest] = (group.name, None)
    sections.append(
        {
            "id": "general",
            "title": "Evaluation and output",
            "help": "Choose training labels, evaluation metrics, reproducibility and output settings.",
        }
    )
    fields = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        section, condition = membership.get(action.dest, ("general", None))
        boolean = isinstance(
            action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
        )
        default = action.default
        # Boolean defaults made by Argument.from_dataclass are strings in the CLI.
        if boolean:
            default = str(default).lower() == "true"
        elif default is None or str(default).lower() == "none":
            default = ""
        kind = "text"
        if action.type is int:
            kind = "int"
        elif action.type is float:
            kind = "float"
        elif action.type == HPSearchConfig.from_str:
            kind = "hyperparameter"
        elif action.dest == "max_train_samples":
            kind = "optional-int"
        elif action.dest == "jac_chunk_size":
            kind = "auto-int"
        elif action.dest == "atomic_energies":
            kind = "atomic"
        fields.append(
            {
                "id": action.dest,
                "flag": action.option_strings[0],
                "label": readable_label(action.option_strings[0]),
                "section": section,
                "condition": condition,
                "help": action.help or "",
                "default": default,
                "required": action.required or action.dest in required,
                "choices": list(action.choices) if action.choices is not None else None,
                "multiple": action.nargs in ("+", "*"),
                "boolean": boolean,
                "const": action.const if boolean else None,
                "kind": kind,
            }
        )
    return {"command": "franken.autotune", "sections": sections, "fields": fields}


def validate_argv(argv):
    """Validate without loading datasets/models or running autotune."""
    if not isinstance(argv, list) or not all(isinstance(v, str) for v in argv):
        return {"valid": False, "error": "Expected a list of argument strings."}
    # Do not let argparse's help action bypass validation.
    if any(v in ("-h", "--help") for v in argv):
        return {"valid": False, "error": "Help is not a configuration option."}
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            parse_cli(argv)
    except SystemExit:
        message = stderr.getvalue().rsplit("error:", 1)[-1].strip()
        return {"valid": False, "error": message or "Invalid arguments."}
    except (TypeError, ValueError, SyntaxError) as exc:
        return {"valid": False, "error": str(exc)}
    return {"valid": True, "error": ""}


def generate_html():
    # Escape HTML-sensitive characters even inside the JSON script element.
    schema = (
        json.dumps(build_schema())
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return TEMPLATE.replace("__SCHEMA__", schema)


def serve(html, port, host="127.0.0.1"):
    # Mark only pages delivered by our server for authoritative validation.
    # Generated files embedded in docs must retain offline validation.
    html = html.replace("const served = false;", "const served = true;")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.respond(html.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/validate":
                self.send_error(404)
                return
            # No CORS: only the same-origin builder can use this endpoint.
            if self.headers.get("Origin") not in (
                None,
                f"http://{self.headers.get('Host')}",
            ):
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    self.send_error(413)
                    return
                argv = json.loads(self.rfile.read(size))
                result = validate_argv(argv)
            except (ValueError, UnicodeError) as exc:
                result = {"valid": False, "error": str(exc)}
            self.respond(json.dumps(result).encode(), "application/json")

        def respond(self, data, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer((host, port), Handler)
    print(f"Open http://{host}:{server.server_port} (Ctrl+C to stop)", flush=True)
    if host == "0.0.0.0":
        print(
            "For LAN access, replace 0.0.0.0 with this machine's IP address.",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Franken autotune command builder</title>
<style>
* { box-sizing: border-box; }
body { font: 16px/1.5 system-ui, sans-serif; color: #243044; background: #f5f7fa; margin: 0; }
main { max-width: 1150px; margin: auto; padding: 24px; }
h1 { margin-bottom: 8px; } h2 { font-size: 1.3rem; margin-top: 0; }
section, .result { background: white; border: 1px solid #d6dde6; border-radius: 8px; padding: 20px; margin: 20px 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
.field { min-width: 0; } label, legend { font-weight: 600; }
input:not([type=checkbox]), select, textarea { display: block; width: 100%; padding: 9px; border: 1px solid #8894a6; border-radius: 4px; font: inherit; margin: 6px 0; }
input[type=checkbox] { margin-right: 8px; } fieldset { margin: 6px 0; border: 1px solid #d6dde6; }
fieldset label { display: block; font-weight: normal; }
.flag { font: .8rem/1.5 monospace; color: #687589; margin: 2px 0 6px; overflow-wrap: anywhere; }
.help, .default, .hint { font-size: .88rem; color: #526075; white-space: pre-line; }
.default { margin-top: 5px; } .error { color: #a51919; font-size: .9rem; min-height: 1.4em; }
[aria-invalid=true] { border-color: #a51919 !important; } [hidden] { display: none !important; }
button { cursor: pointer; padding: 10px 16px; border: 1px solid #8894a6; border-radius: 5px; font: inherit; background: white; }
#copy { display: block; width: 100%; background: #205bcc; color: white; font-size: 1.15rem; font-weight: 600; padding: 16px; margin-top: 15px; }
button:disabled { opacity: .5; cursor: not-allowed; } #command { min-height: 140px; font-family: monospace; }
#status { white-space: pre-line; } .hp-values { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 8px; }
a { color: #205bcc; }
</style>
</head>
<body><main>
<h1>Build your autotune command</h1>
<p>Work through the steps below. All settings are visible, and CLI defaults are already selected.
Choose a training dataset, a backbone checkpoint and a number of random features to get started.</p>
<p class="hint" id="mode"></p>
<form id="builder" novalidate></form>
<div class="result">
<h2>Copy and run</h2>
<p>Paste this command into a POSIX shell (Bash, Zsh, or a Linux/macOS terminal) with Franken installed.</p>
<div id="status" role="status" aria-live="polite"></div>
<label for="command">Autotune command</label><textarea id="command" readonly spellcheck="false"></textarea>
<button type="button" id="copy" disabled>Copy autotune command</button>
<p id="copy-status" role="status" aria-live="polite"></p>
<button type="button" id="reset">Reset to CLI defaults</button>
</div>
</main>
<script id="schema" type="application/json">__SCHEMA__</script>
<script>
'use strict';
const schema = JSON.parse(document.getElementById('schema').textContent);
const form = document.getElementById('builder');
const controls = new Map();
const served = false;
document.getElementById('mode').textContent = served
  ? 'Live validation uses the actual Python CLI parser. No datasets or models are loaded, and nothing is launched.'
  : 'Offline mode checks required fields and supported value formats. For full CLI parser validation, run: python scripts/autotune_ui.py --serve. Paths and checkpoint availability are checked when you run autotune.';
function el(tag, text, attrs = {}) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}
function input(id, value, placeholder = '') {
  return el('input', undefined, {id, type: 'text', value, placeholder, spellcheck: 'false'});
}
function select(id, choices, value) {
  const node = el('select', undefined, {id});
  for (const choice of choices) node.append(el('option', choice, {value: choice}));
  node.value = value;
  return node;
}
function hpParts(value) {
  const raw = String(value).trim();
  if (/^[\[(]/.test(raw)) {
    const parts = raw.slice(1, -1).split(',').map(x => x.trim().replace(/^['"]|['"]$/g, ''));
    if (parts.length === 4 && ['log', 'linear'].includes(parts[3])) return {mode: parts[3], values: parts.slice(0, 3)};
    return {mode: 'list', values: [parts.join(', ')]};
  }
  return {mode: 'single', values: [raw]};
}
function makeHP(field, container) {
  const parsed = hpParts(field.default);
  const mode = select(field.id, ['single', 'list', 'linear', 'log'], parsed.mode);
  container.append(mode);
  const holder = el('div', undefined, {class: 'hp-values'}); container.append(holder);
  function render() {
    holder.replaceChildren();
    const names = mode.value === 'single' ? ['Value'] : mode.value === 'list' ? ['Values, separated by commas'] : ['Start', 'Stop', 'Count'];
    for (let i = 0; i < names.length; i++) {
      const box = el('div');
      const id = field.id + '-hp-' + i;
      box.append(el('label', names[i], {for: id}));
      box.append(input(id, mode.value === parsed.mode ? (parsed.values[i] || '') : '', names[i]));
      holder.append(box);
    }
  }
  mode.addEventListener('change', render); render();
  container.append(el('p', 'Linear ranges include both endpoints. Log ranges use powers of 10: start −6, stop −2 means 10⁻⁶ to 10⁻².', {class: 'hint'}));
  return {node: mode, get: () => {
    const values = Array.from(holder.querySelectorAll('input')).map(n => n.value.trim());
    if (mode.value === 'single') return values[0];
    if (mode.value === 'list') return '[' + values[0] + ']';
    return '(' + values.join(', ') + ', ' + mode.value + ')';
  }, validate: () => {
    const values = Array.from(holder.querySelectorAll('input')).map(n => n.value.trim());
    if (mode.value === 'single') return isNumber(values[0]) ? '' : 'Enter a number.';
    if (mode.value === 'list') return values[0].split(',').filter((x, i, a) => !(i === a.length - 1 && !x.trim())).every(x => isNumber(x.trim())) && values[0].trim() ? '' : 'Enter a comma-separated list of numbers.';
    if (!isNumber(values[0]) || !isNumber(values[1])) return 'Enter numeric start and stop values.';
    if (!isInteger(values[2]) || Number(values[2]) < 1) return 'Count must be a positive integer.';
    return '';
  }};
}
const numeric = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
function isNumber(s) { return numeric.test(s) && Number.isFinite(Number(s)); }
function isInteger(s) { return /^[+-]?\d+$/.test(s); }
for (const [index, section] of schema.sections.entries()) {
  const panel = el('section'); panel.append(el('h2', (index + 1) + '. ' + section.title));
  if (section.help) panel.append(el('p', section.help));
  if (section.id === 'dataset') panel.append(el('p', 'Set a registered dataset name or a training path. A training path takes precedence. Add a validation path to compare trials on separate data.', {class: 'hint'}));
  const grid = el('div', undefined, {class: 'grid'}); panel.append(grid); form.append(panel);
  for (const field of schema.fields.filter(f => f.section === section.id)) {
    const wrap = el('div', undefined, {class: 'field'});
    const title = field.label + (field.required ? ' (required)' : '');
    wrap.append(el('label', title, {for: field.id}));
    wrap.append(el('div', field.flag, {class: 'flag'}));
    let control;
    if (field.kind === 'hyperparameter') control = makeHP(field, wrap);
    else if (field.boolean) {
      const node = el('input', undefined, {id: field.id, type: 'checkbox'});
      // Checkbox always means "include the flag", including store_false flags.
      node.checked = false;
      const label = el('label', 'Enable this flag'); label.prepend(node); wrap.append(label);
      control = {node, get: () => node.checked};
    } else if (field.multiple && field.choices) {
      const box = el('fieldset', undefined, {id: field.id});
      box.append(el('legend', field.id === 'metrics' ? 'Leave unchecked for automatic metrics' : 'Select one or more'));
      for (const choice of field.choices) {
        const node = el('input', undefined, {type: 'checkbox', value: choice});
        node.checked = Array.isArray(field.default) && field.default.includes(choice);
        const label = el('label', choice); label.prepend(node); box.append(label);
      }
      wrap.append(box); control = {node: box, get: () => Array.from(box.querySelectorAll('input:checked')).map(n => n.value)};
    } else {
      const node = field.choices ? select(field.id, field.required ? [''].concat(field.choices) : field.choices, field.default)
        : input(field.id, Array.isArray(field.default) ? field.default.join('\n') : field.default, field.multiple ? 'One value per line' : '');
      if (field.multiple && !field.choices) {
        const area = el('textarea', undefined, {id: field.id, placeholder: 'One metric name per line'});
        area.value = field.default.join('\n'); wrap.append(area);
        control = {node: area, get: () => area.value.split('\n').map(s => s.trim()).filter(Boolean)};
      } else { wrap.append(node); control = {node, get: () => node.value.trim()}; }
    }
    const helpId = field.id + '-help', errorId = field.id + '-error';
    wrap.append(el('div', field.help, {id: helpId, class: 'help'}));
    const defaultText = field.boolean ? 'flag off (configured value: ' + field.default + ')' : Array.isArray(field.default) ? field.default.join(', ') || 'automatic' : field.default || 'unset';
    wrap.append(el('div', 'CLI default: ' + defaultText, {class: 'default'}));
    const error = el('div', '', {id: errorId, class: 'error'}); wrap.append(error);
    control.node.setAttribute('aria-describedby', helpId + ' ' + errorId);
    if (field.required) control.node.setAttribute('aria-required', 'true');
    controls.set(field.id, {...control, wrap, error}); grid.append(wrap);
  }
}
function active(field) { return !field.condition || controls.get(field.section).get() === field.condition; }
function atomicError(value) {
  if (!value || value.toLowerCase() === 'none') return '';
  if (!/^\{[\s\S]*\}$/.test(value)) return 'Use a dictionary, e.g. {1: -0.5, 8: -75.3}, or None.';
  const entries = value.slice(1, -1).trim();
  if (!entries) return '';
  const parts = entries.replace(/,\s*$/, '').split(',');
  for (const part of parts) {
    const pair = part.split(':').map(s => s.trim());
    if (pair.length !== 2 || !isInteger(pair[0].replace(/^['"]|['"]$/g, '')) || !isNumber(pair[1].replace(/^['"]|['"]$/g, ''))) return 'Each entry must map an atomic number to a numeric energy.';
  }
  return '';
}
function fieldError(field, control, value) {
  if (field.required && (value === '' || Array.isArray(value) && !value.length)) return 'This option is required.';
  if (control.validate) return control.validate();
  if (field.id === 'train_targets' && !value.length) return 'Choose at least one training target.';
  if (value === '') return '';
  if (field.kind === 'int' && !isInteger(value)) return 'Enter an integer.';
  if (field.kind === 'optional-int' && value.toLowerCase() !== 'none' && !isInteger(value)) return 'Enter an integer or None.';
  if (field.kind === 'float' && !isNumber(value)) return 'Enter a number.';
  if (field.kind === 'auto-int' && value !== 'auto' && !isInteger(value)) return 'Enter an integer or auto.';
  if (field.kind === 'atomic') return atomicError(value);
  if (String(value).includes('\0')) return 'Null characters cannot be used in shell arguments.';
  return '';
}
// Single quotes preserve spaces, quotes, dollar signs and shell substitutions literally.
function quote(value) { return "'" + String(value).replace(/'/g, "'\\''") + "'"; }
function commandFor(argv) { return schema.command + argv.map(v => ' ' + quote(v)).join(''); }
let revision = 0, timer;
async function update() {
  const current = ++revision; clearTimeout(timer);
  document.getElementById('copy-status').textContent = '';
  const argv = [], errors = [];
  for (const field of schema.fields) {
    const control = controls.get(field.id), enabled = active(field);
    control.wrap.hidden = !enabled; control.error.textContent = '';
    control.node.removeAttribute('aria-invalid');
    if (!enabled) continue;
    const value = control.get(), error = fieldError(field, control, value);
    if (error) { control.error.textContent = error; control.node.setAttribute('aria-invalid', 'true'); errors.push(field.flag + ': ' + error); }
    if (field.boolean) { if (value) argv.push(field.flag); continue; }
    // Omit unchanged optional defaults; still show them in the form.
    if (!field.required && (value === '' || Array.isArray(value) && !value.length || JSON.stringify(value) === JSON.stringify(field.default))) continue;
    // Use --flag=value to allow paths/values starting with a hyphen.
    if (field.multiple) {
      if (value.some(v => v.startsWith('-'))) errors.push(field.flag + ': values cannot start with a hyphen.');
      argv.push(field.flag, ...value);
    } else argv.push(field.flag + '=' + value);
  }
  const dataset = controls.get('dataset_name').get(), train = controls.get('train_path').get();
  if ((!dataset || dataset.toLowerCase() === 'none') && (!train || train.toLowerCase() === 'none')) errors.push('Set --dataset-name or --train-path to choose your training data.');
  const command = document.getElementById('command'), copy = document.getElementById('copy'), status = document.getElementById('status');
  command.value = commandFor(argv); copy.disabled = true;
  if (errors.length) { status.textContent = errors.join('\n'); return; }
  if (!served) { status.textContent = 'Required fields and supported formats are valid. Ready to copy (offline validation).'; copy.disabled = false; return; }
  status.textContent = 'Checking with the Python CLI parser…';
  timer = setTimeout(async () => {
    try {
      const response = await fetch('/validate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(argv)});
      if (!response.ok) throw new Error('Validation server returned ' + response.status);
      const result = await response.json();
      if (current !== revision) return;
      status.textContent = result.valid ? 'Valid configuration according to the autotune CLI parser. Ready to copy.' : result.error;
      copy.disabled = !result.valid;
    } catch (error) {
      if (current === revision) status.textContent = 'Parser validation unavailable. Keep the Python server running. ' + error.message;
    }
  }, 200);
}
form.addEventListener('input', update); form.addEventListener('change', update);
form.addEventListener('submit', event => event.preventDefault());
document.getElementById('reset').addEventListener('click', () => location.reload());
document.getElementById('copy').addEventListener('click', async () => {
  const command = document.getElementById('command');
  try {
    await navigator.clipboard.writeText(command.value);
    document.getElementById('copy-status').textContent = 'Command copied.';
  } catch {
    command.focus(); command.select();
    const copied = document.execCommand('copy');
    document.getElementById('copy-status').textContent = copied ? 'Command copied.'
      : 'Clipboard access is unavailable. The command is selected; press Ctrl+C (or ⌘C) to copy.';
  }
});
update();
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/autotune.html"),
        help="Destination for the self-contained HTML page.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve locally with live validation through the Python parser.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Address to bind; use 0.0.0.0 to share on the local network.",
    )
    args = parser.parse_args()
    html = generate_html()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Generated {args.output}", flush=True)
    if args.serve:
        serve(html, args.port, args.host)


if __name__ == "__main__":
    main()
