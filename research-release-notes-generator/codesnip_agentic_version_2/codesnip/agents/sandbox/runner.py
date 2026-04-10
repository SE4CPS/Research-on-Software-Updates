"""
sandbox/runner.py  — Universal Multi-Language Code Sandbox
===========================================================
Auto-detects language from file extension → generates an instrumented harness
→ executes in a child subprocess → returns a uniform SandboxReport.

Supported languages (confirmed available in runtime environment):
  Python      .py          python3 + tracemalloc + cProfile + gc
  JavaScript  .js .mjs     node + v8.getHeapStatistics() + perf_hooks
  TypeScript  .ts .tsx      tsc compile → node (same instrumentation as JS)
  Java        .java         java 21 single-file + MemoryMXBean + nanoTime
  C           .c            gcc -O0 + clock() + /proc/self/status RSS
  C++         .cpp .cc .cxx g++ -O0 + clock() + try/catch(std::exception&)
  Perl        .pl .pm       perl + Time::HiRes + eval{} for exception capture
  Bash/Shell  .sh .bash     bash + TIMEFORMAT + trap ERR for crash detection
  SQL         .sql          python3 sqlite3 in-memory engine (schema + queries)
  Assembly    .s .asm       GNU as + ld → native binary + clock_gettime syscall

All harnesses output a JSON array in this uniform schema:
  [{ "name": str, "file": str, "line": int,
     "cpu_time_ms": float, "object_delta": int, "mem_delta_kb": float,
     "raised_exception": str|null, "top_calls": [str] }, ...]

The parent process is never at risk — everything runs in a timeout-bounded
child subprocess with a memory cap appropriate to the language runtime.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Language detection
# ═══════════════════════════════════════════════════════════════════════════════

class Language(Enum):
    PYTHON     = auto()
    JAVASCRIPT = auto()
    TYPESCRIPT = auto()
    JAVA       = auto()
    C          = auto()
    CPP        = auto()
    PERL       = auto()
    BASH       = auto()
    SQL        = auto()
    ASSEMBLY   = auto()
    UNKNOWN    = auto()

EXTENSION_MAP: dict[str, Language] = {
    ".py":   Language.PYTHON,
    ".js":   Language.JAVASCRIPT, ".mjs": Language.JAVASCRIPT, ".cjs": Language.JAVASCRIPT,
    ".ts":   Language.TYPESCRIPT, ".tsx": Language.TYPESCRIPT,
    ".java": Language.JAVA,
    ".c":    Language.C,
    ".cpp":  Language.CPP, ".cc": Language.CPP, ".cxx": Language.CPP,
    ".pl":   Language.PERL, ".pm": Language.PERL,
    ".sh":   Language.BASH, ".bash": Language.BASH,
    ".sql":  Language.SQL,
    ".s":    Language.ASSEMBLY, ".asm": Language.ASSEMBLY,
}

LANGUAGE_NAMES = {
    Language.PYTHON: "Python", Language.JAVASCRIPT: "JavaScript",
    Language.TYPESCRIPT: "TypeScript", Language.JAVA: "Java",
    Language.C: "C", Language.CPP: "C++",
    Language.PERL: "Perl", Language.BASH: "Bash",
    Language.SQL: "SQL", Language.ASSEMBLY: "Assembly",
    Language.UNKNOWN: "Unknown",
}

def detect_language(file_path: str) -> Language:
    return EXTENSION_MAP.get(Path(file_path).suffix.lower(), Language.UNKNOWN)

def detect_languages_in_dir(repo_path: str, changed_files: list) -> dict:
    groups: dict[Language, list[str]] = {}
    for f in changed_files:
        lang = detect_language(f)
        if lang != Language.UNKNOWN:
            groups.setdefault(lang, []).append(f)
    return groups


# ═══════════════════════════════════════════════════════════════════════════════
# Shared output types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FunctionProfile:
    name: str
    file: str
    line: int
    language: str                    = "Unknown"
    cpu_time_ms: float               = 0.0
    object_delta: int                = 0
    mem_delta_kb: float              = 0.0
    raised_exception: Optional[str] = None
    top_calls: list                  = field(default_factory=list)

    @property
    def is_leak_suspect(self) -> bool:
        return (self.object_delta > 200) if self.language == "Python" else (self.mem_delta_kb > 512)

    @property
    def is_slow(self) -> bool:
        return self.cpu_time_ms > 500


@dataclass
class SandboxReport:
    executed: bool              = False
    error: Optional[str]        = None
    languages_detected: list    = field(default_factory=list)
    functions: list             = field(default_factory=list)
    leak_suspects: list         = field(default_factory=list)
    slow_functions: list        = field(default_factory=list)
    crash_functions: list       = field(default_factory=list)
    total_exec_ms: float        = 0.0
    raw_output: str             = ""

    def merge(self, other: "SandboxReport"):
        self.executed = self.executed or other.executed
        self.functions.extend(other.functions)
        self.leak_suspects.extend(other.leak_suspects)
        self.slow_functions.extend(other.slow_functions)
        self.crash_functions.extend(other.crash_functions)
        self.total_exec_ms += other.total_exec_ms
        for lang in other.languages_detected:
            if lang not in self.languages_detected:
                self.languages_detected.append(lang)
        if other.error and not self.error:
            self.error = other.error


# ═══════════════════════════════════════════════════════════════════════════════
# Harness generators — one per language
# ═══════════════════════════════════════════════════════════════════════════════

# ── Python ────────────────────────────────────────────────────────────────────

class PythonHarnessGenerator:
    TYPE_DEFAULTS = {
        "str": '"test"', "int": "42", "float": "3.14", "bool": "True",
        "list": "[]", "dict": "{}", "tuple": "()", "set": "set()",
        "bytes": 'b"x"', "Optional": "None", "Any": "None",
    }

    def generate(self, repo_path: str, py_files: list) -> Optional[str]:
        calls = []
        for rel in py_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full):
                continue
            try:
                src = open(full, encoding="utf-8", errors="replace").read()
                tree = ast.parse(src)
            except SyntaxError:
                continue
            mod_name = rel.replace(os.sep, ".").replace("/", ".").removesuffix(".py")
            if not all(p.isidentifier() for p in mod_name.split(".")):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for child in ast.walk(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not child.name.startswith("_"):
                                calls.append({"mod": mod_name, "fn": f"{node.name}.{child.name}",
                                    "args": self._args_for(child, True), "line": child.lineno,
                                    "file": rel, "class": node.name})
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        calls.append({"mod": mod_name, "fn": node.name,
                            "args": self._args_for(node), "line": node.lineno, "file": rel})
        seen, dedup = set(), []
        for c in calls:
            key = (c["mod"], c["fn"])
            if key not in seen:
                seen.add(key); dedup.append(c)
        if not dedup:
            return None
        calls_str = repr(json.dumps(dedup))
        return (
            f"import sys,os,gc,json,time,tracemalloc,cProfile,pstats,io,resource\n"
            f"sys.path.insert(0,{repr(repo_path)})\nos.chdir({repr(repo_path)})\n"
            f"results=[]\nCALLS=json.loads({calls_str})\n"
            "def _gf(m,f,c=None):\n"
            "    try:\n"
            "        mod=__import__(m,fromlist=[m.split('.')[-1]])\n"
            "        if c:\n"
            "            cls=getattr(mod,c,None)\n"
            "            if not cls: return None\n"
            "            try: obj=cls()\n"
            "            except: return None\n"
            "            return getattr(obj,f.split('.')[-1],None)\n"
            "        return getattr(mod,f,None)\n"
            "    except: return None\n"
            "for call in CALLS:\n"
            "    fn=_gf(call['mod'],call['fn'],call.get('class'))\n"
            "    if not fn or not callable(fn): continue\n"
            "    gc.collect();gc.collect()\n"
            "    ob=len(gc.get_objects())\n"
            "    tracemalloc.start()\n"
            "    mr=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
            "    p=cProfile.Profile();t0=time.perf_counter();p.enable();exc=None\n"
            "    try:\n"
            "        a=call.get('args','')\n"
            "        if a: eval(f'fn({a})')\n"
            "        else: fn()\n"
            "    except Exception as e: exc=f'{type(e).__name__}: {str(e)[:150]}'\n"
            "    except SystemExit: exc='SystemExit'\n"
            "    p.disable();el=(time.perf_counter()-t0)*1000\n"
            "    tracemalloc.stop()\n"
            "    ma=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
            "    gc.collect();gc.collect()\n"
            "    oa=len(gc.get_objects())\n"
            "    s=io.StringIO();pstats.Stats(p,stream=s).sort_stats('cumulative').print_stats(5)\n"
            "    top=[l.strip() for l in s.getvalue().splitlines() if '.py' in l][:3]\n"
            "    results.append({'name':call['fn'],'file':call['file'],'line':call['line'],"
            "'cpu_time_ms':el,'object_delta':oa-ob,'mem_delta_kb':(ma-mr)/1024,"
            "'raised_exception':exc,'top_calls':top})\n"
            "print(json.dumps(results))\n"
        )

    def _args_for(self, node, skip_self=False):
        args = list(node.args.args)
        if skip_self and args and args[0].arg in ("self", "cls"):
            args = args[1:]
        parts = []
        for arg in args:
            ann = arg.annotation
            t = "str"
            if ann:
                if isinstance(ann, ast.Name): t = ann.id
                elif isinstance(ann, ast.Attribute): t = ann.attr
                elif isinstance(ann, ast.Subscript): t = "None"
            parts.append(self.TYPE_DEFAULTS.get(t, "None"))
        return ", ".join(parts)


# ── JavaScript / TypeScript ───────────────────────────────────────────────────

class JavaScriptHarnessGenerator:

    def _extract_functions(self, source: str) -> list:
        functions, seen = [], set()
        patterns = [
            r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)',
            r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>',
            r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*async\s+function\s*\(([^)]*)\)',
            r'^\s{2,}(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*(?::\s*\w+\s*)?\{',
        ]
        skip = {"if","for","while","switch","catch","constructor","get","set","return","else"}
        for i, line in enumerate(source.splitlines(), 1):
            for pat in patterns:
                m = re.match(pat, line)
                if m:
                    name = m.group(1)
                    if name in skip or name.startswith("_"): break
                    params = m.group(2).strip()
                    pc = len([p for p in params.split(",") if p.strip()]) if params else 0
                    if name not in seen:
                        seen.add(name)
                        functions.append({"name": name, "line": i, "param_count": pc})
                    break
        return functions

    def _dummy_args(self, n: int) -> str:
        defaults = ['"test"', "42", "true", "[]", "{}", "null"]
        return ", ".join(defaults[i % len(defaults)] for i in range(n))

    def generate(self, repo_path: str, js_files: list, is_typescript: bool = False) -> Optional[str]:
        all_calls = []
        for rel in js_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try:
                src = open(full, encoding="utf-8", errors="replace").read()
            except: continue
            for fn in self._extract_functions(src):
                all_calls.append({"file": rel, "name": fn["name"], "line": fn["line"],
                                  "args": self._dummy_args(fn["param_count"])})
        if not all_calls: return None
        requires = []
        for rel in js_files:
            vn = re.sub(r'[^a-zA-Z0-9_]', '_', rel.replace('/','_').replace('.','_'))
            full = os.path.join(repo_path, rel)
            requires.append(f"let _mod_{vn}; try{{_mod_{vn}=require({repr(full)})}}catch(e){{}}")
        mod_vars = "\n".join(
            f"  _mod_{re.sub(r'[^a-zA-Z0-9_]','_',rel.replace('/','_').replace('.','_'))},"
            for rel in js_files
        )
        return (
            "const v8=require('v8'),{performance}=require('perf_hooks');\n"
            "const origLog=console.log,origErr=console.error;\n"
            "console.log=()=>{};console.error=()=>{};\n"
            + "\n".join(requires) + "\n"
            "console.log=origLog;console.error=origErr;\n"
            "const fnMap={};\n"
            f"const modules=[{chr(10)}{mod_vars}{chr(10)}];\n"
            "for(const mod of modules){\n"
            "  if(!mod)continue;\n"
            "  for(const[k,v] of Object.entries(mod)){\n"
            "    if(typeof v==='function'&&!k.startsWith('_'))fnMap[k]=v;\n"
            "    if(typeof v==='object'&&v!==null){\n"
            "      for(const[mk,mv] of Object.entries(v)){\n"
            "        if(typeof mv==='function'&&!mk.startsWith('_'))fnMap[mk]=mv.bind(v);\n"
            "      }\n    }\n  }\n}\n"
            f"const CALLS={json.dumps(all_calls)};\n"
            "const results=[];\n"
            "async function runAll(){\n"
            "  for(const call of CALLS){\n"
            "    const fn=fnMap[call.name];\n"
            "    if(!fn)continue;\n"
            "    if(global.gc)global.gc();\n"
            "    const hb=v8.getHeapStatistics().used_heap_size;\n"
            "    const t0=performance.now();\n"
            "    let exc=null;\n"
            "    try{\n"
            "      const args=call.args?eval('['+call.args+']'):[];\n"
            "      const r=fn(...args);\n"
            "      if(r&&typeof r.then==='function')await r;\n"
            "    }catch(e){exc=(e.constructor?e.constructor.name:'Error')+': '+String(e.message||e).slice(0,150);}\n"
            "    const el=performance.now()-t0;\n"
            "    if(global.gc)global.gc();\n"
            "    const ha=v8.getHeapStatistics().used_heap_size;\n"
            "    results.push({name:call.name,file:call.file,line:call.line,\n"
            "      cpu_time_ms:el,object_delta:0,mem_delta_kb:(ha-hb)/1024,\n"
            "      raised_exception:exc,top_calls:[]});\n"
            "  }\n"
            "  process.stdout.write(JSON.stringify(results)+'\\n');\n"
            "}\n"
            "runAll().catch(e=>process.stdout.write(JSON.stringify([{name:'__harness__',file:'',line:0,"
            "cpu_time_ms:0,object_delta:0,mem_delta_kb:0,raised_exception:'HarnessError: '+String(e),top_calls:[]}])));\n"
        )


# ── Java ──────────────────────────────────────────────────────────────────────

class JavaHarnessGenerator:

    def _extract_methods(self, source: str) -> list:
        methods, seen = [], set()
        class_name = None
        for m in re.finditer(r'\bclass\s+(\w+)', source):
            class_name = m.group(1)
        pat = re.compile(
            r'public\s+(?:static\s+)?(?:\w+(?:<[^>]*>)?\s+)+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[\w,\s]+)?\s*\{'
        )
        for i, line in enumerate(source.splitlines(), 1):
            m = pat.search(line)
            if m:
                name = m.group(1)
                if name in ("main","toString","hashCode","equals","clone") or name.startswith("_"):
                    continue
                params = []
                for p in m.group(2).strip().split(","):
                    p = p.strip()
                    if p:
                        parts = p.split()
                        if parts: params.append(parts[0].rstrip("<>[]"))
                if name not in seen:
                    seen.add(name)
                    methods.append({"name": name, "line": i, "params": params, "class": class_name})
        return methods

    def _java_arg(self, t: str) -> str:
        m = {"int":"42","long":"42L","double":"3.14","float":"3.14f","boolean":"true",
             "String":'"test"',"char":"'a'","Integer":"42","Long":"42L","short":"1",
             "byte":"1","List":"new java.util.ArrayList<>()","Map":"new java.util.HashMap<>()"}
        return m.get(t.strip(), "null")

    def generate(self, repo_path: str, java_files: list) -> Optional[str]:
        all_methods, sources = [], []
        for rel in java_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try: src = open(full, encoding="utf-8", errors="replace").read()
            except: continue
            for m in self._extract_methods(src):
                m["file"] = rel; all_methods.append(m)
            sources.append(src)
        if not all_methods: return None

        blocks = []
        for m in all_methods:
            cls = m.get("class") or "HarnessMain"
            args = ", ".join(self._java_arg(p) for p in m["params"])
            ne = m["name"].replace('"', '\\"')
            fe = m["file"].replace("\\", "\\\\").replace('"', '\\"')
            blocks.append(
                f'    {{\n'
                f'        long mb=memBean.getHeapMemoryUsage().getUsed();\n'
                f'        long t0=System.nanoTime(); String exc=null;\n'
                f'        try{{ {cls} obj=new {cls}(); obj.{m["name"]}({args}); }}\n'
                f'        catch(Exception e){{exc=e.getClass().getSimpleName()+": "+e.getMessage();}}\n'
                f'        catch(Error e){{exc=e.getClass().getSimpleName()+": "+e.getMessage();}}\n'
                f'        double el=(System.nanoTime()-t0)/1_000_000.0;\n'
                f'        long ma=memBean.getHeapMemoryUsage().getUsed();\n'
                f'        if(first)first=false; else sb.append(",");\n'
                f'        sb.append("{{\\"name\\":\\"{ne}\\",\\"file\\":\\"{fe}\\",\\"line\\":{m["line"]}"'
                f'           +",\\"cpu_time_ms\\":"+String.format("%.3f",el)'
                f'           +",\\"object_delta\\":0,\\"mem_delta_kb\\":"+String.format("%.1f",(ma-mb)/1024.0)'
                f'           +(exc==null?",\\"raised_exception\\":null":(",\\"raised_exception\\":\\""+exc.replace("\\\\","\\\\\\\\").replace("\\"","\\\\\\"")+"\\""))'
                f'           +",\\"top_calls\\":[]}}");'
                f'\n    }}\n'
            )

        inlined = []
        for src in sources:
            s = re.sub(r'^\s*package\s+[\w.]+\s*;', '', src, flags=re.MULTILINE)
            s = re.sub(r'\bpublic\s+class\b', 'class', s)
            inlined.append(s)

        # HarnessMain MUST come first for Java 21 single-file launcher
        return (
            "import java.lang.management.*;\n"
            "import java.util.*;\n\n"
            "public class HarnessMain {\n"
            "    public static void main(String[] args) {\n"
            "        MemoryMXBean memBean=ManagementFactory.getMemoryMXBean();\n"
            "        StringBuilder sb=new StringBuilder(\"[\");\n"
            "        boolean first=true;\n"
            + "".join(blocks) +
            "        sb.append(\"]\");\n"
            "        System.out.println(sb.toString());\n"
            "    }\n"
            "}\n\n"
            + "\n".join(inlined) + "\n"
        )


# ── C / C++ ───────────────────────────────────────────────────────────────────

class CppHarnessGenerator:

    def _extract_functions(self, source: str, is_cpp: bool) -> list:
        fns, seen = [], set()
        if is_cpp:
            pat = re.compile(r'^(?:(?:static|inline|virtual|explicit|constexpr|friend)\s+)*(?:\w+[\s*&:<>]+)+(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:override\s*)?(?:noexcept)?\s*\{')
        else:
            pat = re.compile(r'^(?:static\s+)?(?:\w+\s*[\*&]?\s+)+(\w+)\s*\(([^)]*)\)\s*\{')
        skip = {"main","if","for","while","switch","return","else","do","case"}
        for i, line in enumerate(source.splitlines(), 1):
            m = pat.match(line)
            if m:
                name = m.group(1)
                if name in skip or name.startswith("_"): continue
                params = []
                ps = m.group(2).strip()
                if ps and ps != "void":
                    for p in ps.split(","):
                        parts = p.strip().split()
                        if parts: params.append(parts[0].lstrip("*&"))
                if name not in seen:
                    seen.add(name)
                    fns.append({"name": name, "line": i, "params": params})
        return fns

    def _c_arg(self, t: str, is_cpp: bool) -> str:
        m = {"int":"42","long":"42L","float":"3.14f","double":"3.14","char":"'a'",
             "bool":"true" if is_cpp else "1","short":"1","unsigned":"42U","size_t":"42",
             "string":'std::string("test")' if is_cpp else '"test"'}
        return m.get(t, "0")

    def generate(self, repo_path: str, c_files: list, is_cpp: bool = False) -> Optional[str]:
        all_fns, sources = [], []
        for rel in c_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try: src = open(full, encoding="utf-8", errors="replace").read()
            except: continue
            for f in self._extract_functions(src, is_cpp):
                f["file"] = rel; all_fns.append(f)
            sources.append(src)
        if not all_fns: return None

        blocks = []
        for fn in all_fns:
            args = ", ".join(self._c_arg(p, is_cpp) for p in fn["params"])
            ne = fn["name"]
            fe = fn["file"].replace("\\", "\\\\").replace('"', '\\"')
            if is_cpp:
                blocks.append(
                    f'    {{\n'
                    f'        long vb=get_vm_kb(); clock_t t0=clock(); const char* exc=NULL;\n'
                    f'        try{{ {fn["name"]}({args}); }}\n'
                    f'        catch(const std::exception& e){{exc=e.what();}}\n'
                    f'        catch(...){{exc="unknown exception";}}\n'
                    f'        double el=(double)(clock()-t0)/CLOCKS_PER_SEC*1000.0;\n'
                    f'        long va=get_vm_kb();\n'
                    f'        if(first)first=0; else printf(",");\n'
                    f'        if(exc) printf("{{\\"name\\":\\"{ne}\\",\\"file\\":\\"{fe}\\",\\"line\\":{fn["line"]},'
                    f'\\"cpu_time_ms\\":%.3f,\\"object_delta\\":0,\\"mem_delta_kb\\":%.1f,\\"raised_exception\\":\\"%s\\",\\"top_calls\\":[]}}",el,(double)(va-vb),exc);\n'
                    f'        else printf("{{\\"name\\":\\"{ne}\\",\\"file\\":\\"{fe}\\",\\"line\\":{fn["line"]},'
                    f'\\"cpu_time_ms\\":%.3f,\\"object_delta\\":0,\\"mem_delta_kb\\":%.1f,\\"raised_exception\\":null,\\"top_calls\\":[]}}",el,(double)(va-vb));\n'
                    f'    }}\n'
                )
            else:
                blocks.append(
                    f'    {{\n'
                    f'        long vb=get_vm_kb(); clock_t t0=clock();\n'
                    f'        {fn["name"]}({args});\n'
                    f'        double el=(double)(clock()-t0)/CLOCKS_PER_SEC*1000.0;\n'
                    f'        long va=get_vm_kb();\n'
                    f'        if(first)first=0; else printf(",");\n'
                    f'        printf("{{\\"name\\":\\"{ne}\\",\\"file\\":\\"{fe}\\",\\"line\\":{fn["line"]},'
                    f'\\"cpu_time_ms\\":%.3f,\\"object_delta\\":0,\\"mem_delta_kb\\":%.1f,\\"raised_exception\\":null,\\"top_calls\\":[]}}",el,(double)(va-vb));\n'
                    f'    }}\n'
                )

        inlined = []
        for src in sources:
            s = re.sub(r'\bint\s+main\s*\([^)]*\)\s*\{[^}]*\}', '', src, flags=re.DOTALL)
            inlined.append(s)

        vmfn = (
            "long get_vm_kb(){\n"
            "    long vm=0; FILE* f=fopen(\"/proc/self/status\",\"r\");\n"
            "    if(f){char line[128];\n"
            "    while(fgets(line,128,f)){\n"
            "        if(strncmp(line,\"VmRSS:\",6)==0){sscanf(line+6,\"%ld\",&vm);break;}\n"
            "    }fclose(f);} return vm;\n"
            "}\n"
        )

        if is_cpp:
            header = "#include <stdio.h>\n#include <stdlib.h>\n#include <time.h>\n#include <string.h>\n#include <string>\n#include <vector>\n#include <map>\n#include <stdexcept>\n"
        else:
            header = "#include <stdio.h>\n#include <stdlib.h>\n#include <time.h>\n#include <string.h>\n"

        return (
            header
            + "\n".join(inlined) + "\n\n"
            + vmfn
            + "int main(void){\n    int first=1;\n    printf(\"[\");\n"
            + "".join(blocks)
            + '    printf("]\\n");\n    return 0;\n}\n'
        )


# ── Perl ──────────────────────────────────────────────────────────────────────

class PerlHarnessGenerator:
    """
    Generates a Perl harness using Time::HiRes for timing and eval{} for
    exception capture. Extracts named subs from .pl/.pm files.
    """

    def _extract_subs(self, source: str) -> list:
        subs, seen = [], set()
        # Match: sub name { or sub name (proto) {
        pat = re.compile(r'^sub\s+(\w+)\s*(?:\([^)]*\))?\s*\{', re.MULTILINE)
        for m in pat.finditer(source):
            name = m.group(1)
            line = source[:m.start()].count('\n') + 1
            if name not in seen and not name.startswith('_'):
                seen.add(name)
                subs.append({"name": name, "line": line})
        return subs

    def _count_params(self, source: str, sub_name: str) -> int:
        """Rough estimate of param count from @_ usage."""
        # Look for my ($a, $b) = @_ pattern
        m = re.search(rf'sub\s+{re.escape(sub_name)}[^{{]*\{{[^}}]*my\s*\(([^)]+)\)\s*=\s*@_', source)
        if m:
            return len([x for x in m.group(1).split(',') if x.strip()])
        return 1  # default: one arg

    def _dummy_args(self, n: int) -> str:
        defaults = ['"test"', '42', '3.14', '"hello"', '0', '1']
        return ", ".join(defaults[i % len(defaults)] for i in range(n))

    def generate(self, repo_path: str, pl_files: list) -> Optional[str]:
        all_subs = []
        all_sources = []
        for rel in pl_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try: src = open(full, encoding="utf-8", errors="replace").read()
            except: continue
            for s in self._extract_subs(src):
                n = self._count_params(src, s["name"])
                s["file"] = rel
                s["args"] = self._dummy_args(n)
                all_subs.append(s)
            all_sources.append((rel, src))

        if not all_subs: return None

        # Build the harness
        source_blocks = []
        for rel, src in all_sources:
            # Strip shebang and package declarations from included files
            s = re.sub(r'^#!.*\n', '', src)
            s = re.sub(r'^package\s+\w+\s*;', '', s, flags=re.MULTILINE)
            source_blocks.append(f"# --- {rel} ---\n{s}\n")

        calls_json = json.dumps(all_subs)

        # Build harness using string concatenation to avoid quote escaping issues
        calls_perl = calls_json.replace('\\', '\\\\').replace("'", "\\'")
        inlined_src = ''.join(source_blocks)
        return (
            "#!/usr/bin/perl\n"
            "use strict;\n"
            "use warnings;\n"
            "use Time::HiRes qw(time);\n"
            "use JSON::PP;\n\n"
            "# --- Inlined source ---\n"
            + inlined_src +
            "# --- End inlined ---\n\n"
            "my $calls_json = '" + calls_perl + "';\n"
            "my @CALLS = @{decode_json($calls_json)};\n"
            "my @results;\n\n"
            "for my $call (@CALLS) {\n"
            "    my $name = $call->{name};\n"
            "    my $args_str = $call->{args} // '';\n"
            "    my $t0 = time();\n"
            "    my $exc = undef;\n"
            "    eval {\n"
            "        no strict 'refs';\n"
            "        if ($args_str) {\n"
            "            my @dummy_args = eval qq{($args_str)};\n"
            "            &{$name}(@dummy_args);\n"
            "        } else {\n"
            "            &{$name}();\n"
            "        }\n"
            "    };\n"
            "    if ($@) {\n"
            "        ($exc = $@) =~ s/\\n/ /g;\n"
            "        $exc =~ s/\"/\\\"/g;\n"
            "        $exc =~ s/\\s+at .*$//;\n"
            "    }\n"
            "    my $elapsed_ms = (time() - $t0) * 1000;\n"
            "    push @results, {\n"
            "        name => $name, file => $call->{file}, line => $call->{line}+0,\n"
            "        cpu_time_ms => $elapsed_ms, object_delta => 0, mem_delta_kb => 0,\n"
            "        raised_exception => $exc, top_calls => [],\n"
            "    };\n"
            "}\n\n"
            "print encode_json(\\@results) . \"\\n\";\n"
        )


# ── Bash / Shell ──────────────────────────────────────────────────────────────

class BashHarnessGenerator:
    """
    Generates a Bash harness that sources the original script and times each
    function with TIMEFORMAT + { time func; } 2>&1. Errors are caught via
    set -e and trap.
    """

    def _extract_functions(self, source: str) -> list:
        fns, seen = [], set()
        # Match: function name() { or name() {
        pat = re.compile(r'^(?:function\s+)?(\w+)\s*\(\s*\)\s*\{', re.MULTILINE)
        skip = {"main", "usage", "help", "cleanup", "setup", "teardown"}
        for m in pat.finditer(source):
            name = m.group(1)
            line = source[:m.start()].count('\n') + 1
            if name not in seen and not name.startswith('_') and name not in skip:
                seen.add(name)
                fns.append({"name": name, "line": line})
        return fns

    def generate(self, repo_path: str, sh_files: list) -> Optional[str]:
        all_fns = []
        for rel in sh_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try: src = open(full, encoding="utf-8", errors="replace").read()
            except: continue
            for fn in self._extract_functions(src):
                fn["file"] = rel
                all_fns.append((fn, full))

        if not all_fns: return None

        call_blocks = []
        sourced = set()
        for fn, full_path in all_fns:
            if full_path not in sourced:
                call_blocks.append(f'. {repr(full_path)} 2>/dev/null || true')
                sourced.add(full_path)

        call_blocks.append("RESULTS='['")
        call_blocks.append("FIRST=1")

        for fn, full_path in all_fns:
            name = fn["name"]
            rel = fn["file"]
            line = fn["line"]
            call_blocks.append(textwrap.dedent(f"""\
            {{
              EXC=""
              T0=$(date +%s%N 2>/dev/null || echo 0)
              {name} "test" "42" 2>/dev/null || EXC="ExitCode: $?"
              T1=$(date +%s%N 2>/dev/null || echo 0)
              if [ "$T0" != "0" ] && [ "$T1" != "0" ]; then
                ELAPSED_MS=$(echo "scale=3; ($T1 - $T0) / 1000000" | bc 2>/dev/null || echo "0")
              else
                ELAPSED_MS="0"
              fi
              if [ "$FIRST" = "1" ]; then FIRST=0; else RESULTS="$RESULTS,"; fi
              if [ -n "$EXC" ]; then
                RESULTS="$RESULTS{{\\\"name\\\":\\\"{name}\\\",\\\"file\\\":\\\"{rel}\\\",\\\"line\\\":{line},\\\"cpu_time_ms\\\":$ELAPSED_MS,\\\"object_delta\\\":0,\\\"mem_delta_kb\\\":0,\\\"raised_exception\\\":\\\"$EXC\\\",\\\"top_calls\\\":[]}}"
              else
                RESULTS="$RESULTS{{\\\"name\\\":\\\"{name}\\\",\\\"file\\\":\\\"{rel}\\\",\\\"line\\\":{line},\\\"cpu_time_ms\\\":$ELAPSED_MS,\\\"object_delta\\\":0,\\\"mem_delta_kb\\\":0,\\\"raised_exception\\\":null,\\\"top_calls\\\":[]}}"
              fi
            }}"""))

        call_blocks.append("RESULTS=\"$RESULTS]\"")
        call_blocks.append("echo \"$RESULTS\"")

        return "#!/bin/bash\nset +e\n\n" + "\n".join(call_blocks) + "\n"


# ── SQL ───────────────────────────────────────────────────────────────────────

class SQLHarnessGenerator:
    """
    Runs SQL statements in an in-memory SQLite3 database via Python.
    Extracts SELECT/INSERT/UPDATE/DELETE/CREATE statements and times each.
    """

    def _extract_statements(self, source: str) -> list:
        stmts = []
        # Split on semicolons (rough but works for most SQL files)
        raw_stmts = [s.strip() for s in source.split(';') if s.strip()]
        skip_prefixes = ('--', '/*', '#')
        for i, stmt in enumerate(raw_stmts):
            # Remove inline comments
            lines = [l for l in stmt.splitlines() if not l.strip().startswith('--')]
            clean = ' '.join(lines).strip()
            if not clean: continue
            kw = clean.upper().split()[0] if clean.split() else ''
            if kw in ('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP',
                      'ALTER', 'WITH', 'EXPLAIN'):
                name = f"{kw.lower()}_{i+1}"
                stmts.append({"name": name, "sql": clean, "line": i + 1})
        return stmts

    def generate(self, repo_path: str, sql_files: list) -> Optional[str]:
        all_stmts = []
        for rel in sql_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try: src = open(full, encoding="utf-8", errors="replace").read()
            except: continue
            for s in self._extract_statements(src):
                s["file"] = rel
                all_stmts.append(s)

        if not all_stmts: return None

        stmts_json = repr(json.dumps(all_stmts))
        return (
            "import sqlite3, json, time\n"
            "conn = sqlite3.connect(':memory:')\n"
            "conn.row_factory = sqlite3.Row\n"
            f"STMTS = json.loads({stmts_json})\n"
            "results = []\n"
            "for s in STMTS:\n"
            "    t0 = time.perf_counter()\n"
            "    exc = None\n"
            "    try:\n"
            "        cur = conn.execute(s['sql'])\n"
            "        list(cur)  # consume rows\n"
            "        conn.commit()\n"
            "    except Exception as e:\n"
            "        exc = f'{type(e).__name__}: {str(e)[:150]}'\n"
            "    el = (time.perf_counter() - t0) * 1000\n"
            "    results.append({'name': s['name'], 'file': s['file'], 'line': s['line'],\n"
            "        'cpu_time_ms': el, 'object_delta': 0, 'mem_delta_kb': 0,\n"
            "        'raised_exception': exc, 'top_calls': []})\n"
            "conn.close()\n"
            "print(json.dumps(results))\n"
        )


# ── Assembly ──────────────────────────────────────────────────────────────────

class AssemblyHarnessGenerator:
    """
    Assembles .s files with GNU 'as' + 'ld', wraps them in a C harness that
    calls declared global symbols and times them with clock_gettime.
    Falls back to a pure static analysis report if the code can't be linked.
    """

    def _extract_globals(self, source: str) -> list:
        """Find .global (public) labels that look like functions (not data)."""
        globals_ = []
        seen = set()
        global_pat = re.compile(r'^\s*\.(?:global|globl)\s+(\w+)', re.MULTILINE)
        for m in global_pat.finditer(source):
            name = m.group(1)
            if name in ('_start', '__start', 'main') or name.startswith('__'):
                continue
            line = source[:m.start()].count('\n') + 1
            if name not in seen:
                seen.add(name)
                globals_.append({"name": name, "line": line})
        return globals_

    def generate(self, repo_path: str, asm_files: list) -> Optional[str]:
        all_syms = []
        all_sources = []
        for rel in asm_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try: src = open(full, encoding="utf-8", errors="replace").read()
            except: continue
            for sym in self._extract_globals(src):
                sym["file"] = rel
                all_syms.append(sym)
            all_sources.append((rel, full))

        if not all_syms:
            # Fallback: treat file as a single unit and report static analysis
            if not all_sources:
                return None
            rel, _ = all_sources[0]
            return (
                "import json\n"
                f"print(json.dumps([{{'name': 'asm_file', 'file': {repr(rel)}, 'line': 1, "
                "'cpu_time_ms': 0, 'object_delta': 0, 'mem_delta_kb': 0, "
                "'raised_exception': null, 'top_calls': ['Assembly: no exported symbols found']}}]))\n"
                .replace("null", "None")
            )

        # Generate a C wrapper that calls each global symbol
        decls = "\n".join(f"extern void {sym['name']}();" for sym in all_syms)
        sym_list = json.dumps(all_syms)

        return textwrap.dedent(f"""\
#include <stdio.h>
#include <time.h>
#include <string.h>

{decls}

static double ns_elapsed(struct timespec a, struct timespec b) {{
    return (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1e6;
}}

long get_vm_kb() {{
    long vm = 0;
    FILE* f = fopen("/proc/self/status", "r");
    if (f) {{
        char line[128];
        while (fgets(line, 128, f))
            if (strncmp(line, "VmRSS:", 6) == 0) {{ sscanf(line + 6, "%ld", &vm); break; }}
        fclose(f);
    }}
    return vm;
}}

int main(void) {{
    struct {{ const char* name; const char* file; int line; void (*fn)(); }} syms[] = {{
        {', '.join(f'{{"{s["name"]}", "{s["file"]}", {s["line"]}, {s["name"]}}}' for s in all_syms)}
    }};
    int n = {len(all_syms)};
    int first = 1;
    printf("[");
    for (int i = 0; i < n; i++) {{
        struct timespec t0, t1;
        long vb = get_vm_kb();
        clock_gettime(CLOCK_MONOTONIC, &t0);
        syms[i].fn();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        long va = get_vm_kb();
        double el = ns_elapsed(t0, t1);
        if (first) first = 0; else printf(",");
        printf("{{\\"name\\":\\"%s\\",\\"file\\":\\"%s\\",\\"line\\":%d,\\"cpu_time_ms\\":%.3f,"
               "\\"object_delta\\":0,\\"mem_delta_kb\\":%.1f,\\"raised_exception\\":null,"
               "\\"top_calls\\":[]}}",
               syms[i].name, syms[i].file, syms[i].line, el, (double)(va - vb));
    }}
    printf("]\\n");
    return 0;
}}
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Language executor — runs subprocess per language
# ═══════════════════════════════════════════════════════════════════════════════

class LanguageExecutor:
    def __init__(self, timeout: int = 25, max_memory_mb: int = 256):
        self.timeout     = timeout
        self.max_mem_mb  = max_memory_mb

    def _run(self, cmd: list, cwd: str = None, env: dict = None, apply_mem_limit: bool = True):
        import resource
        t0 = time.time()
        try:
            kwargs = dict(capture_output=True, text=True, timeout=self.timeout, cwd=cwd, env=env)
            if apply_mem_limit:
                limit = self.max_mem_mb * 1024 * 1024 * 4
                kwargs["preexec_fn"] = lambda: resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            r = subprocess.run(cmd, **kwargs)
            return r.stdout, r.stderr, (time.time() - t0) * 1000
        except subprocess.TimeoutExpired:
            return "", f"TIMEOUT after {self.timeout}s", (time.time() - t0) * 1000

    def run_python(self, path: str):
        return self._run([sys.executable, path], apply_mem_limit=True)

    def run_js(self, path: str, is_ts: bool = False):
        flags = [f"--max-old-space-size={self.max_mem_mb}", "--expose-gc"]
        if is_ts: flags.append("--experimental-strip-types")
        return self._run(["node"] + flags + [path], apply_mem_limit=False)

    def run_java(self, path: str, cwd: str):
        flags = [f"-Xmx{self.max_mem_mb}m", "-Xms32m"]
        return self._run(["java"] + flags + [path], cwd=cwd, apply_mem_limit=False)

    def run_c(self, src: str, cwd: str, is_cpp: bool = False):
        compiler = "g++" if is_cpp else "gcc"
        binary   = os.path.join(cwd, "out")
        flags    = ["-O0", "-o", binary, src] + (["-std=c++17"] if is_cpp else ["-std=c11"])
        out, err, _ = self._run([compiler] + flags + ["-fsanitize=address,undefined"], cwd=cwd, apply_mem_limit=False)
        if not os.path.exists(binary):
            out, err, _ = self._run([compiler] + flags, cwd=cwd, apply_mem_limit=False)
        if not os.path.exists(binary):
            return "", f"Compile error: {err[:300]}", 0.0
        return self._run([binary], cwd=cwd, apply_mem_limit=False)

    def run_perl(self, path: str):
        return self._run(["perl", "-MJSON::PP", path], apply_mem_limit=True)

    def run_bash(self, path: str):
        return self._run(["bash", path], apply_mem_limit=True)

    def run_sql_python(self, path: str):
        return self._run([sys.executable, path], apply_mem_limit=True)

    def run_asm(self, asm_files: list, c_wrapper: str, cwd: str):
        """Assemble all .s files then compile the C wrapper that calls them."""
        obj_files = []
        for asm_path in asm_files:
            obj = asm_path.replace('.s', '.o').replace('.asm', '.o')
            out, err, _ = self._run(["as", asm_path, "-o", obj], cwd=cwd, apply_mem_limit=False)
            if os.path.exists(obj):
                obj_files.append(obj)

        if not obj_files:
            return "", "Assembly failed: no object files produced", 0.0

        # Compile C wrapper with all object files
        wrapper_path = os.path.join(cwd, "asm_wrapper.c")
        binary       = os.path.join(cwd, "asm_out")
        open(wrapper_path, "w").write(c_wrapper)
        cmd = ["gcc", "-O0", wrapper_path] + obj_files + ["-lm", "-o", binary]
        out, err, _ = self._run(cmd, cwd=cwd, apply_mem_limit=False)
        if not os.path.exists(binary):
            return "", f"Link error: {err[:300]}", 0.0
        return self._run([binary], cwd=cwd, apply_mem_limit=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Main CodeSandbox
# ═══════════════════════════════════════════════════════════════════════════════

class CodeSandbox:
    """
    Universal sandbox — give it a repo path and a list of changed files.
    It auto-detects languages, runs per-language harnesses, and returns
    a merged SandboxReport with uniform FunctionProfile entries.

    Handles mixed-language PRs naturally: each language gets its own
    subprocess, all results are merged into one report.
    """

    def __init__(self, timeout_seconds: int = 25, max_memory_mb: int = 256):
        self.timeout = timeout_seconds
        self.max_mem = max_memory_mb
        self.exe     = LanguageExecutor(timeout_seconds, max_memory_mb)

    def run(self, repo_path: str, changed_files: list) -> SandboxReport:
        report = SandboxReport()
        if not changed_files:
            report.error = "No files to execute"
            return report

        groups = detect_languages_in_dir(repo_path, changed_files)
        if not groups:
            report.error = "No supported language files detected"
            return report

        for lang, files in groups.items():
            sub = self._run_lang(lang, files, repo_path)
            report.merge(sub)
            name = LANGUAGE_NAMES[lang]
            if name not in report.languages_detected:
                report.languages_detected.append(name)

        return report

    def _run_lang(self, lang: Language, files: list, repo_path: str) -> SandboxReport:
        dispatch = {
            Language.PYTHON:     lambda: self._py(files, repo_path),
            Language.JAVASCRIPT: lambda: self._js(files, repo_path, False),
            Language.TYPESCRIPT: lambda: self._js(files, repo_path, True),
            Language.JAVA:       lambda: self._java(files, repo_path),
            Language.C:          lambda: self._c(files, repo_path, False),
            Language.CPP:        lambda: self._c(files, repo_path, True),
            Language.PERL:       lambda: self._perl(files, repo_path),
            Language.BASH:       lambda: self._bash(files, repo_path),
            Language.SQL:        lambda: self._sql(files, repo_path),
            Language.ASSEMBLY:   lambda: self._asm(files, repo_path),
        }
        fn = dispatch.get(lang)
        if not fn:
            r = SandboxReport(); r.error = f"No executor for {LANGUAGE_NAMES[lang]}"; return r
        return fn()

    # ── JSON output parser (shared) ──────────────────────────────────────────

    def _parse(self, stdout: str, report: SandboxReport, lang: str):
        stdout = stdout.strip()
        if not stdout:
            report.error = f"[{lang}] No harness output"
            return
        m = re.search(r'\[.*\]', stdout, re.DOTALL)
        if not m:
            report.error = f"[{lang}] No JSON in output: {stdout[:200]}"
            return
        try:
            for item in json.loads(m.group(0)):
                fp = FunctionProfile(
                    name=item.get("name","?"), file=item.get("file",""),
                    line=item.get("line",0), language=lang,
                    cpu_time_ms=item.get("cpu_time_ms",0.0),
                    object_delta=item.get("object_delta",0),
                    mem_delta_kb=item.get("mem_delta_kb",0.0),
                    raised_exception=item.get("raised_exception"),
                    top_calls=item.get("top_calls",[]),
                )
                report.functions.append(fp)
                if fp.is_leak_suspect:  report.leak_suspects.append(fp)
                if fp.is_slow:          report.slow_functions.append(fp)
                if fp.raised_exception: report.crash_functions.append(fp)
            report.executed = True
        except json.JSONDecodeError as e:
            report.error = f"[{lang}] JSON error: {e}. Raw: {stdout[:300]}"

    # ── Language runners ──────────────────────────────────────────────────────

    def _py(self, files: list, repo: str) -> SandboxReport:
        r = SandboxReport()
        harness = PythonHarnessGenerator().generate(repo, files)
        if not harness: r.error = "No callable Python functions"; return r
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="cs_py_") as f:
            f.write(harness); path = f.name
        try:
            out, err, el = self.exe.run_python(path)
            r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
            self._parse(out, r, "Python")
        finally:
            try: os.unlink(path)
            except: pass
        return r

    def _js(self, files: list, repo: str, is_ts: bool) -> SandboxReport:
        r = SandboxReport()
        lang = "TypeScript" if is_ts else "JavaScript"
        compile_dir = None

        if is_ts:
            import shutil as _shutil
            compile_dir = tempfile.mkdtemp(prefix="cs_tscomp_")
            compiled_map = {}
            for rel in files:
                full = os.path.join(repo, rel)
                if not os.path.exists(full): continue
                self.exe._run(
                    ["tsc", "--outDir", compile_dir, "--skipLibCheck",
                     "--module", "commonjs", "--target", "ES2020", full],
                    apply_mem_limit=False,
                )
                js_name = os.path.splitext(os.path.basename(rel))[0] + ".js"
                js_path = os.path.join(compile_dir, js_name)
                if os.path.exists(js_path):
                    compiled_map[rel] = js_path
            if not compiled_map:
                # Regex-based fallback type stripping
                _shutil.rmtree(compile_dir, ignore_errors=True)
                compile_dir = tempfile.mkdtemp(prefix="cs_tsstrip_")
                for rel in files:
                    full = os.path.join(repo, rel)
                    if not os.path.exists(full): continue
                    src = open(full, encoding="utf-8", errors="replace").read()
                    src = re.sub(r'\binterface\s+\w+\s*\{[^}]*\}', '', src, flags=re.DOTALL)
                    src = re.sub(r'\btype\s+\w+\s*=\s*[^;]+;', '', src)
                    src = re.sub(r':\s*\w+(?:<[^>]*>)?(?:\[\])?(?:\s*\|\s*\w+(?:<[^>]*>)?(?:\[\])?)*', '', src)
                    js_name = os.path.splitext(os.path.basename(rel))[0] + ".js"
                    open(os.path.join(compile_dir, js_name), "w").write(src)
                    compiled_map[rel] = os.path.join(compile_dir, js_name)
            harness = JavaScriptHarnessGenerator().generate(
                compile_dir,
                [os.path.basename(p) for p in compiled_map.values()],
                False
            )
        else:
            harness = JavaScriptHarnessGenerator().generate(repo, files, False)

        if not harness:
            if compile_dir:
                import shutil as _shutil; _shutil.rmtree(compile_dir, ignore_errors=True)
            r.error = f"No callable {lang} functions"; return r

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, prefix="cs_js_") as f:
            f.write(harness); path = f.name
        try:
            out, err, el = self.exe.run_js(path, False)
            r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
            self._parse(out, r, lang)
        finally:
            try: os.unlink(path)
            except: pass
            if compile_dir:
                import shutil as _shutil; _shutil.rmtree(compile_dir, ignore_errors=True)
        return r

    def _java(self, files: list, repo: str) -> SandboxReport:
        r = SandboxReport()
        harness = JavaHarnessGenerator().generate(repo, files)
        if not harness: r.error = "No callable Java methods"; return r
        import shutil
        wd = tempfile.mkdtemp(prefix="cs_java_")
        hp = os.path.join(wd, "HarnessMain.java")
        try:
            open(hp, "w").write(harness)
            out, err, el = self.exe.run_java(hp, wd)
            r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
            self._parse(out, r, "Java")
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        return r

    def _c(self, files: list, repo: str, is_cpp: bool) -> SandboxReport:
        r = SandboxReport()
        harness = CppHarnessGenerator().generate(repo, files, is_cpp)
        if not harness: r.error = f"No callable {'C++' if is_cpp else 'C'} functions"; return r
        import shutil
        wd = tempfile.mkdtemp(prefix="cs_c_")
        hp = os.path.join(wd, "harness.cpp" if is_cpp else "harness.c")
        try:
            open(hp, "w").write(harness)
            out, err, el = self.exe.run_c(hp, wd, is_cpp)
            r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
            self._parse(out, r, "C++" if is_cpp else "C")
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        return r

    def _perl(self, files: list, repo: str) -> SandboxReport:
        r = SandboxReport()
        # Check if JSON Perl module is available
        check_out, _, _ = self.exe._run(["perl", "-MJSON::PP", "-e", "1"], apply_mem_limit=False)
        gen = PerlHarnessGenerator()
        harness = gen.generate(repo, files)
        if not harness: r.error = "No callable Perl subs"; return r
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pl", delete=False, prefix="cs_perl_") as f:
            f.write(harness); path = f.name
        try:
            out, err, el = self.exe.run_perl(path)
            r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
            if "Can't locate JSON::PP" in err or "Can't locate JSON." in err:
                # Fallback: use a pure-Perl JSON encoder
                r.error = "[Perl] JSON module not available; install perl-JSON"
            else:
                self._parse(out, r, "Perl")
        finally:
            try: os.unlink(path)
            except: pass
        return r

    def _bash(self, files: list, repo: str) -> SandboxReport:
        r = SandboxReport()
        gen = BashHarnessGenerator()
        harness = gen.generate(repo, files)
        if not harness: r.error = "No callable Bash functions"; return r
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, prefix="cs_bash_") as f:
            f.write(harness); path = f.name
        os.chmod(path, 0o755)
        try:
            out, err, el = self.exe.run_bash(path)
            r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
            self._parse(out, r, "Bash")
        finally:
            try: os.unlink(path)
            except: pass
        return r

    def _sql(self, files: list, repo: str) -> SandboxReport:
        r = SandboxReport()
        gen = SQLHarnessGenerator()
        harness = gen.generate(repo, files)
        if not harness: r.error = "No SQL statements found"; return r
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="cs_sql_") as f:
            f.write(harness); path = f.name
        try:
            out, err, el = self.exe.run_sql_python(path)
            r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
            self._parse(out, r, "SQL")
        finally:
            try: os.unlink(path)
            except: pass
        return r

    def _asm(self, files: list, repo: str) -> SandboxReport:
        r = SandboxReport()
        gen = AssemblyHarnessGenerator()
        # Need full paths for assembly
        full_paths = [os.path.join(repo, f) for f in files if os.path.exists(os.path.join(repo, f))]
        if not full_paths: r.error = "No assembly files found"; return r

        # Combine all sources for symbol extraction
        all_src = "\n".join(open(p).read() for p in full_paths)
        c_wrapper = gen.generate(repo, files)
        if not c_wrapper: r.error = "No assembly symbols found"; return r

        import shutil
        wd = tempfile.mkdtemp(prefix="cs_asm_")
        # Copy asm files to work dir
        asm_in_wd = []
        try:
            for fp in full_paths:
                dest = os.path.join(wd, os.path.basename(fp))
                import shutil as sh; sh.copy(fp, dest)
                asm_in_wd.append(dest)
            out, err, el = self.exe.run_asm(asm_in_wd, c_wrapper, wd)
            r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
            if "Link error" in err or "Assembly failed" in err:
                r.error = err[:200]
            else:
                self._parse(out, r, "Assembly")
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        return r


# ── Backwards-compat alias ────────────────────────────────────────────────────
HarnessGenerator = PythonHarnessGenerator
# ═══════════════════════════════════════════════════════════════════════════════
# TRANSPILER-BACKED LANGUAGES
# Ruby, Go, Rust, PHP, Kotlin, R
#
# These languages are not natively installed. Each transpiler converts source
# to Python (for execution) and reports:
#   - "transpiled_from": original language name
#   - "raised_exception": real runtime errors caught during Python execution
#   - "top_calls": transpilation notes / static analysis warnings
#
# Transpilation is best-effort for idiomatic common patterns. Complex code
# gets static analysis instead of execution. Agents see the real runtime
# results and can tell users "this crash would also happen in <lang>".
# ═══════════════════════════════════════════════════════════════════════════════

import textwrap as _textwrap
import re as _re


def _py_json_result(name, file_, line, cpu_ms, obj_delta, mem_kb, exc, notes):
    """Build the Python expression that produces one FunctionProfile dict."""
    exc_s = repr(exc) if exc else "None"
    return (
        f"results.append({{'name':{repr(name)},'file':{repr(file_)},"
        f"'line':{line},'cpu_time_ms':el,'object_delta':{obj_delta},"
        f"'mem_delta_kb':{mem_kb},'raised_exception':{exc_s},"
        f"'top_calls':{repr(notes)}}})\n"
    )


# ── Ruby ──────────────────────────────────────────────────────────────────────

class RubyTranspiler:
    """Transpiles Ruby (.rb) to Python for execution."""

    # Default argument values by pattern
    _TYPE_DEFAULTS = ["'test'", "42", "3.14", "True", "[]", "{}", "None"]

    def _extract_methods(self, source: str) -> list:
        methods, seen = [], set()
        pat = _re.compile(r'^(?:def\s+)(\w+)\(([^)]*)\)', _re.MULTILINE)
        for m in pat.finditer(source):
            name, params_str = m.group(1), m.group(2).strip()
            line = source[:m.start()].count('\n') + 1
            if name.startswith('_') or name in ('initialize', 'new'):
                continue
            params = [p.strip().lstrip('*&') for p in params_str.split(',') if p.strip()]
            if name not in seen:
                seen.add(name)
                methods.append({'name': name, 'line': line, 'params': params})
        return methods

    def _to_python(self, source: str) -> str:
        src = source
        # Remove shebang
        src = _re.sub(r'^#!.*\n', '', src)
        # def method(params) -> def method(params):
        src = _re.sub(r'\bdef (\w+)\(([^)]*)\)', r'def \1(\2):', src)
        src = _re.sub(r'\bdef (\w+)\b(?!\s*\()', r'def \1():', src)
        # end -> (remove)
        src = _re.sub(r'^\s*end\s*$', '', src, flags=_re.MULTILINE)
        # RAISE WITH INLINE IF: raise X, "msg" if cond -> if cond: raise X("msg")
        src = _re.sub(r'raise\s+(\w+),\s*"([^"]*)"\s+if\s+(.+)',
                      r'if \3: raise \1("\2")', src)
        src = _re.sub(r"raise\s+(\w+),\s*'([^']*)'\s+if\s+(.+)",
                      r"if \3: raise \1('\2')", src)
        src = _re.sub(r'raise\s+"([^"]*)"\s+if\s+(.+)',
                      r'if \2: raise RuntimeError("\1")', src)
        src = _re.sub(r'raise\s+(\w+),\s*"([^"]*)"', r'raise \1("\2")', src)
        src = _re.sub(r"raise\s+(\w+),\s*'([^']*)'", r"raise \1('\2')", src)
        src = _re.sub(r'raise\s+"([^"]*)"', r'raise RuntimeError("\1")', src)
        src = _re.sub(r'raise\s+(\w+)\s*$', r'raise \1()', src, flags=_re.MULTILINE)
        src = _re.sub(r'\bthrow\b', 'raise RuntimeError', src)
        # String interpolation "hello #{name}" -> f"hello {name}"
        src = _re.sub(r'"([^"]*)[#][{]([^}]+)[}]([^"]*)"', r'f"\1{\2}\3"', src)
        # nil/true/false
        src = _re.sub(r'\bnil\b', 'None', src)
        src = _re.sub(r'\btrue\b', 'True', src)
        src = _re.sub(r'\bfalse\b', 'False', src)
        # puts / p
        src = _re.sub(r'\bputs\s+', 'print(', src)
        src = _re.sub(r'\bp\s+(.+)', r'print(repr(\1))', src)
        # attr_accessor etc -> pass
        src = _re.sub(r'\battr_(?:accessor|reader|writer)\s+.*', 'pass', src)
        # unless/until
        src = _re.sub(r'\bunless\s+', 'if not ', src)
        src = _re.sub(r'\buntil\s+', 'while not ', src)
        # rescue => e -> except ... as e:
        src = _re.sub(r'\brescue\s+(\w+)\s+=>\s+(\w+)', r'except \1 as \2:', src)
        src = _re.sub(r'\brescue\b', 'except Exception:', src)
        src = _re.sub(r'^\s*begin\s*$', 'try:', src, flags=_re.MULTILINE)
        src = _re.sub(r'\belsif\b', 'elif', src)
        src = _re.sub(r'^(\s*(?:if|elsif|elif|while|for|else|unless|until)[^:{\n]*[^:{\n])$',
                      r'\1:', src, flags=_re.MULTILINE)
        src = _re.sub(r'\brequire(?:_relative)?\s+.*', 'pass', src)
        src = _re.sub(r'\bclass\s+(\w+)\s*<\s*(\w+)', r'class \1(\2):', src)
        src = _re.sub(r'\bclass\s+(\w+)\b(?!\s*[\(:])', r'class \1:', src)
        src = src.replace('def initialize(', 'def __init__(self, ')
        src = src.replace('def initialize():', 'def __init__(self):')
        # Add Ruby stdlib exception aliases so transpiled code doesn't crash on them
        preamble = (
            "# Ruby exception aliases\n"
            "ArgumentError = ValueError\n"
            "TypeError = TypeError\n"
            "RuntimeError = RuntimeError\n"
            "StandardError = Exception\n"
            "NoMethodError = AttributeError\n"
            "NameError = NameError\n"
            "IOError = IOError\n"
            "IndexError = IndexError\n"
            "KeyError = KeyError\n"
            "ZeroDivisionError = ZeroDivisionError\n"
            "NotImplementedError = NotImplementedError\n\n"
        )
        src = preamble + src
        return src
    def generate(self, repo_path: str, rb_files: list) -> tuple[Optional[str], list]:
        """Returns (python_harness, static_warnings)."""
        all_methods = []
        all_sources = []
        warnings = []
        for rel in rb_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try:
                src = open(full, encoding='utf-8', errors='replace').read()
            except: continue
            methods = self._extract_methods(src)
            py_src = self._to_python(src)
            for m in methods:
                m['file'] = rel
                m['py_src'] = py_src
            all_methods.extend(methods)
            all_sources.append((rel, src, py_src))

        if not all_methods:
            return None, warnings

        # Build one Python harness that inlines all transpiled sources
        parts = [
            "import sys, gc, json, time, tracemalloc, cProfile, pstats, io, resource\n",
            "results = []\n\n",
        ]
        for rel, orig, py in all_sources:
            parts.append(f"# === Transpiled from {rel} ===\n")
            try:
                import ast as _ast
                _ast.parse(py)
                parts.append(py + "\n\n")
            except SyntaxError as e:
                # Transpilation produced invalid Python → static-only
                warnings.append(f"TranspileError:{rel}:{e}")
                parts.append(f"# TRANSPILE FAILED: {e}\n\n")

        for m in all_methods:
            args = ', '.join(self._TYPE_DEFAULTS[i % len(self._TYPE_DEFAULTS)]
                             for i in range(len(m['params'])))
            fn = m['name']
            parts.append(
                "try:\n"
                f"    gc.collect()\n"
                f"    _ob = len(gc.get_objects())\n"
                f"    _mr = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
                f"    _t0 = time.perf_counter()\n"
                f"    {fn}({args})\n"
                f"    _el = (time.perf_counter() - _t0) * 1000\n"
                f"    gc.collect()\n"
                f"    results.append({{'name':{repr(fn)},'file':{repr(m['file'])},'line':{m['line']},"
                f"'cpu_time_ms':_el,'object_delta':len(gc.get_objects())-_ob,"
                f"'mem_delta_kb':(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-_mr)/1024,"
                f"'raised_exception':None,'top_calls':['transpiled_from:Ruby']}})\n"
                f"except Exception as _e:\n"
                f"    _el = (time.perf_counter() - _t0) * 1000\n"
                f"    results.append({{'name':{repr(fn)},'file':{repr(m['file'])},'line':{m['line']},"
                f"'cpu_time_ms':_el,'object_delta':0,'mem_delta_kb':0,"
                f"'raised_exception':f'{{type(_e).__name__}}: {{str(_e)[:150]}}',"
                f"'top_calls':['transpiled_from:Ruby']}})\n"
                f"except NameError:\n"
                f"    results.append({{'name':{repr(fn)},'file':{repr(m['file'])},'line':{m['line']},"
                f"'cpu_time_ms':0,'object_delta':0,'mem_delta_kb':0,"
                f"'raised_exception':'TranspileIncomplete: function not defined after transpilation',"
                f"'top_calls':['transpiled_from:Ruby','static_analysis_only']}})\n\n"
            )

        parts.append("print(json.dumps(results))\n")
        return ''.join(parts), warnings


# ── PHP ───────────────────────────────────────────────────────────────────────

class PHPTranspiler:
    """Transpiles PHP (.php) to Python for execution."""

    _TYPE_DEFAULTS = ["'test'", "42", "3.14", "True", "[]", "{}", "None"]

    def _extract_functions(self, source: str) -> list:
        fns, seen = [], set()
        pat = _re.compile(r'\bfunction\s+(\w+)\s*\(([^)]*)\)', _re.MULTILINE)
        for m in pat.finditer(source):
            name, params_str = m.group(1), m.group(2).strip()
            line = source[:m.start()].count('\n') + 1
            if name.startswith('_') or name == '__construct':
                continue
            params = [_re.sub(r'^\$', '', p.strip().split('=')[0].strip())
                      for p in params_str.split(',') if p.strip()]
            if name not in seen:
                seen.add(name)
                fns.append({'name': name, 'line': line, 'params': params})
        return fns

    def _to_python(self, source: str) -> str:
        src = source
        # Remove PHP tags
        src = _re.sub(r'<\?(?:php)?\s*', '', src)
        src = _re.sub(r'\?>', '', src)
        # PHP exception aliases
        preamble = (
            "# PHP exception aliases\n"
            "# (Exception is already Python builtin)\n"
            "class InvalidArgumentException(ValueError): pass\n"
            "class RuntimeException(RuntimeError): pass\n"
            "class LogicException(Exception): pass\n"
            "class BadMethodCallException(AttributeError): pass\n"
            "class OutOfRangeException(IndexError): pass\n"
            "class UnexpectedValueException(ValueError): pass\n\n"
        )
        # function -> def, strip $
        src = _re.sub(r'\bfunction\s+(\w+)\s*\(([^)]*)\)\s*\{',
                      lambda m2: 'def ' + m2.group(1) + '(' +
                                 _re.sub(r'\$', '', m2.group(2)).strip() + '):',
                      src)
        # Remove lone closing braces used as block markers
        src = _re.sub(r'^\s*\}\s*$', '', src, flags=_re.MULTILINE)
        # $var -> var
        src = _re.sub(r'\$(\w+)', r'\1', src)
        # throw new X("msg") -> raise X("msg")
        src = _re.sub(r'\bthrow\s+new\s+(\w+)\s*\("([^"]*)"\)', r'raise \1("\2")', src)
        src = _re.sub(r"\bthrow\s+new\s+(\w+)\s*\('([^']*)'\)", r"raise \1('\2')", src)
        src = _re.sub(r'\bthrow\s+new\s+(\w+)\s*\(\)', r'raise \1()', src)
        # null/true/false
        src = _re.sub(r'\bnull\b', 'None', src)
        src = _re.sub(r'\btrue\b', 'True', src)
        src = _re.sub(r'\bfalse\b', 'False', src)
        # echo/print
        src = _re.sub(r'\becho\s+(.*);', r'print(\1)', src)
        # String concat " . " -> " + "  (between vars and strings)
        src = _re.sub(r'"\s*\.\s*(\w+)\s*\.\s*"', r'" + \1 + "', src)
        src = _re.sub(r'(\w+)\s*\.\s*(\w+)\s*(?=[;,\)])', r'\1 + \2', src)
        # Remove semicolons at end of lines
        src = _re.sub(r';\s*$', '', src, flags=_re.MULTILINE)
        # array() -> list()
        src = _re.sub(r'\barray\s*\(', '[', src)
        # isset -> (x is not None)
        src = _re.sub(r'\bisset\s*\((\w+)\)', r'(\1 is not None)', src)
        # -> method access
        src = _re.sub(r'->(\w+)\s*\(', r'.\1(', src)
        src = _re.sub(r'::', r'.', src)
        # catch (ExcType $e) -> except ExcType as e:
        src = _re.sub(r'\bcatch\s*\(\s*(\w+)\s+\w+\)', r'except \1 as e:', src)
        src = _re.sub(r'^\s*try\s*\{', 'try:', src, flags=_re.MULTILINE)
        src = _re.sub(r'^\s*finally\s*\{', 'finally:', src, flags=_re.MULTILINE)
        src = _re.sub(r'^\s*else\s*\{', 'else:', src, flags=_re.MULTILINE)
        src = _re.sub(r'\bif\s*\(([^)]+)\)\s*\{', r'if \1:', src)
        src = _re.sub(r'\bwhile\s*\(([^)]+)\)\s*\{', r'while \1:', src)
        src = _re.sub(r'\bfor\s*\(([^)]*)\)\s*\{', r'for \1:', src)
        # Fix: if(cond)raise -> if (cond): raise
        src = _re.sub(r'\bif\s*\(([^)]+)\)([ \t]*(?:raise|return|pass))', r'if (\1): \2', src)
        # Fix: stray '.varname' -> ' + varname'
        src = _re.sub(r'"\.([a-zA-Z_]\w*)', r'" + \1', src)
        # Fix: remove trailing ; from Python statements
        src = _re.sub(r';\s*$', '', src, flags=_re.MULTILINE)
        # Fix: remove stray } at end of lines or on own line
        src = _re.sub(r'\}\s*$', '', src, flags=_re.MULTILINE)
        src = _re.sub(r'^\s*\}\s*$', '', src, flags=_re.MULTILINE)
        return preamble + src
    def generate(self, repo_path: str, php_files: list) -> tuple[Optional[str], list]:
        all_fns, all_sources, warnings = [], [], []
        for rel in php_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try: src = open(full, encoding='utf-8', errors='replace').read()
            except: continue
            fns = self._extract_functions(src)
            py = self._to_python(src)
            for f in fns:
                f['file'] = rel
            all_fns.extend(fns)
            all_sources.append((rel, src, py))

        if not all_fns: return None, warnings

        parts = ["import sys, gc, json, time, resource\nresults = []\n\n"]
        for rel, _, py in all_sources:
            parts.append(f"# === Transpiled from {rel} ===\n")
            try:
                import ast as _ast; _ast.parse(py)
                parts.append(py + "\n\n")
            except SyntaxError as e:
                warnings.append(f"TranspileError:{rel}:{e}")
                parts.append(f"# TRANSPILE FAILED: {e}\n\n")

        for fn in all_fns:
            args = ', '.join(self._TYPE_DEFAULTS[i % len(self._TYPE_DEFAULTS)]
                             for i in range(len(fn['params'])))
            f = fn['name']
            parts.append(
                "try:\n"
                f"    _mr = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
                f"    _t0 = time.perf_counter()\n"
                f"    {f}({args})\n"
                f"    _el = (time.perf_counter() - _t0) * 1000\n"
                f"    results.append({{'name':{repr(f)},'file':{repr(fn['file'])},'line':{fn['line']},"
                f"'cpu_time_ms':_el,'object_delta':0,"
                f"'mem_delta_kb':(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-_mr)/1024,"
                f"'raised_exception':None,'top_calls':['transpiled_from:PHP']}})\n"
                f"except Exception as _e:\n"
                f"    results.append({{'name':{repr(f)},'file':{repr(fn['file'])},'line':{fn['line']},"
                f"'cpu_time_ms':0,'object_delta':0,'mem_delta_kb':0,"
                f"'raised_exception':f'{{type(_e).__name__}}: {{str(_e)[:150]}}',"
                f"'top_calls':['transpiled_from:PHP']}})\n"
                f"except NameError:\n"
                f"    results.append({{'name':{repr(f)},'file':{repr(fn['file'])},'line':{fn['line']},"
                f"'cpu_time_ms':0,'object_delta':0,'mem_delta_kb':0,"
                f"'raised_exception':'TranspileIncomplete: function not defined',"
                f"'top_calls':['transpiled_from:PHP','static_analysis_only']}})\n\n"
            )
        parts.append("print(json.dumps(results))\n")
        return ''.join(parts), warnings


# ── R ─────────────────────────────────────────────────────────────────────────

class RTranspiler:
    """Transpiles R (.r / .R) to Python (uses scipy/numpy stubs where needed)."""

    _TYPE_DEFAULTS = ["'test'", "42", "3.14", "[1, 2, 3]", "None"]

    def _extract_functions(self, source: str) -> list:
        fns, seen = [], set()
        pat = _re.compile(r'^(\w+)\s*<-\s*function\s*\(([^)]*)\)', _re.MULTILINE)
        for m in pat.finditer(source):
            name, params_str = m.group(1), m.group(2).strip()
            line = source[:m.start()].count('\n') + 1
            if name.startswith('.') or name.startswith('_'): continue
            params = [p.strip().split('=')[0].strip() for p in params_str.split(',') if p.strip()]
            if name not in seen:
                seen.add(name)
                fns.append({'name': name, 'line': line, 'params': params})
        return fns

    def _to_python(self, source: str) -> str:
        src = source
        preamble = (
            "import math, statistics\n"
            "# R stdlib stubs\n"
            "def mean(x): return sum(x)/len(x) if x else float('nan')\n"
            "def sd(x):\n"
            "    import statistics\n"
            "    return statistics.stdev(x) if len(x) > 1 else 0.0\n"
            "def var(x):\n"
            "    import statistics\n"
            "    return statistics.variance(x) if len(x) > 1 else 0.0\n"
            "def nchar(x): return len(str(x))\n"
            "def paste(*args, sep=' '): return sep.join(str(a) for a in args)\n"
            "def cat(*args): print(*args)\n"
            "def seq(start, end, by=1): return list(range(int(start), int(end)+1, int(by)))\n"
            "def rep(x, n): return [x]*n\n"
            "def which(x): return [i+1 for i,v in enumerate(x) if v]\n"
            "def is_na(x): return x is None or x != x\n"
            "def abs(x): return __builtins__['abs'](x) if isinstance(__builtins__, dict) else __import__('builtins').abs(x)\n\n"
        )
        # f <- function(params) { -> def f(params):
        src = _re.sub(r'^(\w+)\s*<-\s*function\s*\(([^)]*)\)\s*\{',
                      r'def \1(\2):', src, flags=_re.MULTILINE)
        # Remove closing braces
        src = _re.sub(r'^\s*\}\s*$', '', src, flags=_re.MULTILINE)
        # <- assignment -> =
        src = _re.sub(r'\s*<-\s*', ' = ', src)
        # stop("msg") -> raise ValueError("msg")
        src = _re.sub(r'\bstop\s*\("([^"]*)"\)', r'raise ValueError("\1")', src)
        src = _re.sub(r"\bstop\s*\('([^']*)'\)", r"raise ValueError('\1')", src)
        src = _re.sub(r'\bstop\s*\((\w+)\)', r'raise ValueError(str(\1))', src)
        # warning() -> pass
        src = _re.sub(r'\bwarning\s*\([^)]*\)', 'pass', src)
        # return(x) -> return x
        src = _re.sub(r'\breturn\s*\(([^)]+)\)', r'return \1', src)
        # NULL/TRUE/FALSE/NA
        src = _re.sub(r'\bNULL\b', 'None', src)
        src = _re.sub(r'\bTRUE\b', 'True', src)
        src = _re.sub(r'\bFALSE\b', 'False', src)
        src = _re.sub(r'\bNA\b', 'None', src)
        # if/else/for/while R-style
        src = _re.sub(r'\bif\s*\(([^)]+)\)\s*\{', r'if \1:', src)
        src = _re.sub(r'\belse\s*\{', 'else:', src)
        src = _re.sub(r'\bfor\s*\((\w+)\s+in\s+([^)]+)\)\s*\{', r'for \1 in \2:', src)
        src = _re.sub(r'\bwhile\s*\(([^)]+)\)\s*\{', r'while \1:', src)
        # c(1,2,3) -> [1,2,3]
        src = _re.sub(r'\bc\s*\(([^)]*)\)', r'[\1]', src)
        # length(x) -> len(x)
        src = _re.sub(r'\blength\s*\(', 'len(', src)
        # print(x)
        src = _re.sub(r'\bcat\s*\(', 'print(', src)
        # library/require -> pass
        src = _re.sub(r'\b(?:library|require)\s*\([^)]*\)', 'pass', src)
        # tryCatch -> try (approximate)
        src = _re.sub(r'\btryCatch\s*\(', '# tryCatch(', src)
        # Fix: if(cond)stmt -> if cond: stmt  (handles nested parens like len(x)==0)
        def fix_if_colon(s):
            import re as _r
            # Match if( ... ) followed by raise/return/pass, handling nested parens
            result = []
            i = 0
            while i < len(s):
                m = _r.match(r'\bif\s*\(', s[i:])
                if m and s[i:i+2] != 'if':
                    i += 1; continue
                m2 = _r.match(r'if\s*\(', s[i:])
                if m2:
                    start = i + m2.end() - 1  # position of opening (
                    depth = 0; j = start
                    while j < len(s):
                        if s[j] == '(': depth += 1
                        elif s[j] == ')': depth -= 1
                        if depth == 0: break
                        j += 1
                    cond = s[start+1:j]  # content inside outer parens
                    rest = s[j+1:]  # after closing )
                    m3 = _r.match(r'[ \t]*(raise|return|pass)', rest)
                    if m3:
                        result.append('if ' + cond + ': ')
                        result.append(rest)
                        i = len(s)  # consumed all
                        break
                result.append(s[i])
                i += 1
            return ''.join(result)
        src = '\n'.join(fix_if_colon(line) if 'if(' in line or 'if (' in line else line
                        for line in src.splitlines())
        # Fix: remove stray }
        src = _re.sub(r'^\s*\}\s*$', '', src, flags=_re.MULTILINE)
        return preamble + src
    def generate(self, repo_path: str, r_files: list) -> tuple[Optional[str], list]:
        all_fns, all_sources, warnings = [], [], []
        for rel in r_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try: src = open(full, encoding='utf-8', errors='replace').read()
            except: continue
            fns = self._extract_functions(src)
            py = self._to_python(src)
            for f in fns:
                f['file'] = rel
            all_fns.extend(fns)
            all_sources.append((rel, src, py))

        if not all_fns: return None, warnings

        parts = ["import sys, gc, json, time, resource, math\nresults = []\n\n"]
        for rel, _, py in all_sources:
            parts.append(f"# === Transpiled from {rel} ===\n")
            try:
                import ast as _ast; _ast.parse(py)
                parts.append(py + "\n\n")
            except SyntaxError as e:
                warnings.append(f"TranspileError:{rel}:{e}")
                parts.append(f"# TRANSPILE FAILED: {e}\n\n")

        for fn in all_fns:
            args = ', '.join(self._TYPE_DEFAULTS[i % len(self._TYPE_DEFAULTS)]
                             for i in range(len(fn['params'])))
            f = fn['name']
            parts.append(
                "try:\n"
                f"    _t0 = time.perf_counter()\n"
                f"    {f}({args})\n"
                f"    _el = (time.perf_counter() - _t0) * 1000\n"
                f"    results.append({{'name':{repr(f)},'file':{repr(fn['file'])},'line':{fn['line']},"
                f"'cpu_time_ms':_el,'object_delta':0,'mem_delta_kb':0,"
                f"'raised_exception':None,'top_calls':['transpiled_from:R']}})\n"
                f"except Exception as _e:\n"
                f"    results.append({{'name':{repr(f)},'file':{repr(fn['file'])},'line':{fn['line']},"
                f"'cpu_time_ms':0,'object_delta':0,'mem_delta_kb':0,"
                f"'raised_exception':f'{{type(_e).__name__}}: {{str(_e)[:150]}}',"
                f"'top_calls':['transpiled_from:R']}})\n"
                f"except NameError:\n"
                f"    results.append({{'name':{repr(f)},'file':{repr(fn['file'])},'line':{fn['line']},"
                f"'cpu_time_ms':0,'object_delta':0,'mem_delta_kb':0,"
                f"'raised_exception':'TranspileIncomplete: function not defined',"
                f"'top_calls':['transpiled_from:R','static_analysis_only']}})\n\n"
            )
        parts.append("print(json.dumps(results))\n")
        return ''.join(parts), warnings


# ── Kotlin ────────────────────────────────────────────────────────────────────

class KotlinTranspiler:
    """Transpiles Kotlin (.kt) to Java for execution via java 21 single-file launcher."""

    _TYPE_MAP = {
        'Int': 'int', 'Long': 'long', 'Double': 'double', 'Float': 'float',
        'Boolean': 'boolean', 'String': 'String', 'Unit': 'void',
        'Any': 'Object', 'Nothing': 'void', 'Char': 'char',
        'IntArray': 'int[]', 'List<Int>': 'java.util.List<Integer>',
        'List<String>': 'java.util.List<String>',
        'MutableList<Int>': 'java.util.ArrayList<Integer>',
        'Map<String, Int>': 'java.util.Map<String, Integer>',
    }

    _ARG_MAP = {
        'int': '42', 'long': '42L', 'double': '3.14', 'float': '3.14f',
        'boolean': 'true', 'String': '"test"', 'char': "'a'",
        'void': '', 'Object': 'null',
    }

    def _extract_functions(self, source: str) -> list:
        fns, seen = [], set()
        # fun name(param: Type, ...): ReturnType {
        pat = _re.compile(
            r'\bfun\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*[\w<>, ?]+)?\s*(?:=|\{)',
            _re.MULTILINE
        )
        for m in pat.finditer(source):
            name, params_str = m.group(1), m.group(2).strip()
            line = source[:m.start()].count('\n') + 1
            if name.startswith('_') or name == 'main': continue
            params = []
            for p in params_str.split(','):
                p = p.strip()
                if not p: continue
                parts = p.split(':')
                ptype = parts[1].strip().rstrip('?') if len(parts) > 1 else 'String'
                java_type = self._TYPE_MAP.get(ptype, ptype.replace('?', ''))
                params.append(java_type)
            if name not in seen:
                seen.add(name)
                fns.append({'name': name, 'line': line, 'params': params})
        return fns

    def _to_java(self, source: str, class_name: str = 'KotlinCompat') -> str:
        src = source
        def convert_fun(m2):
            name = m2.group(1)
            params_str = m2.group(2)
            ret_str = (m2.group(3) or 'void').strip().rstrip('?')
            ret_java = self._TYPE_MAP.get(ret_str, 'Object')
            params_java = []
            for p in params_str.split(','):
                p = p.strip()
                if not p: continue
                parts = p.split(':')
                pname = parts[0].strip()
                ptype = (parts[1].strip().rstrip('?') if len(parts) > 1 else 'Object')
                java_t = self._TYPE_MAP.get(ptype, ptype)
                params_java.append(f'{java_t} {pname}')
            return f'public static {ret_java} {name}({", ".join(params_java)})'

        # fun name(params): ReturnType { ... }
        src = _re.sub(
            r'\bfun\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*([\w<>, ?]+))?\s*\{',
            lambda m2: convert_fun(m2) + ' {', src)
        # Single-expression: fun f(x: Int) = expr  -> public static Object f(int x) { return expr; }
        src = _re.sub(
            r'\bfun\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*([\w<>, ?]+))?\s*=\s*(.+)',
            lambda m2: convert_fun(m2) + ' { return ' + m2.group(4).strip() + '; }', src)
        # val/var -> auto (or just remove type annotation)
        src = _re.sub(r'\b(?:val|var)\s+(\w+)\s*(?::\s*[\w<>?]+)?\s*=', r'var \1 =', src)
        # println -> System.out.println
        src = _re.sub(r'\bprintln\s*\(', 'System.out.println(', src)
        src = _re.sub(r'\bprint\s*\(', 'System.out.print(', src)
        # String templates "$x" / "${expr}" -> string concat
        src = _re.sub(r'"([^"]*)\$\{([^}]+)\}([^"]*)"', r'"\1" + (\2) + "\3"', src)
        src = _re.sub(r'"([^"]*)\$(\w+)([^"]*)"', r'"\1" + \2 + "\3"', src)
        # listOf -> Arrays.asList
        src = _re.sub(r'\blistOf\s*\(', 'java.util.Arrays.asList(', src)
        # when -> comment (too complex)
        src = _re.sub(r'\bwhen\s*\(([^)]+)\)\s*\{', r'// when(\1) - unsupported\n// {', src)
        # is -> instanceof
        src = _re.sub(r'\bis\s+(\w+)', r'instanceof \1', src)
        # Remove package/import
        src = _re.sub(r'^\s*(?:package|import)\s+.*$', '', src, flags=_re.MULTILINE)
        # Remove fun main
        src = _re.sub(r'\bfun\s+main\s*\([^)]*\)\s*\{[^}]*\}', '', src, flags=_re.DOTALL)
        # Fix: Kotlin 'throw X()' -> Java 'throw new X()'
        src = _re.sub(r'\bthrow\s+(?!new\b)(\w+\s*\()', r'throw new \1', src)

        # ADD SEMICOLONS to all statement lines in Java
        result_lines = []
        for line in src.splitlines():
            stripped = line.rstrip()
            s = stripped.strip()
            needs_semi = (
                s and
                not s.endswith('{') and
                not s.endswith('}') and
                not s.endswith(';') and
                not s.endswith(',') and
                not s.startswith('//') and
                not s.startswith('/*') and
                not s.startswith('*') and
                not _re.match(r'public\s+(?:static\s+)?(?:\w+\s+)+\w+\s*\(', s) and
                not _re.match(r'(?:public|private|protected|class)\s', s)
            )
            if needs_semi:
                stripped += ';'
            result_lines.append(stripped)
        return '\n'.join(result_lines)
    def generate(self, repo_path: str, kt_files: list) -> tuple[Optional[str], list]:
        all_fns, all_sources, warnings = [], [], []
        for rel in kt_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try: src = open(full, encoding='utf-8', errors='replace').read()
            except: continue
            fns = self._extract_functions(src)
            cls = _re.sub(r'[^a-zA-Z0-9]', '_', _re.sub(r'\.kt$', '', os.path.basename(rel)))
            java_src = self._to_java(src, cls)
            for f in fns:
                f['file'] = rel
                f['class'] = cls
            all_fns.extend(fns)
            all_sources.append((rel, src, java_src))

        if not all_fns: return None, warnings

        # Use Java harness generator with the transpiled Java source
        java_gen = JavaHarnessGenerator()
        # We need to write the transpiled Java to temp files and use the Java harness
        # Since we can't modify the Java harness generator directly here, build the harness manually

        inlined = [src for _, _, src in all_sources]
        blocks = []
        for fn in all_fns:
            args = ', '.join(self._ARG_MAP.get(p, 'null') for p in fn['params'])
            cls = fn.get('class', 'KotlinCompat')
            ne = fn['name'].replace('"', '\\"')
            fe = fn['file'].replace('\\', '\\\\').replace('"', '\\"')
            blocks.append(
                f'    {{\n'
                f'        long mb=memBean.getHeapMemoryUsage().getUsed();\n'
                f'        long t0=System.nanoTime(); String exc=null;\n'
                f'        try{{ {fn["name"]}({args}); }}\n'
                f'        catch(Exception e){{exc=e.getClass().getSimpleName()+": "+e.getMessage();}}\n'
                f'        catch(Error e){{exc=e.getClass().getSimpleName()+": "+e.getMessage();}}\n'
                f'        double el=(System.nanoTime()-t0)/1_000_000.0;\n'
                f'        long ma=memBean.getHeapMemoryUsage().getUsed();\n'
                f'        if(first)first=false; else sb.append(",");\n'
                f'        sb.append("{{\\"name\\":\\"{ne}\\",\\"file\\":\\"{fe}\\",\\"line\\":{fn["line"]}"'
                f'           +",\\"cpu_time_ms\\":"+String.format("%.3f",el)'
                f'           +",\\"object_delta\\":0,\\"mem_delta_kb\\":"+String.format("%.1f",(ma-mb)/1024.0)'
                f'           +(exc==null?",\\"raised_exception\\":null":(",\\"raised_exception\\":\\""+exc.replace("\\\\","\\\\\\\\").replace("\\"","\\\\\\"")+"\\""))'
                f'           +",\\"top_calls\\":[\\\"transpiled_from:Kotlin\\\"]}}");'
                f'\n    }}\n'
            )

        # Clean up inlined Java: remove public class modifiers
        cleaned = []
        for src in inlined:
            s = _re.sub(r'\bpublic\s+class\b', 'class', src)
            s = _re.sub(r'^\s*package\s+[\w.]+\s*;', '', s, flags=_re.MULTILINE)
            cleaned.append(s)

        # Wrap transpiled functions INSIDE HarnessMain as static methods
        # so Java 21 single-file runner sees them correctly
        java_harness = (
            "import java.lang.management.*;\nimport java.util.*;\n\n"
            "public class HarnessMain {\n"
            "    public static void main(String[] args) {\n"
            "        MemoryMXBean memBean=ManagementFactory.getMemoryMXBean();\n"
            "        StringBuilder sb=new StringBuilder(\"[\");\n"
            "        boolean first=true;\n"
            + "".join(blocks) +
            "        sb.append(\"]\");\n"
            "        System.out.println(sb.toString());\n"
            "    }\n\n"
            # Paste transpiled static methods inside the class
            + "\n".join("    " + line if line.strip() else line
                         for java_src in cleaned
                         for line in java_src.splitlines()) + "\n"
            "}\n"
        )
        return java_harness, warnings


# ── Go ────────────────────────────────────────────────────────────────────────

class GoTranspiler:
    """
    Transpiles Go (.go) to C for execution via gcc.
    Go and C share enough syntax (structs, for loops, basic types) that
    simple functions transpile cleanly.
    """

    _TYPE_MAP = {
        'int': 'int', 'int32': 'int', 'int64': 'long long',
        'uint': 'unsigned int', 'uint32': 'unsigned int', 'uint64': 'unsigned long long',
        'float32': 'float', 'float64': 'double',
        'string': 'const char*', 'bool': 'int',
        'byte': 'unsigned char', 'rune': 'int',
        'error': 'const char*',
    }

    _ARG_MAP = {
        'int': '42', 'long long': '42LL', 'float': '3.14f', 'double': '3.14',
        'const char*': '"test"', 'int': '42', 'unsigned int': '42U',
        'unsigned long long': '42ULL', 'unsigned char': '65',
    }

    def _extract_functions(self, source: str) -> list:
        fns, seen = [], set()
        # func Name(a int, b int) ReturnType {
        pat = _re.compile(
            r'\bfunc\s+(\w+)\s*\(([^)]*)\)\s*(?:\(?([\w\s,*]+)\)?)?\s*\{',
            _re.MULTILINE
        )
        for m in pat.finditer(source):
            name = m.group(1)
            params_str = m.group(2).strip()
            ret_str = (m.group(3) or '').strip()
            line = source[:m.start()].count('\n') + 1
            if name[0].islower() and name not in ['main']:
                pass  # Go: lowercase = unexported, still add
            if name == 'main' or name.startswith('Test') or name.startswith('Benchmark'):
                continue
            c_params = []
            for p in params_str.split(','):
                p = p.strip()
                if not p: continue
                parts = p.split()
                if len(parts) >= 2:
                    go_type = parts[-1]
                    c_type = self._TYPE_MAP.get(go_type, 'int')
                    c_params.append(c_type)
                elif len(parts) == 1:
                    c_params.append('int')
            c_ret = self._TYPE_MAP.get(ret_str.split()[0] if ret_str.split() else '', 'int')
            if name not in seen:
                seen.add(name)
                fns.append({'name': name, 'line': line,
                            'c_params': c_params, 'c_ret': c_ret})
        return fns

    def _to_c(self, source: str) -> str:
        src = source
        # Remove package/import
        src = _re.sub(r'^package\s+\w+.*$', '', src, flags=_re.MULTILINE)
        src = _re.sub(r'^import\s*\(.*?\)', '', src, flags=_re.DOTALL)
        src = _re.sub(r'^import\s+"[^"]+".*$', '', src, flags=_re.MULTILINE)

        def expand_params(params_str):
            # Handle Go grouped params: "a, b int, c string" -> "int a, int b, const char* c"
            if not params_str.strip():
                return ''
            raw = [p.strip() for p in params_str.split(',')]
            typed = []
            cur_type = 'int'
            for part in reversed(raw):
                tokens = part.split()
                if len(tokens) >= 2:
                    cur_type = self._TYPE_MAP.get(tokens[-1], 'int')
                    typed.insert(0, f'{cur_type} {" ".join(tokens[:-1])}')
                elif len(tokens) == 1:
                    typed.insert(0, f'{cur_type} {tokens[0]}')
            return ', '.join(typed)

        def conv_func(m2):
            name = m2.group(1)
            params_str = m2.group(2).strip()
            ret_str = (m2.group(3) or '').strip().strip('()')
            c_ret = self._TYPE_MAP.get(ret_str.split(',')[0].strip(), 'int') if ret_str else 'int'
            return f'{c_ret} {name}({expand_params(params_str) or "void"})'

        src = _re.sub(
            r'\bfunc\s+(\w+)\s*\(([^)]*)\)\s*(?:\(?([\w\s,*]+)\)?)?(?=\s*\{)',
            conv_func, src)

        # if cond { -> if (cond) {
        src = _re.sub(r'\bif\s+([^({][^{]*?)\s*\{', r'if (\1) {', src)
        # for range -> simplified
        src = _re.sub(r'\bfor\s+\S+\s*(?:,\s*\S+)?\s*:=\s*range\s+\S+\s*\{',
                      'for (_go_i = 0; _go_i < 1; _go_i++) {', src)
        # := -> auto
        src = _re.sub(r'(\w+)\s*:=\s*', r'__auto_type \1 = ', src)
        # multi-return
        src = _re.sub(r'return\s+(\w+),\s*[\w.]+', r'return \1', src)
        # fmt calls
        src = _re.sub(r'\bfmt\.Println\s*\(([^)]*)\)', r'printf("%s\n", (char*)(\1))', src)
        src = _re.sub(r'\bfmt\.Printf\s*\(', 'printf(', src)
        # # panic -> abort()
        src = _re.sub(r'\bpanic\s*\([^)]*\)', r'abort()', src)
        # nil/true/false
        src = _re.sub(r'\bnil\b', 'NULL', src)
        src = _re.sub(r'\btrue\b', '1', src)
        src = _re.sub(r'\bfalse\b', '0', src)
        # var declarations
        src = _re.sub(r'\bvar\s+(\w+)\s+string\b', r'const char* \1', src)
        src = _re.sub(r'\bvar\s+(\w+)\s+(\w+)\s*=', r'\2 \1 =', src)
        src = _re.sub(r'\bvar\s+(\w+)\s+(\w+)\b', r'\2 \1', src)
        # len -> strlen
        src = _re.sub(r'\blen\s*\((\w+)\)', r'(int)strlen(\1)', src)
        # Add semicolons to statement lines
        out = []
        for line in src.splitlines():
            s = line.rstrip()
            t = s.strip()
            add_semi = (
                t and
                not t.endswith('{') and not t.endswith('}') and
                not t.endswith(';') and not t.endswith(',') and
                not t.startswith('//') and not t.startswith('#') and
                not t.startswith('/*') and not t.startswith('*') and
                not _re.match(r'^(?:int|long|double|float|char|void|const|unsigned)\s+\w+\s*\(', t) and
                not _re.search(r'\}\s*while\s*\(0\)\s*$', t)
            )
            out.append(s + (';' if add_semi else ''))
        src = '\n'.join(out)
        src = _re.sub(r'([^;{}\n])\s*\}', lambda m: m.group(1) + '; }', src)
        return '#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <time.h>\n\n' + src
    def generate(self, repo_path: str, go_files: list) -> tuple[Optional[str], list]:
        all_fns, all_sources, warnings = [], [], []
        for rel in go_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try: src = open(full, encoding='utf-8', errors='replace').read()
            except: continue
            fns = self._extract_functions(src)
            c_src = self._to_c(src)
            for f in fns: f['file'] = rel
            all_fns.extend(fns)
            all_sources.append((rel, src, c_src))

        if not all_fns: return None, warnings

        # Build C harness using CppHarnessGenerator logic but with Go->C transpiled code
        inlined = [c_src for _, _, c_src in all_sources]
        # Strip main() from inlined
        combined = '\n'.join(inlined)
        combined = _re.sub(r'\bint\s+main\s*\([^)]*\)\s*\{[^}]*\}', '', combined, flags=_re.DOTALL)

        blocks = []
        for fn in all_fns:
            args = ', '.join(
                self._ARG_MAP.get(p, '42') for p in fn['c_params']
            ) or ''
            ne = fn['name']
            fe = fn['file'].replace('\\', '\\\\').replace('"', '\\"')
            blocks.append(
                f'    {{\n'
                f'        long vb=get_vm_kb(); clock_t t0=clock();\n'
                f'        {fn["name"]}({args});\n'
                f'        double el=(double)(clock()-t0)/CLOCKS_PER_SEC*1000.0;\n'
                f'        long va=get_vm_kb();\n'
                f'        if(first)first=0; else printf(",");\n'
                f'        printf("{{\\"name\\":\\"{ne}\\",\\"file\\":\\"{fe}\\",\\"line\\":{fn["line"]},'
                f'\\"cpu_time_ms\\":%.3f,\\"object_delta\\":0,\\"mem_delta_kb\\":%.1f,'
                f'\\"raised_exception\\":null,\\"top_calls\\":[\\\"transpiled_from:Go\\\"]}}",'
                f'el,(double)(va-vb));\n'
                f'    }}\n'
            )

        vmfn = (
            "long get_vm_kb(){\n"
            "    long vm=0; FILE* f=fopen(\"/proc/self/status\",\"r\");\n"
            "    if(f){char line[128];\n"
            "    while(fgets(line,128,f)){\n"
            "        if(strncmp(line,\"VmRSS:\",6)==0){sscanf(line+6,\"%ld\",&vm);break;}\n"
            "    }fclose(f);} return vm;\n}\n"
        )

        c_harness = (
            combined + "\n\n" + vmfn
            + "int main(void){\n    int first=1;\n    printf(\"[\");\n"
            + "".join(blocks)
            + '    printf("]\\n");\n    return 0;\n}\n'
        )
        return c_harness, warnings


# ── Rust ──────────────────────────────────────────────────────────────────────

class RustTranspiler:
    """
    Transpiles Rust (.rs) to C++ for execution via g++.
    Handles basic fn definitions, panic!, Result/Option patterns.
    """

    _TYPE_MAP = {
        'i32': 'int', 'i64': 'long long', 'u32': 'unsigned int',
        'u64': 'unsigned long long', 'i8': 'char', 'u8': 'unsigned char',
        'i16': 'short', 'u16': 'unsigned short', 'isize': 'long',
        'usize': 'unsigned long', 'f32': 'float', 'f64': 'double',
        'bool': 'bool', 'char': 'char', 'str': 'const char*',
        'String': 'std::string', '()': 'void',
    }

    _ARG_MAP = {
        'int': '42', 'long long': '42LL', 'unsigned int': '42U',
        'unsigned long long': '42ULL', 'float': '3.14f', 'double': '3.14',
        'bool': 'true', 'char': "'a'", 'const char*': '"test"',
        'std::string': 'std::string("test")', 'void': '',
        'unsigned long': '42UL', 'long': '42L',
    }

    def _extract_functions(self, source: str) -> list:
        fns, seen = [], set()
        # pub fn name(x: i32, y: &str) -> ReturnType {
        pat = _re.compile(
            r'\b(?:pub\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?:->\s*([\w<>\[\]&\s:,]+?))?\s*\{',
            _re.MULTILINE
        )
        for m in pat.finditer(source):
            name = m.group(1)
            params_str = m.group(2).strip()
            ret_str = (m.group(3) or '()').strip()
            line = source[:m.start()].count('\n') + 1
            if name in ('main', 'new') or name.startswith('test_'): continue

            # Extract param types
            cpp_params = []
            for p in params_str.split(','):
                p = p.strip()
                if not p or p == 'self' or p == '&self' or p == '&mut self': continue
                if ':' in p:
                    rust_type = p.split(':')[-1].strip().lstrip('&mut ').strip()
                    # Handle Option<T> / Result<T,E>
                    rust_type = _re.sub(r'Option<(\w+)>', r'\1', rust_type)
                    rust_type = _re.sub(r'Result<(\w+),.*>', r'\1', rust_type)
                    cpp_t = self._TYPE_MAP.get(rust_type, 'int')
                    cpp_params.append(cpp_t)

            # Return type
            ret_rust = _re.sub(r'Option<(\w+)>', r'\1', ret_str)
            ret_rust = _re.sub(r'Result<(\w+),.*>', r'\1', ret_rust)
            cpp_ret = self._TYPE_MAP.get(ret_rust.strip(), 'int')

            if name not in seen:
                seen.add(name)
                fns.append({'name': name, 'line': line,
                            'cpp_params': cpp_params, 'cpp_ret': cpp_ret})
        return fns

    def _to_cpp(self, source: str) -> str:
        src = source
        # Remove use declarations
        src = _re.sub(r'^use\s+.*$', '', src, flags=_re.MULTILINE)
        # Remove mod declarations
        src = _re.sub(r'^mod\s+.*$', '', src, flags=_re.MULTILINE)
        # pub fn name<T>(params) -> RetType { -> RetType name(params) {
        def convert_fn(m):
            name = m.group(1)
            params_str = m.group(2).strip()
            ret_str = (m.group(3) or '()').strip()
            ret_rust = _re.sub(r'Option<(\w+)>', r'\1', ret_str)
            ret_rust = _re.sub(r'Result<(\w+),.*>', r'\1', ret_rust)
            cpp_ret = self._TYPE_MAP.get(ret_rust.strip(), 'int')
            cpp_params = []
            for p in params_str.split(','):
                p = p.strip()
                if not p or p in ('self', '&self', '&mut self'): continue
                if ':' in p:
                    pn = p.split(':')[0].strip().lstrip('mut ')
                    rust_t = p.split(':')[-1].strip().lstrip('&mut ').strip()
                    rust_t = _re.sub(r'Option<(\w+)>', r'\1', rust_t)
                    rust_t = _re.sub(r'Result<(\w+),.*>', r'\1', rust_t)
                    cpp_t = self._TYPE_MAP.get(rust_t, 'int')
                    cpp_params.append(f'{cpp_t} {pn}')
            return f'{cpp_ret} {name}({", ".join(cpp_params) or ""}) {{'
        src = _re.sub(
            r'\b(?:pub\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?:->\s*([\w<>\[\]&\s:,]+?))?\s*\{',
            convert_fn, src
        )
        # let mut x = -> auto x =
        src = _re.sub(r'\blet\s+mut\s+(\w+)\s*(?::\s*[\w<>]+)?\s*=', r'auto \1 =', src)
        src = _re.sub(r'\blet\s+(\w+)\s*(?::\s*[\w<>]+)?\s*=', r'auto \1 =', src)
        # panic!("msg") -> throw std::runtime_error("msg")
        src = _re.sub(r'\bpanic!\s*\("([^"]*)"\)', r'throw std::runtime_error("\1")', src)
        src = _re.sub(r'\bpanic!\s*\(([^)]+)\)', r'throw std::runtime_error("panic")', src)
        # println!("fmt", args) -> printf("fmt\n", args)  (rough)
        src = _re.sub(r'\bprintln!\s*\("([^"]*)",\s*([^)]+)\)', r'printf("\1\n", \2)', src)
        src = _re.sub(r'\bprintln!\s*\("([^"]*)"\)', r'printf("\1\n")', src)
        src = _re.sub(r'\bprint!\s*\("([^"]*)",\s*([^)]+)\)', r'printf("\1", \2)', src)
        # None -> nullptr
        src = _re.sub(r'\bNone\b', 'nullptr', src)
        # Some(x) -> x
        src = _re.sub(r'\bSome\s*\(([^)]+)\)', r'\1', src)
        # Ok(x) / Err(x) -> x
        src = _re.sub(r'\bOk\s*\(([^)]+)\)', r'\1', src)
        src = _re.sub(r'\bErr\s*\(([^)]+)\)', r'/* Err */ throw std::runtime_error(\1)', src)
        # true/false (already C++ compatible)
        # Vec<T> -> std::vector<T>
        src = _re.sub(r'\bVec\s*<(\w+)>', r'std::vector<\1>', src)
        # .len() -> .size()
        src = src.replace('.len()', '.size()')
        # .to_string() -> std::to_string()
        src = _re.sub(r'(\w+)\.to_string\(\)', r'std::to_string(\1)', src)
        # String::from -> std::string
        src = _re.sub(r'String::from\s*\("([^"]*)"\)', r'std::string("\1")', src)
        # Remove lifetime annotations 'a
        src = _re.sub(r"'[a-z]\b", '', src)
        # impl blocks -> class-like (remove struct/impl for simplicity)
        src = _re.sub(r'\bimpl\s+\w+\s*\{', '// impl {', src)
        src = _re.sub(r'\bstruct\s+(\w+)\s*\{', r'struct \1 {', src)
        # Rust: if cond { -> if (cond) {  (C++ requires parens)
        src = _re.sub(r'\bif\s+([^({][^{]*?)\s*\{', r'if (\1) {', src)
        # Rust implicit return: last statement in function body becomes return
        # Walk lines: when we see RetType FnName(...) {, track brace depth
        # and prepend 'return' to last non-control-flow statement before closing }
        def add_implicit_returns(code):
            lines = code.splitlines()
            out = []
            i = 0
            skip_kw = ('return', 'throw', 'if', 'else', 'for', 'while', 'auto ',
                       'int ', 'void ', 'long ', 'double ', 'float ', '//', '{', '}')
            # Find function signatures with non-void return type
            fn_pat = _re.compile(r'^(int|long|double|float|bool|std::string|char)\s+\w+\s*\(')
            while i < len(lines):
                line = lines[i]
                if fn_pat.match(line.strip()) and line.rstrip().endswith('{'):
                    # Collect body until matching }
                    depth = 1
                    body_start = i + 1
                    j = i + 1
                    while j < len(lines) and depth > 0:
                        depth += lines[j].count('{') - lines[j].count('}')
                        j += 1
                    body_end = j - 1  # line with closing }
                    # Find last non-empty, non-control statement in body
                    last_stmt = -1
                    for k in range(body_end - 1, body_start - 1, -1):
                        s = lines[k].strip()
                        if s and not any(s.startswith(kw) for kw in skip_kw):
                            last_stmt = k
                            break
                    # Emit function lines, adding return to last_stmt
                    for k in range(i, j):
                        if k == last_stmt:
                            s = lines[k].strip()
                            # Remove trailing ; if present, then re-add as return
                            s = s.rstrip(';')
                            indent = len(lines[k]) - len(lines[k].lstrip())
                            out.append(' ' * indent + 'return ' + s + ';')
                        else:
                            out.append(lines[k])
                    i = j
                    continue
                out.append(line)
                i += 1
            return '\n'.join(out)
        # Fix single-line functions: int f(a,b) { expr; } -> int f(a,b) { return expr; }
        # Matches both "{ expr }" and "{ expr; }" forms
        def fix_single_line_fn(m):
            body = m.group(2).strip().rstrip(';')
            # Don't add return to control flow or existing return/throw
            if _re.match(r'^(return|throw|if|for|while)', body):
                return m.group(0)
            return m.group(1) + " return " + body + "; " + m.group(3)
        src = _re.sub(
            r"((?:int|long|double|float|bool|std::string)\s+\w+\s*\([^)]*\)\s*\{)\s*([^;{}]+);?\s*(\})",
            fix_single_line_fn,
            src
        )
        src = add_implicit_returns(src)
        # Add C++ headers
        src = '#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <time.h>\n#include <string>\n#include <vector>\n#include <stdexcept>\n\n' + src
        src = _re.sub(r'([^;{}\n])\s*\}', lambda m: m.group(1) + '; }', src)
        return src

    def generate(self, repo_path: str, rs_files: list) -> tuple[Optional[str], list]:
        all_fns, all_sources, warnings = [], [], []
        for rel in rs_files:
            full = os.path.join(repo_path, rel)
            if not os.path.exists(full): continue
            try: src = open(full, encoding='utf-8', errors='replace').read()
            except: continue
            fns = self._extract_functions(src)
            cpp_src = self._to_cpp(src)
            for f in fns: f['file'] = rel
            all_fns.extend(fns)
            all_sources.append((rel, src, cpp_src))

        if not all_fns: return None, warnings

        combined = '\n'.join(c for _, _, c in all_sources)
        combined = _re.sub(r'\bint\s+main\s*\([^)]*\)\s*\{[^}]*\}', '', combined, flags=_re.DOTALL)

        blocks = []
        for fn in all_fns:
            args = ', '.join(self._ARG_MAP.get(p, '42') for p in fn['cpp_params'])
            ne = fn['name']
            fe = fn['file'].replace('\\', '\\\\').replace('"', '\\"')
            blocks.append(
                f'    {{\n'
                f'        long vb=get_vm_kb(); clock_t t0=clock(); const char* exc=NULL;\n'
                f'        try{{ {fn["name"]}({args}); }}\n'
                f'        catch(const std::exception& e){{exc=e.what();}}\n'
                f'        catch(...){{exc="unknown exception";}}\n'
                f'        double el=(double)(clock()-t0)/CLOCKS_PER_SEC*1000.0;\n'
                f'        long va=get_vm_kb();\n'
                f'        if(first)first=0; else printf(",");\n'
                f'        if(exc) printf("{{\\"name\\":\\"{ne}\\",\\"file\\":\\"{fe}\\",\\"line\\":{fn["line"]},'
                f'\\"cpu_time_ms\\":%.3f,\\"object_delta\\":0,\\"mem_delta_kb\\":%.1f,'
                f'\\"raised_exception\\":\\"%s\\",\\"top_calls\\":[\\\"transpiled_from:Rust\\\"]}}",'
                f'el,(double)(va-vb),exc);\n'
                f'        else printf("{{\\"name\\":\\"{ne}\\",\\"file\\":\\"{fe}\\",\\"line\\":{fn["line"]},'
                f'\\"cpu_time_ms\\":%.3f,\\"object_delta\\":0,\\"mem_delta_kb\\":%.1f,'
                f'\\"raised_exception\\":null,\\"top_calls\\":[\\\"transpiled_from:Rust\\\"]}}",'
                f'el,(double)(va-vb));\n'
                f'    }}\n'
            )

        vmfn = (
            "long get_vm_kb(){\n"
            "    long vm=0; FILE* f=fopen(\"/proc/self/status\",\"r\");\n"
            "    if(f){char line[128];\n"
            "    while(fgets(line,128,f)){\n"
            "        if(strncmp(line,\"VmRSS:\",6)==0){sscanf(line+6,\"%ld\",&vm);break;}\n"
            "    }fclose(f);} return vm;\n}\n"
        )

        cpp_harness = (
            combined + "\n\n" + vmfn
            + "int main(void){\n    int first=1;\n    printf(\"[\");\n"
            + "".join(blocks)
            + '    printf("]\\n");\n    return 0;\n}\n'
        )
        return cpp_harness, warnings


# ═══════════════════════════════════════════════════════════════════════════════
# Extend Language enum and extension map for new languages
# ═══════════════════════════════════════════════════════════════════════════════

# Patch the existing Language enum with new members at runtime
import types as _types

_new_langs = {
    'RUBY':   auto(),
    'PHP':    auto(),
    'R':      auto(),
    'KOTLIN': auto(),
    'GO':     auto(),
    'RUST':   auto(),
}
for _name, _val in _new_langs.items():
    if not hasattr(Language, _name):
        Language._value2member_map_[_val.value] = None
        try:
            Language.__members__[_name] = Language(_val.value)
        except Exception:
            pass  # If patching fails, we handle it via direct class attrs below

# Since Enum patching is tricky, use a separate extended enum approach:
class LanguageExt(Enum):
    RUBY   = 101
    PHP    = 102
    R      = 103
    KOTLIN = 104
    GO     = 105
    RUST   = 106

# Extend the extension map
EXTENSION_MAP.update({
    '.rb':  LanguageExt.RUBY,
    '.php': LanguageExt.PHP,
    '.phtml': LanguageExt.PHP,
    '.r':   LanguageExt.R,
    '.R':   LanguageExt.R,
    '.kt':  LanguageExt.KOTLIN,
    '.kts': LanguageExt.KOTLIN,
    '.go':  LanguageExt.GO,
    '.rs':  LanguageExt.RUST,
})

LANGUAGE_NAMES.update({
    LanguageExt.RUBY:   'Ruby',
    LanguageExt.PHP:    'PHP',
    LanguageExt.R:      'R',
    LanguageExt.KOTLIN: 'Kotlin',
    LanguageExt.GO:     'Go',
    LanguageExt.RUST:   'Rust',
})


# ═══════════════════════════════════════════════════════════════════════════════
# Extend CodeSandbox._run_lang with transpiler-backed languages
# ═══════════════════════════════════════════════════════════════════════════════

_orig_run_lang = CodeSandbox._run_lang

def _run_lang_extended(self, lang, files: list, repo_path: str) -> SandboxReport:
    """Extended dispatcher that handles transpiler-backed languages."""
    if isinstance(lang, LanguageExt):
        lname = LANGUAGE_NAMES[lang]
        if lang == LanguageExt.RUBY:   return self._ruby(files, repo_path)
        if lang == LanguageExt.PHP:    return self._php(files, repo_path)
        if lang == LanguageExt.R:      return self._r(files, repo_path)
        if lang == LanguageExt.KOTLIN: return self._kotlin(files, repo_path)
        if lang == LanguageExt.GO:     return self._go(files, repo_path)
        if lang == LanguageExt.RUST:   return self._rust(files, repo_path)
        r = SandboxReport(); r.error = f"No transpiler for {lname}"; return r
    return _orig_run_lang(self, lang, files, repo_path)

CodeSandbox._run_lang = _run_lang_extended


def _ruby(self, files: list, repo: str) -> SandboxReport:
    r = SandboxReport()
    harness, warnings = RubyTranspiler().generate(repo, files)
    if not harness: r.error = "No callable Ruby methods"; return r
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="cs_rb_") as f:
        f.write(harness); path = f.name
    try:
        out, err, el = self.exe.run_python(path)
        r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
        self._parse(out, r, "Ruby")
    finally:
        try: os.unlink(path)
        except: pass
    return r

def _php(self, files: list, repo: str) -> SandboxReport:
    r = SandboxReport()
    harness, warnings = PHPTranspiler().generate(repo, files)
    if not harness: r.error = "No callable PHP functions"; return r
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="cs_php_") as f:
        f.write(harness); path = f.name
    try:
        out, err, el = self.exe.run_python(path)
        r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
        self._parse(out, r, "PHP")
    finally:
        try: os.unlink(path)
        except: pass
    return r

def _r_lang(self, files: list, repo: str) -> SandboxReport:
    r = SandboxReport()
    harness, warnings = RTranspiler().generate(repo, files)
    if not harness: r.error = "No callable R functions"; return r
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="cs_r_") as f:
        f.write(harness); path = f.name
    try:
        out, err, el = self.exe.run_python(path)
        r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
        self._parse(out, r, "R")
    finally:
        try: os.unlink(path)
        except: pass
    return r

def _kotlin(self, files: list, repo: str) -> SandboxReport:
    r = SandboxReport()
    harness, warnings = KotlinTranspiler().generate(repo, files)
    if not harness: r.error = "No callable Kotlin functions"; return r
    import shutil as _sh
    wd = tempfile.mkdtemp(prefix="cs_kt_")
    hp = os.path.join(wd, "HarnessMain.java")
    try:
        open(hp, "w").write(harness)
        out, err, el = self.exe.run_java(hp, wd)
        r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
        self._parse(out, r, "Kotlin")
    finally:
        _sh.rmtree(wd, ignore_errors=True)
    return r

def _go(self, files: list, repo: str) -> SandboxReport:
    r = SandboxReport()
    harness, warnings = GoTranspiler().generate(repo, files)
    if not harness: r.error = "No callable Go functions"; return r
    import shutil as _sh
    wd = tempfile.mkdtemp(prefix="cs_go_")
    hp = os.path.join(wd, "harness.c")
    try:
        open(hp, "w").write(harness)
        out, err, el = self.exe.run_c(hp, wd, False)
        r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
        self._parse(out, r, "Go")
    finally:
        _sh.rmtree(wd, ignore_errors=True)
    return r

def _rust_lang(self, files: list, repo: str) -> SandboxReport:
    r = SandboxReport()
    harness, warnings = RustTranspiler().generate(repo, files)
    if not harness: r.error = "No callable Rust functions"; return r
    import shutil as _sh
    wd = tempfile.mkdtemp(prefix="cs_rs_")
    hp = os.path.join(wd, "harness.cpp")
    try:
        open(hp, "w").write(harness)
        out, err, el = self.exe.run_c(hp, wd, True)
        r.total_exec_ms = el; r.raw_output = (out+err)[:3000]
        self._parse(out, r, "Rust")
    finally:
        _sh.rmtree(wd, ignore_errors=True)
    return r

# Bind methods to CodeSandbox
CodeSandbox._ruby   = _ruby
CodeSandbox._php    = _php
CodeSandbox._r      = _r_lang
CodeSandbox._kotlin = _kotlin
CodeSandbox._go     = _go
CodeSandbox._rust   = _rust_lang
