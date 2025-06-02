const fs = require('fs');
const path = require('path');
const { include_dir, symbols } = require('node-api-headers');


// Parse function signature from header files
function parseFunctionSignature(headerContent, functionName) {
  // First remove C-style comments /* ... */ and line comments // ...
  let cleanContent = headerContent
    .replace(/\/\*[\s\S]*?\*\//g, ' ')  // Remove /* ... */ comments
    .replace(/\/\/.*$/gm, ' ')           // Remove // ... line comments
    .replace(/\s+/g, ' ')               // Normalize whitespace
    .trim();

  // Regex patterns to match NAPI function declarations
  const patterns = [
    // Standard pattern: NAPI_EXTERN return_type NAPI_CDECL function_name(params);
    new RegExp(`NAPI_EXTERN\\s+(\\w+(?:\\s*\\**)?)\\s+NAPI_CDECL\\s+${functionName}\\s*\\(([^)]*)\\)`, 'g'),
    // Pattern without NAPI_CDECL: NAPI_EXTERN return_type function_name(params);
    new RegExp(`NAPI_EXTERN\\s+(\\w+(?:\\s*\\**)?)\\s+${functionName}\\s*\\(([^)]*)\\)`, 'g'),
    // Pattern for void functions
    new RegExp(`NAPI_EXTERN\\s+(void)\\s+NAPI_CDECL\\s+${functionName}\\s*\\(([^)]*)\\)`, 'g'),
    // Pattern with NAPI_NO_RETURN
    new RegExp(`NAPI_EXTERN\\s+NAPI_NO_RETURN\\s+(void)\\s+NAPI_CDECL\\s+${functionName}\\s*\\(([^)]*)\\)`, 'g')
  ];

  for (const pattern of patterns) {
    pattern.lastIndex = 0; // Reset regex state
    const match = pattern.exec(cleanContent);
    if (match) {
      const returnType = match[1].trim();
      const paramsStr = match[2].trim();
      
      // Parse parameters
      const args = [];
      if (paramsStr && paramsStr !== 'void') {
        // Split by comma, but be careful with function pointers and nested types
        const params = [];
        let current = '';
        let parenCount = 0;
        let i = 0;
        
        while (i < paramsStr.length) {
          const char = paramsStr[i];
          if (char === '(') {
            parenCount++;
          } else if (char === ')') {
            parenCount--;
          } else if (char === ',' && parenCount === 0) {
            params.push(current.trim());
            current = '';
            i++;
            continue;
          }
          current += char;
          i++;
        }
        if (current.trim()) {
          params.push(current.trim());
        }

        for (const param of params) {
          const cleanParam = param.trim();
          if (!cleanParam) continue;
          
          // Handle function pointer parameters like "napi_callback cb"
          if (cleanParam.includes('(') && cleanParam.includes(')')) {
            // Function pointer parameter - extract the parameter name
            const fpMatch = cleanParam.match(/(\w+)\s*$/);
            if (fpMatch) {
              args.push({
                name: fpMatch[1],
                type: cleanParam.replace(/\s+\w+\s*$/, '').trim()
              });
            }
          } else {
            // Regular parameter: type name or type* name or const type* name
            // Match patterns like: "const char*", "napi_env env", "size_t* argc", etc.
            const paramMatch = cleanParam.match(/^(.+?)\s+(\w+)\s*$/);
            if (paramMatch) {
              args.push({
                name: paramMatch[2],
                type: paramMatch[1].trim()
              });
            } else {
              // Fallback - try to extract from complex patterns
              const parts = cleanParam.split(/\s+/);
              if (parts.length >= 2) {
                const lastPart = parts[parts.length - 1];
                // Check if last part looks like a parameter name (not a type modifier)
                if (/^[a-zA-Z_]\w*$/.test(lastPart) && !['const', 'unsigned', 'signed'].includes(lastPart)) {
                  const name = lastPart;
                  const type = parts.slice(0, -1).join(' ');
                  args.push({ name, type });
                } else {
                  // If we can't parse it properly, use a fallback
                  console.warn(`Warning: Could not parse parameter: ${cleanParam} for function ${functionName}`);
                  args.push({
                    name: 'param' + args.length,
                    type: cleanParam
                  });
                }
              }
            }
          }
        }
      }

      return {
        args,
        return: returnType
      };
    }
  }

  return null;
}

// Read header files
function readHeaderFiles() {
  const jsNativeApiPath = path.join(include_dir, 'js_native_api.h');
  const nodeApiPath = path.join(include_dir, 'node_api.h');
  
  const jsNativeApi = fs.readFileSync(jsNativeApiPath, 'utf8');
  const nodeApi = fs.readFileSync(nodeApiPath, 'utf8');
  
  return jsNativeApi + '\n' + nodeApi;
}

// Generate napiFunctions from symbols and headers
function generateNapiFunctions() {
  const headerContent = readHeaderFiles();
  const napiFunctions = {};

  // Process each version
  for (let version = 1; version <= 9; version++) {
    if (!symbols[`v${version}`]) continue;
    
    napiFunctions[version] = {};
    
    // Combine js_native_api_symbols and node_api_symbols
    const allSymbols = [
      ...symbols[`v${version}`].js_native_api_symbols,
      ...symbols[`v${version}`].node_api_symbols
    ];

    for (const functionName of allSymbols) {
      const signature = parseFunctionSignature(headerContent, functionName);
      if (signature) {
        napiFunctions[version][functionName] = signature;
      } else {
        // Fallback for functions not found in headers
        console.warn(`Warning: Could not parse signature for ${functionName}`);
        napiFunctions[version][functionName] = {
          args: [{ name: 'env', type: 'napi_env' }],
          return: 'napi_status'
        };
      }
    }
  }

  return napiFunctions;
}

// Generate napiFunctions dynamically
const napiFunctions = generateNapiFunctions();

// Helper functions
function getFunctionsUpToVersion(targetVersion) {
  let allFunctions = [];
  for (let version = 1; version <= targetVersion; version++) {
    if (napiFunctions[version]) {
      allFunctions = allFunctions.concat(Object.keys(napiFunctions[version]));
    }
  }
  return [...new Set(allFunctions)];
}

function getParamsForFunction(funcName) {
  for (let version = 1; version <= 9; version++) {
    if (napiFunctions[version] && napiFunctions[version][funcName]) {
      return napiFunctions[version][funcName];
    }
  }
  return null;
}

function solveParams(type, argName = null) {
  if (type.includes('(')) {
    return argName ? type.replace(/\(\*\w*\)/, `(*${argName})`) : type.replace(/\(\*\w*\)/, '(*)');
  }
  return argName ? `${type} ${argName}` : type;
}

function generateFunctionPointers(functions) {
  return functions.map(func => {
    const funcInfo = getParamsForFunction(func);
    if (!funcInfo) return `static napi_status (*${func}_ptr)(napi_env env, ...) = NULL;`;
    
    const paramStr = funcInfo.args.map(p => solveParams(p.type)).join(', ');
    return `static ${funcInfo.return} (*${func}_ptr)(${paramStr}) = NULL;`;
  }).join('\n');
}

function generateFunctionEntries(functions) {
  const entries = functions.map(func => `    {"${func}", (void**)&${func}_ptr}`);
  return entries.join(',\n') + ',\n    {NULL, NULL} // END MARK';
}

function generateWrapperFunctions(functions) {
  return functions.map(func => {
    const funcInfo = getParamsForFunction(func);
    
    if (!funcInfo) {
      return `
napi_status ${func}(napi_env env, ...) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!${func}_ptr) {
        return napi_generic_failure;
    }
    
    return ${func}_ptr(env);
}`;
    }

    const paramDef = funcInfo.args.map(p => solveParams(p.type, p.name)).join(', ');
    const paramCall = funcInfo.args.map(p => p.name).join(', ');
    
    const failureReturn = funcInfo.return === 'void' ? 'return;' : 'return napi_generic_failure;';
    const successCall = funcInfo.return === 'void' ? 
      `${func}_ptr(${paramCall});` : 
      `return ${func}_ptr(${paramCall});`;
    
    return `
${funcInfo.return} ${func}(${paramDef}) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        ${failureReturn}
    }
    
    if (!${func}_ptr) {
        ${failureReturn}
    }
    
    ${successCall}
}`;
  }).join('\n');
}

function generateDynamicLoadCode(napiVersion = 9) {
  const functions = getFunctionsUpToVersion(napiVersion);
  
  const template = `#ifdef _WIN32

#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <node_api.h>

#undef NAPI_EXTERN

${generateFunctionPointers(functions)}

static HMODULE napi_module_handle = NULL;
static int napi_functions_loaded = 0;

typedef struct {
    const char* name;
    void** func_ptr;
} napi_function_entry;

static napi_function_entry napi_functions[] = {
${generateFunctionEntries(functions)}
};

static int load_napi_functions_from_module(HMODULE module) {
    if (!module) return 0;
    
    int loaded_count = 0;
    for (int i = 0; napi_functions[i].name != NULL; i++) {
        FARPROC proc = GetProcAddress(module, napi_functions[i].name);
        if (proc) {
            *(napi_functions[i].func_ptr) = (void*)proc;
            loaded_count++;
        }
    }
    
    return loaded_count;
}

static int discover_and_load_napi_functions() {
    if (napi_functions_loaded) {
        return 1;
    }
    
    const char* module_names[] = {
        "node.exe",
        "node.dll",
        "electron.exe",
        "nw.exe",
        NULL
    };
    
    HMODULE current_process = GetModuleHandle(NULL);
    if (load_napi_functions_from_module(current_process) > 0) {
        napi_module_handle = current_process;
        napi_functions_loaded = 1;
        return 1;
    }
    
    for (int i = 0; module_names[i] != NULL; i++) {
        HMODULE module = GetModuleHandle(module_names[i]);
        if (module && load_napi_functions_from_module(module) > 0) {
            napi_module_handle = module;
            napi_functions_loaded = 1;
            return 1;
        }
    }
    
    HMODULE node_dll = LoadLibrary("node.dll");
    if (node_dll && load_napi_functions_from_module(node_dll) > 0) {
        napi_module_handle = node_dll;
        napi_functions_loaded = 1;
        return 1;
    }
    
    return 0;
}

${generateWrapperFunctions(functions)}

__attribute__((constructor))
static void init_napi_dynamic_load() {
    discover_and_load_napi_functions();
}

__attribute__((destructor))
static void cleanup_napi_dynamic_load() {
    for (int i = 0; napi_functions[i].name != NULL; i++) {
        *(napi_functions[i].func_ptr) = NULL;
    }
    
    napi_module_handle = NULL;
    napi_functions_loaded = 0;
}

int napi_dynamic_load_init() {
    return discover_and_load_napi_functions();
}

int napi_dynamic_load_is_loaded() {
    return napi_functions_loaded;
}

int napi_dynamic_load_get_function_count() {
    int count = 0;
    for (int i = 0; napi_functions[i].name != NULL; i++) {
        if (*(napi_functions[i].func_ptr) != NULL) {
            count++;
        }
    }
    return count;
}

int napi_dynamic_load_is_function_available(const char* function_name) {
    if (!function_name) return 0;
    
    for (int i = 0; napi_functions[i].name != NULL; i++) {
        if (strcmp(napi_functions[i].name, function_name) == 0) {
            return *(napi_functions[i].func_ptr) != NULL;
        }
    }
    
    return 0;
}

const char* napi_dynamic_load_get_version(void) {
    return "1.0.0 (NAPI v${napiVersion} compatible)";
}

int napi_dynamic_load_get_status_info(char* buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) {
        return -1;
    }
    
    char temp[2048] = {0};
    int len = 0;
    
    len += snprintf(temp + len, sizeof(temp) - len, 
        "NAPI Dynamic Load Status:\\n");
    len += snprintf(temp + len, sizeof(temp) - len, 
        "  Loaded: %s\\n", napi_functions_loaded ? "Yes" : "No");
    len += snprintf(temp + len, sizeof(temp) - len, 
        "  Module Handle: %p\\n", (void*)napi_module_handle);
    
    int total_functions = 0;
    int loaded_functions = 0;
    
    for (int i = 0; napi_functions[i].name != NULL; i++) {
        total_functions++;
        if (*(napi_functions[i].func_ptr) != NULL) {
            loaded_functions++;
        }
    }
    
    len += snprintf(temp + len, sizeof(temp) - len, 
        "  Functions: %d/%d loaded\\n", loaded_functions, total_functions);
    
    if (len >= buffer_size) {
        return -((int)buffer_size);
    }
    
    strcpy(buffer, temp);
    return len;
}

int napi_dynamic_load_reload(void) {
    napi_functions_loaded = 0;
    napi_module_handle = NULL;
    
    for (int i = 0; napi_functions[i].name != NULL; i++) {
        *(napi_functions[i].func_ptr) = NULL;
    }
    
    return discover_and_load_napi_functions();
}

__attribute__((visibility("default")))
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
  switch (fdwReason) {
    case DLL_PROCESS_ATTACH:
      napi_dynamic_load_init();
      break;
  }
  return TRUE;
}

#endif // _WIN32
`;

  return template;
}

const args = process.argv.slice(2);
const napiVersion = args[0] ? parseInt(args[0]) : 9;

if (napiVersion < 1 || napiVersion > 9) {
console.error('Error: Invalid NAPI version');
process.exit(1);
}

console.log(`Generating NAPI v${napiVersion} dynamic load code...`);
console.log(`Found ${Object.keys(napiFunctions).length} NAPI versions`);

const functions = getFunctionsUpToVersion(napiVersion);
console.log(`Processing ${functions.length} functions for NAPI v${napiVersion}`);

const code = generateDynamicLoadCode(napiVersion);
const outputPath = path.join(__dirname, '..', 'src', 'win_dynamic_load.c');

fs.writeFileSync(outputPath, code, 'utf8');
console.log(`Generated dynamic load code: ${outputPath}`);
