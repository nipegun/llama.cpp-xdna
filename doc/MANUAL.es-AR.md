# Manual

Este MANUAL también está disponible en:\
[en-GB](MANUAL.en-GB.md), [en-US](MANUAL.md), es-AR, [es-ES](MANUAL.es-ES.md)

Cómo se instala y se usa llama.cpp-xdna, con la NPU de un procesador AMD Ryzen AI o sin ella.

## Índice

1. [Qué hace](#1-qué-hace)
2. [Qué necesita](#2-qué-necesita)
3. [Instalación](#3-instalación)
4. [Modelos](#4-modelos)
5. [Cómo se usa](#5-cómo-se-usa)
6. [Qué va a la NPU](#6-qué-va-a-la-npu)
7. [Comprobar que la NPU funciona](#7-comprobar-que-la-npu-funciona)
8. [Ajustes del backend de la NPU](#8-ajustes-del-backend-de-la-npu)
9. [Velocidad](#9-velocidad)
10. [Varias computadoras](#10-varias-computadoras)
11. [El servidor web](#11-el-servidor-web)
12. [Cuando algo falla](#12-cuando-algo-falla)
13. [Herramientas para desarrolladores](#13-herramientas-para-desarrolladores)

## 1. Qué hace

llama.cpp-xdna es un fork de [llama.cpp](https://github.com/ggml-org/llama.cpp): los mismos programas (`llama-cli`, `llama-server`, `llama-tts`, `llama-bench`, `llama-perplexity`, `llama-quantize`…), los mismos archivos GGUF y tres backends:

- **XDNA**, escrito para este proyecto, para la NPU de los procesadores AMD Ryzen AI. Hoy funciona con la generación de NPU que AMD llama XDNA2 (8 columnas de 4 núcleos); se llama así para que pueda admitir también XDNA1 y XDNA3. Habla directamente con el controlador `amdxdna` de Linux (no hace falta XRT).
- **CPU**, con una biblioteca para cada generación de procesadores x86; al arrancar un programa se carga la mejor para la computadora.
- **RPC**, para repartir un modelo entre varias computadoras.

Los demás backends de llama.cpp se eliminaron.

Con NPU, el backend XDNA lee los prompts y genera el texto y la voz en ella (ver [Qué va a la NPU](#6-qué-va-a-la-npu)), y el procesador hace el resto. Sin NPU, o sin su controlador, el backend se queda fuera y todo funciona en el procesador, como en llama.cpp.

## 2. Qué necesita

Para usar la NPU:

- **Un procesador AMD Ryzen AI con NPU XDNA2** (Strix Point, Krackan Point…). Se ha desarrollado en un Ryzen AI 7 PRO 350 (Krackan Point, NPU `1022:17f0` revisión `0x20`).
- **Linux 6.14 o posterior con el controlador `amdxdna`** (`CONFIG_DRM_ACCEL_AMDXDNA`). Debian 13 trae Linux 6.12, que no lo tiene: hace falta un kernel más nuevo. La NPU aparece como `/dev/accel/accel0`.
- **El firmware de la NPU, versión 1.1.2.64 o posterior**, de [linux-firmware](https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/tree/amdnpu). Con el controlador de Linux 7.x y el firmware 1.0.0.63 que trae Debian 13, todas las órdenes de la NPU terminan en `abort`. Instalalo como `npu_7.sbin` en la carpeta de tu NPU (en esta, `/lib/firmware/amdnpu/17f0_10/npu_7.sbin`; el controlador busca primero ese nombre) y reiniciá.
- **Acceso a `/dev/accel/accel0`**: pertenece al grupo `render`. Agregá tu usuario a él (`usermod -aG render <usuario>`) y volvé a iniciar la sesión.
- **Memoria**: la NPU guarda sus propias copias de los pesos de un modelo, además del archivo (como mucho el 40 % de la memoria de la computadora, si se fija `GGML_XDNA_WEIGHTS_GB` o `GGML_XDNA_COPY_WEIGHTS`, o automáticamente hasta el 48 % / 72 % en modelos densos y el 76–78 % en mezclas de expertos cuando `MADV_DONTNEED` libera las páginas proyectadas de los tensores 2D y de los expertos 3D; un modelo cuyas copias no caben va en parte en la NPU y en parte en el procesador): en coma flotante por bloques para los prompts (1,125 bytes por peso, u opcionalmente en `Q6_K` 4-en-3 a 0,84375 bytes por peso / `Q8_0` mediante `GGML_XDNA_PROMPT_FORMAT`) y en el formato del archivo para generar texto y para los expertos enrutados (`Q4_K`, `Q5_K`, `Q5_1`/`Q5_0`, `Q6_K` y `Q8_0`). Qwen3.5 4B Q8_0 (4,3 GiB) y Qwen3.5 9B Q8_0 (9,1 GiB) mantienen ambas copias en la NPU; Qwen3.8 27B Q6_K (20,9 GiB) y Q8_0 (27,1 GiB) mantienen el 100 % de sus 64 capas en generación encadenada y transmiten al vuelo solo las capas de prompt que exceden el límite del 72 % (o mantienen estáticas sus 750 matrices en `45,0 GB` con `GGML_XDNA_PROMPT_FORMAT=q6k`); Qwen3.6 35B-A3B hasta Q8_0 (35,2 GiB) y Flint 40B-A2.5B hasta Q8_0 (39,4 GiB) mantienen el 100 % de sus capas de expertos en la NPU (sustituyendo en memoria a las páginas del archivo); y los modelos mayores que la memoria física como Qwen3.8 Flash Next (`qwen4exp`, 105,4 GiB `Q4_K_XL` con 61,5 GiB de RAM) mantienen en la NPU el 100 % de sus proyecciones 2D densas más todas las capas de expertos que caben (23/48 sin MTP, o 17/48 más la capa MTP `blk.48` con `--mtp 1`), leyendo el resto directamente desde `mmap` sin duplicarlos con `CPU_REPACK`. Tanto los estados recurrentes como las cachés KV en `F16` hasta 4 GiB viven en búferes compartidos del controlador (`XDNA_State`). La memoria que la BIOS reserva para la GPU integrada no está disponible para la NPU.

Para compilarlo: Debian 13 o Alpine 3.21 o posterior, conexión a Internet, unos 5 GB de disco y, para la interfaz web de `llama-server`, unos 4 GB de memoria libre.

## 3. Instalación

### Con el instalador

Como `root`:

```bash
# Debian 13
curl -fsSL https://raw.githubusercontent.com/nipegun/llama.cpp-xdna/main/deploy/install-update-reinstall-debian.sh \
  | bash -s -- --install

# Alpine Linux 3.21 o posterior
wget -qO- https://raw.githubusercontent.com/nipegun/llama.cpp-xdna/main/deploy/install-update-reinstall-alpine.sh \
  | sh -s -- --install
```

El instalador instala las dependencias de la compilación, descarga las fuentes más recientes, las compila con los backends CPU, RPC y XDNA y la interfaz web, y las instala en `/usr/local`: los programas, los backends (`libggml-*.so`) y los overlays de la NPU en `/usr/local/bin`, y las bibliotecas en `/usr/local/lib`. Registra lo que hace en `/root/llama.cpp-xdna-install.log`.

| Opción | Qué hace |
|---|---|
| `--install`, `--update` | Compila las fuentes más recientes y las instala (encima de la instalación anterior, si la hay) |
| `--reinstall` | Lo mismo, borrando antes los archivos de la instalación anterior |
| `--uninstall` | Borra los archivos que instaló |
| `--no-ui` | Sin la interfaz web de `llama-server` (entonces solo sirve su API, y no hace falta Node.js) |
| `--jobs <n>` | Trabajos en paralelo de la compilación (por defecto, los procesadores de la computadora) |
| `--source <carpeta>` | Compila una copia local de las fuentes en vez de descargarlas |

En Debian, la interfaz web necesita Node.js 22, y Debian 13 trae el 20: el instalador descarga el Node.js 22 oficial de nodejs.org en la carpeta de la compilación, lo comprueba con su suma SHA-256 y lo borra con la compilación. En Alpine usa el Node.js de Alpine, que es el 22 desde Alpine 3.21.

### A mano

Como root en Debian 13:

```bash
# 1. Las dependencias de la compilación y las de las herramientas de Python
apt-get -y update
apt-get -y install git build-essential cmake libssl-dev curl xz-utils python3-numpy python3-yaml python3-tqdm

# 2. Las fuentes
git clone https://github.com/nipegun/llama.cpp-xdna.git
cd llama.cpp-xdna

# 3. Node.js 22 para la interfaz web, comprobado y solo para esta compilación
curl -fsSLO https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt
NODE_TAR=$(grep -o 'node-v22[^ ]*-linux-x64\.tar\.xz' SHASUMS256.txt)
curl -fsSLO "https://nodejs.org/dist/latest-v22.x/$NODE_TAR"
grep " $NODE_TAR\$" SHASUMS256.txt | sha256sum -c -
tar -xJf "$NODE_TAR"
export PATH="$PWD/${NODE_TAR%.tar.xz}/bin:$PATH"

# 4. Compilar
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_RPC=ON -DGGML_XDNA=ON -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_TESTS_INSTALL=OFF
NODE_OPTIONS=--max-old-space-size=4096 cmake --build build -j "$(nproc)"

# 5. Instalar (programas, backends y overlays en /usr/local/bin, bibliotecas en /usr/local/lib)
cmake --install build
ldconfig
```

- Sin la interfaz web, salteá el paso 3 y configurá con `-DLLAMA_BUILD_UI=OFF`. La interfaz web necesita unos 3,2 GB de memoria para compilarse: `NODE_OPTIONS=--max-old-space-size=4096` deja que Node.js use hasta 4 GB.
- `-DGGML_XDNA=ON` compila el backend XDNA y sus herramientas de desarrollo, y pone los tres overlays de la NPU (`xdna2-gemm-bfp16.*`, `xdna2-gemv-chain.*` y `xdna2-moe.*`) junto a los programas. Vienen compilados en `ggml/src/ggml-xdna/overlays/`; para volver a compilarlos desde sus diseños, ver [CODE.es-AR.md](CODE.es-AR.md#7-puntos-de-extensión).
- En lugar de instalarlo, se puede copiar a otra computadora la carpeta `build/bin/` entera (programas, bibliotecas, backends y overlays).
- Los programas cargan los backends solo desde su propia carpeta (`/usr/local/bin`, o `build/bin/`), nunca desde la carpeta en la que se ejecutan.
- `python3-numpy`, `python3-yaml` y `python3-tqdm` solo hacen falta para las herramientas de Python de `gguf-py` (`gguf_dump.py`, `gguf_set_metadata.py`…), que leen y editan archivos GGUF.
- Los tests (`test-backend-ops`…) no se instalan; se ejecutan desde `build/bin/`.

## 4. Modelos

Seis arquitecturas:

- **`qwen35`**: Qwen3.5 0.8B, 2B, 4B, 9B y 27B (Qwen3.8 27B), con su capa de predicción de varios tokens (MTP), solo texto.
- **`qwen35moe`**: Qwen3.5 y 3.6 con mezcla de expertos (Qwen3.6 35B-A3B: 256 expertos, de los que cada token usa 8), solo texto. Sus expertos enrutados van en la NPU hasta el presupuesto de memoria (`GGML_XDNA_WEIGHTS_GB`), y su enrutador va en el procesador.
- **`qwen4exp`**: Qwen3.8 Flash Next (`512x56B` / `179,55B`: 48 capas híbridas `3:1` de atención lineal y atención dispersa por bloques QSA más 1 capa MTP, Hyper-Connections de 4 flujos, incrustaciones por capa PLE bajo demanda en `blk.1`, y 512 expertos de anchura oculta 640 con 10 expertos enrutados + 1 compartido por token), solo texto.
- **`granite`**: los modelos densos de IBM Granite 4.2 (Granite 4.2 3B, `ibm-granite/granite-4.2-3b-GGUF`).
- **`flint`**: Flint 40B-A2.5B, con capas con una ventana deslizante de 513 tokens y capas globales, sumideros de atención y mezcla de expertos (256 expertos, de los que cada token usa 6), solo texto. Sus expertos enrutados van en la NPU hasta el presupuesto de memoria, y su enrutador va en el procesador.
- **`qwen3tts`**: síntesis de voz Qwen3-TTS 12Hz a 24 kHz (`llama-tts`), junto con su proyector de audio (`mmproj-Qwen3-TTS-12Hz-*.gguf`), en sus tres variantes: **`Base`** (`0.6B` y `1.7B`, clonación instantánea de voz a partir de una muestra de audio con `--tts-speaker-file`), **`CustomVoice`** (`0.6B` y `1.7B`, 9 voces integradas — `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu`, `vivian` — elegidas con `--tts-voice`, más instrucciones de estilo o emoción con `--tts-instruct` en `1.7B`) y **`VoiceDesign`** (`1.7B`, creación de voces nuevas desde una descripción en texto con `--tts-instruct`).

Formatos: Q8_0, Q6_K, Q5_K_M, Q5_1, Q5_0 y Q4_K_M / Q4_K_XL, y también BF16, F16 y F32. Para generar texto, la NPU descomprime directamente en sus núcleos los pesos `Q4_K`, `Q5_K`, `Q6_K` y `Q8_0`, y para los expertos enrutados (`xdna2-moe`) descomprime también `Q5_1` (y `Q5_0` convertido sin pérdida a `Q5_1`) con teselas parciales exactas; los demás tipos van en coma flotante por bloques (8 bits y un exponente cada 8 valores), que cuesta más o menos la precisión de Q8_0. La caché KV tiene que ser F16 (la de por defecto) para que la atención vaya a la NPU.

## 5. Cómo se usa

Los programas son los de llama.cpp. Por ejemplo:

```bash
# chat en la terminal
llama-cli -m Qwen3.5-4B-Q4_K_M.gguf

# un servidor con su interfaz web en http://127.0.0.1:8080
llama-server -m Qwen3.5-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8080

# síntesis de voz a 24 kHz: clonar una voz (Base), crearla desde texto (VoiceDesign) o elegir una integrada (CustomVoice)
llama-tts -m Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf --mmproj mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --tts-speaker-file voz-ref.wav --tts-lang es -c 2048 -p "Hola, esta voz sale de la NPU." -o clonada.wav
llama-tts -m Qwen3-TTS-12Hz-1.7B-VoiceDesign-Q4_K_M.gguf --mmproj mmproj-Qwen3-TTS-12Hz-1.7B-VoiceDesign-Q8_0.gguf \
  --tts-instruct "Voz masculina grave, cálida y serena de narrador de documentales." \
  --tts-lang es -c 2048 -p "Hola, esta voz ha sido diseñada desde cero en la NPU." -o creada.wav
llama-tts -m Qwen3-TTS-12Hz-1.7B-CustomVoice-Q4_K_M.gguf --mmproj mmproj-Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf \
  --tts-voice serena --tts-instruct "Habla con entusiasmo y cercanía." \
  --tts-lang es -c 2048 -p "Hola, hablo con una voz integrada de Qwen3-TTS." -o integrada.wav

# velocidad de los prompts y de la generación
llama-bench -m Qwen3.5-4B-Q4_K_M.gguf -p 512 -n 64
```

No hay que indicar nada para que se use la NPU: el backend XDNA va primero cuando están la NPU, su controlador y los overlays. Flash attention está activada por defecto (`-fa auto`), y la atención en la NPU la necesita. `-t 8` fija los hilos del procesador (la parte del anfitrión del backend también los usa).

La primera vez que se usa un modelo, el backend convierte sus pesos para la NPU (unos segundos con un 4B). Los pesos del archivo no se copian: se quedan en su proyección en memoria (`mmap`), como en el procesador.

## 6. Qué va a la NPU

- **Prompts** (lotes de 40 tokens o más):
  - las multiplicaciones del modelo por sus pesos, con las normas, la SwiGLU, las sumas residuales y la convolución corta de la atención lineal de Qwen3.5 hechas al convertir sus filas o al leer sus resultados;
  - la atención (`FLASH_ATTN_EXT`), como dos multiplicaciones por cabeza de claves; en los modelos sin norma de q y k (Granite), a partir de 1.024 claves de contexto;
  - la recurrencia de las capas de atención lineal de Qwen3.5, en lotes de 128 tokens o más.
- **Con mezcla de expertos** (`qwen35moe`, `qwen4exp`, `flint`): todas las capas de expertos enrutados (`MUL_MAT_ID`) que caben en el presupuesto automático (40/40 en Qwen3.6 35B-A3B, 47/47 en Flint 40B-A2.5B y 23/48 — o 17/48 más la capa MTP `blk.48` al activar `--mtp` — en Qwen3.8 Flash Next `Q4_K_XL` con 61,5 GiB de RAM), junto con su SwiGLU, sus expertos compartidos, su reducción ponderada y su suma residual, corren en el overlay de expertos (`xdna2-moe`), y al generar texto también lo hacen las proyecciones de entrada de atención, Hyper-Connections y PLE, los bloques feed-forward densos iniciales y los logits (incluida la vista parcial `n_vocab_mtp`) con 0 cambios de overlay (o en el overlay encadenado cuando los expertos de 1 token corren en el procesador con `GGML_XDNA_MOE_MIN_TOKENS=2`); su enrutador discreto y la proyección de salida de la atención se quedan en el procesador.
- **Generar texto** (de 1 a 8 tokens a la vez): cada capa en una orden del overlay encadenado, con sus multiplicaciones, sus normas, su SwiGLU y sus sumas residuales; la última, con los logits.
- **Síntesis de voz** (`qwen3tts` + `mmproj`): el Talker de 28 capas y el predictor `code_gen` de 5 capas y 16 pasos por fotograma (con sus 15 cabezas de libro de códigos) se encadenan en `xdna2-gemv-chain` con 0 cambios de overlay entre fotogramas, y el vocoder `code2wav` ejecuta en `xdna2-gemm-bfp16` su pre-transformer de 8 capas, las proyecciones puntuales de ConvNeXt y todas las convoluciones DAC causales y traspuestas multicanal (1 cambio de overlay por bloque de 72 fotogramas); en los modelos `Base`, el codificador de hablante `ECAPA-TDNN` corre una sola vez en F32 exacto en el procesador.
- **En el procesador**: los lotes de 9 a 39 tokens (con ellos es más rápido), la recurrencia y la convolución de la atención lineal al generar texto, la RoPE, las normas de q y k, la atención de un token que se genera, las copias a la caché `XDNA_State` y la elección del token siguiente (`GGML_XDNA_FULL=1` reúne todo el grafo de cálculo en un único subgrafo XDNA con `splits = 1`, y `GGML_XDNA_FULL=2` lleva también la atención de 1 token al overlay de prompts).

## 7. Comprobar que la NPU funciona

```bash
# lo que dice el controlador de la NPU (columnas, firmware, relojes); no ejecuta nada en ella
llama-xdna-info

# los mensajes del backend: por qué no usa la NPU, o qué hizo, cuando se libera
llama-cli -m modelo.gguf -p "Hola" -n 16 -no-cnv -lv 4
llama-bench -m modelo.gguf -p 512 -n 0 -v
```

Sin NPU o si falta algo, el backend dice por qué con `-lv 4` (por ejemplo `XDNA: the NPU is not used: cannot read the overlay ...`). Al liberarse, informa de lo que hizo: las multiplicaciones y órdenes de la NPU y sus tiempos, las atenciones, las recurrencias, los expertos enrutados y cuántas veces cambió entre sus overlays.

## 8. Ajustes del backend de la NPU

Variables de entorno. Las habituales:

| Variable | Qué hace |
|---|---|
| `GGML_XDNA_DISABLE=1` | El backend no ofrece la NPU: todo funciona en el procesador |
| `GGML_XDNA_DEVICE` | El nodo de dispositivo de la NPU (`/dev/accel/accel0` por defecto) |
| `GGML_XDNA_OVERLAY_DIR` | La carpeta de los overlays (por defecto, la del programa; nunca la carpeta actual) |
| `GGML_XDNA_MIN_BATCH` | Tokens que necesita un lote para que sus multiplicaciones y su atención vayan a la NPU (40 por defecto) |
| `GGML_XDNA_COPY_WEIGHTS=1` | Copia los pesos del modelo a búferes del backend en vez de dejarlos en la proyección del archivo |
| `GGML_XDNA_NO_ATTN=1` | La atención de los prompts en el procesador |
| `GGML_XDNA_ATTN_SPLIT=0` o `1` | Q y K de la atención en uno o en dos trozos de coma flotante por bloques (por defecto, dos cuando q y k no van normalizados) |
| `GGML_XDNA_NO_GDN=1` | La recurrencia de los prompts en el procesador |
| `GGML_XDNA_NO_CHAIN=1` | Genera con el overlay de generación sin encadenar, si está |
| `GGML_XDNA_NO_GEMV=1` | Genera en el procesador |
| `GGML_XDNA_LINEAR=1` | La capa entera de atención lineal en la NPU al generar texto (funciona, a un 6,2 % del procesador al omitir vaciados redundantes) |
| `GGML_XDNA_PROMPT_FORMAT` | Formato de los pesos de prompt en la NPU: `auto` (por defecto `bfp16`, o `q6k` 4-en-3 bajo un límite estricto de `GGML_XDNA_WEIGHTS_GB`), `bfp16`, `q6k` (un 25 % menos de memoria estática) o `q8_0` |
| `GGML_XDNA_FULL=1` o `2` | Reclama todas las operaciones del grafo para ejecutar prompts y generación en un solo subgrafo XDNA (`splits = 1`); `2` envía también `FLASH_ATTN_EXT` de 1 token a `xdna2-gemm-bfp16` |
| `GGML_XDNA_NO_STATE_BUFFERS=1` | Los estados recurrentes y las cachés KV en memoria del procesador en vez de en búferes `XDNA_State` del controlador |
| `GGML_XDNA_WEIGHTS_GB` | La memoria, en GB, que pueden ocupar las copias de los pesos para la NPU (por defecto, el 40 % de la memoria de la computadora, o automáticamente hasta el 48 % / 72 % en modelos densos y el 76–78 % en mezclas de expertos cuando `MADV_DONTNEED` libera las páginas convertidas del archivo). Un modelo cuyas copias superan esos límites va en parte en la NPU y en parte en el procesador |
| `GGML_XDNA_NO_MOE=1` | Con mezcla de expertos, deja los expertos enrutados (`MUL_MAT_ID`) en el procesador |
| `GGML_XDNA_MOE_MIN_TOKENS` | Mínimo de tokens de un lote para que los expertos enrutados vayan a la NPU (1 por defecto; 2 deja los expertos de un token en el procesador y encadena el experto compartido en `xdna2-gemv-chain`) |
| `GGML_XDNA_NO_MOE_DENSE=1` | Con mezcla de expertos, deja en el procesador las proyecciones densas y los expertos compartidos al generar |
| `GGML_XDNA_MOE_ATTN_OUT=1` | Con mezcla de expertos, ejecuta también en la NPU `ssm_out` y `attn_output` de las capas de expertos al generar |
| `GGML_XDNA_MOE_GEMV=1` | Con mezcla de expertos, envía todas las multiplicaciones densas de la generación por `xdna2-gemv-chain` |
| `GGML_XDNA_EMULATE=1` | El procesador hace lo que haría la NPU, con sus formatos numéricos; no abre la NPU |

Para comparar (cada una desactiva una mejora): `GGML_XDNA_NO_FUSE`, `GGML_XDNA_NO_GLU`, `GGML_XDNA_NO_OVERLAP`, `GGML_XDNA_NO_PACK_GLU`, `GGML_XDNA_NO_GATED`, `GGML_XDNA_NO_GDN_FOLD`, `GGML_XDNA_NO_L2`, `GGML_XDNA_NO_PARK`, `GGML_XDNA_NO_CHAIN_FFN`, `GGML_XDNA_NO_CHAIN_NEXT`, `GGML_XDNA_NO_EARLY_DESCRIPTORS`, `GGML_XDNA_NO_USER_FLUSH`, `GGML_XDNA_NO_FILLS_FIRST`, `GGML_XDNA_NO_ATTN_OVERLAP`, `GGML_XDNA_NO_CONV`, `GGML_XDNA_NO_CONV_FOLD` y `GGML_XDNA_GDN_PARTS` (de 1 a 4 órdenes por recurrencia). Qué hace cada una está en [XDNA2.md](XDNA2.md) (en inglés).

## 9. Velocidad

Medido con `llama-bench -fa 1` en un Ryzen AI 7 PRO 350 (8 hilos), en tokens por segundo. La velocidad de la notebook cambia varios puntos de una hora a otra, así que solo se comparan bien las medidas tomadas una detrás de otra.

Qwen3.5 4B, generando (`-n 64`):

| Archivo | NPU | Procesador solo |
|---|---|---|
| Q4_K_M | 21,3 | 15,1 |
| Q5_K_M | 19,1 | 12,9 |
| Q6_K | 18,0 | 12,3 |
| Q8_0 | 14,9 | 10,8 |

Modelos densos y de mezcla de expertos en `Q8_0`, `Q6_K` y `Q4_K_XL` (`llama-bench -p 512 -n 64 -r 3 --delay 3 -t 8 -fa 1` en `pp512` / `tg64`, y `llama-speculative-simple -n 64 --temp 0 -t 8 -fa on --mtp 1 --runs 5 --prompt-cycle` en `--mtp 1` para los modelos con capa MTP):

| Modelo | Prompt de 512 (`pp512`), NPU / procesador | Generación (`tg64`), NPU / procesador | Generación (`--mtp 1`), NPU / procesador |
|---|---|---|---|
| Qwen3.5 0.8B `Q8_0` | 4.459,1 / 720,2 (6,19×) | 65,57 / 58,35 (1,12×) | 61,34 / 64,68 (0,95×) |
| Qwen3.5 2B `Q8_0` | 2.947,6 / 272,8 (10,81×) | 31,30 / 23,36 (1,34×) | 36,71 / 27,93 (1,31×) |
| Granite 4.2 3B `Q8_0` | 1.367,3 / 115,7 (11,82×) | 18,53 / 13,54 (1,37×) | — |
| Qwen3.5 4B `Q8_0` | 1.650,8 / 100,7 (16,40×) | 14,90 / 10,82 (1,38×) | 18,55 / 14,09 (1,32×) |
| Qwen3.5 9B `Q8_0` | 1.190,3 / 51,5 (23,12×) | 7,42 / 5,61 (1,32×) | 12,55 / 7,96 (1,58×) |
| Qwen3.8 27B `Q6_K` (64/64 encadenado, prompt híbrido estático+transmitido) | 210,2 / 36,2 (5,81×) | 3,02 / 2,15 (1,40×) | 4,25 / 3,27 (1,30×) |
| Qwen3.6 35B-A3B `Q6_K` (40/40 capas de expertos) | 392,4 / 182,0 (2,16×) | 18,09 / 15,73 (1,15×) | 18,05 / 16,93 (1,07×) |
| Flint 40B-A2.5B `Q6_K` (47/47 de expertos + 2 densas) | 311,1 / 168,5 (1,85×) | 19,12 / 18,92 (1,01×) | — |
| Qwen3.8 Flash Next `Q4_K_XL` (105,4 GiB en NVMe, 61,5 GiB RAM, 23/48 expertos) | 63,53 / 61,13 (1,04×) | 5,34 / 5,18 (1,03×) | 5,78 / 4,78 (1,21×) |

Con contexto en la caché (`-d`), Qwen3.5 4B Q4_K_M (`pp512`), la atención en la NPU y en el procesador:

| Contexto | Atención en el procesador | Atención en la NPU |
|---|---|---|
| ninguno | 1.089 | 1.149 |
| 1.024 tokens | 793 | 1.118 |
| 2.048 tokens | 696 | 1.048 |
| 8.192 tokens | 294 | 829 |

Qwen3-TTS 12Hz (`0.6B-Base`, `1.7B-Base`, `0.6B-CustomVoice`, `1.7B-CustomVoice` y `1.7B-VoiceDesign`): en `llama-bench` con `Q8_0` nativo (2026/11/20), el Talker lee un prompt de 512 tokens de 2.385 a 2.870 (`0.6B`, de 3,9 a 4,7× los 615,2 del procesador) y de 2.086 a 2.097 (`1.7B`, 9,6× los 218,1 del procesador), y genera de 94,2 a 94,5 (`0.6B` `Q8_0` frente a 89,0, hasta 105,2 en `Q4_K_M` y 1,89× en `BF16`) y de 42,3 a 42,5 (`1.7B` `Q8_0` frente a 30,1, 1,41×; hasta 57,9 en `Q4_K_M` y 2,55× en `BF16`). De extremo a extremo en `llama-tts` (72 fotogramas = 5,76 s de voz a 24 kHz), `0.6B Q8_0` sintetiza de `1,33` a `1,35×` el tiempo real (`4,15–4,27 s`), `1.7B-Base Q8_0` supera el tiempo real con `1,06×` (`5,45 s` frente a `6,91 s` antes de `Q8_0` nativo), y `1.7B-CustomVoice`/`VoiceDesign Q8_0` alcanzan `0,98×` en frío (`5,86–5,90 s`, de `1,07` a `1,08×` tras el prompt).

Los prompts de 40 tokens o más van más rápido en la NPU (con 64, Qwen3.5 4B va a 265 frente a 170). La GPU integrada del mismo procesador (Radeon 860M, con el backend Vulkan de llama.cpp original) leía un prompt de 512 tokens del 4B a unos 340 tokens por segundo y generaba hasta a 23,9 (medido el 2026/11/15).

## 10. Varias computadoras

El backend RPC reparte un modelo entre varias computadoras: arrancá un `ggml-rpc-server` en cada computadora extra y agregala con `--rpc`.

Arrancá el servidor con un token en `GGML_RPC_TOKEN` y dale el mismo token al cliente: entonces el servidor solo atiende a los clientes que lo conocen, y el token nunca viaja por la red. El tráfico no va cifrado, así que cualquiera que pueda verlo puede leer los prompts y las respuestas: usá el servidor RPC solo en una red de confianza o a través de un túnel SSH. Por ejemplo:

```bash
# en la otra computadora (el token se crea una vez, por ejemplo con: openssl rand -hex 32)
GGML_RPC_TOKEN='<token>' ggml-rpc-server -H 0.0.0.0 -p 50052

# en el cliente
GGML_RPC_TOKEN='<token>' llama-server -m modelo.gguf --rpc 192.168.1.20:50052 -ngl 99
```

Sin `GGML_RPC_TOKEN`, cualquier cliente que llegue al servidor puede usarlo, así que dejalo en `127.0.0.1` (el valor por defecto) salvo en una red de confianza.

Las capas se reparten entre los dispositivos (`-ts` fija la proporción), así que un modelo que no cabe en una computadora puede caber en varias, pero cada token pasa por todos uno detrás de otro, y el más lento marca la velocidad.

Con Qwen3.5, `-sm tensor` (experimental) reparte cada capa entre los servidores, para que trabajen a la vez, y los servidores suman entre ellos sus resultados parciales. Arrancá un `ggml-rpc-server` por dispositivo, también el local, y usá solo esos:

```bash
GGML_RPC_COMM_ENDPOINT=127.0.0.1:51052 ggml-rpc-server -H 127.0.0.1 -p 50052 &
GGML_RPC_COMM_ENDPOINT=127.0.0.1:51052 ggml-rpc-server -H 127.0.0.1 -p 50053 &
llama-server -m modelo.gguf --rpc 127.0.0.1:50052,127.0.0.1:50053 -dev RPC0,RPC1 -sm tensor -ngl 99
```

La comunicación entre servidores está desactivada salvo que `GGML_RPC_COMM_ENDPOINT` tenga, en todos los servidores que participan, la misma dirección IPv4 y el mismo puerto del rango cero. Usá el primer servidor RPC de la lista del cliente, con su puerto más 1000 (`51052` arriba). La dirección tiene que ser una IPv4 literal, alcanzable desde los demás servidores y en la que el rango cero pueda escuchar; se rechazan los nombres DNS y las direcciones comodín. El rango cero solo escucha en esa dirección, y los demás rangos solo se conectan a ella. El reparto normal por capas no necesita este ajuste. Granite no admite `-sm tensor`.

El cliente pregunta a cada servidor qué operaciones puede ejecutar su dispositivo, y ejecuta las demás en otro dispositivo.

## 11. El servidor web

`llama-server` solo deja usar su API desde un navegador a las páginas de `localhost` y a su propia interfaz web (`--cors-origins localhost`; el llama.cpp oficial lo permite a cualquier sitio web): para usarla desde una aplicación web en otra dirección, agregá esa dirección con `--cors-origins`. Además comprueba la cabecera HTTP `Host` antes de servir cualquier ruta: se admiten localhost, la dirección local del socket (también una IP de la red local) y los nombres dados con `--host`. Para un nombre DNS público, o un proxy inverso que lo conserve, agregá `--allowed-hosts llama.example.com` (solo nombres, separados por comas). Los ajustes de CORS no desactivan la comprobación de `Host`. Más en [SECURITY.md](../SECURITY.md).

## 12. Cuando algo falla

- **El backend no usa la NPU**: ejecutá el programa con `-lv 4` y leé la línea `XDNA: the NPU is not used: ...`. Lo habitual: no hay `/dev/accel/accel0` (no hay controlador, o el kernel es anterior a 6.14), no hay permiso sobre él (el grupo `render`) o no están los overlays junto al programa (o en `GGML_XDNA_OVERLAY_DIR`).
- **Todas las órdenes terminan en `abort`**: el firmware de la NPU es demasiado antiguo para el controlador (ver [Qué necesita](#2-qué-necesita)).
- **La NPU se cuelga**: una orden que nunca termina deja la NPU inutilizable (sus consultas dan `EINVAL` y crear un contexto da `ENODEV`) hasta que se apaga y se vuelve a prender la computadora. En ese estado no descargues el controlador (`rmmod amdxdna`): con la NPU así se queda bloqueado para siempre dentro del kernel, y entonces la computadora tampoco se apaga (hay que mantener apretado el botón de encendido). `llama-xdna-run --pdi <overlay> --insts <flujo> --repeat 0` comprueba la NPU sin ejecutar nada: solo crea un contexto. El backend nunca vuelve a enviar una orden después de una que no terminó.
- **Falta memoria**: las copias de los pesos para la NPU van en búferes del controlador (ver [Qué necesita](#2-qué-necesita)). Ayuda un formato menor, un modelo más pequeño o menos memoria reservada en la BIOS para la GPU integrada.
- **Para comparar con el procesador**: `GGML_XDNA_DISABLE=1` (o `--no-xdna` en `llama-bench` y `llama-speculative-simple`) lo ejecuta todo en él.

## 13. Herramientas para desarrolladores

Se compilan con `-DGGML_XDNA=ON`:

| Herramienta | Qué hace |
|---|---|
| `llama-xdna-info` | Lo que dice el controlador de la NPU; `--check-buffers` crea y sincroniza un búfer de cada tipo. No ejecuta nada en la NPU |
| `llama-xdna-run` | Ejecuta un diseño compilado con mlir-aie (`--pdi`, `--insts`, `--in`, `--out`) y lo cronometra |
| `llama-xdna-insts` | Escribe e imprime los flujos de instrucciones que van a la NPU con una orden; no toca la NPU |
| `llama-xdna-gemm` | Multiplica matrices aleatorias en el overlay de los prompts y compara los resultados con el procesador; `gdn`, la recurrencia de un prompt |
| `llama-xdna-gemv` | Lo mismo con el overlay de generación (`--ffn`, `--gdn`, `--la`) y el de expertos (`--moe`) |
| `llama-xdna-probe` | Lo que cuestan una orden, una cadena de órdenes y un cambio de overlay |

Cómo está hecho el backend está en [CODE.es-AR.md](CODE.es-AR.md), y las notas técnicas sobre la NPU, con todas las mediciones, en [XDNA2.md](XDNA2.md) (en inglés).
